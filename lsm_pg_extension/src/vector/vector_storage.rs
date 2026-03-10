//! Pluggable vector storage backends for HNSW index.
//!
//! Decouples the HNSW graph structure from where vector data lives:
//! - `InMemoryVectorStorage`: HashMap in RAM (testing, small datasets)
//! - `LsmVectorStorage`: LSM engine → S3 (production, unlimited scale)

use lsm_engine::{LsmConfig, LsmStore};
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::runtime::Runtime;

// ─────────────────────────────────────────────────────────────────────
// LRU Vector Cache
// ─────────────────────────────────────────────────────────────────────

struct VectorCache {
    inner: RwLock<VectorCacheInner>,
    capacity_bytes: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

struct VectorCacheInner {
    map: HashMap<u64, Vec<f32>>,
    order: VecDeque<u64>,
    current_bytes: usize,
}

impl VectorCache {
    fn new(capacity_bytes: usize) -> Self {
        Self {
            inner: RwLock::new(VectorCacheInner {
                map: HashMap::new(),
                order: VecDeque::new(),
                current_bytes: 0,
            }),
            capacity_bytes,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    fn get(&self, id: u64) -> Option<Vec<f32>> {
        let mut inner = self.inner.write();
        if let Some(vec) = inner.map.get(&id).cloned() {
            self.hits.fetch_add(1, Ordering::Relaxed);
            promote_id(&mut inner.order, id);
            Some(vec)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    fn insert(&self, id: u64, vector: &[f32]) {
        let entry_bytes = vector.len() * 4 + 32; // f32 data + overhead
        if entry_bytes > self.capacity_bytes {
            return;
        }

        let mut inner = self.inner.write();

        if let Some(old) = inner.map.get(&id) {
            inner.current_bytes -= old.len() * 4 + 32;
            promote_id(&mut inner.order, id);
        } else {
            inner.order.push_front(id);
        }

        while inner.current_bytes + entry_bytes > self.capacity_bytes {
            if let Some(victim) = inner.order.pop_back() {
                if let Some(evicted) = inner.map.remove(&victim) {
                    inner.current_bytes -= evicted.len() * 4 + 32;
                }
            } else {
                break;
            }
        }

        inner.current_bytes += entry_bytes;
        inner.map.insert(id, vector.to_vec());
    }

    fn remove(&self, id: u64) {
        let mut inner = self.inner.write();
        if let Some(vec) = inner.map.remove(&id) {
            inner.current_bytes -= vec.len() * 4 + 32;
            if let Some(pos) = inner.order.iter().position(|&k| k == id) {
                inner.order.remove(pos);
            }
        }
    }

    fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }
}

fn promote_id(order: &mut VecDeque<u64>, id: u64) {
    if let Some(pos) = order.iter().position(|&k| k == id) {
        order.remove(pos);
    }
    order.push_front(id);
}

/// Trait for storing and retrieving vectors by node ID.
///
/// Implementations must be Send + Sync for use in concurrent HNSW operations.
pub trait VectorStorage: Send + Sync {
    /// Store a vector with the given ID.
    fn store(&self, id: u64, vector: &[f32]) -> Result<(), String>;

    /// Load a vector by ID.
    fn load(&self, id: u64) -> Result<Vec<f32>, String>;

    /// Batch-load multiple vectors by ID.
    /// Returns (id, vector) pairs for all found vectors.
    /// Missing IDs are silently skipped.
    fn batch_load(&self, ids: &[u64]) -> Result<Vec<(u64, Vec<f32>)>, String>;

    /// Delete a vector by ID.
    fn delete(&self, id: u64) -> Result<(), String>;
}

// ─────────────────────────────────────────────────────────────────────
// In-Memory Backend (testing)
// ─────────────────────────────────────────────────────────────────────

/// Stores all vectors in a HashMap in RAM.
/// Used for testing and small datasets.
pub struct InMemoryVectorStorage {
    vectors: RwLock<HashMap<u64, Vec<f32>>>,
}

impl InMemoryVectorStorage {
    pub fn new() -> Self {
        Self {
            vectors: RwLock::new(HashMap::new()),
        }
    }

    /// Return a clone of all stored vectors (for snapshotting).
    pub fn all_vectors(&self) -> HashMap<u64, Vec<f32>> {
        self.vectors.read().clone()
    }

    /// Bulk-insert vectors from a snapshot into storage.
    pub fn restore_vectors(&self, vectors: &HashMap<u64, Vec<f32>>) {
        let mut store = self.vectors.write();
        for (id, vec) in vectors {
            store.insert(*id, vec.clone());
        }
    }
}

impl VectorStorage for InMemoryVectorStorage {
    fn store(&self, id: u64, vector: &[f32]) -> Result<(), String> {
        self.vectors.write().insert(id, vector.to_vec());
        Ok(())
    }

    fn load(&self, id: u64) -> Result<Vec<f32>, String> {
        self.vectors
            .read()
            .get(&id)
            .cloned()
            .ok_or_else(|| format!("Vector {} not found", id))
    }

    fn batch_load(&self, ids: &[u64]) -> Result<Vec<(u64, Vec<f32>)>, String> {
        let vectors = self.vectors.read();
        Ok(ids
            .iter()
            .filter_map(|&id| vectors.get(&id).map(|v| (id, v.clone())))
            .collect())
    }

    fn delete(&self, id: u64) -> Result<(), String> {
        self.vectors.write().remove(&id);
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// LSM Engine Backend (production — S3-backed)
// ─────────────────────────────────────────────────────────────────────

/// Default vector cache size: 256 MB.
const DEFAULT_VECTOR_CACHE_BYTES: usize = 256 * 1024 * 1024;

/// Stores vectors in the LSM engine, persisted to S3.
///
/// An LRU cache sits in front of the LSM store so that hot vectors
/// (e.g. frequently-visited HNSW neighbours) are served from RAM
/// without any `block_on` / object-store overhead.
pub struct LsmVectorStorage {
    store: Arc<LsmStore>,
    runtime: Arc<Runtime>,
    cache: Arc<VectorCache>,
}

impl LsmVectorStorage {
    /// Create a new LSM-backed vector storage with default cache size.
    pub fn new(config: LsmConfig, runtime: Arc<Runtime>) -> Result<Self, String> {
        Self::with_cache(config, runtime, DEFAULT_VECTOR_CACHE_BYTES)
    }

    /// Create a new LSM-backed vector storage with explicit cache capacity.
    pub fn with_cache(
        config: LsmConfig,
        runtime: Arc<Runtime>,
        cache_bytes: usize,
    ) -> Result<Self, String> {
        let store = runtime
            .block_on(async { LsmStore::open(config).await })
            .map_err(|e| format!("Failed to open vector store: {}", e))?;

        Ok(Self {
            store: Arc::new(store),
            runtime,
            cache: Arc::new(VectorCache::new(cache_bytes)),
        })
    }

    /// Cache hit / miss stats for diagnostics.
    pub fn cache_stats(&self) -> (u64, u64) {
        (self.cache.hits(), self.cache.misses())
    }

    /// Flush the underlying LSM store's MemTable to object storage.
    pub fn flush(&self) -> Result<(), String> {
        self.runtime
            .block_on(async { self.store.flush().await })
            .map_err(|e| format!("Vector store flush error: {}", e))
    }

    /// Encode a vector ID as a storage key.
    fn key(id: u64) -> Vec<u8> {
        let mut k = Vec::with_capacity(2 + 8);
        k.extend_from_slice(b"v:");
        k.extend_from_slice(&id.to_be_bytes());
        k
    }

    /// Encode f32 slice as raw bytes (little-endian).
    fn encode_vector(vector: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(vector.len() * 4);
        for &val in vector {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Decode raw bytes back to Vec<f32>.
    fn decode_vector(bytes: &[u8]) -> Result<Vec<f32>, String> {
        if bytes.len() % 4 != 0 {
            return Err(format!(
                "Invalid vector bytes length: {} (not multiple of 4)",
                bytes.len()
            ));
        }

        let dim = bytes.len() / 4;
        let mut data = Vec::with_capacity(dim);
        for i in 0..dim {
            let offset = i * 4;
            let val = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            data.push(val);
        }
        Ok(data)
    }
}

impl VectorStorage for LsmVectorStorage {
    fn store(&self, id: u64, vector: &[f32]) -> Result<(), String> {
        let key = Self::key(id);
        let value = Self::encode_vector(vector);
        self.runtime
            .block_on(async { self.store.put(&key, &value).await })
            .map_err(|e| format!("Vector store error: {}", e))?;

        self.cache.insert(id, vector);
        Ok(())
    }

    fn load(&self, id: u64) -> Result<Vec<f32>, String> {
        if let Some(vec) = self.cache.get(id) {
            return Ok(vec);
        }

        let key = Self::key(id);
        let bytes = self
            .runtime
            .block_on(async { self.store.get(&key).await })
            .map_err(|e| format!("Vector load error: {}", e))?
            .ok_or_else(|| format!("Vector {} not found", id))?;
        let vec = Self::decode_vector(&bytes)?;

        self.cache.insert(id, &vec);
        Ok(vec)
    }

    fn batch_load(&self, ids: &[u64]) -> Result<Vec<(u64, Vec<f32>)>, String> {
        let mut results = Vec::with_capacity(ids.len());
        let mut miss_ids = Vec::new();

        for &id in ids {
            if let Some(vec) = self.cache.get(id) {
                results.push((id, vec));
            } else {
                miss_ids.push(id);
            }
        }

        if !miss_ids.is_empty() {
            let store = self.store.clone();
            let cache = self.cache.clone();

            let fetched: Vec<(u64, Vec<f32>)> = self.runtime.block_on(async {
                let futs: Vec<_> = miss_ids.iter().map(|&id| {
                    let store = store.clone();
                    async move {
                        let key = LsmVectorStorage::key(id);
                        match store.get(&key).await {
                            Ok(Some(bytes)) => LsmVectorStorage::decode_vector(&bytes)
                                .ok()
                                .map(|vec| (id, vec)),
                            _ => None,
                        }
                    }
                }).collect();

                futures::future::join_all(futs).await
                    .into_iter()
                    .flatten()
                    .collect()
            });

            for (id, vec) in &fetched {
                cache.insert(*id, vec);
            }
            results.extend(fetched);
        }

        Ok(results)
    }

    fn delete(&self, id: u64) -> Result<(), String> {
        self.cache.remove(id);
        let key = Self::key(id);
        self.runtime
            .block_on(async { self.store.delete(&key).await })
            .map_err(|e| format!("Vector delete error: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── InMemoryVectorStorage tests ──

    #[test]
    fn test_in_memory_store_load() {
        let storage = InMemoryVectorStorage::new();
        let vec = vec![1.0, 2.0, 3.0];
        storage.store(1, &vec).unwrap();
        let loaded = storage.load(1).unwrap();
        assert_eq!(loaded, vec);
    }

    #[test]
    fn test_in_memory_load_missing() {
        let storage = InMemoryVectorStorage::new();
        assert!(storage.load(999).is_err());
    }

    #[test]
    fn test_in_memory_batch_load() {
        let storage = InMemoryVectorStorage::new();
        storage.store(1, &[1.0, 0.0]).unwrap();
        storage.store(2, &[0.0, 1.0]).unwrap();
        storage.store(3, &[1.0, 1.0]).unwrap();

        let results = storage.batch_load(&[1, 2, 999]).unwrap();
        assert_eq!(results.len(), 2); // 999 is missing, silently skipped
        assert_eq!(results[0].0, 1);
        assert_eq!(results[1].0, 2);
    }

    #[test]
    fn test_in_memory_delete() {
        let storage = InMemoryVectorStorage::new();
        storage.store(1, &[1.0]).unwrap();
        storage.delete(1).unwrap();
        assert!(storage.load(1).is_err());
    }

    #[test]
    fn test_in_memory_overwrite() {
        let storage = InMemoryVectorStorage::new();
        storage.store(1, &[1.0]).unwrap();
        storage.store(1, &[2.0]).unwrap();
        assert_eq!(storage.load(1).unwrap(), vec![2.0]);
    }

    // ── LsmVectorStorage tests ──

    #[test]
    fn test_lsm_store_load() {
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        );
        let config = LsmConfig::in_memory("/test_vectors");
        let storage = LsmVectorStorage::new(config, rt).unwrap();

        let vec = vec![1.0, -2.5, 3.14, 0.0];
        storage.store(42, &vec).unwrap();
        let loaded = storage.load(42).unwrap();
        assert_eq!(loaded, vec);
    }

    #[test]
    fn test_lsm_load_missing() {
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        );
        let config = LsmConfig::in_memory("/test_vectors_missing");
        let storage = LsmVectorStorage::new(config, rt).unwrap();
        assert!(storage.load(999).is_err());
    }

    #[test]
    fn test_lsm_batch_load() {
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        );
        let config = LsmConfig::in_memory("/test_vectors_batch");
        let storage = LsmVectorStorage::new(config, rt).unwrap();

        storage.store(1, &[1.0, 0.0]).unwrap();
        storage.store(2, &[0.0, 1.0]).unwrap();

        let results = storage.batch_load(&[1, 2, 99]).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_lsm_delete() {
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        );
        let config = LsmConfig::in_memory("/test_vectors_del");
        let storage = LsmVectorStorage::new(config, rt).unwrap();

        storage.store(1, &[1.0, 2.0]).unwrap();
        storage.delete(1).unwrap();
        assert!(storage.load(1).is_err());
    }

    // ── Encode/decode tests ──

    #[test]
    fn test_encode_decode_roundtrip() {
        let original = vec![1.0_f32, -2.5, 3.14159, 0.0, f32::MAX, f32::MIN];
        let encoded = LsmVectorStorage::encode_vector(&original);
        let decoded = LsmVectorStorage::decode_vector(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_decode_invalid_length() {
        let bad_bytes = vec![1, 2, 3]; // Not a multiple of 4
        assert!(LsmVectorStorage::decode_vector(&bad_bytes).is_err());
    }

    // ── Vector Cache tests ──

    #[test]
    fn test_lsm_cache_hit_on_reload() {
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        );
        let config = LsmConfig::in_memory("/test_cache_hit");
        let storage = LsmVectorStorage::with_cache(config, rt, 1024 * 1024).unwrap();

        let vec = vec![1.0, 2.0, 3.0];
        storage.store(1, &vec).unwrap();

        let (hits_before, _) = storage.cache_stats();
        let loaded = storage.load(1).unwrap();
        assert_eq!(loaded, vec);
        let (hits_after, _) = storage.cache_stats();
        assert_eq!(hits_after, hits_before + 1, "Second load should hit cache");
    }

    #[test]
    fn test_lsm_cache_eviction() {
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        );
        let config = LsmConfig::in_memory("/test_cache_evict");
        // Tiny cache: only ~1 vector fits (3 f32s = 12 bytes + 32 overhead = 44 bytes)
        let storage = LsmVectorStorage::with_cache(config, rt, 80).unwrap();

        storage.store(1, &[1.0, 2.0, 3.0]).unwrap();
        storage.store(2, &[4.0, 5.0, 6.0]).unwrap();

        // Vector 1 should have been evicted (cache only holds one)
        let loaded2 = storage.load(2).unwrap();
        assert_eq!(loaded2, vec![4.0, 5.0, 6.0]);
        let (hits, _) = storage.cache_stats();
        assert!(hits >= 1, "Load of vector 2 should hit cache");

        // Vector 1 must fall through to LSM store
        let loaded1 = storage.load(1).unwrap();
        assert_eq!(loaded1, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_lsm_cache_invalidate_on_delete() {
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        );
        let config = LsmConfig::in_memory("/test_cache_delete");
        let storage = LsmVectorStorage::with_cache(config, rt, 1024 * 1024).unwrap();

        storage.store(1, &[1.0]).unwrap();
        let _ = storage.load(1).unwrap(); // populate cache
        storage.delete(1).unwrap();
        assert!(storage.load(1).is_err(), "Deleted vector should not be in cache");
    }
}
