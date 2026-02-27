//! Pluggable vector storage backends for HNSW index.
//!
//! Decouples the HNSW graph structure from where vector data lives:
//! - `InMemoryVectorStorage`: HashMap in RAM (testing, small datasets)
//! - `LsmVectorStorage`: LSM engine → S3 (production, unlimited scale)

use lsm_engine::{LsmConfig, LsmStore};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::runtime::Runtime;

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

/// Stores vectors in the LSM engine, persisted to S3.
///
/// Vectors are stored as raw f32 byte arrays with keys like `v:{id}`.
/// Uses a dedicated LsmStore separate from user table data.
pub struct LsmVectorStorage {
    store: Arc<LsmStore>,
    runtime: Arc<Runtime>,
}

impl LsmVectorStorage {
    /// Create a new LSM-backed vector storage.
    ///
    /// Opens (or creates) an LsmStore at the given path within object storage.
    pub fn new(config: LsmConfig, runtime: Arc<Runtime>) -> Result<Self, String> {
        let store = runtime
            .block_on(async { LsmStore::open(config).await })
            .map_err(|e| format!("Failed to open vector store: {}", e))?;

        Ok(Self {
            store: Arc::new(store),
            runtime,
        })
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
            .map_err(|e| format!("Vector store error: {}", e))
    }

    fn load(&self, id: u64) -> Result<Vec<f32>, String> {
        let key = Self::key(id);
        let bytes = self
            .runtime
            .block_on(async { self.store.get(&key).await })
            .map_err(|e| format!("Vector load error: {}", e))?
            .ok_or_else(|| format!("Vector {} not found", id))?;
        Self::decode_vector(&bytes)
    }

    fn batch_load(&self, ids: &[u64]) -> Result<Vec<(u64, Vec<f32>)>, String> {
        let store = self.store.clone();
        let ids = ids.to_vec();

        self.runtime.block_on(async {
            let futs: Vec<_> = ids.iter().map(|&id| {
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

            Ok(futures::future::join_all(futs).await
                .into_iter()
                .flatten()
                .collect())
        })
    }

    fn delete(&self, id: u64) -> Result<(), String> {
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
}
