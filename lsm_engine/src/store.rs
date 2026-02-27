//! The main LsmStore: a key-value store backed by object storage.
//!
//! Orchestrates MemTable, SSTable, Compaction, Manifest, and WAL to provide
//! a complete LSM-tree based storage engine with crash-safe durability.

use crate::block_cache::BlockCache;
use crate::compaction::{CompactionConfig, Compactor};
use crate::config::LsmConfig;
use crate::manifest::{Manifest, ManifestStore};
use crate::memtable::MemTable;
use crate::sstable::{SSTableBuilder, SSTableReader};
use crate::wal::{WalReader, WalWriter};
use crate::{KvEntry, LsmError, LsmResult, WriteOp};
use object_store::path::Path;
use object_store::ObjectStore;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex as AsyncMutex, Notify};

/// The main LSM key-value store.
///
/// Thread-safe and async-ready. Supports concurrent reads and serialized writes.
pub struct LsmStore {
    config: LsmConfig,

    /// Active (mutable) MemTable for current writes.
    active_memtable: Arc<RwLock<Arc<MemTable>>>,

    /// Frozen MemTables awaiting flush to object storage.
    frozen_memtables: Arc<RwLock<Vec<Arc<MemTable>>>>,

    /// The LSM manifest tracking all SSTables.
    manifest: Arc<RwLock<Manifest>>,

    /// Manifest persistence.
    manifest_store: Arc<ManifestStore>,

    /// Object store for SSTable I/O.
    object_store: Arc<dyn ObjectStore>,

    /// Root path in object storage.
    root_path: String,

    /// Flag indicating the store is open.
    is_open: Arc<AtomicBool>,

    /// Notify channel for flush requests.
    flush_notify: Arc<Notify>,

    /// Write-Ahead Log writer (None if WAL disabled).
    wal_writer: Option<Arc<AsyncMutex<WalWriter>>>,

    /// Handle to the background flush task.
    flush_handle: parking_lot::Mutex<Option<tokio::task::JoinHandle<()>>>,

    /// Handle to the background compaction task.
    compaction_handle: parking_lot::Mutex<Option<tokio::task::JoinHandle<()>>>,

    /// LRU cache for SSTable bytes fetched from object storage.
    block_cache: Arc<BlockCache>,
}

impl LsmStore {
    /// Open or create an LSM store at the given path in object storage.
    pub async fn open(config: LsmConfig) -> LsmResult<Self> {
        let manifest_store = Arc::new(ManifestStore::new(
            config.object_store.clone(),
            &config.root_path,
        ));

        let mut manifest = manifest_store.load_or_create().await?;

        // Fencing check: ensure no other writer has claimed this store
        if let Some(ref our_token) = config.fencing_token {
            if let Some(ref existing) = manifest.fencing_token {
                if existing != our_token {
                    return Err(LsmError::FencingViolation(format!(
                        "manifest fencing token '{}' does not match ours '{}'",
                        existing, our_token
                    )));
                }
            }
            manifest.fencing_token = Some(our_token.clone());
            manifest_store.save(&manifest).await?;
        }

        let mut start_seq = manifest.next_sequence;

        let active_memtable = Arc::new(RwLock::new(Arc::new(MemTable::new(start_seq))));
        let frozen_memtables: Arc<RwLock<Vec<Arc<MemTable>>>> = Arc::new(RwLock::new(Vec::new()));

        // WAL recovery: replay any unflushed WAL segments
        let mut wal_next_seg = 1u64;
        let mut discovered_segments = Vec::new();
        if config.wal_enabled {
            let wal_reader = WalReader::new(config.object_store.clone(), &config.root_path);
            let entries = wal_reader.replay_all().await?;
            if !entries.is_empty() {
                tracing::info!("WAL recovery: replaying {} entries", entries.len());
                let memtable = active_memtable.read().clone();
                for entry in &entries {
                    match &entry.op {
                        WriteOp::Put { key, value } => { memtable.put(key, value)?; }
                        WriteOp::Delete { key } => { memtable.delete(key)?; }
                    }
                    if entry.sequence >= start_seq {
                        start_seq = entry.sequence + 1;
                    }
                }
                tracing::info!("WAL recovery complete, next_seq={}", start_seq);
            }

            // Determine next WAL segment seq
            discovered_segments = wal_reader.discover_segments().await?;
            if let Some(&last) = discovered_segments.last() {
                wal_next_seg = last + 1;
            }
        }

        // Create WAL writer
        let wal_writer = if config.wal_enabled {
            let mut writer = WalWriter::new(
                config.object_store.clone(),
                &config.root_path,
                wal_next_seg,
                config.wal_segment_size,
            );
            for &seg in &discovered_segments {
                writer.add_discovered_segment(seg);
            }
            Some(Arc::new(AsyncMutex::new(writer)))
        } else {
            None
        };

        let manifest = Arc::new(RwLock::new(manifest));
        let is_open = Arc::new(AtomicBool::new(true));
        let flush_notify = Arc::new(Notify::new());
        let block_cache = Arc::new(BlockCache::new(config.block_cache_size));

        let store = Self {
            config: config.clone(),
            active_memtable: active_memtable.clone(),
            frozen_memtables: frozen_memtables.clone(),
            manifest: manifest.clone(),
            manifest_store: manifest_store.clone(),
            object_store: config.object_store.clone(),
            root_path: config.root_path.clone(),
            is_open: is_open.clone(),
            flush_notify: flush_notify.clone(),
            wal_writer: wal_writer.clone(),
            flush_handle: parking_lot::Mutex::new(None),
            compaction_handle: parking_lot::Mutex::new(None),
            block_cache,
        };

        // Start background flush task
        let flush_handle = {
            let _active_mt = active_memtable.clone();
            let frozen_mt = frozen_memtables.clone();
            let manifest_arc = manifest.clone();
            let ms = manifest_store.clone();
            let os = config.object_store.clone();
            let rp = config.root_path.clone();
            let open = is_open.clone();
            let notify = flush_notify.clone();
            let flush_interval = config.flush_interval;
            let block_size = config.block_size;
            let wal_w = wal_writer.clone();
            let compress = config.enable_compression;

            tokio::spawn(async move {
                loop {
                    // Wait for flush signal or timeout
                    tokio::select! {
                        _ = notify.notified() => {},
                        _ = tokio::time::sleep(flush_interval) => {},
                    }

                    if !open.load(Ordering::SeqCst) {
                        // Flush remaining on shutdown
                        let _ = Self::do_flush(
                            &frozen_mt, &manifest_arc, &ms, &os, &rp, block_size, &wal_w, compress
                        ).await;
                        break;
                    }

                    if let Err(e) = Self::do_flush(
                        &frozen_mt, &manifest_arc, &ms, &os, &rp, block_size, &wal_w, compress
                    ).await {
                        tracing::error!("Flush error: {}", e);
                    }
                }
            })
        };
        *store.flush_handle.lock() = Some(flush_handle);

        // Start background compaction task
        let compaction_handle = {
            let manifest_arc = manifest.clone();
            let ms = manifest_store.clone();
            let os = config.object_store.clone();
            let rp = config.root_path.clone();
            let open = is_open.clone();
            let compaction_config = CompactionConfig {
                l0_threshold: config.l0_compaction_threshold,
                block_size: config.block_size,
                target_sstable_size: config.sstable_target_size,
                level_size_multiplier: 10,
                l1_max_tables: 10,
                enable_compression: config.enable_compression,
            };

            tokio::spawn(async move {
                let compactor = Compactor::new(compaction_config, os, rp);
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

                    if !open.load(Ordering::SeqCst) {
                        break;
                    }

                    let manifest_snap = manifest_arc.read().clone();
                    if compactor.needs_compaction(&manifest_snap) {
                        // L0 → L1 compaction
                        match compactor.compact_l0(&manifest_snap).await {
                            Ok(Some(result)) => {
                                if let Err(e) = compactor.apply_result(&result).await {
                                    tracing::error!("Compaction apply error: {}", e);
                                    continue;
                                }
                                let manifest_snapshot = {
                                    let mut m = manifest_arc.write();
                                    m.apply_compaction(
                                        result.source_level,
                                        &result.source_ids,
                                        result.target_level,
                                        result.new_tables,
                                    );
                                    m.clone()
                                };
                                if let Err(e) = ms.save(&manifest_snapshot).await {
                                    tracing::error!("Manifest save error after compaction: {}", e);
                                }
                            }
                            Ok(None) => {}
                            Err(e) => {
                                tracing::error!("L0 compaction error: {}", e);
                            }
                        }

                        // Higher-level compactions (L1→L2, L2→L3, etc.)
                        for level in 1..6u32 {
                            let snap = manifest_arc.read().clone();
                            if compactor.needs_level_compaction(&snap, level as usize) {
                                match compactor.compact_level(&snap, level as usize).await {
                                    Ok(Some(result)) => {
                                        if let Err(e) = compactor.apply_result(&result).await {
                                            tracing::error!("L{} compaction apply error: {}", level, e);
                                            continue;
                                        }
                                        let manifest_snapshot = {
                                            let mut m = manifest_arc.write();
                                            m.apply_compaction(
                                                result.source_level,
                                                &result.source_ids,
                                                result.target_level,
                                                result.new_tables,
                                            );
                                            m.clone()
                                        };
                                        if let Err(e) = ms.save(&manifest_snapshot).await {
                                            tracing::error!("Manifest save error after L{} compaction: {}", level, e);
                                        }
                                    }
                                    Ok(None) => {}
                                    Err(e) => {
                                        tracing::error!("L{} compaction error: {}", level, e);
                                    }
                                }
                            }
                        }
                    }
                }
            })
        };
        *store.compaction_handle.lock() = Some(compaction_handle);

        tracing::info!("LSM store opened at {}", config.root_path);
        Ok(store)
    }

    /// Put a key-value pair.
    pub async fn put(&self, key: &[u8], value: &[u8]) -> LsmResult<()> {
        if !self.is_open.load(Ordering::SeqCst) {
            return Err(LsmError::StoreClosed);
        }

        // Allocate sequence under WAL lock to ensure WAL and MemTable agree.
        // This serializes writes through the WAL, which is necessary for
        // correct ordering anyway.
        let memtable = self.active_memtable.read().clone();

        if let Some(ref wal) = self.wal_writer {
            let mut wal_guard = wal.lock().await;
            let seq = memtable.next_sequence();
            let op = WriteOp::Put { key: key.to_vec(), value: value.to_vec() };
            wal_guard.append(&op, seq).await?;
            memtable.put(key, value)?;
        } else {
            memtable.put(key, value)?;
        }

        // Check if MemTable should be rotated
        if memtable.approximate_size() >= self.config.memtable_size_limit {
            self.rotate_memtable();
        }

        Ok(())
    }

    /// Get a value by key. Checks MemTable first, then SSTables (newest to oldest).
    pub async fn get(&self, key: &[u8]) -> LsmResult<Option<Vec<u8>>> {
        if !self.is_open.load(Ordering::SeqCst) {
            return Err(LsmError::StoreClosed);
        }

        // 1. Check active MemTable
        {
            let memtable = self.active_memtable.read().clone();
            if let Some(val) = memtable.get(key) {
                return Ok(val); // Some(Some(v)) → found, Some(None) → tombstone
            }
        }

        // 2. Check frozen MemTables (newest first)
        {
            let frozen = self.frozen_memtables.read();
            for mt in frozen.iter().rev() {
                if let Some(val) = mt.get(key) {
                    return Ok(val);
                }
            }
        }

        // 3. Check SSTables (L0 first, then L1, etc.)
        let manifest = self.manifest.read().clone();
        for level in &manifest.levels {
            // Within a level, check newest SSTables first
            for table_meta in level.iter().rev() {
                // Quick check: is key in range?
                if key < table_meta.min_key.as_slice() || key > table_meta.max_key.as_slice() {
                    continue;
                }

                let data = if let Some(cached) = self.block_cache.get(&table_meta.path) {
                    cached
                } else {
                    let path = Path::from(table_meta.path.clone());
                    match self.object_store.get(&path).await {
                        Ok(result) => {
                            let fetched = result.bytes().await
                                .map_err(|e| LsmError::ObjectStore(e))?;
                            self.block_cache.insert(&table_meta.path, fetched.clone());
                            fetched
                        }
                        Err(object_store::Error::NotFound { .. }) => continue,
                        Err(e) => return Err(LsmError::ObjectStore(e)),
                    }
                };
                let reader = SSTableReader::open(data)?;
                match reader.get(key)? {
                    Some(Some(val)) => return Ok(Some(val)),
                    Some(None) => return Ok(None),
                    None => {}
                }
            }
        }

        Ok(None)
    }

    /// Delete a key by writing a tombstone.
    pub async fn delete(&self, key: &[u8]) -> LsmResult<()> {
        if !self.is_open.load(Ordering::SeqCst) {
            return Err(LsmError::StoreClosed);
        }

        let memtable = self.active_memtable.read().clone();

        if let Some(ref wal) = self.wal_writer {
            let mut wal_guard = wal.lock().await;
            let seq = memtable.next_sequence();
            let op = WriteOp::Delete { key: key.to_vec() };
            wal_guard.append(&op, seq).await?;
            memtable.delete(key)?;
        } else {
            memtable.delete(key)?;
        }

        Ok(())
    }

    /// Scan entries in a key range [start, end).
    pub async fn scan(&self, start: &[u8], end: &[u8]) -> LsmResult<Vec<KvEntry>> {
        if !self.is_open.load(Ordering::SeqCst) {
            return Err(LsmError::StoreClosed);
        }

        let mut results = Vec::new();

        // Collect from active MemTable
        {
            let memtable = self.active_memtable.read().clone();
            results.extend(memtable.scan(start, end));
        }

        // Collect from frozen MemTables
        {
            let frozen = self.frozen_memtables.read();
            for mt in frozen.iter().rev() {
                results.extend(mt.scan(start, end));
            }
        }

        // Collect from SSTables
        let manifest = self.manifest.read().clone();
        for level in &manifest.levels {
            for table_meta in level.iter().rev() {
                if table_meta.max_key.as_slice() < start || table_meta.min_key.as_slice() >= end {
                    continue;
                }

                let data = if let Some(cached) = self.block_cache.get(&table_meta.path) {
                    cached
                } else {
                    let path = Path::from(table_meta.path.clone());
                    match self.object_store.get(&path).await {
                        Ok(result) => {
                            let fetched = result.bytes().await
                                .map_err(|e| LsmError::ObjectStore(e))?;
                            self.block_cache.insert(&table_meta.path, fetched.clone());
                            fetched
                        }
                        Err(object_store::Error::NotFound { .. }) => continue,
                        Err(e) => return Err(LsmError::ObjectStore(e)),
                    }
                };
                let reader = SSTableReader::open(data)?;
                results.extend(reader.scan(start, end)?);
            }
        }

        // Deduplicate: keep only the newest version of each key
        results.sort_by(|a, b| a.key.cmp(&b.key).then(b.sequence.cmp(&a.sequence)));
        results.dedup_by(|a, b| a.key == b.key);

        Ok(results)
    }

    /// Force a flush of the current MemTable to object storage.
    pub async fn flush(&self) -> LsmResult<()> {
        // Sync WAL before flush to ensure all entries are persisted
        if let Some(ref wal) = self.wal_writer {
            wal.lock().await.sync().await?;
        }

        self.rotate_memtable();
        self.flush_notify.notify_one();

        // Wait briefly for flush to complete
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Do a synchronous flush as well
        Self::do_flush(
            &self.frozen_memtables,
            &self.manifest,
            &self.manifest_store,
            &self.object_store,
            &self.root_path,
            self.config.block_size,
            &self.wal_writer,
            self.config.enable_compression,
        ).await
    }

    /// Close the store, flushing all pending data.
    pub async fn close(&self) -> LsmResult<()> {
        tracing::info!("Closing LSM store at {}", self.root_path);

        // Sync WAL
        if let Some(ref wal) = self.wal_writer {
            wal.lock().await.sync().await?;
        }

        // Rotate and flush
        self.rotate_memtable();
        Self::do_flush(
            &self.frozen_memtables,
            &self.manifest,
            &self.manifest_store,
            &self.object_store,
            &self.root_path,
            self.config.block_size,
            &self.wal_writer,
            self.config.enable_compression,
        ).await?;

        // Signal background tasks to stop
        self.is_open.store(false, Ordering::SeqCst);
        self.flush_notify.notify_one();

        // Wait for background tasks to finish
        if let Some(handle) = self.flush_handle.lock().take() {
            let _ = handle.await;
        }
        if let Some(handle) = self.compaction_handle.lock().take() {
            let _ = handle.await;
        }

        Ok(())
    }

    /// Rotate the active MemTable: freeze it and create a new one.
    fn rotate_memtable(&self) {
        let mut active = self.active_memtable.write();
        let current = active.clone();

        if current.is_empty() {
            return;
        }

        // The new MemTable must continue from the old one's sequence counter,
        // not the manifest's next_sequence (which is only updated on flush).
        let new_seq = current.current_sequence();

        current.freeze();

        let mut frozen = self.frozen_memtables.write();
        frozen.push(current);

        *active = Arc::new(MemTable::new(new_seq));

        self.flush_notify.notify_one();
    }

    /// Flush all frozen MemTables to SSTables on object storage.
    async fn do_flush(
        frozen_memtables: &Arc<RwLock<Vec<Arc<MemTable>>>>,
        manifest: &Arc<RwLock<Manifest>>,
        manifest_store: &Arc<ManifestStore>,
        object_store: &Arc<dyn ObjectStore>,
        root_path: &str,
        block_size: usize,
        wal_writer: &Option<Arc<AsyncMutex<WalWriter>>>,
        enable_compression: bool,
    ) -> LsmResult<()> {
        loop {
            // Take the oldest frozen MemTable
            let memtable = {
                let mut frozen = frozen_memtables.write();
                if frozen.is_empty() {
                    return Ok(());
                }
                frozen.remove(0)
            };

            if memtable.is_empty() {
                continue;
            }

            // Build SSTable from MemTable entries
            let entries = memtable.entries();
            let mut builder = SSTableBuilder::new(block_size, enable_compression);

            for (key, seq, value) in entries {
                builder.add(key, seq, value);
            }

            let (data, mut meta) = builder.build()?;

            // Determine path and save
            let next_seq = memtable.current_sequence();
            let sstable_path = format!("{}/L0/{}.sst", root_path, next_seq);
            meta.path = sstable_path.clone();

            let path = Path::from(sstable_path);
            object_store.put(&path, data.into()).await
                .map_err(|e| LsmError::ObjectStore(e))?;

            // Update manifest (scope the write guard so it's dropped before await)
            let manifest_snapshot = {
                let mut m = manifest.write();
                m.add_l0_sstable(meta);
                m.next_sequence = next_seq;
                m.clone()
            };

            // Persist manifest
            manifest_store.save(&manifest_snapshot).await?;

            // Trim WAL: after successful flush, old WAL segments are no longer needed
            if let Some(ref wal) = wal_writer {
                let mut w = wal.lock().await;
                let current = w.current_segment_seq();
                let old_segs: Vec<u64> = w.active_segments()
                    .iter()
                    .filter(|&&s| s < current)
                    .copied()
                    .collect();
                if !old_segs.is_empty() {
                    w.trim(&old_segs).await?;
                    tracing::info!("Trimmed {} WAL segments after flush", old_segs.len());
                }
            }

            tracing::info!("Flushed MemTable to L0 SSTable (seq={})", next_seq);
        }
    }

    /// Get store statistics.
    pub fn stats(&self) -> StoreStats {
        let manifest = self.manifest.read();
        let frozen_count = self.frozen_memtables.read().len();
        let active_size = self.active_memtable.read().approximate_size();

        StoreStats {
            active_memtable_size: active_size,
            frozen_memtable_count: frozen_count,
            l0_sstable_count: manifest.l0_count(),
            total_sstable_count: manifest.all_tables().len(),
            manifest_version: manifest.version,
            cache_hits: self.block_cache.hits(),
            cache_misses: self.block_cache.misses(),
            cache_hit_rate: self.block_cache.hit_rate(),
        }
    }
}

/// Store statistics for monitoring.
#[derive(Debug, Clone)]
pub struct StoreStats {
    pub active_memtable_size: usize,
    pub frozen_memtable_count: usize,
    pub l0_sstable_count: usize,
    pub total_sstable_count: usize,
    pub manifest_version: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_rate: f64,
}

impl std::fmt::Display for StoreStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MemTable: {}B | Frozen: {} | L0: {} | Total SSTables: {} | Manifest v{} | Cache: {}/{} ({:.1}%)",
            self.active_memtable_size,
            self.frozen_memtable_count,
            self.l0_sstable_count,
            self.total_sstable_count,
            self.manifest_version,
            self.cache_hits,
            self.cache_hits + self.cache_misses,
            self.cache_hit_rate * 100.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_store() -> LsmStore {
        let config = LsmConfig::in_memory("/test")
            .with_memtable_size(1024); // Small for testing
        LsmStore::open(config).await.unwrap()
    }

    #[tokio::test]
    async fn test_put_get() {
        let store = test_store().await;
        store.put(b"hello", b"world").await.unwrap();

        let val = store.get(b"hello").await.unwrap();
        assert_eq!(val, Some(b"world".to_vec()));

        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_get_missing() {
        let store = test_store().await;
        let val = store.get(b"missing").await.unwrap();
        assert_eq!(val, None);
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_delete() {
        let store = test_store().await;
        store.put(b"key", b"val").await.unwrap();
        store.delete(b"key").await.unwrap();

        let val = store.get(b"key").await.unwrap();
        assert_eq!(val, None);

        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_overwrite() {
        let store = test_store().await;
        store.put(b"key", b"v1").await.unwrap();
        store.put(b"key", b"v2").await.unwrap();

        let val = store.get(b"key").await.unwrap();
        assert_eq!(val, Some(b"v2".to_vec()));

        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_flush_and_read() {
        let store = test_store().await;
        store.put(b"key1", b"val1").await.unwrap();
        store.put(b"key2", b"val2").await.unwrap();

        // Force flush to object storage
        store.flush().await.unwrap();

        // Read should still work (from SSTable)
        let val = store.get(b"key1").await.unwrap();
        assert_eq!(val, Some(b"val1".to_vec()));

        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_scan() {
        let store = test_store().await;
        store.put(b"a", b"1").await.unwrap();
        store.put(b"b", b"2").await.unwrap();
        store.put(b"c", b"3").await.unwrap();
        store.put(b"d", b"4").await.unwrap();

        let entries = store.scan(b"b", b"d").await.unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].key, b"b".to_vec());
        assert_eq!(entries[1].key, b"c".to_vec());

        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_stats() {
        let store = test_store().await;
        let stats = store.stats();
        assert_eq!(stats.active_memtable_size, 0);
        assert_eq!(stats.l0_sstable_count, 0);

        store.put(b"key", b"val").await.unwrap();
        let stats = store.stats();
        assert!(stats.active_memtable_size > 0);

        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_wal_recovery() {
        // Use a shared object store to simulate crash + reopen
        let shared_store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());

        // Phase 1: Write data, then "crash" (drop without flush)
        {
            let config = LsmConfig {
                root_path: "/recovery_test".to_string(),
                object_store: shared_store.clone(),
                memtable_size_limit: 1024 * 1024, // Large so it doesn't auto-flush
                flush_interval: std::time::Duration::from_secs(3600), // No auto-flush
                l0_compaction_threshold: 4,
                sstable_target_size: 64 * 1024 * 1024,
                block_size: 4096,
                block_cache_size: 256 * 1024 * 1024,
                enable_compression: false,
                enable_bloom_filters: true,
                wal_enabled: true,
                wal_segment_size: 16 * 1024 * 1024,
                fencing_token: None,
            };

            let store = LsmStore::open(config).await.unwrap();

            store.put(b"crash_key_1", b"crash_val_1").await.unwrap();
            store.put(b"crash_key_2", b"crash_val_2").await.unwrap();
            store.put(b"crash_key_3", b"crash_val_3").await.unwrap();
            store.delete(b"crash_key_2").await.unwrap();

            // Sync WAL but DON'T flush MemTable to SSTables
            if let Some(ref wal) = store.wal_writer {
                wal.lock().await.sync().await.unwrap();
            }

            // Simulate crash: stop background tasks without flushing
            store.is_open.store(false, Ordering::SeqCst);
            store.flush_notify.notify_one();
            // Drop store — data is only in WAL, not in SSTables
        }

        // Phase 2: Reopen and verify recovery from WAL
        {
            let config = LsmConfig {
                root_path: "/recovery_test".to_string(),
                object_store: shared_store.clone(),
                memtable_size_limit: 1024 * 1024,
                flush_interval: std::time::Duration::from_secs(3600),
                l0_compaction_threshold: 4,
                sstable_target_size: 64 * 1024 * 1024,
                block_size: 4096,
                block_cache_size: 256 * 1024 * 1024,
                enable_compression: false,
                enable_bloom_filters: true,
                wal_enabled: true,
                wal_segment_size: 16 * 1024 * 1024,
                fencing_token: None,
            };

            let store = LsmStore::open(config).await.unwrap();

            // Key 1 should be recovered
            let v1 = store.get(b"crash_key_1").await.unwrap();
            assert_eq!(v1, Some(b"crash_val_1".to_vec()), "Key 1 should be recovered from WAL");

            // Key 2 was deleted
            let v2 = store.get(b"crash_key_2").await.unwrap();
            assert_eq!(v2, None, "Key 2 should be deleted (tombstone recovered)");

            // Key 3 should be recovered
            let v3 = store.get(b"crash_key_3").await.unwrap();
            assert_eq!(v3, Some(b"crash_val_3".to_vec()), "Key 3 should be recovered from WAL");

            store.close().await.unwrap();
        }
    }
}
