//! Storage bridge: manages per-table LsmStore instances and TID mapping.
//!
//! This module maintains a global registry of LSM stores, one per table,
//! plus a stable TID <-> key mapping that enables index scans and
//! tuple-level DML (DELETE, UPDATE) via Postgres's ItemPointer mechanism.

use lsm_engine::{LsmConfig, LsmStore};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Global storage registry, accessible from any SQL function.
static GLOBAL_STORAGE: Lazy<TableStorage> = Lazy::new(|| {
    TableStorage::new()
});

// ─────────────────────────────────────────────────────────────────────
// TID ↔ u64 conversion
//
// PostgreSQL TIDs are (BlockNumber u32, OffsetNumber u16).
// OffsetNumber must be >= 1.  We pack them into a u64 using
//   id = block * 65535 + offset
// which gives ~2.8 × 10^14 unique TIDs.
// ─────────────────────────────────────────────────────────────────────

/// Convert a TID u64 id into (block, offset) for `ItemPointerSet`.
pub fn tid_to_block_offset(id: u64) -> (u32, u16) {
    debug_assert!(id >= 1, "TID ids start at 1");
    let id0 = id - 1;
    let block = (id0 / 65535) as u32;
    let offset = ((id0 % 65535) + 1) as u16;
    (block, offset)
}

/// Convert (block, offset) from an `ItemPointer` back to a TID u64.
pub fn block_offset_to_tid(block: u32, offset: u16) -> u64 {
    (block as u64) * 65535 + (offset as u64)
}

/// Per-table TID management.
///
/// Maps between synthetic Postgres TIDs and LSM key bytes.
/// Uses a monotonically increasing u64 counter, split into
/// (BlockNumber, OffsetNumber) when written to ItemPointers.
struct TidManager {
    next_id: AtomicU64,
    tid_to_key: RwLock<HashMap<u64, Vec<u8>>>,
    key_to_tid: RwLock<HashMap<Vec<u8>, u64>>,
}

impl TidManager {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            tid_to_key: RwLock::new(HashMap::new()),
            key_to_tid: RwLock::new(HashMap::new()),
        }
    }

    fn assign_or_get(&self, key: &[u8]) -> u64 {
        {
            let map = self.key_to_tid.read();
            if let Some(&id) = map.get(key) {
                return id;
            }
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        self.tid_to_key.write().insert(id, key.to_vec());
        self.key_to_tid.write().insert(key.to_vec(), id);
        id
    }

    fn key_for_tid(&self, id: u64) -> Option<Vec<u8>> {
        self.tid_to_key.read().get(&id).cloned()
    }

    fn remove(&self, id: u64) {
        if let Some(key) = self.tid_to_key.write().remove(&id) {
            self.key_to_tid.write().remove(&key);
        }
    }

    fn clear(&self) {
        self.tid_to_key.write().clear();
        self.key_to_tid.write().clear();
        self.next_id.store(1, Ordering::SeqCst);
    }

    /// Re-insert a TID mapping (used during transaction rollback).
    fn restore(&self, id: u64, key: &[u8]) {
        self.tid_to_key.write().insert(id, key.to_vec());
        self.key_to_tid.write().insert(key.to_vec(), id);
    }
}

/// Registry of per-table LSM stores.
pub struct TableStorage {
    stores: RwLock<HashMap<String, Arc<LsmStore>>>,
    tid_managers: RwLock<HashMap<String, Arc<TidManager>>>,
    runtime: Arc<Runtime>,
}

impl TableStorage {
    fn new() -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime for LSM storage");

        Self {
            stores: RwLock::new(HashMap::new()),
            tid_managers: RwLock::new(HashMap::new()),
            runtime: Arc::new(runtime),
        }
    }

    /// Get the global storage instance.
    pub fn global() -> &'static TableStorage {
        &GLOBAL_STORAGE
    }

    /// Get a handle to the shared tokio runtime.
    pub fn runtime(&self) -> Arc<Runtime> {
        self.runtime.clone()
    }

    fn tid_manager(&self, table_name: &str) -> Arc<TidManager> {
        {
            let mgrs = self.tid_managers.read();
            if let Some(mgr) = mgrs.get(table_name) {
                return mgr.clone();
            }
        }
        let mgr = Arc::new(TidManager::new());
        self.tid_managers.write().insert(table_name.to_string(), mgr.clone());
        mgr
    }

    /// Get or create an LSM store for the given table.
    pub fn get_or_create(&self, table_name: &str) -> Result<Arc<LsmStore>, String> {
        {
            let stores = self.stores.read();
            if let Some(store) = stores.get(table_name) {
                return Ok(store.clone());
            }
        }

        let config = LsmConfig::in_memory(&format!("/tables/{}", table_name))
            .with_memtable_size(64 * 1024 * 1024);

        let store = self.runtime.block_on(async {
            LsmStore::open(config).await
        }).map_err(|e| format!("Failed to open LSM store: {}", e))?;

        let store = Arc::new(store);
        let mut stores = self.stores.write();
        stores.insert(table_name.to_string(), store.clone());

        Ok(store)
    }

    /// Insert a key-value pair and return the assigned TID id.
    pub fn insert_with_tid(&self, table_name: &str, key: &[u8], value: &[u8]) -> Result<u64, String> {
        let store = self.get_or_create(table_name)?;
        self.runtime.block_on(async {
            store.put(key, value).await
        }).map_err(|e| format!("{}", e))?;

        let mgr = self.tid_manager(table_name);
        Ok(mgr.assign_or_get(key))
    }

    /// Insert a key-value pair into a table.
    pub fn insert(&self, table_name: &str, key: &[u8], value: &[u8]) -> Result<(), String> {
        self.insert_with_tid(table_name, key, value)?;
        Ok(())
    }

    /// Get a value by key from a table.
    pub fn get(&self, table_name: &str, key: &[u8]) -> Result<Option<Vec<u8>>, String> {
        let store = self.get_or_create(table_name)?;
        self.runtime.block_on(async {
            store.get(key).await
        }).map_err(|e| format!("{}", e))
    }

    /// Delete a key from a table.
    pub fn delete(&self, table_name: &str, key: &[u8]) -> Result<(), String> {
        let store = self.get_or_create(table_name)?;
        self.runtime.block_on(async {
            store.delete(key).await
        }).map_err(|e| format!("{}", e))
    }

    /// Delete by TID id. Returns the key that was deleted.
    pub fn delete_by_tid(&self, table_name: &str, tid_id: u64) -> Result<Option<Vec<u8>>, String> {
        let mgr = self.tid_manager(table_name);
        let key = match mgr.key_for_tid(tid_id) {
            Some(k) => k,
            None => return Ok(None),
        };

        let store = self.get_or_create(table_name)?;
        self.runtime.block_on(async {
            store.delete(&key).await
        }).map_err(|e| format!("{}", e))?;

        mgr.remove(tid_id);
        Ok(Some(key))
    }

    /// Fetch a (key, value) pair by TID id for index_fetch_tuple.
    pub fn fetch_by_tid(&self, table_name: &str, tid_id: u64) -> Result<Option<(Vec<u8>, Vec<u8>)>, String> {
        let mgr = self.tid_manager(table_name);
        let key = match mgr.key_for_tid(tid_id) {
            Some(k) => k,
            None => return Ok(None),
        };

        let store = self.get_or_create(table_name)?;
        let value = self.runtime.block_on(async {
            store.get(&key).await
        }).map_err(|e| format!("{}", e))?;

        match value {
            Some(v) => Ok(Some((key, v))),
            None => Ok(None),
        }
    }

    /// Get the TID id for a key (or assign one).
    pub fn tid_for_key(&self, table_name: &str, key: &[u8]) -> u64 {
        let mgr = self.tid_manager(table_name);
        mgr.assign_or_get(key)
    }

    /// Scan all entries in a table. Returns (key, value, tid_id) triples.
    pub fn scan_all_with_tids(&self, table_name: &str) -> Result<Vec<(Vec<u8>, Vec<u8>, u64)>, String> {
        let store = self.get_or_create(table_name)?;
        let entries = self.runtime.block_on(async {
            store.scan(&[], &[0xFF; 32]).await
        }).map_err(|e| format!("{}", e))?;

        let mgr = self.tid_manager(table_name);
        Ok(entries
            .into_iter()
            .map(|e| {
                let offset = mgr.assign_or_get(&e.key);
                (e.key, e.value, offset)
            })
            .collect())
    }

    /// Scan all keys (without values) in a table. Returns (key, tid_id) pairs.
    /// Used by sequential scan to reduce peak memory — values are fetched on demand.
    pub fn scan_keys_with_tids(&self, table_name: &str) -> Result<Vec<(Vec<u8>, u64)>, String> {
        let store = self.get_or_create(table_name)?;
        let entries = self.runtime.block_on(async {
            store.scan(&[], &[0xFF; 32]).await
        }).map_err(|e| format!("{}", e))?;

        let mgr = self.tid_manager(table_name);
        Ok(entries
            .into_iter()
            .map(|e| {
                let offset = mgr.assign_or_get(&e.key);
                (e.key, offset)
            })
            .collect())
    }

    /// Scan all entries in a table (legacy, without TIDs).
    pub fn scan_all(&self, table_name: &str) -> Result<Vec<(Vec<u8>, Vec<u8>)>, String> {
        let store = self.get_or_create(table_name)?;
        let entries = self.runtime.block_on(async {
            store.scan(&[], &[0xFF; 32]).await
        }).map_err(|e| format!("{}", e))?;

        Ok(entries.into_iter().map(|e| (e.key, e.value)).collect())
    }

    /// Scan entries in a key range.
    pub fn scan_range(
        &self,
        table_name: &str,
        start: &[u8],
        end: &[u8],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>, String> {
        let store = self.get_or_create(table_name)?;
        let entries = self.runtime.block_on(async {
            store.scan(start, end).await
        }).map_err(|e| format!("{}", e))?;

        Ok(entries.into_iter().map(|e| (e.key, e.value)).collect())
    }

    /// Get stats for a table.
    pub fn stats(&self, table_name: &str) -> Result<String, String> {
        let store = self.get_or_create(table_name)?;
        Ok(store.stats().to_string())
    }

    /// Force flush a table's MemTable.
    pub fn flush(&self, table_name: &str) -> Result<(), String> {
        let store = self.get_or_create(table_name)?;
        self.runtime.block_on(async {
            store.flush().await
        }).map_err(|e| format!("{}", e))
    }

    /// Flush all tables. Returns the number of tables flushed.
    pub fn flush_all(&self) -> Result<usize, String> {
        let stores: Vec<(String, Arc<LsmStore>)> = {
            let map = self.stores.read();
            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        };
        let count = stores.len();
        for (name, store) in &stores {
            self.runtime.block_on(async {
                store.flush().await
            }).map_err(|e| format!("Flush failed for '{}': {}", name, e))?;
        }
        Ok(count)
    }

    /// Restore a specific TID mapping (used by txn rollback).
    pub fn restore_tid(&self, table_name: &str, tid_id: u64, key: &[u8]) {
        let mgr = self.tid_manager(table_name);
        mgr.restore(tid_id, key);
    }

    /// Truncate a table: delete all data and reset TID mapping.
    pub fn truncate(&self, table_name: &str) -> Result<(), String> {
        // Remove the existing store
        self.stores.write().remove(table_name);
        // Reset TID mapping
        if let Some(mgr) = self.tid_managers.read().get(table_name) {
            mgr.clear();
        }
        // Re-create an empty store
        self.get_or_create(table_name)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_storage_basic() {
        let storage = TableStorage::new();

        storage.insert("test", b"key1", b"val1").unwrap();
        let val = storage.get("test", b"key1").unwrap();
        assert_eq!(val, Some(b"val1".to_vec()));
    }

    #[test]
    fn test_table_storage_separate_tables() {
        let storage = TableStorage::new();

        storage.insert("t1", b"key", b"v1").unwrap();
        storage.insert("t2", b"key", b"v2").unwrap();

        assert_eq!(storage.get("t1", b"key").unwrap(), Some(b"v1".to_vec()));
        assert_eq!(storage.get("t2", b"key").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn test_table_storage_scan() {
        let storage = TableStorage::new();

        storage.insert("scan_test", b"a", b"1").unwrap();
        storage.insert("scan_test", b"b", b"2").unwrap();
        storage.insert("scan_test", b"c", b"3").unwrap();

        let results = storage.scan_all("scan_test").unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_tid_mapping_basic() {
        let storage = TableStorage::new();

        let tid = storage.insert_with_tid("tid_test", b"k1", b"v1").unwrap();
        assert!(tid >= 1);

        let fetched = storage.fetch_by_tid("tid_test", tid).unwrap();
        assert_eq!(fetched, Some((b"k1".to_vec(), b"v1".to_vec())));
    }

    #[test]
    fn test_tid_mapping_delete() {
        let storage = TableStorage::new();

        let tid = storage.insert_with_tid("del_tid", b"k1", b"v1").unwrap();
        let deleted_key = storage.delete_by_tid("del_tid", tid).unwrap();
        assert_eq!(deleted_key, Some(b"k1".to_vec()));

        let fetched = storage.fetch_by_tid("del_tid", tid).unwrap();
        assert_eq!(fetched, None);
    }

    #[test]
    fn test_scan_with_tids() {
        let storage = TableStorage::new();

        storage.insert("scan_tid", b"a", b"1").unwrap();
        storage.insert("scan_tid", b"b", b"2").unwrap();

        let results = storage.scan_all_with_tids("scan_tid").unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].2 >= 1);
        assert!(results[1].2 >= 1);
        assert_ne!(results[0].2, results[1].2);
    }

    #[test]
    fn test_truncate() {
        let storage = TableStorage::new();

        storage.insert("trunc_test", b"k1", b"v1").unwrap();
        storage.insert("trunc_test", b"k2", b"v2").unwrap();
        assert_eq!(storage.scan_all("trunc_test").unwrap().len(), 2);

        storage.truncate("trunc_test").unwrap();
        assert_eq!(storage.scan_all("trunc_test").unwrap().len(), 0);
    }

    #[test]
    fn test_flush_all() {
        let storage = TableStorage::new();
        storage.insert("flush_all_1", b"k", b"v").unwrap();
        storage.insert("flush_all_2", b"k", b"v").unwrap();

        let count = storage.flush_all().unwrap();
        assert_eq!(count, 2);
    }
}
