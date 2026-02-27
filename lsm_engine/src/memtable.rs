//! In-memory write buffer (MemTable) using a concurrent skip list.
//!
//! The MemTable accumulates writes in sorted order. When it reaches the size
//! threshold, it is frozen and flushed as an SSTable to object storage.

use crate::{KvEntry, LsmResult};
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// A value entry in the MemTable, supporting both puts and deletes.
#[derive(Debug, Clone)]
pub enum MemTableValue {
    /// A live value.
    Put(Vec<u8>),
    /// A tombstone marker (deletion).
    Delete,
}

/// An in-memory sorted write buffer.
///
/// Uses a BTreeMap for ordered key-value storage with concurrent access
/// via a RwLock. Tracks approximate size for flush threshold decisions.
#[derive(Debug)]
pub struct MemTable {
    /// Sorted key-value data. Key is the user key, value includes sequence number.
    data: RwLock<BTreeMap<Vec<u8>, (u64, MemTableValue)>>,

    /// Approximate size of all keys + values in bytes.
    approximate_size: AtomicUsize,

    /// Monotonically increasing sequence number for ordering writes.
    next_sequence: AtomicU64,

    /// Whether this MemTable is frozen (read-only, pending flush).
    frozen: RwLock<bool>,
}

impl MemTable {
    /// Create a new empty MemTable.
    pub fn new(start_sequence: u64) -> Self {
        Self {
            data: RwLock::new(BTreeMap::new()),
            approximate_size: AtomicUsize::new(0),
            next_sequence: AtomicU64::new(start_sequence),
            frozen: RwLock::new(false),
        }
    }

    /// Insert a key-value pair into the MemTable.
    /// Returns the sequence number assigned to this write.
    pub fn put(&self, key: &[u8], value: &[u8]) -> LsmResult<u64> {
        let frozen = self.frozen.read();
        if *frozen {
            return Err(crate::LsmError::StoreClosed);
        }
        drop(frozen);

        let seq = self.next_sequence.fetch_add(1, Ordering::SeqCst);
        let size_delta = key.len() + value.len() + 8; // 8 bytes for seq number

        let mut data = self.data.write();
        let old = data.insert(key.to_vec(), (seq, MemTableValue::Put(value.to_vec())));
        drop(data);

        if let Some((_, old_val)) = old {
            let old_size = key.len() + match &old_val {
                MemTableValue::Put(v) => v.len(),
                MemTableValue::Delete => 0,
            } + 8;
            let _ = self.approximate_size.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(old_size))
            });
        }
        self.approximate_size.fetch_add(size_delta, Ordering::Relaxed);

        Ok(seq)
    }

    /// Mark a key as deleted with a tombstone.
    pub fn delete(&self, key: &[u8]) -> LsmResult<u64> {
        let frozen = self.frozen.read();
        if *frozen {
            return Err(crate::LsmError::StoreClosed);
        }
        drop(frozen);

        let seq = self.next_sequence.fetch_add(1, Ordering::SeqCst);

        let mut data = self.data.write();
        let old = data.insert(key.to_vec(), (seq, MemTableValue::Delete));
        drop(data);

        let size_delta = key.len() + 8;
        if let Some((_, old_val)) = old {
            let old_size = key.len() + match &old_val {
                MemTableValue::Put(v) => v.len(),
                MemTableValue::Delete => 0,
            } + 8;
            let _ = self.approximate_size.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(old_size))
            });
        }
        self.approximate_size.fetch_add(size_delta, Ordering::Relaxed);

        Ok(seq)
    }

    /// Get a value by key from the MemTable.
    /// Returns None if the key is not found, or Some(None) if it was deleted (tombstone).
    pub fn get(&self, key: &[u8]) -> Option<Option<Vec<u8>>> {
        let data = self.data.read();
        data.get(key).map(|(_, val)| match val {
            MemTableValue::Put(v) => Some(v.clone()),
            MemTableValue::Delete => None,
        })
    }

    /// Scan entries in the MemTable within the given key range.
    pub fn scan(&self, start: &[u8], end: &[u8]) -> Vec<KvEntry> {
        let data = self.data.read();
        data.range(start.to_vec()..end.to_vec())
            .filter_map(|(k, (seq, val))| match val {
                MemTableValue::Put(v) => Some(KvEntry::new(k.clone(), v.clone(), *seq)),
                MemTableValue::Delete => None,
            })
            .collect()
    }

    /// Get all entries in sorted order (for flushing to SSTable).
    pub fn entries(&self) -> Vec<(Vec<u8>, u64, MemTableValue)> {
        let data = self.data.read();
        data.iter()
            .map(|(k, (seq, val))| (k.clone(), *seq, val.clone()))
            .collect()
    }

    /// Get the approximate size of the MemTable in bytes.
    pub fn approximate_size(&self) -> usize {
        self.approximate_size.load(Ordering::Relaxed)
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.data.read().len()
    }

    /// Check if the MemTable is empty.
    pub fn is_empty(&self) -> bool {
        self.data.read().is_empty()
    }

    /// Freeze the MemTable, making it read-only.
    /// After freezing, no new writes are accepted and the table is ready for flushing.
    pub fn freeze(&self) {
        let mut frozen = self.frozen.write();
        *frozen = true;
    }

    /// Check if the MemTable is frozen.
    pub fn is_frozen(&self) -> bool {
        *self.frozen.read()
    }

    /// Get the current sequence number.
    pub fn current_sequence(&self) -> u64 {
        self.next_sequence.load(Ordering::SeqCst)
    }

    /// Get the next sequence number that will be assigned (peek, no increment).
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_put_get() {
        let mt = MemTable::new(0);
        mt.put(b"hello", b"world").unwrap();

        let val = mt.get(b"hello");
        assert_eq!(val, Some(Some(b"world".to_vec())));
    }

    #[test]
    fn test_get_missing() {
        let mt = MemTable::new(0);
        assert_eq!(mt.get(b"missing"), None);
    }

    #[test]
    fn test_delete() {
        let mt = MemTable::new(0);
        mt.put(b"key", b"val").unwrap();
        mt.delete(b"key").unwrap();

        // Tombstone: key exists but value is None
        assert_eq!(mt.get(b"key"), Some(None));
    }

    #[test]
    fn test_overwrite() {
        let mt = MemTable::new(0);
        mt.put(b"key", b"v1").unwrap();
        mt.put(b"key", b"v2").unwrap();
        assert_eq!(mt.get(b"key"), Some(Some(b"v2".to_vec())));
    }

    #[test]
    fn test_scan() {
        let mt = MemTable::new(0);
        mt.put(b"a", b"1").unwrap();
        mt.put(b"b", b"2").unwrap();
        mt.put(b"c", b"3").unwrap();
        mt.put(b"d", b"4").unwrap();

        let entries = mt.scan(b"b", b"d");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].key, b"b".to_vec());
        assert_eq!(entries[1].key, b"c".to_vec());
    }

    #[test]
    fn test_freeze() {
        let mt = MemTable::new(0);
        mt.put(b"key", b"val").unwrap();
        mt.freeze();

        assert!(mt.is_frozen());
        assert!(mt.put(b"new", b"val").is_err());

        // Reads still work
        assert_eq!(mt.get(b"key"), Some(Some(b"val".to_vec())));
    }

    #[test]
    fn test_approximate_size() {
        let mt = MemTable::new(0);
        assert_eq!(mt.approximate_size(), 0);

        mt.put(b"key", b"value").unwrap();
        assert!(mt.approximate_size() > 0);
    }

    #[test]
    fn test_sequence_numbers() {
        let mt = MemTable::new(100);
        let s1 = mt.put(b"a", b"1").unwrap();
        let s2 = mt.put(b"b", b"2").unwrap();
        assert_eq!(s1, 100);
        assert_eq!(s2, 101);
    }
}
