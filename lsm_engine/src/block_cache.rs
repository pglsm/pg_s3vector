//! LRU cache for SSTable bytes fetched from object storage.
//!
//! Caches full SSTable file contents keyed by their object store path,
//! avoiding repeated GETs to S3/MinIO for hot SSTables.

use bytes::Bytes;
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct BlockCache {
    entries: RwLock<CacheInner>,
    capacity_bytes: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

struct CacheInner {
    map: HashMap<String, Bytes>,
    order: VecDeque<String>,
    current_bytes: usize,
}

impl BlockCache {
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            entries: RwLock::new(CacheInner {
                map: HashMap::new(),
                order: VecDeque::new(),
                current_bytes: 0,
            }),
            capacity_bytes,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Look up cached SSTable bytes by path. Promotes the entry to MRU on hit.
    pub fn get(&self, path: &str) -> Option<Bytes> {
        let mut inner = self.entries.write();
        if let Some(data) = inner.map.get(path).cloned() {
            self.hits.fetch_add(1, Ordering::Relaxed);
            promote(&mut inner.order, path);
            Some(data)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert SSTable bytes. Evicts LRU entries if capacity would be exceeded.
    /// Entries larger than the entire cache capacity are not cached.
    pub fn insert(&self, path: &str, data: Bytes) {
        let entry_size = entry_overhead(path, &data);
        if entry_size > self.capacity_bytes {
            return;
        }

        let mut inner = self.entries.write();

        if let Some(old) = inner.map.get(path) {
            inner.current_bytes -= entry_overhead(path, old);
            promote(&mut inner.order, path);
        } else {
            inner.order.push_front(path.to_owned());
        }

        while inner.current_bytes + entry_size > self.capacity_bytes {
            if let Some(victim) = inner.order.pop_back() {
                if let Some(evicted) = inner.map.remove(&victim) {
                    inner.current_bytes -= entry_overhead(&victim, &evicted);
                }
            } else {
                break;
            }
        }

        inner.current_bytes += entry_size;
        inner.map.insert(path.to_owned(), data);
    }

    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    pub fn hit_rate(&self) -> f64 {
        let h = self.hits() as f64;
        let m = self.misses() as f64;
        let total = h + m;
        if total == 0.0 {
            0.0
        } else {
            h / total
        }
    }
}

/// Estimate memory footprint of a cache entry (path allocation + Bytes overhead).
fn entry_overhead(path: &str, data: &Bytes) -> usize {
    path.len() + data.len() + 64
}

/// Move an existing key to the front (MRU position) of the LRU deque.
fn promote(order: &mut VecDeque<String>, key: &str) {
    if let Some(pos) = order.iter().position(|k| k == key) {
        order.remove(pos);
    }
    order.push_front(key.to_owned());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let cache = BlockCache::new(4096);
        let data = Bytes::from(vec![1u8; 100]);
        cache.insert("sst/001.sst", data.clone());

        let got = cache.get("sst/001.sst");
        assert_eq!(got, Some(data));
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn test_miss() {
        let cache = BlockCache::new(4096);
        assert_eq!(cache.get("nope"), None);
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn test_eviction() {
        // Each entry: path.len() + data.len() + 64 overhead
        // "a" (1) + 100 + 64 = 165 bytes per entry
        let cache = BlockCache::new(400);
        cache.insert("a", Bytes::from(vec![0u8; 100]));
        cache.insert("b", Bytes::from(vec![1u8; 100]));

        // Both should fit (165 * 2 = 330 < 400)
        assert!(cache.get("a").is_some());
        assert!(cache.get("b").is_some());

        // Third pushes over capacity, "a" was LRU after accessing both above
        // (access order: get(a), get(b) — b is MRU, a is next, then we insert c)
        cache.insert("c", Bytes::from(vec![2u8; 100]));
        // 330 + 165 = 495 > 400 → evict LRU ("a")
        assert!(cache.get("a").is_none());
        assert!(cache.get("b").is_some());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn test_oversize_entry_skipped() {
        let cache = BlockCache::new(100);
        cache.insert("huge", Bytes::from(vec![0u8; 200]));
        assert!(cache.get("huge").is_none());
    }

    #[test]
    fn test_hit_rate() {
        let cache = BlockCache::new(4096);
        cache.insert("x", Bytes::from(vec![0u8; 10]));

        cache.get("x"); // hit
        cache.get("y"); // miss
        cache.get("x"); // hit

        assert_eq!(cache.hits(), 2);
        assert_eq!(cache.misses(), 1);
        assert!((cache.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_update_existing() {
        let cache = BlockCache::new(4096);
        cache.insert("k", Bytes::from(vec![1u8; 10]));
        cache.insert("k", Bytes::from(vec![2u8; 20]));

        let got = cache.get("k").unwrap();
        assert_eq!(got.len(), 20);
        assert_eq!(got[0], 2);
    }
}
