//! Configuration for the LSM storage engine.

use object_store::memory::InMemory;
use object_store::ObjectStore;
use std::sync::Arc;
use std::time::Duration;

/// Configuration for the LSM store.
#[derive(Clone)]
pub struct LsmConfig {
    /// Root path in object storage for this store's data.
    pub root_path: String,

    /// The object store backend (S3, MinIO, in-memory, etc.)
    pub object_store: Arc<dyn ObjectStore>,

    /// Maximum size of the in-memory MemTable before flushing (bytes).
    /// Default: 64 MB
    pub memtable_size_limit: usize,

    /// How often to flush the MemTable to object storage.
    /// Default: 5 seconds
    pub flush_interval: Duration,

    /// Maximum number of L0 SSTables before triggering compaction.
    /// Default: 4
    pub l0_compaction_threshold: usize,

    /// Target size for SSTable files on S3 (bytes).
    /// Default: 64 MB (batching 8KB pages into large objects)
    pub sstable_target_size: usize,

    /// Block size within SSTables (bytes).
    /// Default: 4 KB
    pub block_size: usize,

    /// Size of the in-memory block cache (bytes).
    /// Default: 256 MB
    pub block_cache_size: usize,

    /// Whether to enable compression for SSTables.
    pub enable_compression: bool,

    /// Whether to enable bloom filters for SSTables.
    pub enable_bloom_filters: bool,

    /// Whether to enable the Write-Ahead Log for durability.
    /// Default: true
    pub wal_enabled: bool,

    /// Maximum size of a WAL segment before rotation (bytes).
    /// Default: 16 MB
    pub wal_segment_size: usize,

    /// Optional fencing token for single-writer enforcement.
    /// When set, the store verifies this matches the manifest's fencing token
    /// on open, preventing split-brain from stale writers.
    pub fencing_token: Option<String>,
}

impl LsmConfig {
    /// Create a config for in-memory object storage (testing).
    pub fn in_memory(root_path: &str) -> Self {
        Self {
            root_path: root_path.to_string(),
            object_store: Arc::new(InMemory::new()),
            memtable_size_limit: 64 * 1024 * 1024,  // 64 MB
            flush_interval: Duration::from_secs(5),
            l0_compaction_threshold: 4,
            sstable_target_size: 64 * 1024 * 1024,   // 64 MB
            block_size: 4096,                          // 4 KB
            block_cache_size: 256 * 1024 * 1024,      // 256 MB
            enable_compression: false,
            enable_bloom_filters: true,
            wal_enabled: true,
            wal_segment_size: 16 * 1024 * 1024,        // 16 MB
            fencing_token: None,
        }
    }

    /// Create a config for S3-compatible storage.
    pub fn s3(
        root_path: &str,
        object_store: Arc<dyn ObjectStore>,
    ) -> Self {
        Self {
            root_path: root_path.to_string(),
            object_store,
            memtable_size_limit: 64 * 1024 * 1024,
            flush_interval: Duration::from_secs(5),
            l0_compaction_threshold: 4,
            sstable_target_size: 64 * 1024 * 1024,
            block_size: 4096,
            block_cache_size: 256 * 1024 * 1024,
            enable_compression: true,
            enable_bloom_filters: true,
            wal_enabled: true,
            wal_segment_size: 16 * 1024 * 1024,
            fencing_token: None,
        }
    }

    /// Builder-style: set fencing token.
    pub fn with_fencing_token(mut self, token: String) -> Self {
        self.fencing_token = Some(token);
        self
    }

    /// Builder-style: set memtable size limit.
    pub fn with_memtable_size(mut self, size: usize) -> Self {
        self.memtable_size_limit = size;
        self
    }

    /// Builder-style: set flush interval.
    pub fn with_flush_interval(mut self, interval: Duration) -> Self {
        self.flush_interval = interval;
        self
    }

    /// Builder-style: set L0 compaction threshold.
    pub fn with_l0_compaction_threshold(mut self, threshold: usize) -> Self {
        self.l0_compaction_threshold = threshold;
        self
    }

    /// Builder-style: enable/disable WAL.
    pub fn with_wal(mut self, enabled: bool) -> Self {
        self.wal_enabled = enabled;
        self
    }
}

impl std::fmt::Debug for LsmConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LsmConfig")
            .field("root_path", &self.root_path)
            .field("memtable_size_limit", &self.memtable_size_limit)
            .field("flush_interval", &self.flush_interval)
            .field("l0_compaction_threshold", &self.l0_compaction_threshold)
            .field("sstable_target_size", &self.sstable_target_size)
            .field("block_size", &self.block_size)
            .field("block_cache_size", &self.block_cache_size)
            .finish()
    }
}
