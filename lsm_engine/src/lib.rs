//! LSM Engine: A Log-Structured Merge-tree storage engine for S3-backed storage.
//!
//! This crate provides a standalone key-value store that persists data to object storage
//! (S3, MinIO, or in-memory for testing) using an LSM-tree architecture.
//!
//! # Architecture
//!
//! ```text
//! Writes → MemTable (RAM) → L0 SSTables (Local) → Compacted SSTables (S3)
//! ```
//!
//! # Usage
//!
//! ```rust,no_run
//! use lsm_engine::{LsmStore, LsmConfig};
//!
//! # async fn example() -> Result<(), lsm_engine::LsmError> {
//! let config = LsmConfig::in_memory("/test");
//! let store = LsmStore::open(config).await?;
//!
//! store.put(b"key1", b"value1").await?;
//! let val = store.get(b"key1").await?;
//! assert_eq!(val, Some(b"value1".to_vec()));
//!
//! store.close().await?;
//! # Ok(())
//! # }
//! ```

pub mod block_cache;
pub mod config;
pub mod store;
pub mod memtable;
pub mod sstable;
pub mod compaction;
pub mod manifest;
pub mod wal;

pub use config::LsmConfig;
pub use store::LsmStore;

use thiserror::Error;

/// Errors that can occur in the LSM engine.
#[derive(Error, Debug)]
pub enum LsmError {
    #[error("Object store error: {0}")]
    ObjectStore(#[from] object_store::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Store is closed")]
    StoreClosed,

    #[error("Writer fencing violation: {0}")]
    FencingViolation(String),

    #[error("Manifest corruption: {0}")]
    ManifestCorruption(String),

    #[error("WAL corruption: {0}")]
    WalCorruption(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

/// Result type alias for LSM operations.
pub type LsmResult<T> = Result<T, LsmError>;

/// A key-value entry returned by scan operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvEntry {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    pub sequence: u64,
}

impl KvEntry {
    pub fn new(key: Vec<u8>, value: Vec<u8>, sequence: u64) -> Self {
        Self { key, value, sequence }
    }
}

/// Represents a write operation in the LSM engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WriteOp {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
}
