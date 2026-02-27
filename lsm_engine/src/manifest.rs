//! LSM state manifest for tracking SSTables across levels.
//!
//! The manifest tracks which SSTables exist at each level of the LSM tree.
//! It is persisted to object storage for crash recovery.

use crate::sstable::SSTableMeta;
use crate::{LsmError, LsmResult};
use bytes::Bytes;
use object_store::path::Path;
use object_store::ObjectStore;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The LSM manifest: a snapshot of the current state of the LSM tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Monotonically increasing version number.
    pub version: u64,

    /// Next SSTable ID to assign.
    pub next_sstable_id: u64,

    /// Next sequence number (for writes after recovery).
    pub next_sequence: u64,

    /// SSTables at each level. Index 0 = L0 (most recent).
    pub levels: Vec<Vec<SSTableMeta>>,

    /// Writer fencing token (ensures single-writer invariant).
    pub fencing_token: Option<String>,
}

impl Manifest {
    /// Create a new empty manifest.
    pub fn new() -> Self {
        Self {
            version: 0,
            next_sstable_id: 1,
            next_sequence: 0,
            levels: vec![Vec::new(); 7], // L0 through L6
            fencing_token: None,
        }
    }

    /// Add an SSTable to L0.
    pub fn add_l0_sstable(&mut self, mut meta: SSTableMeta) -> u64 {
        let id = self.next_sstable_id;
        self.next_sstable_id += 1;
        meta.id = id;
        meta.level = 0;
        self.levels[0].push(meta);
        self.version += 1;
        id
    }

    /// Move SSTables from one level to another (after compaction).
    pub fn apply_compaction(
        &mut self,
        source_level: u32,
        source_ids: &[u64],
        target_level: u32,
        new_tables: Vec<SSTableMeta>,
    ) {
        // Remove source tables
        let sl = source_level as usize;
        self.levels[sl].retain(|t| !source_ids.contains(&t.id));

        // Add new tables at target level
        let tl = target_level as usize;
        for mut table in new_tables {
            table.id = self.next_sstable_id;
            self.next_sstable_id += 1;
            table.level = target_level;
            self.levels[tl].push(table);
        }

        self.version += 1;
    }

    /// Get all L0 SSTables (newest first, by sequence).
    pub fn l0_tables(&self) -> &[SSTableMeta] {
        &self.levels[0]
    }

    /// Get the number of L0 SSTables.
    pub fn l0_count(&self) -> usize {
        self.levels[0].len()
    }

    /// Get all SSTables across all levels.
    pub fn all_tables(&self) -> Vec<&SSTableMeta> {
        self.levels.iter().flat_map(|l| l.iter()).collect()
    }

    /// Serialize the manifest to JSON bytes.
    pub fn to_bytes(&self) -> LsmResult<Bytes> {
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| LsmError::Serialization(e.to_string()))?;
        Ok(Bytes::from(json))
    }

    /// Deserialize a manifest from JSON bytes.
    pub fn from_bytes(data: &[u8]) -> LsmResult<Self> {
        serde_json::from_slice(data)
            .map_err(|e| LsmError::ManifestCorruption(e.to_string()))
    }
}

impl Default for Manifest {
    fn default() -> Self {
        Self::new()
    }
}

/// Persistent manifest storage backed by object storage.
pub struct ManifestStore {
    object_store: Arc<dyn ObjectStore>,
    manifest_path: Path,
}

impl ManifestStore {
    /// Create a new manifest store.
    pub fn new(object_store: Arc<dyn ObjectStore>, root_path: &str) -> Self {
        let manifest_path = Path::from(format!("{}/manifest.json", root_path));
        Self {
            object_store,
            manifest_path,
        }
    }

    /// Load the manifest from object storage, or create a new one if none exists.
    pub async fn load_or_create(&self) -> LsmResult<Manifest> {
        match self.object_store.get(&self.manifest_path).await {
            Ok(result) => {
                let data = result.bytes().await
                    .map_err(|e| LsmError::ObjectStore(e))?;
                Manifest::from_bytes(&data)
            }
            Err(object_store::Error::NotFound { .. }) => {
                let manifest = Manifest::new();
                self.save(&manifest).await?;
                Ok(manifest)
            }
            Err(e) => Err(LsmError::ObjectStore(e)),
        }
    }

    /// Save the manifest to object storage.
    pub async fn save(&self, manifest: &Manifest) -> LsmResult<()> {
        let data = manifest.to_bytes()?;
        self.object_store
            .put(&self.manifest_path, data.into())
            .await
            .map_err(|e| LsmError::ObjectStore(e))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_manifest() {
        let m = Manifest::new();
        assert_eq!(m.version, 0);
        assert_eq!(m.next_sstable_id, 1);
        assert_eq!(m.levels.len(), 7);
        assert!(m.l0_tables().is_empty());
    }

    #[test]
    fn test_add_l0() {
        let mut m = Manifest::new();
        let meta = SSTableMeta {
            id: 0,
            level: 0,
            min_key: b"a".to_vec(),
            max_key: b"z".to_vec(),
            entry_count: 100,
            size_bytes: 4096,
            path: "test.sst".to_string(),
            min_sequence: 0,
            max_sequence: 99,
        };

        let id = m.add_l0_sstable(meta);
        assert_eq!(id, 1);
        assert_eq!(m.l0_count(), 1);
        assert_eq!(m.version, 1);
    }

    #[test]
    fn test_serialize_roundtrip() {
        let mut m = Manifest::new();
        m.add_l0_sstable(SSTableMeta {
            id: 0,
            level: 0,
            min_key: b"a".to_vec(),
            max_key: b"z".to_vec(),
            entry_count: 100,
            size_bytes: 4096,
            path: "test.sst".to_string(),
            min_sequence: 0,
            max_sequence: 99,
        });

        let bytes = m.to_bytes().unwrap();
        let m2 = Manifest::from_bytes(&bytes).unwrap();
        assert_eq!(m2.version, m.version);
        assert_eq!(m2.l0_count(), m.l0_count());
    }

    #[tokio::test]
    async fn test_manifest_store() {
        let store = Arc::new(object_store::memory::InMemory::new());
        let ms = ManifestStore::new(store.clone(), "/test");

        // Load or create (creates new)
        let manifest = ms.load_or_create().await.unwrap();
        assert_eq!(manifest.version, 0);

        // Modify and save
        let mut modified = manifest;
        modified.next_sequence = 42;
        ms.save(&modified).await.unwrap();

        // Reload
        let reloaded = ms.load_or_create().await.unwrap();
        assert_eq!(reloaded.next_sequence, 42);
    }
}
