//! Background compaction for the LSM tree.
//!
//! Compaction merges SSTables from lower levels (L0) into higher levels,
//! reducing read amplification and reclaiming space from tombstoned entries.
//!
//! Strategy: Size-Tiered Compaction
//! - When L0 has too many SSTables, merge them all into a single L1 SSTable.
//! - When L1 has too many, merge overlapping ranges into L2, etc.

use crate::manifest::Manifest;
use crate::sstable::{SSTableBuilder, SSTableMeta, SSTableReader};
use crate::{LsmError, LsmResult};
use bytes::Bytes;
use object_store::path::Path;
use object_store::ObjectStore;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Compaction configuration.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Number of L0 SSTables that triggers compaction.
    pub l0_threshold: usize,
    /// Target block size for compacted SSTables.
    pub block_size: usize,
    /// Target SSTable size for compacted output.
    pub target_sstable_size: usize,
    /// Each level is this many times larger than the previous (default 10).
    pub level_size_multiplier: usize,
    /// Maximum SSTables in L1 before triggering L1→L2 compaction (default 10).
    pub l1_max_tables: usize,
    /// Whether to compress output SSTables with zstd.
    pub enable_compression: bool,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            l0_threshold: 4,
            block_size: 4096,
            target_sstable_size: 64 * 1024 * 1024, // 64 MB
            level_size_multiplier: 10,
            l1_max_tables: 10,
            enable_compression: false,
        }
    }
}

/// Result of a compaction operation.
#[derive(Debug)]
pub struct CompactionResult {
    /// IDs of source SSTables that were merged.
    pub source_ids: Vec<u64>,
    /// Object storage paths of source SSTables (for deletion after apply).
    pub source_paths: Vec<String>,
    /// Source level.
    pub source_level: u32,
    /// Target level.
    pub target_level: u32,
    /// New SSTables produced.
    pub new_tables: Vec<SSTableMeta>,
    /// Bytes of the new SSTables (to be written to object storage).
    pub new_table_data: Vec<Bytes>,
}

/// Determines if compaction is needed and performs it.
pub struct Compactor {
    config: CompactionConfig,
    object_store: Arc<dyn ObjectStore>,
    root_path: String,
}

impl Compactor {
    /// Create a new compactor.
    pub fn new(
        config: CompactionConfig,
        object_store: Arc<dyn ObjectStore>,
        root_path: String,
    ) -> Self {
        Self {
            config,
            object_store,
            root_path,
        }
    }

    /// Check if any level needs compaction.
    pub fn needs_compaction(&self, manifest: &Manifest) -> bool {
        if manifest.l0_count() >= self.config.l0_threshold {
            return true;
        }
        for level in 1..6 {
            if self.needs_level_compaction(manifest, level) {
                return true;
            }
        }
        false
    }

    /// Check if a specific level (L1+) needs compaction into the next level.
    pub fn needs_level_compaction(&self, manifest: &Manifest, level: usize) -> bool {
        if level == 0 || level >= manifest.levels.len() - 1 {
            return false;
        }
        let count = manifest.levels[level].len();
        let max = if level == 1 {
            self.config.l1_max_tables
        } else {
            self.config.l1_max_tables
                * self.config.level_size_multiplier.saturating_pow((level - 1) as u32)
        };
        count > max
    }

    /// Perform LN → L(N+1) compaction for levels >= 1.
    ///
    /// Picks the SSTable in level N with the oldest min_sequence, finds all
    /// overlapping SSTables in level N+1 by key range, merges them (newest
    /// version wins), and writes new SSTable(s) to level N+1.
    pub async fn compact_level(
        &self,
        manifest: &Manifest,
        level: usize,
    ) -> LsmResult<Option<CompactionResult>> {
        if level == 0 || level >= manifest.levels.len() - 1 {
            return Ok(None);
        }
        if !self.needs_level_compaction(manifest, level) {
            return Ok(None);
        }

        let source_tables = &manifest.levels[level];
        if source_tables.is_empty() {
            return Ok(None);
        }

        // Pick the SSTable with the oldest min_sequence
        let picked = source_tables
            .iter()
            .min_by_key(|t| t.min_sequence)
            .unwrap();

        let picked_min = &picked.min_key;
        let picked_max = &picked.max_key;

        // Find all overlapping SSTables in level N+1
        let next_level = level + 1;
        let overlapping: Vec<&SSTableMeta> = manifest.levels[next_level]
            .iter()
            .filter(|t| t.min_key <= *picked_max && t.max_key >= *picked_min)
            .collect();

        let mut source_ids = vec![picked.id];
        let mut source_paths = vec![picked.path.clone()];
        for t in &overlapping {
            source_ids.push(t.id);
            source_paths.push(t.path.clone());
        }

        tracing::info!(
            "Starting L{}→L{} compaction: 1 source + {} overlapping targets",
            level,
            next_level,
            overlapping.len()
        );

        // Read and merge all entries (newest version of each key wins)
        let mut merged: BTreeMap<Vec<u8>, (u64, bool, Vec<u8>)> = BTreeMap::new();

        // Helper closure to read an SSTable and merge its entries
        let read_and_merge = |data: Bytes, merged: &mut BTreeMap<Vec<u8>, (u64, bool, Vec<u8>)>| -> LsmResult<()> {
            let reader = SSTableReader::open(data)?;
            let entries = reader.scan_all_with_tombstones(&[], &[0xFF; 32])?;
            for (key, seq, is_tombstone, value) in entries {
                let should_insert = match merged.get(&key) {
                    Some((existing_seq, _, _)) => seq > *existing_seq,
                    None => true,
                };
                if should_insert {
                    merged.insert(key, (seq, is_tombstone, value));
                }
            }
            Ok(())
        };

        // Read the picked source SSTable
        let picked_path = Path::from(picked.path.clone());
        let picked_data = self
            .object_store
            .get(&picked_path)
            .await
            .map_err(LsmError::ObjectStore)?
            .bytes()
            .await
            .map_err(LsmError::ObjectStore)?;
        read_and_merge(picked_data, &mut merged)?;

        // Read overlapping target SSTables
        for table_meta in &overlapping {
            let path = Path::from(table_meta.path.clone());
            let data = self
                .object_store
                .get(&path)
                .await
                .map_err(LsmError::ObjectStore)?
                .bytes()
                .await
                .map_err(LsmError::ObjectStore)?;
            read_and_merge(data, &mut merged)?;
        }

        if merged.is_empty() {
            return Ok(None);
        }

        // Drop tombstones only if there are no levels below that could have the key.
        // "Below" = levels > next_level. If next_level is the last level, we can drop them.
        let is_bottom_level = next_level >= manifest.levels.len() - 1
            || manifest.levels[next_level + 1..].iter().all(|l| l.is_empty());

        let mut builder = SSTableBuilder::new(self.config.block_size, false);
        for (key, (seq, is_tombstone, value)) in &merged {
            if *is_tombstone && is_bottom_level {
                continue;
            }
            if *is_tombstone {
                builder.add(key.clone(), *seq, crate::memtable::MemTableValue::Delete);
            } else {
                builder.add(
                    key.clone(),
                    *seq,
                    crate::memtable::MemTableValue::Put(value.clone()),
                );
            }
        }

        let (data, mut meta) = builder.build()?;
        let sstable_path = format!(
            "{}/L{}/{}.sst",
            self.root_path, next_level, meta.min_sequence
        );
        meta.path = sstable_path;
        meta.level = next_level as u32;

        Ok(Some(CompactionResult {
            source_ids,
            source_paths,
            source_level: level as u32,
            target_level: next_level as u32,
            new_tables: vec![meta],
            new_table_data: vec![data],
        }))
    }

    /// Perform L0 → L1 compaction.
    ///
    /// Reads all L0 SSTables, merges them in sorted order (newest wins),
    /// removes tombstones for keys not present in lower levels, and writes
    /// new L1 SSTables.
    pub async fn compact_l0(&self, manifest: &Manifest) -> LsmResult<Option<CompactionResult>> {
        if !self.needs_compaction(manifest) {
            return Ok(None);
        }

        let l0_tables = manifest.l0_tables();
        tracing::info!("Starting L0 compaction: {} SSTables", l0_tables.len());

        // Read all L0 SSTables and merge them, preserving tombstones
        let mut merged: BTreeMap<Vec<u8>, (u64, bool, Vec<u8>)> = BTreeMap::new();
        let mut source_ids = Vec::new();
        let mut source_paths = Vec::new();

        for table_meta in l0_tables {
            source_ids.push(table_meta.id);
            source_paths.push(table_meta.path.clone());

            let path = Path::from(table_meta.path.clone());
            let data = self.object_store.get(&path).await
                .map_err(|e| LsmError::ObjectStore(e))?
                .bytes().await
                .map_err(|e| LsmError::ObjectStore(e))?;

            let reader = SSTableReader::open(data)?;

            let entries = reader.scan_all_with_tombstones(&[], &[0xFF; 32])?;
            for (key, seq, is_tombstone, value) in entries {
                let should_insert = match merged.get(&key) {
                    Some((existing_seq, _, _)) => seq > *existing_seq,
                    None => true,
                };
                if should_insert {
                    merged.insert(key, (seq, is_tombstone, value));
                }
            }
        }

        if merged.is_empty() {
            return Ok(None);
        }

        // Build new SSTable(s) from merged data
        let mut builder = SSTableBuilder::new(self.config.block_size, self.config.enable_compression);
        for (key, (seq, is_tombstone, value)) in &merged {
            if *is_tombstone {
                builder.add(key.clone(), *seq, crate::memtable::MemTableValue::Delete);
            } else {
                builder.add(key.clone(), *seq, crate::memtable::MemTableValue::Put(value.clone()));
            }
        }

        let (data, mut meta) = builder.build()?;
        let sstable_path = format!("{}/L1/{}.sst", self.root_path, meta.min_sequence);
        meta.path = sstable_path;
        meta.level = 1;

        Ok(Some(CompactionResult {
            source_ids,
            source_paths,
            source_level: 0,
            target_level: 1,
            new_tables: vec![meta],
            new_table_data: vec![data],
        }))
    }

    /// Write compaction results to object storage and delete source SSTables.
    pub async fn apply_result(&self, result: &CompactionResult) -> LsmResult<()> {
        for (meta, data) in result.new_tables.iter().zip(result.new_table_data.iter()) {
            let path = Path::from(meta.path.clone());
            self.object_store
                .put(&path, data.clone().into())
                .await
                .map_err(|e| LsmError::ObjectStore(e))?;
        }

        for source_path in &result.source_paths {
            let p = Path::from(source_path.clone());
            match self.object_store.delete(&p).await {
                Ok(()) => {}
                Err(object_store::Error::NotFound { .. }) => {}
                Err(e) => tracing::warn!("Failed to delete compacted SSTable {}: {}", source_path, e),
            }
        }

        tracing::info!(
            "Compaction complete: merged {} L{} SSTables into {} L{} SSTables, deleted {} source files",
            result.source_ids.len(),
            result.source_level,
            result.new_tables.len(),
            result.target_level,
            result.source_paths.len(),
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memtable::MemTableValue;
    use crate::sstable::SSTableBuilder;

    async fn create_test_sstable(
        store: &Arc<dyn ObjectStore>,
        path: &str,
        entries: Vec<(&[u8], &[u8], u64)>,
    ) -> SSTableMeta {
        let mut builder = SSTableBuilder::new(256, false);
        for (k, v, seq) in &entries {
            builder.add(k.to_vec(), *seq, MemTableValue::Put(v.to_vec()));
        }
        let (data, mut meta) = builder.build().unwrap();
        meta.path = path.to_string();
        store.put(&Path::from(path), data.into()).await.unwrap();
        meta
    }

    #[tokio::test]
    async fn test_needs_compaction() {
        let store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let config = CompactionConfig { l0_threshold: 2, ..Default::default() };
        let compactor = Compactor::new(config, store.clone(), "/test".to_string());

        let mut manifest = Manifest::new();
        assert!(!compactor.needs_compaction(&manifest));

        // Add 2 L0 tables to trigger compaction
        let meta1 = create_test_sstable(&store, "/test/L0/1.sst", vec![
            (b"a", b"1", 1), (b"c", b"3", 3),
        ]).await;
        manifest.add_l0_sstable(meta1);

        let meta2 = create_test_sstable(&store, "/test/L0/2.sst", vec![
            (b"b", b"2", 2), (b"d", b"4", 4),
        ]).await;
        manifest.add_l0_sstable(meta2);

        assert!(compactor.needs_compaction(&manifest));
    }

    #[tokio::test]
    async fn test_compact_l0() {
        let store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let config = CompactionConfig { l0_threshold: 2, block_size: 256, ..Default::default() };
        let compactor = Compactor::new(config, store.clone(), "/test".to_string());

        let mut manifest = Manifest::new();

        let meta1 = create_test_sstable(&store, "/test/L0/1.sst", vec![
            (b"a", b"v1", 1), (b"c", b"v3", 3),
        ]).await;
        manifest.add_l0_sstable(meta1);

        let meta2 = create_test_sstable(&store, "/test/L0/2.sst", vec![
            (b"a", b"v1_new", 10), (b"b", b"v2", 2),
        ]).await;
        manifest.add_l0_sstable(meta2);

        let result = compactor.compact_l0(&manifest).await.unwrap().unwrap();
        assert_eq!(result.source_ids.len(), 2);
        assert_eq!(result.new_tables.len(), 1);

        // Verify the compacted SSTable has the correct merged data
        let reader = SSTableReader::open(result.new_table_data[0].clone()).unwrap();
        assert_eq!(reader.get(b"a").unwrap(), Some(Some(b"v1_new".to_vec()))); // Newer wins
        assert_eq!(reader.get(b"b").unwrap(), Some(Some(b"v2".to_vec())));
        assert_eq!(reader.get(b"c").unwrap(), Some(Some(b"v3".to_vec())));
    }

    #[tokio::test]
    async fn test_compact_l1_to_l2() {
        let store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let config = CompactionConfig {
            l0_threshold: 4,
            block_size: 256,
            target_sstable_size: 64 * 1024 * 1024,
            level_size_multiplier: 10,
            l1_max_tables: 2,
            enable_compression: false,
        };
        let compactor = Compactor::new(config, store.clone(), "/test".to_string());

        let mut manifest = Manifest::new();

        // Create 3 L1 SSTables (exceeds l1_max_tables=2) with overlapping key ranges
        let mut meta1 = create_test_sstable(&store, "/test/L1/1.sst", vec![
            (b"a", b"v1", 1), (b"b", b"v2", 2),
        ]).await;
        meta1.level = 1;
        meta1.id = manifest.next_sstable_id;
        manifest.next_sstable_id += 1;
        manifest.levels[1].push(meta1);

        let mut meta2 = create_test_sstable(&store, "/test/L1/2.sst", vec![
            (b"c", b"v3", 3), (b"d", b"v4", 4),
        ]).await;
        meta2.level = 1;
        meta2.id = manifest.next_sstable_id;
        manifest.next_sstable_id += 1;
        manifest.levels[1].push(meta2);

        let mut meta3 = create_test_sstable(&store, "/test/L1/3.sst", vec![
            (b"a", b"v1_new", 10), (b"e", b"v5", 5),
        ]).await;
        meta3.level = 1;
        meta3.id = manifest.next_sstable_id;
        manifest.next_sstable_id += 1;
        manifest.levels[1].push(meta3);

        // Also create an overlapping L2 SSTable
        let mut l2_meta = create_test_sstable(&store, "/test/L2/100.sst", vec![
            (b"a", b"v1_old", 0), (b"b", b"v2_old", 0),
        ]).await;
        l2_meta.level = 2;
        l2_meta.id = manifest.next_sstable_id;
        manifest.next_sstable_id += 1;
        manifest.levels[2].push(l2_meta);

        assert!(compactor.needs_level_compaction(&manifest, 1));

        let result = compactor.compact_level(&manifest, 1).await.unwrap().unwrap();
        assert_eq!(result.source_level, 1);
        assert_eq!(result.target_level, 2);
        assert_eq!(result.new_tables.len(), 1);

        // The picked SSTable is the one with the oldest min_sequence.
        // meta1 has min_seq=1, meta2 has min_seq=3, meta3 has min_seq=5.
        // So meta1 (keys a,b) is picked. The overlapping L2 table (keys a,b) is merged.
        // Source IDs should be: meta1.id (L1) + l2_meta.id (L2 overlap)
        assert_eq!(result.source_ids.len(), 2);

        // Verify merged output: newest version of each key wins
        let reader = SSTableReader::open(result.new_table_data[0].clone()).unwrap();
        // key "a": seq 1 from meta1 vs seq 0 from L2 → meta1 wins (v1)
        assert_eq!(reader.get(b"a").unwrap(), Some(Some(b"v1".to_vec())));
        // key "b": seq 2 from meta1 vs seq 0 from L2 → meta1 wins (v2)
        assert_eq!(reader.get(b"b").unwrap(), Some(Some(b"v2".to_vec())));
    }

    #[tokio::test]
    async fn test_needs_level_compaction() {
        let store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let config = CompactionConfig {
            l1_max_tables: 2,
            level_size_multiplier: 10,
            ..Default::default()
        };
        let compactor = Compactor::new(config, store.clone(), "/test".to_string());

        let mut manifest = Manifest::new();
        // L0 check should not be done via needs_level_compaction
        assert!(!compactor.needs_level_compaction(&manifest, 0));

        // 2 tables in L1 → not over threshold (max=2)
        for i in 0..2u64 {
            let mut meta = create_test_sstable(
                &store,
                &format!("/test/L1/{}.sst", i),
                vec![(b"a", b"v", i)],
            ).await;
            meta.level = 1;
            meta.id = manifest.next_sstable_id;
            manifest.next_sstable_id += 1;
            manifest.levels[1].push(meta);
        }
        assert!(!compactor.needs_level_compaction(&manifest, 1));

        // 3rd table → over threshold
        let mut extra = create_test_sstable(
            &store, "/test/L1/extra.sst", vec![(b"z", b"v", 99)],
        ).await;
        extra.level = 1;
        extra.id = manifest.next_sstable_id;
        manifest.next_sstable_id += 1;
        manifest.levels[1].push(extra);
        assert!(compactor.needs_level_compaction(&manifest, 1));
    }
}
