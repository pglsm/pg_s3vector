//! Write-Ahead Log (WAL) for crash-safe durability.
//!
//! Every write operation is appended to a WAL segment on object storage
//! *before* it enters the MemTable. On recovery, unflushed WAL segments
//! are replayed to restore the MemTable state.
//!
//! ## Segment Format
//!
//! ```text
//! [magic: 4B][version: 4B][entry][entry]...[entry]
//! ```
//!
//! Each entry:
//! ```text
//! [crc32: 4B][op_type: 1B][key_len: 4B][key: N][val_len: 4B][value: M][sequence: 8B]
//! ```

use crate::{LsmError, LsmResult, WriteOp};
use bytes::{BufMut, BytesMut};
use object_store::path::Path;
use object_store::ObjectStore;
use std::sync::Arc;

const WAL_MAGIC: u32 = 0x57414C31; // "WAL1"
const WAL_VERSION: u32 = 1;
const HEADER_SIZE: usize = 8; // magic + version

/// Op type markers in the WAL entry.
const OP_PUT: u8 = 1;
const OP_DELETE: u8 = 2;

/// A single WAL entry deserialized from a segment.
#[derive(Debug, Clone)]
pub struct WalEntry {
    pub op: WriteOp,
    pub sequence: u64,
}

/// Writes WAL entries to object storage segments.
pub struct WalWriter {
    object_store: Arc<dyn ObjectStore>,
    root_path: String,
    /// Current segment buffer (accumulated before PUT).
    buffer: BytesMut,
    /// Current segment sequence number.
    segment_seq: u64,
    /// Maximum segment size before rotation.
    segment_size_limit: usize,
    /// Sequence numbers of active (untrimmed) segments.
    active_segments: Vec<u64>,
}

impl WalWriter {
    /// Create a new WAL writer.
    pub fn new(
        object_store: Arc<dyn ObjectStore>,
        root_path: &str,
        start_segment_seq: u64,
        segment_size_limit: usize,
    ) -> Self {
        let mut buffer = BytesMut::with_capacity(segment_size_limit);
        // Write segment header
        buffer.put_u32(WAL_MAGIC);
        buffer.put_u32(WAL_VERSION);

        Self {
            object_store,
            root_path: root_path.to_string(),
            buffer,
            segment_seq: start_segment_seq,
            segment_size_limit,
            active_segments: vec![start_segment_seq],
        }
    }

    /// Append a write operation to the WAL.
    ///
    /// Returns the segment sequence that contains this entry.
    pub async fn append(&mut self, op: &WriteOp, sequence: u64) -> LsmResult<u64> {
        let entry_bytes = Self::encode_entry(op, sequence);
        
        // Check if we need to rotate
        if self.buffer.len() + entry_bytes.len() > self.segment_size_limit {
            self.flush_segment().await?;
            self.rotate();
        }

        self.buffer.put_slice(&entry_bytes);
        Ok(self.segment_seq)
    }

    /// Flush the current segment buffer to object storage.
    pub async fn flush_segment(&mut self) -> LsmResult<()> {
        if self.buffer.len() <= HEADER_SIZE {
            return Ok(()); // Nothing to flush (only header)
        }

        let path = self.segment_path(self.segment_seq);
        let data = self.buffer.clone().freeze();
        self.object_store
            .put(&path, data.into())
            .await
            .map_err(LsmError::ObjectStore)?;

        Ok(())
    }

    /// Sync: flush current segment and ensure it's persisted.
    pub async fn sync(&mut self) -> LsmResult<()> {
        self.flush_segment().await
    }

    /// Rotate to a new segment.
    fn rotate(&mut self) {
        self.segment_seq += 1;
        self.buffer.clear();
        self.buffer.put_u32(WAL_MAGIC);
        self.buffer.put_u32(WAL_VERSION);
        self.active_segments.push(self.segment_seq);
    }

    /// Trim (delete) WAL segments that have been fully flushed to SSTables.
    ///
    /// `up_to_seq` is the sequence number up to which data has been flushed.
    /// All segments with seq < current can be safely deleted.
    pub async fn trim(&mut self, segments_to_remove: &[u64]) -> LsmResult<()> {
        for &seg_seq in segments_to_remove {
            let path = self.segment_path(seg_seq);
            match self.object_store.delete(&path).await {
                Ok(()) => {}
                Err(object_store::Error::NotFound { .. }) => {} // Already deleted
                Err(e) => return Err(LsmError::ObjectStore(e)),
            }
            self.active_segments.retain(|&s| s != seg_seq);
        }
        Ok(())
    }

    /// Register a previously-discovered WAL segment so it can be trimmed after flush.
    pub fn add_discovered_segment(&mut self, seq: u64) {
        if !self.active_segments.contains(&seq) {
            self.active_segments.push(seq);
            self.active_segments.sort();
        }
    }

    /// Get all active segment sequence numbers.
    pub fn active_segments(&self) -> &[u64] {
        &self.active_segments
    }

    /// Get the current segment sequence.
    pub fn current_segment_seq(&self) -> u64 {
        self.segment_seq
    }

    /// Encode a write op + sequence into a WAL entry.
    fn encode_entry(op: &WriteOp, sequence: u64) -> Vec<u8> {
        let mut buf = BytesMut::new();

        // Reserve space for CRC (we'll fill it in at the end)
        let crc_pos = 0;
        buf.put_u32(0); // placeholder for CRC

        match op {
            WriteOp::Put { key, value } => {
                buf.put_u8(OP_PUT);
                buf.put_u32(key.len() as u32);
                buf.put_slice(key);
                buf.put_u32(value.len() as u32);
                buf.put_slice(value);
            }
            WriteOp::Delete { key } => {
                buf.put_u8(OP_DELETE);
                buf.put_u32(key.len() as u32);
                buf.put_slice(key);
                buf.put_u32(0); // no value
            }
        }

        buf.put_u64(sequence);

        // Compute CRC over everything after the CRC field
        let crc = crc32_compute(&buf[4..]);
        let crc_bytes = crc.to_be_bytes();
        buf[crc_pos] = crc_bytes[0];
        buf[crc_pos + 1] = crc_bytes[1];
        buf[crc_pos + 2] = crc_bytes[2];
        buf[crc_pos + 3] = crc_bytes[3];

        buf.to_vec()
    }

    /// Path for a WAL segment.
    fn segment_path(&self, seq: u64) -> Path {
        Path::from(format!("{}/wal/segment-{:016}.wal", self.root_path, seq))
    }
}

/// Reads and replays WAL segments for recovery.
pub struct WalReader {
    object_store: Arc<dyn ObjectStore>,
    root_path: String,
}

impl WalReader {
    pub fn new(object_store: Arc<dyn ObjectStore>, root_path: &str) -> Self {
        Self {
            object_store,
            root_path: root_path.to_string(),
        }
    }

    /// Discover all WAL segments on object storage.
    pub async fn discover_segments(&self) -> LsmResult<Vec<u64>> {
        let prefix = Path::from(format!("{}/wal/", self.root_path));
        let mut segments = Vec::new();

        let list = self.object_store
            .list(Some(&prefix));

        use futures::TryStreamExt;
        let objects: Vec<_> = list.try_collect().await
            .map_err(LsmError::ObjectStore)?;

        for obj in objects {
            let name = obj.location.filename().unwrap_or("");
            if let Some(seq_str) = name.strip_prefix("segment-").and_then(|s| s.strip_suffix(".wal")) {
                if let Ok(seq) = seq_str.parse::<u64>() {
                    segments.push(seq);
                }
            }
        }

        segments.sort();
        Ok(segments)
    }

    /// Replay all WAL segments, returning entries in order.
    pub async fn replay_all(&self) -> LsmResult<Vec<WalEntry>> {
        let segments = self.discover_segments().await?;
        let mut all_entries = Vec::new();

        for seg_seq in segments {
            let entries = self.replay_segment(seg_seq).await?;
            all_entries.extend(entries);
        }

        // Sort by sequence to ensure correct replay order
        all_entries.sort_by_key(|e| e.sequence);
        Ok(all_entries)
    }

    /// Replay a single WAL segment.
    pub async fn replay_segment(&self, seg_seq: u64) -> LsmResult<Vec<WalEntry>> {
        let path = Path::from(format!("{}/wal/segment-{:016}.wal", self.root_path, seg_seq));

        let data = match self.object_store.get(&path).await {
            Ok(result) => result.bytes().await.map_err(LsmError::ObjectStore)?,
            Err(object_store::Error::NotFound { .. }) => return Ok(Vec::new()),
            Err(e) => return Err(LsmError::ObjectStore(e)),
        };

        if data.len() < HEADER_SIZE {
            return Err(LsmError::WalCorruption("Segment too small".into()));
        }

        // Validate header
        let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        if magic != WAL_MAGIC {
            return Err(LsmError::WalCorruption("Invalid WAL magic".into()));
        }

        let version = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        if version != WAL_VERSION {
            return Err(LsmError::WalCorruption(
                format!("Unsupported WAL version: {}", version),
            ));
        }

        // Parse entries
        let mut entries = Vec::new();
        let mut pos = HEADER_SIZE;

        while pos < data.len() {
            match Self::decode_entry(&data[pos..]) {
                Ok((entry, bytes_consumed)) => {
                    entries.push(entry);
                    pos += bytes_consumed;
                }
                Err(e) => {
                    // Partial/corrupted entry at end — stop here (crash mid-write)
                    tracing::warn!("WAL segment {} truncated at offset {}: {}", seg_seq, pos, e);
                    break;
                }
            }
        }

        Ok(entries)
    }

    /// Decode a single WAL entry from a byte slice.
    /// Returns (entry, bytes_consumed).
    fn decode_entry(data: &[u8]) -> LsmResult<(WalEntry, usize)> {
        // Minimum entry: crc(4) + op(1) + key_len(4) + val_len(4) + seq(8) = 21
        if data.len() < 21 {
            return Err(LsmError::WalCorruption("Entry too small".into()));
        }

        let stored_crc = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let op_type = data[4];
        let key_len = u32::from_be_bytes([data[5], data[6], data[7], data[8]]) as usize;

        let key_end = 9 + key_len;
        if data.len() < key_end + 4 {
            return Err(LsmError::WalCorruption("Key truncated".into()));
        }

        let key = data[9..key_end].to_vec();
        let val_len = u32::from_be_bytes([
            data[key_end],
            data[key_end + 1],
            data[key_end + 2],
            data[key_end + 3],
        ]) as usize;

        let val_start = key_end + 4;
        let val_end = val_start + val_len;
        if data.len() < val_end + 8 {
            return Err(LsmError::WalCorruption("Value/sequence truncated".into()));
        }

        let value = data[val_start..val_end].to_vec();
        let sequence = u64::from_be_bytes([
            data[val_end],
            data[val_end + 1],
            data[val_end + 2],
            data[val_end + 3],
            data[val_end + 4],
            data[val_end + 5],
            data[val_end + 6],
            data[val_end + 7],
        ]);

        let total_size = val_end + 8;

        // Validate CRC
        let computed_crc = crc32_compute(&data[4..total_size]);
        if stored_crc != computed_crc {
            return Err(LsmError::WalCorruption(format!(
                "CRC mismatch: stored={:#x}, computed={:#x}",
                stored_crc, computed_crc
            )));
        }

        let op = match op_type {
            OP_PUT => WriteOp::Put { key, value },
            OP_DELETE => WriteOp::Delete { key },
            _ => return Err(LsmError::WalCorruption(format!("Unknown op type: {}", op_type))),
        };

        Ok((WalEntry { op, sequence }, total_size))
    }
}

/// Simple CRC32 (IEEE) computation.
fn crc32_compute(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    fn test_store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    #[tokio::test]
    async fn test_wal_write_read_roundtrip() {
        let store = test_store();
        let mut writer = WalWriter::new(store.clone(), "/test", 1, 1024 * 1024);

        let op1 = WriteOp::Put { key: b"k1".to_vec(), value: b"v1".to_vec() };
        let op2 = WriteOp::Put { key: b"k2".to_vec(), value: b"v2".to_vec() };
        let op3 = WriteOp::Delete { key: b"k1".to_vec() };

        writer.append(&op1, 1).await.unwrap();
        writer.append(&op2, 2).await.unwrap();
        writer.append(&op3, 3).await.unwrap();
        writer.sync().await.unwrap();

        // Read back
        let reader = WalReader::new(store, "/test");
        let entries = reader.replay_all().await.unwrap();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].sequence, 1);
        assert_eq!(entries[1].sequence, 2);
        assert_eq!(entries[2].sequence, 3);

        match &entries[0].op {
            WriteOp::Put { key, value } => {
                assert_eq!(key, b"k1");
                assert_eq!(value, b"v1");
            }
            _ => panic!("Expected Put"),
        }

        match &entries[2].op {
            WriteOp::Delete { key } => assert_eq!(key, b"k1"),
            _ => panic!("Expected Delete"),
        }
    }

    #[tokio::test]
    async fn test_wal_segment_rotation() {
        let store = test_store();
        // Very small segment size to force rotation
        let mut writer = WalWriter::new(store.clone(), "/test", 1, 64);

        for i in 0..10 {
            let op = WriteOp::Put {
                key: format!("key-{}", i).into_bytes(),
                value: format!("val-{}", i).into_bytes(),
            };
            writer.append(&op, i as u64).await.unwrap();
        }
        writer.sync().await.unwrap();

        // Should have created multiple segments
        let reader = WalReader::new(store, "/test");
        let segments = reader.discover_segments().await.unwrap();
        assert!(segments.len() > 1, "Expected multiple segments, got {}", segments.len());

        // All entries should be recoverable
        let entries = reader.replay_all().await.unwrap();
        assert_eq!(entries.len(), 10);
    }

    #[tokio::test]
    async fn test_wal_trim() {
        let store = test_store();
        let mut writer = WalWriter::new(store.clone(), "/test", 1, 64);

        for i in 0..5 {
            let op = WriteOp::Put { key: vec![i], value: vec![i] };
            writer.append(&op, i as u64).await.unwrap();
        }
        writer.sync().await.unwrap();

        let reader = WalReader::new(store.clone(), "/test");
        let segments_before = reader.discover_segments().await.unwrap();
        assert!(!segments_before.is_empty());

        // Trim all but the current segment
        let to_trim: Vec<u64> = segments_before.iter()
            .filter(|&&s| s < writer.current_segment_seq())
            .copied()
            .collect();
        writer.trim(&to_trim).await.unwrap();

        let segments_after = reader.discover_segments().await.unwrap();
        assert!(segments_after.len() < segments_before.len() || to_trim.is_empty());
    }

    #[tokio::test]
    async fn test_wal_crc_validation() {
        let store = test_store();
        let mut writer = WalWriter::new(store.clone(), "/test", 1, 1024 * 1024);

        let op = WriteOp::Put { key: b"key".to_vec(), value: b"val".to_vec() };
        writer.append(&op, 1).await.unwrap();
        writer.sync().await.unwrap();

        // Corrupt the segment
        let path = Path::from("/test/wal/segment-0000000000000001.wal");
        let data = store.get(&path).await.unwrap().bytes().await.unwrap();
        let mut corrupted = data.to_vec();
        if corrupted.len() > HEADER_SIZE + 5 {
            corrupted[HEADER_SIZE + 5] ^= 0xFF; // Flip a byte in the entry
        }
        store.put(&path, corrupted.into()).await.unwrap();

        // Should detect corruption
        let reader = WalReader::new(store, "/test");
        let entries = reader.replay_all().await.unwrap();
        // Corrupted entry should be skipped (truncated at corruption)
        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn test_crc32() {
        let data = b"hello world";
        let crc = crc32_compute(data);
        // Verify it produces a consistent value
        assert_eq!(crc, crc32_compute(data));
        // Different data should produce different CRC
        assert_ne!(crc, crc32_compute(b"hello worlD"));
    }

    #[test]
    fn test_entry_encode_decode() {
        let op = WriteOp::Put { key: b"test_key".to_vec(), value: b"test_value".to_vec() };
        let encoded = WalWriter::encode_entry(&op, 42);
        let (entry, size) = WalReader::decode_entry(&encoded).unwrap();

        assert_eq!(size, encoded.len());
        assert_eq!(entry.sequence, 42);
        match entry.op {
            WriteOp::Put { key, value } => {
                assert_eq!(key, b"test_key");
                assert_eq!(value, b"test_value");
            }
            _ => panic!("Expected Put"),
        }
    }
}
