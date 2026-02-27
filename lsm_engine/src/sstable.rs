//! SSTable (Sorted String Table) format for persisting MemTable data to object storage.
//!
//! An SSTable is an immutable, sorted file containing key-value pairs organized into
//! fixed-size blocks. Each SSTable has:
//! - Data blocks: sorted key-value pairs
//! - Index block: maps key ranges to data block offsets
//! - Footer: metadata (entry count, key range, bloom filter)

use crate::memtable::MemTableValue;
use crate::{KvEntry, LsmError, LsmResult};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};

/// Metadata about an SSTable stored on object storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSTableMeta {
    /// Unique ID for this SSTable.
    pub id: u64,
    /// The level in the LSM tree (0 = most recent).
    pub level: u32,
    /// Smallest key in this SSTable.
    pub min_key: Vec<u8>,
    /// Largest key in this SSTable.
    pub max_key: Vec<u8>,
    /// Number of entries.
    pub entry_count: u64,
    /// Size on object storage in bytes.
    pub size_bytes: u64,
    /// Object storage path.
    pub path: String,
    /// Sequence number range covered.
    pub min_sequence: u64,
    pub max_sequence: u64,
}

/// A single data block within an SSTable.
#[derive(Debug, Clone)]
pub struct DataBlock {
    /// Sorted key-value entries in this block.
    pub entries: Vec<BlockEntry>,
}

/// An entry within a data block.
#[derive(Debug, Clone)]
pub struct BlockEntry {
    pub key: Vec<u8>,
    pub sequence: u64,
    pub is_tombstone: bool,
    pub value: Vec<u8>,
}

/// Index entry mapping a key to a data block offset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    /// The first key in the referenced data block.
    pub first_key: Vec<u8>,
    /// Byte offset of the data block.
    pub offset: u64,
    /// Size of the data block in bytes.
    pub size: u32,
}

/// Footer of an SSTable containing metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSTableFooter {
    /// Offset of the index block.
    pub index_offset: u64,
    /// Number of data blocks.
    pub data_block_count: u32,
    /// Total number of entries.
    pub entry_count: u64,
    /// Magic number for validation.
    pub magic: u64,
    /// Whether data blocks are zstd-compressed.
    #[serde(default)]
    pub compressed: bool,
}

const SSTABLE_MAGIC: u64 = 0x4C534D5353540001; // "LSMSST\x00\x01"

/// Builder for creating SSTables from sorted key-value entries.
pub struct SSTableBuilder {
    block_size: usize,
    compressed: bool,
    current_block: Vec<BlockEntry>,
    current_block_size: usize,
    data_blocks: Vec<Bytes>,
    index_entries: Vec<IndexEntry>,
    entry_count: u64,
    min_key: Option<Vec<u8>>,
    max_key: Option<Vec<u8>>,
    min_sequence: u64,
    max_sequence: u64,
}

impl SSTableBuilder {
    /// Create a new SSTable builder with the given block size.
    /// If `compressed` is true, data blocks are zstd-compressed.
    pub fn new(block_size: usize, compressed: bool) -> Self {
        Self {
            block_size,
            compressed,
            current_block: Vec::new(),
            current_block_size: 0,
            data_blocks: Vec::new(),
            index_entries: Vec::new(),
            entry_count: 0,
            min_key: None,
            max_key: None,
            min_sequence: u64::MAX,
            max_sequence: 0,
        }
    }

    /// Add an entry to the SSTable. Entries must be added in sorted key order.
    pub fn add(&mut self, key: Vec<u8>, sequence: u64, value: MemTableValue) {
        let (is_tombstone, val_bytes) = match &value {
            MemTableValue::Put(v) => (false, v.clone()),
            MemTableValue::Delete => (true, Vec::new()),
        };

        let entry_size = key.len() + val_bytes.len() + 17; // key + val + seq(8) + tombstone(1) + lengths(8)

        // Update key range
        if self.min_key.is_none() {
            self.min_key = Some(key.clone());
        }
        self.max_key = Some(key.clone());

        // Update sequence range
        self.min_sequence = self.min_sequence.min(sequence);
        self.max_sequence = self.max_sequence.max(sequence);

        let entry = BlockEntry {
            key,
            sequence,
            is_tombstone,
            value: val_bytes,
        };

        self.current_block.push(entry);
        self.current_block_size += entry_size;
        self.entry_count += 1;

        // Flush block if it exceeds the target size
        if self.current_block_size >= self.block_size {
            self.flush_block();
        }
    }

    /// Flush the current data block.
    fn flush_block(&mut self) {
        if self.current_block.is_empty() {
            return;
        }

        let first_key = self.current_block[0].key.clone();
        let block_data = Self::encode_block(&self.current_block, self.compressed);
        let offset = self.data_blocks.iter().map(|b| b.len() as u64).sum();

        self.index_entries.push(IndexEntry {
            first_key,
            offset,
            size: block_data.len() as u32,
        });

        self.data_blocks.push(block_data);
        self.current_block.clear();
        self.current_block_size = 0;
    }

    /// Encode a data block into bytes.
    ///
    /// On-disk format:
    /// - `[flag: 1B]` — 0 = uncompressed, 1 = zstd
    /// - If compressed: `[uncompressed_size: 4B][compressed_data...]`
    /// - If uncompressed: `[raw_data...]`
    fn encode_block(entries: &[BlockEntry], compressed: bool) -> Bytes {
        let mut raw = BytesMut::new();

        raw.put_u32(entries.len() as u32);

        for entry in entries {
            raw.put_u32(entry.key.len() as u32);
            raw.put_slice(&entry.key);
            raw.put_u64(entry.sequence);
            raw.put_u8(if entry.is_tombstone { 1 } else { 0 });
            raw.put_u32(entry.value.len() as u32);
            raw.put_slice(&entry.value);
        }

        let raw = raw.freeze();

        let mut out = BytesMut::new();
        if compressed {
            let compressed_data = zstd::encode_all(&raw[..], 3)
                .expect("zstd compression should not fail on in-memory data");
            out.put_u8(1);
            out.put_u32(raw.len() as u32);
            out.put_slice(&compressed_data);
        } else {
            out.put_u8(0);
            out.put_slice(&raw);
        }
        out.freeze()
    }

    /// Decode a data block from bytes.
    ///
    /// Reads the 1-byte compression flag, decompresses if needed, then
    /// parses the raw entry format.
    pub fn decode_block(data: &[u8]) -> LsmResult<Vec<BlockEntry>> {
        if data.len() < 5 {
            return Err(LsmError::Serialization("Block too small".to_string()));
        }

        let flag = data[0];
        let raw: Vec<u8> = match flag {
            0 => data[1..].to_vec(),
            1 => {
                let mut header = &data[1..5];
                let _uncompressed_size = header.get_u32() as usize;
                zstd::decode_all(&data[5..])
                    .map_err(|e| LsmError::Serialization(format!("zstd decompress: {}", e)))?
            }
            other => {
                return Err(LsmError::Serialization(
                    format!("Unknown block compression flag: {}", other),
                ));
            }
        };

        let mut cursor = std::io::Cursor::new(&raw[..]);
        let mut entries = Vec::new();

        let count = cursor.get_u32() as usize;

        for _ in 0..count {
            let key_len = cursor.get_u32() as usize;
            let mut key = vec![0u8; key_len];
            std::io::Read::read_exact(&mut cursor, &mut key)
                .map_err(|e| LsmError::Serialization(e.to_string()))?;

            let sequence = cursor.get_u64();
            let is_tombstone = cursor.get_u8() == 1;

            let val_len = cursor.get_u32() as usize;
            let mut value = vec![0u8; val_len];
            std::io::Read::read_exact(&mut cursor, &mut value)
                .map_err(|e| LsmError::Serialization(e.to_string()))?;

            entries.push(BlockEntry {
                key,
                sequence,
                is_tombstone,
                value,
            });
        }

        Ok(entries)
    }

    /// Finalize and build the SSTable, returning the complete bytes.
    ///
    /// File layout:
    /// ```text
    /// [data blocks...] [index JSON] [index_len: u32] [footer JSON] [footer_len: u32]
    /// ```
    /// The footer_len u32 is the very last 4 bytes. Read it, then read footer.
    /// The footer contains index_offset and index_length to locate the index.
    pub fn build(mut self) -> LsmResult<(Bytes, SSTableMeta)> {
        // Flush any remaining block
        self.flush_block();

        if self.data_blocks.is_empty() {
            return Err(LsmError::Serialization("Cannot build empty SSTable".to_string()));
        }

        let mut buf = BytesMut::new();

        // Write all data blocks
        for block in &self.data_blocks {
            buf.put_slice(block);
        }

        // Write index block
        let index_offset = buf.len() as u64;
        let index_json = serde_json::to_vec(&self.index_entries)
            .map_err(|e| LsmError::Serialization(e.to_string()))?;
        let index_len = index_json.len() as u32;
        buf.put_slice(&index_json);
        buf.put_u32(index_len); // index length comes AFTER index data

        // Write footer
        let footer = SSTableFooter {
            index_offset,
            data_block_count: self.data_blocks.len() as u32,
            entry_count: self.entry_count,
            magic: SSTABLE_MAGIC,
            compressed: self.compressed,
        };
        let footer_json = serde_json::to_vec(&footer)
            .map_err(|e| LsmError::Serialization(e.to_string()))?;
        let footer_len = footer_json.len() as u32;
        buf.put_slice(&footer_json);
        buf.put_u32(footer_len); // footer length is the VERY LAST 4 bytes

        let total_bytes = buf.freeze();

        let meta = SSTableMeta {
            id: 0, // Will be set by the manifest
            level: 0,
            min_key: self.min_key.unwrap_or_default(),
            max_key: self.max_key.unwrap_or_default(),
            entry_count: self.entry_count,
            size_bytes: total_bytes.len() as u64,
            path: String::new(), // Will be set when persisted
            min_sequence: self.min_sequence,
            max_sequence: self.max_sequence,
        };

        Ok((total_bytes, meta))
    }
}

/// Reader for SSTables stored on object storage.
pub struct SSTableReader {
    data: Bytes,
    footer: SSTableFooter,
    index: Vec<IndexEntry>,
}

impl SSTableReader {
    /// Open an SSTable from raw bytes.
    ///
    /// File layout (from build):
    /// ```text
    /// [data blocks...] [index JSON] [index_len: u32] [footer JSON] [footer_len: u32]
    /// ```
    pub fn open(data: Bytes) -> LsmResult<Self> {
        if data.len() < 8 {
            return Err(LsmError::Serialization("SSTable too small".to_string()));
        }

        let total_len = data.len();

        // 1. Read footer_len from the very last 4 bytes
        let footer_len = {
            let mut slice = &data[total_len - 4..];
            slice.get_u32() as usize
        };

        if footer_len + 4 > total_len {
            return Err(LsmError::Serialization("Footer length exceeds file size".to_string()));
        }

        // 2. Read footer JSON (right before the footer_len u32)
        let footer_end = total_len - 4;
        let footer_start = footer_end - footer_len;
        let footer: SSTableFooter = serde_json::from_slice(&data[footer_start..footer_end])
            .map_err(|e| LsmError::Serialization(format!("Footer: {}", e)))?;

        if footer.magic != SSTABLE_MAGIC {
            return Err(LsmError::Serialization("Invalid SSTable magic number".to_string()));
        }

        // 3. Read index_len (the u32 right before the footer JSON)
        let index_len_end = footer_start;
        let index_len_start = index_len_end - 4;
        let index_len = {
            let mut slice = &data[index_len_start..index_len_end];
            slice.get_u32() as usize
        };

        // 4. Read index JSON (starts at index_offset, length = index_len)
        let idx_start = footer.index_offset as usize;
        let index: Vec<IndexEntry> = serde_json::from_slice(&data[idx_start..idx_start + index_len])
            .map_err(|e| LsmError::Serialization(format!("Index: {}", e)))?;

        Ok(Self { data, footer, index })
    }

    /// Look up a key in the SSTable.
    ///
    /// Returns:
    /// - `Ok(Some(Some(value)))` — key found with a live value
    /// - `Ok(Some(None))` — key found but tombstoned (deleted)
    /// - `Ok(None)` — key not present in this SSTable
    pub fn get(&self, key: &[u8]) -> LsmResult<Option<Option<Vec<u8>>>> {
        // Binary search the index to find the right block
        let block_idx = self.find_block(key);
        if block_idx >= self.index.len() {
            return Ok(None);
        }

        let entry = &self.index[block_idx];
        let block_data = &self.data[entry.offset as usize..(entry.offset as usize + entry.size as usize)];
        let entries = SSTableBuilder::decode_block(block_data)?;

        for e in entries {
            if e.key == key {
                if e.is_tombstone {
                    return Ok(Some(None));
                }
                return Ok(Some(Some(e.value)));
            }
        }

        Ok(None)
    }

    /// Scan entries in the SSTable within a key range.
    pub fn scan(&self, start: &[u8], end: &[u8]) -> LsmResult<Vec<KvEntry>> {
        let mut results = Vec::new();
        let start_block = self.find_block(start);

        for i in start_block..self.index.len() {
            let entry = &self.index[i];

            // If the block's first key is past our end, stop
            if i > start_block && entry.first_key.as_slice() >= end {
                break;
            }

            let block_data = &self.data[entry.offset as usize..(entry.offset as usize + entry.size as usize)];
            let entries = SSTableBuilder::decode_block(block_data)?;

            for e in entries {
                if e.key.as_slice() >= start && e.key.as_slice() < end && !e.is_tombstone {
                    results.push(KvEntry::new(e.key, e.value, e.sequence));
                }
            }
        }

        Ok(results)
    }

    /// Find the block that might contain the given key (binary search).
    fn find_block(&self, key: &[u8]) -> usize {
        match self.index.binary_search_by(|e| e.first_key.as_slice().cmp(key)) {
            Ok(i) => i,
            Err(i) => if i > 0 { i - 1 } else { 0 },
        }
    }

    /// Scan all entries including tombstones (for compaction merges).
    ///
    /// Unlike `scan()`, this returns tombstoned entries so compaction can
    /// propagate deletes across levels.
    pub fn scan_all_with_tombstones(&self, start: &[u8], end: &[u8]) -> LsmResult<Vec<(Vec<u8>, u64, bool, Vec<u8>)>> {
        let mut results = Vec::new();
        let start_block = self.find_block(start);

        for i in start_block..self.index.len() {
            let entry = &self.index[i];

            if i > start_block && entry.first_key.as_slice() >= end {
                break;
            }

            let block_data = &self.data[entry.offset as usize..(entry.offset as usize + entry.size as usize)];
            let entries = SSTableBuilder::decode_block(block_data)?;

            for e in entries {
                if e.key.as_slice() >= start && e.key.as_slice() < end {
                    results.push((e.key, e.sequence, e.is_tombstone, e.value));
                }
            }
        }

        Ok(results)
    }

    /// Get the total number of entries.
    pub fn entry_count(&self) -> u64 {
        self.footer.entry_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memtable::MemTableValue;

    fn build_test_sstable() -> (Bytes, SSTableMeta) {
        let mut builder = SSTableBuilder::new(128, false); // Small block size for testing
        // Keys must be in sorted (lexicographic) order
        builder.add(b"alpha".to_vec(), 1, MemTableValue::Put(b"a_val".to_vec()));
        builder.add(b"beta".to_vec(), 2, MemTableValue::Put(b"b_val".to_vec()));
        builder.add(b"delta".to_vec(), 3, MemTableValue::Delete);
        builder.add(b"epsilon".to_vec(), 4, MemTableValue::Put(b"e_val".to_vec()));
        builder.add(b"gamma".to_vec(), 5, MemTableValue::Put(b"g_val".to_vec()));
        builder.build().unwrap()
    }

    #[test]
    fn test_build_and_read() {
        let (data, meta) = build_test_sstable();
        assert_eq!(meta.entry_count, 5);
        assert_eq!(meta.min_key, b"alpha".to_vec());
        assert_eq!(meta.max_key, b"gamma".to_vec());

        let reader = SSTableReader::open(data).unwrap();
        assert_eq!(reader.entry_count(), 5);
    }

    #[test]
    fn test_point_lookup() {
        let (data, _) = build_test_sstable();
        let reader = SSTableReader::open(data).unwrap();

        assert_eq!(reader.get(b"alpha").unwrap(), Some(Some(b"a_val".to_vec())));
        assert_eq!(reader.get(b"beta").unwrap(), Some(Some(b"b_val".to_vec())));
        assert_eq!(reader.get(b"gamma").unwrap(), Some(Some(b"g_val".to_vec())));

        // Tombstoned key: present but deleted
        assert_eq!(reader.get(b"delta").unwrap(), Some(None));

        // Missing key: not present at all
        assert_eq!(reader.get(b"zzz").unwrap(), None);
    }

    #[test]
    fn test_range_scan() {
        let (data, _) = build_test_sstable();
        let reader = SSTableReader::open(data).unwrap();

        // Range [alpha, gamma): includes alpha, beta, epsilon (delta is tombstoned, gamma excluded)
        let results = reader.scan(b"alpha", b"gamma").unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].key, b"alpha".to_vec());
        assert_eq!(results[1].key, b"beta".to_vec());
        assert_eq!(results[2].key, b"epsilon".to_vec());
    }

    #[test]
    fn test_block_encode_decode() {
        let entries = vec![
            BlockEntry { key: b"k1".to_vec(), sequence: 1, is_tombstone: false, value: b"v1".to_vec() },
            BlockEntry { key: b"k2".to_vec(), sequence: 2, is_tombstone: true, value: vec![] },
        ];

        let encoded = SSTableBuilder::encode_block(&entries, false);
        let decoded = SSTableBuilder::decode_block(&encoded).unwrap();

        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].key, b"k1");
        assert_eq!(decoded[0].value, b"v1");
        assert!(!decoded[0].is_tombstone);
        assert!(decoded[1].is_tombstone);
    }

    #[test]
    fn test_compressed_block_roundtrip() {
        let entries = vec![
            BlockEntry { key: b"key_one".to_vec(), sequence: 1, is_tombstone: false, value: b"value_one".to_vec() },
            BlockEntry { key: b"key_two".to_vec(), sequence: 2, is_tombstone: false, value: b"value_two".to_vec() },
            BlockEntry { key: b"key_three".to_vec(), sequence: 3, is_tombstone: true, value: vec![] },
        ];

        let encoded = SSTableBuilder::encode_block(&entries, true);
        assert_eq!(encoded[0], 1, "First byte should be compression flag = 1 (zstd)");

        let decoded = SSTableBuilder::decode_block(&encoded).unwrap();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].key, b"key_one");
        assert_eq!(decoded[0].value, b"value_one");
        assert_eq!(decoded[1].key, b"key_two");
        assert!(decoded[2].is_tombstone);
    }

    #[test]
    fn test_compressed_sstable_roundtrip() {
        let mut builder = SSTableBuilder::new(128, true);
        builder.add(b"alpha".to_vec(), 1, MemTableValue::Put(b"a_val".to_vec()));
        builder.add(b"beta".to_vec(), 2, MemTableValue::Put(b"b_val".to_vec()));
        builder.add(b"gamma".to_vec(), 3, MemTableValue::Put(b"g_val".to_vec()));

        let (data, meta) = builder.build().unwrap();
        assert_eq!(meta.entry_count, 3);

        let reader = SSTableReader::open(data).unwrap();
        assert_eq!(reader.get(b"alpha").unwrap(), Some(Some(b"a_val".to_vec())));
        assert_eq!(reader.get(b"beta").unwrap(), Some(Some(b"b_val".to_vec())));
        assert_eq!(reader.get(b"gamma").unwrap(), Some(Some(b"g_val".to_vec())));
        assert_eq!(reader.get(b"missing").unwrap(), None);
    }
}
