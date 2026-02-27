# Tutorial 1: The LSM Engine (`lsm_engine` crate)

This tutorial walks through every file in the `lsm_engine` crate, explaining what each struct, function, and design decision does. The engine is a standalone async key-value store — it has no PostgreSQL dependency.

---

## Table of Contents

1. [lib.rs — Public API & Error Types](#1-librs)
2. [config.rs — Configuration](#2-configrs)
3. [memtable.rs — In-Memory Write Buffer](#3-memtablers)
4. [sstable.rs — Sorted String Table Format](#4-sstablers)
5. [wal.rs — Write-Ahead Log](#5-walrs)
6. [manifest.rs — SSTable Metadata Tracking](#6-manifestrs)
7. [block_cache.rs — LRU Read Cache](#7-block_cachrs)
8. [compaction.rs — Background Merging](#8-compactionrs)
9. [store.rs — LsmStore (Orchestrator)](#9-storers)

---

## 1. lib.rs

**File**: `lsm_engine/src/lib.rs`

This is the entry point of the crate. It does three things:

### Module declarations

```rust
pub mod block_cache;
pub mod config;
pub mod store;
pub mod memtable;
pub mod sstable;
pub mod compaction;
pub mod manifest;
pub mod wal;
```

These `pub mod` lines make each submodule accessible. The `pub` keyword means other crates (like `lsm_pg_extension`) can import them.

### Re-exports

```rust
pub use config::LsmConfig;
pub use store::LsmStore;
```

This means users can write `use lsm_engine::LsmStore` instead of `use lsm_engine::store::LsmStore`. Convenience.

### Error types

```rust
#[derive(Error, Debug)]
pub enum LsmError {
    ObjectStore(#[from] object_store::Error),
    Serialization(String),
    Io(#[from] std::io::Error),
    StoreClosed,
    FencingViolation(String),
    ManifestCorruption(String),
    WalCorruption(String),
    Config(String),
}
```

`LsmError` covers every error the engine can produce. The `#[from]` attribute lets you use `?` to automatically convert `object_store::Error` into `LsmError::ObjectStore`. The `thiserror` crate generates the `Display` implementation from the `#[error("...")]` strings.

### Common types

```rust
pub struct KvEntry {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    pub sequence: u64,
}
```

`KvEntry` is returned by scan operations. The `sequence` number tells you the chronological order of writes — higher sequence = newer.

```rust
pub enum WriteOp {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
}
```

`WriteOp` represents a write. The LSM engine doesn't truly delete — it writes a **tombstone** (`Delete`). The actual removal happens during compaction.

---

## 2. config.rs

**File**: `lsm_engine/src/config.rs`

### LsmConfig

Every tunable parameter lives here:

| Field | Default | Purpose |
|-------|---------|---------|
| `root_path` | — | Where to store files in object storage (e.g., `/tables/items`) |
| `object_store` | — | The storage backend (S3, MinIO, or `InMemory` for tests) |
| `memtable_size_limit` | 64 MB | When the MemTable exceeds this, it's frozen and flushed |
| `flush_interval` | 5 seconds | Background flush timer |
| `l0_compaction_threshold` | 4 | Number of L0 SSTables that triggers compaction |
| `sstable_target_size` | 64 MB | Target size for compacted SSTables |
| `block_size` | 4 KB | Size of data blocks within SSTables |
| `block_cache_size` | 256 MB | RAM allocated for caching SSTable reads |
| `enable_compression` | false/true | Whether to zstd-compress data blocks |
| `wal_enabled` | true | Whether to use the write-ahead log |
| `wal_segment_size` | 16 MB | Max size before rotating to a new WAL segment |
| `fencing_token` | None | For single-writer enforcement (prevents split-brain) |

### Constructors

`LsmConfig::in_memory(path)` creates a config backed by `InMemory` object store — used for tests.

`LsmConfig::s3(path, object_store)` creates a config for real S3 — compression enabled by default.

### Builder pattern

```rust
let config = LsmConfig::in_memory("/test")
    .with_memtable_size(1024)
    .with_flush_interval(Duration::from_secs(10))
    .with_fencing_token("my-token".to_string());
```

Each `with_*` method takes `mut self` and returns `Self`, so you can chain them.

---

## 3. memtable.rs

**File**: `lsm_engine/src/memtable.rs`

The MemTable is the **write buffer** — all writes go here first. It's an in-memory sorted data structure.

### Data structure

```rust
pub struct MemTable {
    data: RwLock<BTreeMap<Vec<u8>, (u64, MemTableValue)>>,
    approximate_size: AtomicUsize,
    next_sequence: AtomicU64,
    frozen: RwLock<bool>,
}
```

- `data`: A `BTreeMap` (balanced binary tree) keyed by the user's key bytes. The value is `(sequence_number, Put(bytes) | Delete)`. BTreeMap keeps keys sorted, which is essential for writing sorted SSTables.
- `approximate_size`: Tracks how many bytes are stored (for flush threshold decisions). Uses `AtomicUsize` so it can be read without acquiring the write lock.
- `next_sequence`: Global counter. Every write gets a unique, monotonically increasing number. This is how the engine knows which version of a key is "newest."
- `frozen`: When true, no more writes are accepted. The MemTable is waiting to be flushed as an SSTable.

### MemTableValue

```rust
pub enum MemTableValue {
    Put(Vec<u8>),   // live data
    Delete,         // tombstone marker
}
```

When you delete a key, the engine doesn't remove it from the BTreeMap — it inserts a `Delete` tombstone. This ensures the delete propagates to SSTables during flush. The actual removal happens during compaction.

### Key methods

**`put(key, value)`**: Acquires the sequence counter, inserts into the BTreeMap, and updates the size estimate. If the key already existed, it adjusts the size by subtracting the old entry's size.

**`get(key)`**: Returns `None` (key never seen), `Some(Some(bytes))` (key has a value), or `Some(None)` (key was deleted — tombstone). The `Some(None)` case is critical: it tells the caller to STOP looking in SSTables, because the delete is more recent.

**`scan(start, end)`**: Range scan using BTreeMap's `.range()` method. Skips tombstones.

**`freeze()`**: Marks the MemTable as read-only. After this, `put()` and `delete()` return errors.

**`entries()`**: Returns all entries in sorted order — used when flushing to an SSTable.

### Concurrency model

The BTreeMap is wrapped in `parking_lot::RwLock`. Multiple threads can read concurrently, but writes are serialized. The atomic counters (`approximate_size`, `next_sequence`) avoid lock contention for frequently-read metadata.

---

## 4. sstable.rs

**File**: `lsm_engine/src/sstable.rs`

SSTables are the **on-disk format**. When a MemTable is flushed, its entries are written as an SSTable to S3. SSTables are immutable — once written, they're never modified.

### File layout

```
[data block 0][data block 1]...[data block N]
[index JSON][index_len: 4 bytes]
[footer JSON][footer_len: 4 bytes]   ← last 4 bytes of the file
```

To read an SSTable, you start from the end:
1. Read the last 4 bytes → that's the footer length
2. Read the footer JSON → it tells you where the index is
3. Read the index JSON → it tells you where each data block is
4. Binary search the index to find the right block for your key
5. Decode just that one block

### Data block format

Each block is a sequence of entries encoded as raw bytes:

```
[compression_flag: 1 byte]   ← 0=raw, 1=zstd
  If compressed: [uncompressed_size: 4 bytes][zstd_data...]
  If raw: [raw_data...]

Raw data format:
  [entry_count: 4 bytes]
  For each entry:
    [key_len: 4 bytes][key bytes]
    [sequence: 8 bytes]
    [is_tombstone: 1 byte]
    [value_len: 4 bytes][value bytes]
```

### SSTableBuilder

Builds an SSTable incrementally. You call `add(key, seq, value)` for each entry **in sorted key order**, and it accumulates them into blocks. When a block exceeds the target size (default 4 KB), it's flushed.

`build()` finishes by writing the index and footer, returning the complete byte buffer and metadata.

### SSTableReader

Opens an SSTable from raw bytes and provides:
- `get(key)` — point lookup using binary search on the index
- `scan(start, end)` — range scan
- `scan_all_with_tombstones(start, end)` — used by compaction to preserve tombstones

### SSTableMeta

Metadata about an SSTable that's stored in the manifest:

```rust
pub struct SSTableMeta {
    pub id: u64,              // unique ID
    pub level: u32,           // which level (0, 1, 2, ...)
    pub min_key: Vec<u8>,     // smallest key
    pub max_key: Vec<u8>,     // largest key
    pub entry_count: u64,     // how many entries
    pub size_bytes: u64,      // file size on S3
    pub path: String,         // S3 object path
    pub min_sequence: u64,    // oldest write in this SSTable
    pub max_sequence: u64,    // newest write in this SSTable
}
```

The `min_key`/`max_key` fields enable **range filtering**: when searching for a key, we can skip SSTables whose key range doesn't include it.

---

## 5. wal.rs

**File**: `lsm_engine/src/wal.rs`

The WAL ensures durability. Without it, data in the MemTable would be lost on crash (it's only in RAM). With it, the MemTable can be reconstructed by replaying WAL entries.

### Segment format

WAL data is stored in numbered segments on S3:

```
/test/wal/segment-0000000000000001.wal
/test/wal/segment-0000000000000002.wal
...
```

Each segment starts with a header:
```
[magic: 4 bytes = "WAL1"][version: 4 bytes = 1]
```

Followed by entries:
```
[crc32: 4 bytes][op_type: 1 byte][key_len: 4][key][val_len: 4][value][sequence: 8]
```

### CRC32 checksums

Every WAL entry has a CRC32 checksum computed over everything after the CRC field itself. On recovery, if the checksum doesn't match, the entry is corrupted (probably a crash mid-write) and we stop reading — all entries after the corruption point are lost, but everything before is valid.

The CRC32 implementation is a simple hand-rolled version using the standard IEEE polynomial.

### WalWriter

Accumulates entries in a `BytesMut` buffer. When the buffer exceeds `segment_size_limit` (default 16 MB), it:
1. Flushes the current buffer to S3
2. Rotates to a new segment (increments `segment_seq`, resets buffer)

**`trim(segments)`**: Deletes WAL segments that are no longer needed (their data has been flushed to SSTables). This is called after each successful flush.

### WalReader

Used during recovery to replay WAL entries:
1. `discover_segments()` — lists all WAL segment files on S3
2. `replay_segment(seq)` — reads one segment, validates CRC, returns entries
3. `replay_all()` — reads all segments, sorts entries by sequence number

---

## 6. manifest.rs

**File**: `lsm_engine/src/manifest.rs`

The manifest is the **source of truth** for which SSTables exist and at which levels. It's serialized as JSON and stored on S3 at `{root_path}/manifest.json`.

### Manifest struct

```rust
pub struct Manifest {
    pub version: u64,            // incremented on every change
    pub next_sstable_id: u64,    // counter for unique SSTable IDs
    pub next_sequence: u64,      // for write ordering after recovery
    pub levels: Vec<Vec<SSTableMeta>>,  // levels[0] = L0, levels[1] = L1, ...
    pub fencing_token: Option<String>,  // single-writer enforcement
}
```

The `levels` vector has 7 entries (L0 through L6). Each entry is a list of SSTables at that level.

### Key methods

**`add_l0_sstable(meta)`**: Assigns the next SSTable ID, sets the level to 0, and appends to `levels[0]`.

**`apply_compaction(source_level, source_ids, target_level, new_tables)`**: Removes the source SSTables from their level and adds the new compacted SSTables to the target level. This is called after compaction completes.

### ManifestStore

Handles reading/writing the manifest to S3:
- `load_or_create()` — tries to read from S3; if not found, creates a new empty manifest
- `save(manifest)` — serializes to JSON and writes to S3

---

## 7. block_cache.rs

**File**: `lsm_engine/src/block_cache.rs`

A simple **LRU (Least Recently Used) cache** that stores SSTable file bytes in RAM. Without this, every read would require a GET from S3 (hundreds of milliseconds).

### How it works

```rust
pub struct BlockCache {
    entries: RwLock<CacheInner>,
    capacity_bytes: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

struct CacheInner {
    map: HashMap<String, Bytes>,      // path → SSTable bytes
    order: VecDeque<String>,          // LRU order (front = most recent)
    current_bytes: usize,             // total bytes stored
}
```

**`get(path)`**: If found, increments hits and **promotes** the entry to the front of the LRU queue (it was just used, so it's now "most recently used"). If not found, increments misses.

**`insert(path, data)`**: Adds the entry to the front. If total bytes would exceed `capacity_bytes`, **evicts** entries from the back (least recently used) until there's enough room.

**Entries larger than the cache capacity** are not cached at all — they'd just evict everything else.

### Stats

`hit_rate()` returns `hits / (hits + misses)` as a float. This tells you how effective the cache is. A 90% hit rate means 9 out of 10 reads are served from RAM.

---

## 8. compaction.rs

**File**: `lsm_engine/src/compaction.rs`

Compaction is the background process that merges SSTables to reduce read amplification (fewer files to check) and reclaim space from deleted keys.

### CompactionConfig

```rust
pub struct CompactionConfig {
    pub l0_threshold: usize,          // default 4: compact when L0 has this many SSTables
    pub block_size: usize,            // output SSTable block size
    pub target_sstable_size: usize,   // output SSTable file size
    pub level_size_multiplier: usize, // default 10: each level is 10x larger
    pub l1_max_tables: usize,         // default 10: compact L1 when it has more than this
    pub enable_compression: bool,     // zstd for output SSTables
}
```

### L0 → L1 compaction

When `l0_tables.len() >= l0_threshold`:
1. Read **all** L0 SSTables
2. Merge their entries into a single sorted BTreeMap (newest sequence wins)
3. Preserve tombstones (they still need to mask older values in lower levels)
4. Write one new L1 SSTable

### L1+ → L(N+1) compaction

When a level N has too many SSTables:
1. Pick the SSTable with the **oldest min_sequence** (this is the least-recently-written one)
2. Find all SSTables in level N+1 whose key range **overlaps** with the picked one
3. Merge them all (newest sequence wins)
4. Write new SSTables to level N+1
5. If we're at the **bottom level** (no levels below), drop tombstones entirely — the delete has fully propagated

### apply_result

After compaction, this method:
1. Writes the new SSTables to S3
2. Deletes the old (source) SSTables from S3

The manifest is updated separately by the `store.rs` code.

---

## 9. store.rs

**File**: `lsm_engine/src/store.rs`

`LsmStore` is the **main entry point** — it ties together all the other modules.

### Struct fields

```rust
pub struct LsmStore {
    config: LsmConfig,
    active_memtable: Arc<RwLock<Arc<MemTable>>>,      // current write buffer
    frozen_memtables: Arc<RwLock<Vec<Arc<MemTable>>>>, // waiting for flush
    manifest: Arc<RwLock<Manifest>>,
    manifest_store: Arc<ManifestStore>,
    object_store: Arc<dyn ObjectStore>,
    root_path: String,
    is_open: Arc<AtomicBool>,
    flush_notify: Arc<Notify>,           // wakes the flush task
    wal_writer: Option<Arc<AsyncMutex<WalWriter>>>,
    flush_handle: ...,                   // background task handle
    compaction_handle: ...,              // background task handle
    block_cache: Arc<BlockCache>,
}
```

Note the double-wrapping of the MemTable: `Arc<RwLock<Arc<MemTable>>>`. The outer `RwLock` protects swapping the pointer during rotation. The inner `Arc` lets readers hold a reference to a specific MemTable instance even after rotation replaces the active one.

### open()

1. Load or create the manifest from S3
2. Check the fencing token (prevent split-brain)
3. Replay WAL entries into a fresh MemTable
4. Create the WAL writer (registering discovered segments for later trimming)
5. Create the block cache
6. Spawn two background tasks:
   - **Flush task**: wakes on `flush_notify` or a timer, flushes frozen MemTables to SSTables
   - **Compaction task**: wakes every 10 seconds, checks if compaction is needed

### put()

1. Check store is open
2. Lock the WAL, append the write operation
3. Insert into the active MemTable
4. If the MemTable is over the size limit, rotate it

### get()

Checks in order (most recent data first):
1. Active MemTable
2. Frozen MemTables (newest first)
3. SSTables (L0 first, then L1-L6; within each level, newest first)

For SSTables, it checks the block cache first. On a cache miss, it fetches from S3 and caches the result.

Key range filtering: if the key is outside an SSTable's `[min_key, max_key]` range, that SSTable is skipped entirely.

### delete()

Writes a tombstone to the WAL and MemTable — same path as `put()`, just with `MemTableValue::Delete`.

### scan()

Collects entries from all sources (MemTable + frozen + SSTables), then deduplicates by key (keeping the highest sequence number — newest version wins).

### rotate_memtable()

1. Freeze the active MemTable (no more writes)
2. Push it to the `frozen_memtables` list
3. Create a new empty MemTable (continuing the sequence counter)
4. Notify the flush task

### do_flush()

Takes the oldest frozen MemTable, builds an SSTable from it, writes to S3, updates the manifest, and trims old WAL segments. Repeats until no frozen MemTables remain.

### close()

1. Sync the WAL
2. Rotate and flush all remaining data
3. Signal background tasks to stop
4. Wait for them to finish

---

## How to Run Engine Tests

```bash
# Run all engine-only tests (no PostgreSQL needed)
cargo test -p lsm_engine --lib

# Run a specific test
cargo test -p lsm_engine --lib test_wal_recovery
```

All tests use `InMemory` object storage, so they're fast and hermetic.
