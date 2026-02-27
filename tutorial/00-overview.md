# Tutorial 0: Architecture Overview

Welcome! This document explains the big picture of KriyaDB — what it does, why it exists, and how all the pieces fit together. Read this first before diving into individual modules.

---

## What Is This Project?

KriyaDB is a PostgreSQL extension that lets you store tables and vector indexes on **S3 (object storage)** instead of the local filesystem. It uses an **LSM-tree** (Log-Structured Merge-tree) architecture internally.

In plain English: when you write `CREATE TABLE items (...) USING lsm_s3`, PostgreSQL delegates all storage operations to our Rust code, which writes data to S3 instead of the standard heap files.

It also supports **vector similarity search** (like pgvector) via an HNSW index, so you can do:

```sql
SELECT * FROM items ORDER BY embedding <-> '[1.0, 2.0, 3.0]'::LsmVector LIMIT 5;
```

---

## Workspace Layout

The project is a Rust **workspace** with two crates:

```
lsm_postgres/               ← workspace root
├── Cargo.toml               ← workspace definition (lists both crates)
│
├── lsm_engine/              ← CRATE 1: standalone key-value store
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs           ← public API, error types, re-exports
│       ├── config.rs        ← LsmConfig (all tunable knobs)
│       ├── memtable.rs      ← in-memory sorted write buffer
│       ├── sstable.rs       ← on-disk sorted file format
│       ├── wal.rs           ← write-ahead log for crash safety
│       ├── manifest.rs      ← metadata: which SSTables exist
│       ├── compaction.rs    ← background merge of SSTables
│       ├── block_cache.rs   ← LRU cache for hot SSTables
│       └── store.rs         ← LsmStore: ties everything together
│
├── lsm_pg_extension/        ← CRATE 2: PostgreSQL extension
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs           ← extension entry point, GUC variables
│       ├── tam/             ← Table Access Method (lsm_s3)
│       │   ├── tam_handler.rs  ← PostgreSQL TAM callbacks
│       │   ├── storage.rs      ← per-table LsmStore registry + TID mapping
│       │   └── txn.rs          ← transaction rollback undo log
│       ├── vector/          ← Vector search (lsm_hnsw)
│       │   ├── types.rs        ← LsmVector PostgreSQL type
│       │   ├── distance.rs     ← distance functions + SQL operators
│       │   ├── hnsw.rs         ← HNSW graph algorithm
│       │   ├── iam.rs          ← Index Access Method callbacks
│       │   └── vector_storage.rs ← where vector data lives
│       └── fence/           ← distributed writer lock
│           └── s3_lock.rs
│
└── tutorial/                ← you are here
```

**Why two crates?** Separation of concerns. `lsm_engine` knows nothing about PostgreSQL — it's a pure Rust async key-value store that could be used in any Rust project. `lsm_pg_extension` is the PostgreSQL-specific glue that uses `pgrx` to talk to Postgres.

---

## Key Concepts for a New Intern

### 1. LSM-Tree (Log-Structured Merge-tree)

An LSM-tree is a data structure optimized for **write-heavy workloads**. Traditional databases (like PostgreSQL's heap) update data in-place on disk. LSM-trees never modify existing files — they only **append**. This has huge advantages for object storage (S3) where you can't do random writes.

**The lifecycle of a write:**

```
INSERT INTO items VALUES ('key1', 'hello')
   │
   ▼
1. WAL APPEND ──────── Write to the Write-Ahead Log (crash safety)
   │
   ▼
2. MEMTABLE INSERT ─── Insert into an in-memory sorted buffer (BTreeMap)
   │                    This is FAST (microseconds, RAM only)
   │
   ▼  (when MemTable reaches 64 MB)
3. FREEZE & FLUSH ──── Freeze the MemTable (no more writes)
   │                    Write it as an SSTable file to S3
   │
   ▼  (when too many SSTables accumulate)
4. COMPACTION ──────── Merge multiple SSTables into fewer, larger ones
                        Remove deleted keys, resolve duplicates
```

**The lifecycle of a read:**

```
SELECT * FROM items WHERE id = 'key1'
   │
   ▼
1. CHECK MEMTABLE ────── Is the key in the current write buffer? (fastest)
   │  not found
   ▼
2. CHECK FROZEN ──────── Is it in a MemTable waiting to be flushed?
   │  not found
   ▼
3. CHECK BLOCK CACHE ─── Is the SSTable file cached in RAM?
   │  not found
   ▼
4. FETCH FROM S3 ─────── Download the SSTable, search it, cache it
```

### 2. SSTables (Sorted String Tables)

An SSTable is an **immutable, sorted** file. Once written to S3, it never changes. It contains:

- **Data blocks**: chunks of sorted key-value pairs (optionally zstd-compressed)
- **Index block**: tells you which data block contains which key range
- **Footer**: metadata (entry count, magic number, compression flag)

Because keys are sorted, you can do **binary search** to find any key in O(log N) block reads.

### 3. Levels (L0, L1, L2, ... L6)

SSTables are organized into **levels**:

- **L0**: freshly flushed MemTables. SSTables here may have overlapping key ranges.
- **L1-L6**: compacted SSTables. Within a level, key ranges should not overlap.

When L0 gets too many SSTables (default: 4), they're **compacted** (merged) into L1. When L1 gets too many, they compact into L2, and so on.

### 4. Write-Ahead Log (WAL)

The WAL ensures you don't lose data if PostgreSQL crashes. Before any write enters the MemTable, it's appended to a WAL file on S3. On recovery after a crash, the WAL is replayed to restore the MemTable.

Each WAL entry has a **CRC32 checksum** so corrupted entries (from partial writes during a crash) are detected and skipped.

### 5. PostgreSQL Table Access Method (TAM)

PostgreSQL has a pluggable storage engine interface. The standard one is called "heap" (that's what stores your normal tables). We implement a custom one called `lsm_s3`.

When you write `CREATE TABLE foo (...) USING lsm_s3`, PostgreSQL calls our C-compatible callback functions (written in Rust via `pgrx`) for every operation:
- `scan_begin` / `scan_getnextslot` → SELECT
- `tuple_insert` → INSERT
- `tuple_delete` → DELETE
- `tuple_update` → UPDATE

### 6. PostgreSQL Index Access Method (IAM)

Similarly, PostgreSQL's index system is pluggable. We implement `lsm_hnsw` as a custom index type. When you write `CREATE INDEX ... USING lsm_hnsw`, PostgreSQL uses our callbacks:
- `ambuild` → build the index from existing data
- `aminsert` → add a vector when a row is inserted
- `amrescan` → receive the query vector from `ORDER BY ... <-> ...`
- `amgettuple` → return the next nearest neighbor

### 7. HNSW (Hierarchical Navigable Small World)

HNSW is an algorithm for **approximate nearest neighbor search**. Instead of comparing your query vector against every stored vector (O(N) — too slow), HNSW builds a multi-layer graph where each node is connected to its nearby neighbors. Searching is O(log N) on average.

Think of it like a skip list for geographic proximity: upper layers give you big jumps (continent → country), lower layers give fine-grained steps (neighborhood → street → house).

### 8. TID Mapping

PostgreSQL uses **TIDs** (Tuple IDs) to identify rows. A TID is a pair `(BlockNumber, OffsetNumber)` — basically a page number and a row offset within that page. Since we don't use PostgreSQL's heap pages, we synthesize TIDs from a monotonically increasing counter. The `TidManager` in `storage.rs` maintains a bidirectional map between these synthetic TIDs and our LSM keys.

### 9. pgrx

[pgrx](https://github.com/pgcentralfoundation/pgrx) is a Rust framework for writing PostgreSQL extensions. It provides:
- `#[pg_extern]` — expose a Rust function as a SQL function
- `pg_sys::*` — safe-ish bindings to PostgreSQL's C API
- `PostgresType` — derive macro to create custom SQL types
- `extension_sql!` — embed raw SQL (for CREATE OPERATOR, CREATE ACCESS METHOD, etc.)

---

## How the Crates Connect

```
┌─────────────────────────────────────────────────────────┐
│                    PostgreSQL Server                      │
│                                                          │
│  SQL Query ──→ Planner ──→ Executor                     │
│                               │                          │
│                    ┌──────────┴──────────┐               │
│                    ▼                     ▼               │
│             TAM callbacks          IAM callbacks          │
│          (tam_handler.rs)         (iam.rs)               │
│                    │                     │               │
│                    ▼                     ▼               │
│              TableStorage          HnswIndex             │
│             (storage.rs)          (hnsw.rs)              │
│                    │                     │               │
│                    ▼                     ▼               │
│            ┌──────────────────────────────────┐          │
│            │         lsm_engine crate         │          │
│            │                                  │          │
│            │   LsmStore                       │          │
│            │     ├── MemTable                 │          │
│            │     ├── WAL                      │          │
│            │     ├── SSTable (reader/builder) │          │
│            │     ├── Manifest                 │          │
│            │     ├── Block Cache              │          │
│            │     └── Compactor                │          │
│            │                                  │          │
│            └──────────┬───────────────────────┘          │
│                       │                                  │
└───────────────────────┼──────────────────────────────────┘
                        ▼
                   Object Store
               (S3 / MinIO / In-Memory)
```

---

## Configuration Flow

When the extension loads (`_PG_init`), it registers **GUC variables** (Grand Unified Configuration). These are PostgreSQL settings like `lsm_s3.endpoint`, `lsm_s3.bucket`, etc. You can set them in `postgresql.conf` or with `SET`.

When the first table operation happens, `TableStorage::global()` is called. This creates a shared Tokio runtime and lazily creates `LsmStore` instances per table.

---

## Testing

There are two levels of tests:

1. **Unit tests** (`cargo test -p lsm_engine --lib` and `cargo test -p lsm_pg_extension --lib`): pure Rust tests that don't need PostgreSQL. These test the LSM engine, distance functions, HNSW algorithm, etc.

2. **Integration tests** (`cargo pgrx test pg17`): these are `#[pg_test]` functions that start a real PostgreSQL instance, create tables, run SQL, and verify results. They test the full stack.

---

## Next Steps

- **[Tutorial 1: LSM Engine](./01-lsm-engine.md)** — deep dive into every engine module
- **[Tutorial 2: PostgreSQL Extension](./02-pg-extension.md)** — TAM, storage bridge, transactions
- **[Tutorial 3: Vector Search](./03-vector-search.md)** — LsmVector, distance functions, HNSW, IAM
