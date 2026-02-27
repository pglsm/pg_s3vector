# KriyaDB

S3-native vector database extension for PostgreSQL. Stores tables and vector indexes on object storage (S3/MinIO) using an LSM-tree architecture, with HNSW indexes for approximate nearest-neighbor search.

```sql
CREATE TABLE documents (id TEXT, title TEXT, embedding LsmVector) USING lsm_s3;

CREATE INDEX ON documents USING lsm_hnsw (embedding);

INSERT INTO documents VALUES
  ('doc1', 'Intro to ML', '[0.1, 0.2, 0.3]'),
  ('doc2', 'Deep Learning', '[0.9, 0.8, 0.7]');

SELECT id, title, embedding <-> '[0.1, 0.2, 0.3]'::LsmVector AS dist
FROM documents
ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::LsmVector
LIMIT 5;
```

## Architecture

```
Writes ─→ WAL (durability) ─→ MemTable (RAM)
                                   │
                              flush │
                                   ▼
                            L0 SSTables (S3)
                                   │
                          compact   │
                                   ▼
                         L1..L6 SSTables (S3)

Reads ─→ MemTable → Block Cache → SSTables (S3)
```

**Two crates:**

| Crate | What it does |
|-------|-------------|
| `lsm_engine` | Standalone async key-value store: MemTable, SSTable, WAL, compaction, block cache, zstd compression |
| `lsm_pg_extension` | PostgreSQL extension: Table Access Method (`lsm_s3`), Index Access Method (`lsm_hnsw`), `LsmVector` type, distance operators |

## Features

**Storage Engine**
- LSM-tree with 7-level compaction (L0 through L6)
- Write-ahead log with CRC32 checksums
- zstd block compression
- LRU block cache for SSTable reads
- S3 writer fencing (single-writer invariant)
- Pluggable object storage backend (S3, MinIO, in-memory)

**PostgreSQL Integration**
- `lsm_s3` Table Access Method — use standard SQL (`CREATE TABLE ... USING lsm_s3`)
- Multi-column tables with any PostgreSQL data types
- `INSERT`, `SELECT`, `UPDATE`, `DELETE`, `TRUNCATE`
- Transaction awareness with rollback on `ABORT`

**Vector Search**
- `LsmVector` type with `[1.0, 2.0, 3.0]` text representation
- HNSW index via `CREATE INDEX ... USING lsm_hnsw`
- Three distance metrics:
  - `<->` L2 (Euclidean) — default
  - `<=>` Cosine
  - `<#>` Inner product
- KNN queries: `ORDER BY col <-> query LIMIT K`
- Vectors persisted to S3 via LSM engine

## Prerequisites

- Rust (stable, 1.70+)
- PostgreSQL 17 (or 13-16 with feature flags)
- [pgrx](https://github.com/pgcentralfoundation/pgrx) 0.12.9

```bash
cargo install --locked cargo-pgrx@0.12.9
cargo pgrx init --pg17 download
```

## Build & Install

```bash
# Build everything
cargo build

# Run engine tests (no Postgres needed)
cargo test -p lsm_engine

# Install extension into pgrx-managed Postgres 17
cd lsm_pg_extension
cargo pgrx install --pg-config ~/.pgrx/17.*/pgrx-install/bin/pg_config

# Or run tests (starts a temporary Postgres instance)
cargo pgrx test pg17
```

## Quick Start

Start a pgrx-managed PostgreSQL:

```bash
cargo pgrx start pg17
psql -p 28817 -d postgres
```

```sql
-- Load the extension
CREATE EXTENSION lsm_pg_extension;

-- Create a table on S3 storage
CREATE TABLE items (
    id    TEXT,
    name  TEXT,
    embedding LsmVector
) USING lsm_s3;

-- Insert data
INSERT INTO items VALUES ('1', 'Red shirt',  '[0.1, 0.3, 0.9]');
INSERT INTO items VALUES ('2', 'Blue jeans', '[0.8, 0.1, 0.2]');
INSERT INTO items VALUES ('3', 'Green hat',  '[0.2, 0.9, 0.1]');

-- Create HNSW index (L2 distance, default)
CREATE INDEX items_embedding_idx ON items USING lsm_hnsw (embedding);

-- KNN search
SELECT id, name, embedding <-> '[0.1, 0.3, 0.8]'::LsmVector AS dist
FROM items
ORDER BY embedding <-> '[0.1, 0.3, 0.8]'::LsmVector
LIMIT 2;
```

### Cosine & Inner Product

```sql
-- Cosine distance index
CREATE INDEX items_cos_idx ON items
  USING lsm_hnsw (embedding lsm_vector_cosine_ops);

SELECT id, name, embedding <=> '[0.1, 0.3, 0.8]'::LsmVector AS cos_dist
FROM items
ORDER BY embedding <=> '[0.1, 0.3, 0.8]'::LsmVector
LIMIT 5;

-- Inner product index
CREATE INDEX items_ip_idx ON items
  USING lsm_hnsw (embedding lsm_vector_ip_ops);

SELECT id, name, embedding <#> '[0.1, 0.3, 0.8]'::LsmVector AS ip_dist
FROM items
ORDER BY embedding <#> '[0.1, 0.3, 0.8]'::LsmVector
LIMIT 5;
```

### Multi-Column Tables

```sql
CREATE TABLE products (
    sku      TEXT,
    name     TEXT,
    price    INT,
    in_stock BOOLEAN
) USING lsm_s3;

INSERT INTO products VALUES ('A001', 'Widget', 1299, true);
SELECT * FROM products;
```

### Monitoring

```sql
-- Extension version and config
SELECT lsm_s3_status();

-- Force flush MemTables to S3
SELECT lsm_s3_flush();
```

## Configuration

Set via `postgresql.conf` or `SET`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lsm_s3.endpoint` | `memory` | S3 endpoint URL (`memory` for in-memory testing) |
| `lsm_s3.bucket` | `lsm-postgres` | S3 bucket name |
| `lsm_s3.region` | `us-east-1` | AWS region |
| `lsm_s3.flush_interval_ms` | `5000` | MemTable flush interval (ms) |
| `lsm_s3.memtable_size_mb` | `64` | MemTable size limit before rotation (MB) |
| `lsm_s3.cache_size_mb` | `256` | Block cache size (MB) |

## Project Structure

```
lsm_postgres/
├── lsm_engine/                  # Standalone LSM key-value store
│   └── src/
│       ├── store.rs             # LsmStore: put/get/scan/delete with background flush & compaction
│       ├── memtable.rs          # In-memory sorted write buffer (skip list)
│       ├── sstable.rs           # Sorted String Table format with zstd compression
│       ├── wal.rs               # Write-ahead log with CRC32 checksums
│       ├── compaction.rs        # L0→L1 and L1→L6 leveled compaction
│       ├── manifest.rs          # SSTable metadata tracking + fencing token
│       ├── block_cache.rs       # LRU cache for SSTable reads
│       └── config.rs            # LsmConfig with builder pattern
│
└── lsm_pg_extension/            # PostgreSQL extension (pgrx)
    └── src/
        ├── lib.rs               # Extension entry point, GUC variables
        ├── tam/
        │   ├── tam_handler.rs   # Table Access Method (lsm_s3): scan, insert, delete, update
        │   ├── storage.rs       # Per-table LsmStore registry + TID mapping
        │   └── txn.rs           # Transaction undo log (rollback on ABORT)
        ├── vector/
        │   ├── types.rs         # LsmVector PostgreSQL type
        │   ├── distance.rs      # L2, cosine, inner product + operators (<->, <=>, <#>)
        │   ├── hnsw.rs          # HNSW graph: insert, search, persistence snapshots
        │   ├── iam.rs           # Index Access Method (lsm_hnsw): ambuild, aminsert, amgettuple
        │   └── vector_storage.rs # VectorStorage trait + InMemory/LSM backends
        └── fence/
            └── s3_lock.rs       # Distributed writer lock via S3 conditional PUTs
```

## How It Works

**Writes:** `INSERT` → WAL append (crash safety) → MemTable (sorted in-memory buffer). When the MemTable reaches 64 MB, it's frozen and flushed as an L0 SSTable to S3. Background compaction merges SSTables into deeper levels.

**Reads:** `SELECT` checks MemTable first (fast path), then the block cache, then SSTables on S3 from newest to oldest. Key range checks skip irrelevant SSTables.

**Vector Search:** `ORDER BY embedding <-> query LIMIT K` triggers an HNSW index scan. The graph is traversed in-memory, returning TIDs of the K nearest neighbors. PostgreSQL then fetches the actual rows from the LSM table.

**Durability:** Every write is logged to the WAL before entering the MemTable. On crash recovery, unflushed WAL entries are replayed. Flushed SSTables on S3 are immutable and tracked by the manifest.

## License

MIT
# pg_s3vector
# pg_s3vector
