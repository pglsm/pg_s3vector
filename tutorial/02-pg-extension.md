# Tutorial 2: The PostgreSQL Extension (`lsm_pg_extension` crate)

This tutorial covers the PostgreSQL-specific code: how the extension loads, how tables are created and queried via the Table Access Method (TAM), how TID mapping works, and how transactions are handled.

---

## Table of Contents

1. [lib.rs — Extension Entry Point](#1-librs)
2. [tam/storage.rs — Table Storage Bridge](#2-tamstoragrs)
3. [tam/tam_handler.rs — Table Access Method](#3-tamtam_handlerrs)
4. [tam/txn.rs — Transaction Undo Log](#4-tamtxnrs)

---

## 1. lib.rs

**File**: `lsm_pg_extension/src/lib.rs`

This is the extension entry point — the first code PostgreSQL calls when loading the shared library.

### Module magic

```rust
pgrx::pg_module_magic!();
```

This macro generates the `Pg_magic_func` symbol that PostgreSQL uses to validate the extension was compiled for the correct server version. Without this, PostgreSQL refuses to load the `.so` file.

### GUC Variables

GUC stands for "Grand Unified Configuration" — PostgreSQL's system for configuration parameters. We register six:

```rust
static LSM_S3_ENDPOINT: pgrx::GucSetting<Option<&'static CStr>> =
    pgrx::GucSetting::<Option<&'static CStr>>::new(Some(c"memory"));
```

This creates a string GUC called `lsm_s3.endpoint` with default value `"memory"`. Users can change it with:

```sql
SET lsm_s3.endpoint = 'https://s3.amazonaws.com';
```

| GUC | Default | What it controls |
|-----|---------|------------------|
| `lsm_s3.endpoint` | `memory` | S3 URL. `memory` = in-memory testing mode |
| `lsm_s3.bucket` | `lsm-postgres` | S3 bucket name |
| `lsm_s3.region` | `us-east-1` | AWS region |
| `lsm_s3.flush_interval_ms` | `5000` | How often to flush MemTables (ms) |
| `lsm_s3.memtable_size_mb` | `64` | MemTable size limit (MB) |
| `lsm_s3.cache_size_mb` | `256` | Block cache size (MB) |

### _PG_init

```rust
#[pg_guard]
pub extern "C" fn _PG_init() {
    init_gucs();
}
```

PostgreSQL calls `_PG_init()` when the extension's shared library is loaded. We use it to register our GUC variables. The `#[pg_guard]` attribute ensures Rust panics are caught and converted to PostgreSQL errors instead of crashing the server.

### SQL functions

`lsm_s3_status()` — returns a human-readable string showing the current configuration.

`lsm_s3_flush()` — forces all in-memory data to be flushed to S3.

`lsm_postgres_version()` — returns the version string.

---

## 2. tam/storage.rs

**File**: `lsm_pg_extension/src/tam/storage.rs`

This is the **bridge** between PostgreSQL and the LSM engine. It manages one `LsmStore` per table and handles the TID mapping that PostgreSQL needs.

### Global singleton

```rust
static GLOBAL_STORAGE: Lazy<TableStorage> = Lazy::new(|| TableStorage::new());
```

There's a single `TableStorage` instance for the entire PostgreSQL backend process. It's initialized on first access via `once_cell::sync::Lazy`.

### TableStorage struct

```rust
pub struct TableStorage {
    stores: RwLock<HashMap<String, Arc<LsmStore>>>,   // table_name → store
    tid_managers: RwLock<HashMap<String, Arc<TidManager>>>,  // table_name → TID map
    runtime: Arc<Runtime>,  // shared Tokio runtime
}
```

**Why a Tokio runtime?** The LSM engine is async (it uses `object_store` which is async). PostgreSQL is synchronous. So we create a multi-threaded Tokio runtime and use `runtime.block_on()` to bridge async code into synchronous PostgreSQL callbacks.

### TID mapping

PostgreSQL identifies every row with a **TID** (Tuple ID), which is a `(BlockNumber: u32, OffsetNumber: u16)` pair. Normally this maps to a physical page and offset within that page. Since we don't use heap pages, we synthesize TIDs.

```rust
struct TidManager {
    next_id: AtomicU64,                          // monotonically increasing counter
    tid_to_key: RwLock<HashMap<u64, Vec<u8>>>,   // TID → LSM key
    key_to_tid: RwLock<HashMap<Vec<u8>, u64>>,   // LSM key → TID
}
```

**Why two HashMaps?** We need to go both directions:
- **Key → TID**: When inserting a row, we need to tell PostgreSQL the TID
- **TID → Key**: When PostgreSQL asks us to delete or fetch a specific TID

### TID encoding

A `u64` TID id is split into PostgreSQL's `(BlockNumber, OffsetNumber)`:

```rust
pub fn tid_to_block_offset(id: u64) -> (u32, u16) {
    let id0 = id - 1;       // TIDs start at 1
    let block = (id0 / 65535) as u32;
    let offset = ((id0 % 65535) + 1) as u16;  // OffsetNumber must be >= 1
    (block, offset)
}
```

This gives us ~2.8 × 10^14 unique TIDs before overflow, which is plenty.

### Key methods

**`get_or_create(table_name)`**: Looks up the table's `LsmStore` in the HashMap. If not found, creates a new one with `LsmConfig::in_memory(...)` and inserts it.

**`insert_with_tid(table_name, key, value)`**: Calls `store.put()`, then assigns (or retrieves) a TID for the key.

**`delete_by_tid(table_name, tid_id)`**: Looks up the key for the TID, calls `store.delete()`, then removes the TID mapping.

**`fetch_by_tid(table_name, tid_id)`**: Looks up the key, then calls `store.get()`. Used by index scans — the index returns a TID, and we need to fetch the actual row.

**`scan_keys_with_tids(table_name)`**: Returns `(key, tid)` pairs without values. Used by sequential scans to reduce memory — values are fetched lazily in `getnextslot`.

**`truncate(table_name)`**: Removes the LsmStore, clears the TID mapping, and creates a fresh empty store.

---

## 3. tam/tam_handler.rs

**File**: `lsm_pg_extension/src/tam/tam_handler.rs`

This is the core of the TAM — it implements the ~30 callback functions that PostgreSQL's executor calls. When you write `CREATE TABLE items (...) USING lsm_s3`, PostgreSQL uses these callbacks for all operations on that table.

### Row serialization

PostgreSQL represents rows as arrays of **Datums** (generic values). We need to serialize them to bytes for storage in the LSM engine and deserialize them back.

**Column 0 is special**: It's always the key (TEXT type) and stored separately as the LSM key. Columns 1+ are serialized together as the LSM value.

The serialization format:

```
[num_cols: u16]
For each column (starting from column 1):
  [is_null: u8]           ← 1 = NULL, 0 = has data
  If not null:
    [byval: u8]           ← 1 = pass-by-value, 0 = varlena/pointer
    [data_len: u32]
    [data: raw bytes]
```

**`serialize_row()`**: Walks through each column's datum using the tuple descriptor (`TupleDesc`) to determine the type. For pass-by-value types (integers, booleans), it copies the raw bytes. For varlena types (TEXT, BYTEA, our LsmVector), it detoasts and extracts the payload.

**`deserialize_row()`**: The reverse — reads the byte format and reconstructs datums, calling `palloc()` for varlena types (PostgreSQL manages their memory).

### Sequential scan

**`lsm_scan_begin()`**: Called when `SELECT * FROM items` starts.
1. Gets the table name from the relation
2. Calls `storage.scan_keys_with_tids()` to get all keys (but NOT values — lazy loading)
3. Allocates a custom `LsmScanDesc` struct that extends PostgreSQL's `TableScanDescData`

**`lsm_scan_getnextslot()`**: Called repeatedly to get the next row.
1. Gets the next `(key, tid)` from the list
2. Fetches the actual value bytes on demand from the LSM store
3. Puts the key into column 0 as a TEXT datum
4. Deserializes the remaining columns from the value bytes
5. Stores a virtual tuple in the slot and sets the TID

**`lsm_scan_end()`**: Frees the scan state. Uses `drop_in_place` because we used `palloc0` + `ptr::write` to construct the Rust types inside a C-allocated struct.

### DML operations

**`lsm_tuple_insert()`**: Called for `INSERT`.
1. Extract the key from column 0 (or generate a UUID if NULL)
2. Serialize columns 1+ into value bytes
3. Register a transaction undo callback (for rollback)
4. Insert into the LSM store
5. Record the insert in the undo log
6. Set the TID on the slot

**`lsm_tuple_delete()`**: Called for `DELETE`.
1. Convert the TID from `ItemPointer` to our u64 id
2. Save the old key-value pair (for rollback)
3. Delete by TID from the LSM store
4. Record the delete in the undo log

**`lsm_tuple_update()`**: Called for `UPDATE`. This is a delete-then-insert:
1. Delete the old row by TID (same as delete)
2. Extract key and value from the new slot
3. Insert the new row (same as insert)
4. Signal that indexes need updating

### Index fetch (for index scans)

When the HNSW index returns TIDs, PostgreSQL needs to fetch the actual rows. This goes through:

**`lsm_index_fetch_begin()`**: Allocates our custom `LsmIndexFetchData` with the table name.

**`lsm_index_fetch_tuple()`**: Given a TID, fetches the key-value pair from the LSM store and fills the slot. This is the critical bridge between the HNSW index and the heap data.

### Index build scan

**`lsm_index_build_range_scan()`**: Called during `CREATE INDEX`. Iterates all rows in the table and invokes a callback for each one, passing the column values and TID. This is what feeds data into the HNSW index during `ambuild`.

### The method table

At the bottom of the file, `LSM_TAM_ROUTINE` is a static `TableAmRoutine` struct that maps each required callback to our implementation:

```rust
static LSM_TAM_ROUTINE: pg_sys::TableAmRoutine = pg_sys::TableAmRoutine {
    scan_begin: Some(lsm_scan_begin),
    scan_getnextslot: Some(lsm_scan_getnextslot),
    tuple_insert: Some(lsm_tuple_insert),
    tuple_delete: Some(lsm_tuple_delete),
    tuple_update: Some(lsm_tuple_update),
    // ... ~30 more callbacks
};
```

The handler function returns a pointer to this struct:

```rust
pub extern "C" fn lsm_s3_tam_handler_wrapper(_fcinfo: ...) -> pg_sys::Datum {
    pg_sys::Datum::from(&LSM_TAM_ROUTINE as *const _ as usize)
}
```

And the SQL registration creates the access method:

```sql
CREATE ACCESS METHOD lsm_s3 TYPE TABLE HANDLER lsm_s3_tam_handler_wrapper;
```

### Stubs

Many callbacks are required by PostgreSQL but not meaningful for us:
- `lsm_tuple_lock` — no row-level locking (returns OK)
- `lsm_relation_vacuum` — no vacuum needed (compaction handles it)
- `lsm_relation_size` — returns 0 (our data isn't in PostgreSQL's buffer pool)
- `lsm_relation_needs_toast_table` — false (we handle large values ourselves)

---

## 4. tam/txn.rs

**File**: `lsm_pg_extension/src/tam/txn.rs`

This module provides basic **transaction awareness**. The LSM engine itself has no concept of transactions — once you `put()`, the data is in. PostgreSQL expects that if a transaction ABORTs, all its changes disappear.

### How it works

We maintain a thread-local **undo log**:

```rust
thread_local! {
    static UNDO_LOG: RefCell<Vec<UndoOp>> = RefCell::new(Vec::new());
}
```

Each DML operation in `tam_handler.rs` records what it did:

```rust
enum UndoOp {
    Insert { table: String, tid_id: u64 },
    Delete { table: String, key: Vec<u8>, value: Vec<u8>, tid_id: u64 },
}
```

### Transaction callback

We register a **XactCallback** with PostgreSQL — a function that's called at the end of every transaction:

```rust
unsafe extern "C" fn lsm_xact_callback(event: XactEvent, _arg: *mut c_void) {
    if event == XACT_EVENT_ABORT {
        // Replay undo log in reverse
        for op in ops.drain(..).rev() {
            match op {
                UndoOp::Insert { table, tid_id } => {
                    storage.delete_by_tid(&table, tid_id);  // undo the insert
                }
                UndoOp::Delete { table, key, value, tid_id } => {
                    storage.insert(&table, &key, &value);    // re-insert the deleted row
                    storage.restore_tid(&table, tid_id, &key); // restore TID mapping
                }
            }
        }
    } else {
        // COMMIT — just clear the undo log
        ops.clear();
    }
}
```

**On COMMIT**: The undo log is cleared — changes are permanent.

**On ABORT**: The undo log is replayed in **reverse** order. Inserts are undone by deleting. Deletes are undone by re-inserting the old data and restoring the TID mapping.

### Limitations

- No MVCC: Concurrent transactions can see each other's uncommitted changes
- No savepoints: Can't rollback part of a transaction
- The undo log is in-memory: If PostgreSQL crashes mid-transaction, the undo log is lost (but the WAL preserves the committed data)

### Registration

```rust
pub fn ensure_xact_callback() {
    if !CALLBACK_REGISTERED.swap(true, Ordering::SeqCst) {
        unsafe {
            pg_sys::RegisterXactCallback(Some(lsm_xact_callback), std::ptr::null_mut());
        }
    }
}
```

This is called from every DML callback (`tuple_insert`, `tuple_delete`, `tuple_update`). The `AtomicBool` ensures we only register once — PostgreSQL would get confused if we registered multiple times.
