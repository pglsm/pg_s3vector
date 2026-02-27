# Tutorial 3: Vector Search

This tutorial covers everything related to vector similarity search: the custom `LsmVector` type, distance functions, the HNSW algorithm, and the Index Access Method that makes `ORDER BY ... <-> ... LIMIT K` work.

---

## Table of Contents

1. [vector/types.rs — The LsmVector Type](#1-vectortypesrs)
2. [vector/distance.rs — Distance Functions & Operators](#2-vectordistancers)
3. [vector/vector_storage.rs — Where Vectors Live](#3-vectorvector_storagers)
4. [vector/hnsw.rs — The HNSW Algorithm](#4-vectorhnswrs)
5. [vector/iam.rs — Index Access Method](#5-vectoriamrs)
6. [End-to-End: How a KNN Query Works](#6-end-to-end)

---

## 1. vector/types.rs

**File**: `lsm_pg_extension/src/vector/types.rs`

### LsmVector

```rust
#[derive(Clone, Debug, Serialize, Deserialize, PostgresType)]
#[inoutfuncs]
pub struct LsmVector {
    pub data: Vec<f32>,
}
```

This is a custom PostgreSQL type representing a fixed-dimension float vector. The key attributes:

- **`PostgresType`**: pgrx derive macro that generates the necessary C glue for PostgreSQL to understand this as a native type. Internally, pgrx serializes it as CBOR (Concise Binary Object Representation) inside a varlena.
- **`#[inoutfuncs]`**: Tells pgrx we'll provide custom `input` and `output` functions instead of using the default serialization for text representation.
- **`Serialize, Deserialize`**: serde traits — used for CBOR serialization within PostgreSQL and JSON snapshots.

### Text I/O

```rust
impl InOutFuncs for LsmVector {
    fn input(input: &CStr) -> Self {
        // Parses "[1.0, 2.0, 3.0]" into a Vec<f32>
        // Strips brackets, splits by comma, parses each float
    }

    fn output(&self, buffer: &mut StringInfo) {
        // Writes "[1.0, 2.0, 3.0]" format
    }
}
```

This is what makes `'[1.0, 2.0, 3.0]'::LsmVector` work in SQL. PostgreSQL calls `input()` when it sees a text literal being cast to `LsmVector`, and `output()` when displaying it.

### Binary I/O

`to_bytes()` and `from_bytes()` provide a more compact binary format (4-byte dimension count + raw f32 values) used for storage in the LSM engine.

### SQL helper functions

```rust
#[pg_extern(immutable, parallel_safe)]
fn lsm_vector_dim(v: LsmVector) -> i32 { ... }

#[pg_extern(immutable, parallel_safe)]
fn lsm_vector_magnitude(v: LsmVector) -> f32 { ... }

#[pg_extern(immutable, parallel_safe)]
fn lsm_vector_normalize(v: LsmVector) -> LsmVector { ... }
```

These are SQL functions exposed via `#[pg_extern]`:
- `immutable` — the function always returns the same output for the same input (enables query optimization)
- `parallel_safe` — safe to use in parallel query plans

---

## 2. vector/distance.rs

**File**: `lsm_pg_extension/src/vector/distance.rs`

### Core distance functions

These are pure Rust functions (no PostgreSQL dependency):

**L2 (Euclidean) distance:**
```rust
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| { let d = x - y; d * d }).sum::<f32>().sqrt()
}
```

The "real" distance. `l2_distance_squared()` skips the `sqrt()` for internal comparisons where only relative ordering matters (comparing squared distances gives the same ordering).

**Cosine distance:**
```rust
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot_product(a, b) / (norm(a) * norm(b))
}
```

Measures the angle between vectors, not their magnitude. Two vectors pointing the same direction have distance 0, perpendicular vectors have distance 1.

**Negative inner product:**
```rust
pub fn negative_inner_product(a: &[f32], b: &[f32]) -> f32 {
    -dot_product(a, b)
}
```

Negated so that higher similarity = lower "distance" (PostgreSQL's ORDER BY is ascending).

### SQL functions and operators

Each distance function is exposed as a SQL function:

```rust
#[pg_extern(immutable, parallel_safe)]
fn lsm_vector_l2_distance(a: LsmVector, b: LsmVector) -> f32 { ... }
```

And then **operators** are created in SQL:

```sql
CREATE OPERATOR <-> (
    LEFTARG = LsmVector,
    RIGHTARG = LsmVector,
    FUNCTION = lsm_vector_l2_distance,
    COMMUTATOR = <->
);
```

This means `vec1 <-> vec2` in SQL calls `lsm_vector_l2_distance(vec1, vec2)`. The `COMMUTATOR` tells PostgreSQL that `a <-> b = b <-> a` (the operator is symmetric), which helps the query planner.

The three operators:
| Operator | Metric | Function |
|----------|--------|----------|
| `<->` | L2 (Euclidean) | `lsm_vector_l2_distance` |
| `<=>` | Cosine | `lsm_vector_cosine_distance` |
| `<#>` | Inner product | `lsm_vector_inner_product` |

### Batch operations

`batch_l2_distances()` and `knn_l2()` compute distances from one query vector to many candidates — used by the brute-force fallback and testing.

---

## 3. vector/vector_storage.rs

**File**: `lsm_pg_extension/src/vector/vector_storage.rs`

This file defines **where vector data lives**. The HNSW graph stores only structure (connections between nodes). The actual float arrays are stored separately via this trait.

### VectorStorage trait

```rust
pub trait VectorStorage: Send + Sync {
    fn store(&self, id: u64, vector: &[f32]) -> Result<(), String>;
    fn load(&self, id: u64) -> Result<Vec<f32>, String>;
    fn batch_load(&self, ids: &[u64]) -> Result<Vec<(u64, Vec<f32>)>, String>;
    fn delete(&self, id: u64) -> Result<(), String>;
}
```

`Send + Sync` is required because the HNSW index is accessed from multiple threads.

### InMemoryVectorStorage

```rust
pub struct InMemoryVectorStorage {
    vectors: RwLock<HashMap<u64, Vec<f32>>>,
}
```

Simple HashMap in RAM. Used for testing. All vectors are kept in memory.

### LsmVectorStorage

```rust
pub struct LsmVectorStorage {
    store: Arc<LsmStore>,
    runtime: Arc<Runtime>,
}
```

This is the production backend. Each vector is stored as a key-value pair in a dedicated `LsmStore`:

- **Key**: `v:` prefix + 8-byte big-endian id (e.g., `v:\x00\x00\x00\x00\x00\x00\x00\x2A` for id 42)
- **Value**: raw f32 bytes in little-endian order

The big-endian key format ensures vectors are stored in ID order in SSTables (big-endian preserves numeric sort order for byte comparison).

**`batch_load()`** uses `futures::future::join_all` to fetch multiple vectors in parallel — important for S3 where each GET has high latency but many can run concurrently.

### Why separate storage?

Separating vector data from the graph structure means:
1. The graph snapshot (serialized for persistence) stays small — it doesn't include vector bytes
2. Vectors can be fetched on demand from S3 instead of all being in RAM
3. Different storage backends can be swapped without changing the HNSW algorithm

---

## 4. vector/hnsw.rs

**File**: `lsm_pg_extension/src/vector/hnsw.rs`

This is the HNSW (Hierarchical Navigable Small World) algorithm implementation. It's the core of the vector search engine.

### How HNSW works (conceptual)

Imagine a city map with multiple zoom levels:

- **Layer 3 (top)**: Only major highways connecting distant cities. Very sparse.
- **Layer 2**: Regional roads connecting nearby towns.
- **Layer 1**: Local streets connecting neighborhoods.
- **Layer 0 (bottom)**: Every house is connected to its nearby houses. Dense.

To find the nearest house to a query point:
1. Start at the highway level, jump to the closest major city
2. Drop to regional roads, jump to the closest town
3. Drop to local streets, jump to the closest neighborhood
4. Search the dense bottom layer for the actual nearest houses

Each "jump" narrows the search area dramatically, giving O(log N) search time instead of O(N).

### Configuration

```rust
pub struct HnswConfig {
    pub m: usize,              // max connections per node (default 16)
    pub m_max_0: usize,        // max connections at layer 0 (default 32)
    pub ef_construction: usize, // beam width during build (default 200)
    pub ef_search: usize,      // beam width during search (default 64)
    pub ml: f64,               // level generation factor: 1/ln(m)
    pub max_layers: usize,     // maximum number of layers (default 16)
    pub metric: DistanceMetric, // L2, Cosine, or InnerProduct
}
```

**`m`** (connections per node): Higher = better recall but slower inserts and more memory.

**`ef_construction`**: How many candidates to consider when linking a new node. Higher = better quality graph but slower build.

**`ef_search`**: How many candidates to consider during search. Higher = better recall but slower queries.

**`ml`**: Controls how nodes are assigned to layers. With `ml = 1/ln(m)`, approximately 1/m nodes appear at each successive layer.

### The HnswIndex struct

```rust
pub struct HnswIndex<S: VectorStorage> {
    config: HnswConfig,
    nodes: RwLock<HashMap<u64, HnswNode>>,  // graph structure
    storage: Arc<S>,                         // vector data
    entry_point: RwLock<Option<u64>>,        // topmost node
    max_layer: RwLock<usize>,                // current highest layer
    next_id: AtomicU64,                      // ID counter
    key_to_id: RwLock<HashMap<String, u64>>, // "doc:42" → node ID
    id_to_key: RwLock<HashMap<u64, String>>, // node ID → "doc:42"
}
```

Note it's **generic over `S: VectorStorage`**. In tests, `S = InMemoryVectorStorage`. In production, `S = LsmVectorStorage`.

### HnswNode

```rust
pub struct HnswNode {
    pub id: u64,
    pub connections: Vec<Vec<u64>>,  // connections[layer] = list of neighbor IDs
    pub max_layer: usize,
}
```

A node **does not store its vector**. The vector is in the `VectorStorage` backend. The node only stores graph structure — which other nodes it's connected to at each layer.

### Insert algorithm

`insert_with_key(key, vector)`:

1. **Assign a random layer** using geometric distribution. Most nodes are only at layer 0. A few are at layer 1. Very few at layer 2+.

2. **Store the vector** in the storage backend.

3. **Greedy descent**: Starting from the entry point at the top layer, greedily walk to the nearest node at each layer above the new node's assigned layer.

4. **Connect at each layer**: For each layer the new node participates in:
   - Run a beam search (`search_layer`) to find the closest existing nodes
   - Select the best `M` neighbors
   - Create bidirectional connections
   - If a neighbor now has too many connections, **prune** by keeping only the closest ones

5. **Update entry point** if the new node's layer is the highest.

### Search algorithm

`search(query, k)`:

1. **Greedy descent** from the top layer to layer 1 (using `search_layer_single` — finds only the single nearest node at each layer)

2. **Beam search** at layer 0 using `ef_search` candidates (using `search_layer` — maintains a set of candidates and explores their neighbors)

3. **Return the top K** results sorted by distance

### Beam search (`search_layer`)

The heart of the search. Maintains two heaps:
- `candidates`: min-heap of unexplored candidates (closest first)
- `result`: max-heap of best results found so far (worst first — for easy eviction)

```
while candidates not empty:
    pop closest candidate
    if it's worse than the worst result AND we have enough results: stop
    for each neighbor of this candidate:
        if not visited:
            compute distance
            if better than worst result OR we don't have enough results yet:
                add to candidates AND result
                if result is too large: evict worst
```

### Candidate ordering

```rust
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}
```

Note the **reversed comparison** (`other` vs `self`). Rust's `BinaryHeap` is a max-heap, but we want a min-heap (closest first). By reversing the comparison, the heap pops the smallest distance first.

`MaxCandidate` uses normal ordering — it's a true max-heap for the result set.

### Persistence (GraphSnapshot)

```rust
pub struct GraphSnapshot {
    pub config: HnswConfig,
    pub nodes: HashMap<u64, HnswNode>,
    pub entry_point: Option<u64>,
    pub max_layer: usize,
    pub next_id: u64,
    pub key_to_id: HashMap<String, u64>,
    pub id_to_key: HashMap<u64, String>,
    pub vectors: HashMap<u64, Vec<f32>>,  // legacy, always empty for new snapshots
}
```

`snapshot()` captures the graph state for serialization. Note that `vectors` is always empty for new snapshots — vectors are persisted separately by `LsmVectorStorage`.

`restore()` reconstructs the index from a snapshot. The `vectors` field is iterated for backward compatibility with old snapshots that embedded vectors, but for new snapshots this is a no-op.

---

## 5. vector/iam.rs

**File**: `lsm_pg_extension/src/vector/iam.rs`

This implements the **Index Access Method** — the PostgreSQL interface that makes `CREATE INDEX ... USING lsm_hnsw` and `ORDER BY ... <-> ... LIMIT K` work.

### Global index registry

```rust
static HNSW_INDEXES: Lazy<RwLock<HashMap<pg_sys::Oid, IndexEntry>>> = ...;
```

Each PostgreSQL index has a unique **OID** (Object ID). We map OIDs to HNSW index instances.

### Creating vector storage

```rust
fn make_vector_storage(oid: pg_sys::Oid) -> Arc<LsmVectorStorage> {
    let rt = crate::tam::storage::TableStorage::global().runtime();
    let path = format!("/indexes/hnsw_{}/vectors", oid.as_u32());
    let config = LsmConfig::in_memory(&path);
    Arc::new(LsmVectorStorage::new(config, rt).expect("..."))
}
```

Each HNSW index gets its own `LsmVectorStorage` instance with a unique path. This keeps vector data isolated per index. It reuses the shared Tokio runtime from `TableStorage`.

### Distance metric detection

```rust
unsafe fn detect_metric(index_rel: pg_sys::Relation) -> DistanceMetric {
    // Read the operator family from the index relation
    // Look up the family name in the system catalog cache
    // Match on the name
    match name.as_ref() {
        "lsm_vector_cosine_ops" => DistanceMetric::Cosine,
        "lsm_vector_ip_ops" => DistanceMetric::InnerProduct,
        _ => DistanceMetric::L2,  // default
    }
}
```

When you write `CREATE INDEX ... USING lsm_hnsw (embedding lsm_vector_cosine_ops)`, PostgreSQL stores the operator class. We read it back from the system catalog to determine which distance metric to use.

This uses `SearchSysCache1` (a low-level PostgreSQL cache API) instead of SPI (SQL queries) to avoid interfering with concurrent scan states.

### Extracting vectors from datums

```rust
unsafe fn datum_to_vector(datum: pg_sys::Datum) -> Option<LsmVector> {
    let raw_ptr = datum.cast_mut_ptr::<pg_sys::varlena>();
    let detoasted = pg_sys::pg_detoast_datum(raw_ptr);
    let cbor_bytes = std::slice::from_raw_parts(data_ptr, data_len);
    serde_cbor::from_slice(cbor_bytes).ok()
}
```

pgrx stores `PostgresType` structs as CBOR inside a varlena. The datum may be TOASTed (compressed/out-of-line), so we detoast first, then deserialize the CBOR.

### ambuild — Build index from existing data

Called during `CREATE INDEX`:

1. Create or load the HNSW index for this OID
2. Define a C callback function (`build_callback`)
3. Call the heap table's `index_build_range_scan` — this iterates every row in the table and calls our callback
4. In the callback: extract the vector datum, convert to `LsmVector`, and insert into HNSW with the TID as the key
5. After building, persist the graph snapshot

### aminsert — Insert during DML

Called after an `INSERT` into the table:

1. Check if the column is NULL (skip if so)
2. Extract the vector from the datum
3. Encode the heap TID as a string key
4. Insert into the HNSW index
5. Save the graph snapshot

### amrescan — Receive the query and search

Called when the executor starts an index scan. The ORDER BY expression has been evaluated and the query vector is passed as a scan key:

```rust
unsafe extern "C" fn lsm_hnsw_amrescan(scan, keys, nkeys, orderbys, norderbys) {
    let orderby = &*orderbys.add(0);
    let datum = orderby.sk_argument;    // the query vector
    let query = datum_to_vector(datum);

    let results = index.search_with_keys(&query.data, 100);  // search for top 100

    // Convert HNSW results (TID string keys) to ItemPointerData
    for (tid_key, dist) in results {
        let tid_val = tid_key.parse::<u64>();
        state.results.push((u64_to_tid(tid_val), dist));
    }
}
```

### amgettuple — Return next result

Called repeatedly by the executor:

```rust
unsafe extern "C" fn lsm_hnsw_amgettuple(scan, direction) -> bool {
    let (tid, dist) = state.results[state.position];
    state.position += 1;

    (*scan).xs_heaptid = tid;     // tell PostgreSQL which row to fetch
    (*scan).xs_recheck = false;    // our results are exact, no recheck needed
    (*scan).xs_recheckorderby = false;
    true
}
```

PostgreSQL then calls the TAM's `index_fetch_tuple` to get the actual row data for that TID.

### Operator classes (SQL registration)

```sql
-- Default: L2 distance
CREATE OPERATOR CLASS lsm_vector_l2_ops
DEFAULT FOR TYPE LsmVector USING lsm_hnsw AS
    OPERATOR 1 <-> (LsmVector, LsmVector) FOR ORDER BY float_ops,
    FUNCTION 1 lsm_vector_l2_distance(LsmVector, LsmVector);

-- Opt-in: Cosine distance
CREATE OPERATOR CLASS lsm_vector_cosine_ops
FOR TYPE LsmVector USING lsm_hnsw AS
    OPERATOR 1 <=> (LsmVector, LsmVector) FOR ORDER BY float_ops,
    FUNCTION 1 lsm_vector_cosine_distance(LsmVector, LsmVector);

-- Opt-in: Inner product
CREATE OPERATOR CLASS lsm_vector_ip_ops
FOR TYPE LsmVector USING lsm_hnsw AS
    OPERATOR 1 <#> (LsmVector, LsmVector) FOR ORDER BY float_ops,
    FUNCTION 1 lsm_vector_inner_product(LsmVector, LsmVector);
```

`DEFAULT` means if you write `CREATE INDEX ... USING lsm_hnsw (embedding)` without specifying an operator class, PostgreSQL uses the L2 class. To use cosine: `CREATE INDEX ... USING lsm_hnsw (embedding lsm_vector_cosine_ops)`.

`FOR ORDER BY float_ops` tells PostgreSQL this operator produces float values that can be sorted, enabling `ORDER BY embedding <-> query` syntax.

### IndexAmRoutine

The method table is built at runtime (unlike the TAM which is a static):

```rust
unsafe fn make_index_am_routine() -> *mut pg_sys::IndexAmRoutine {
    let routine = pg_sys::palloc0(...) as *mut pg_sys::IndexAmRoutine;

    (*routine).amcanorderbyop = true;   // we support ORDER BY operator
    (*routine).amoptionalkey = true;    // scan key is optional (ORDER BY only)

    (*routine).ambuild = Some(lsm_hnsw_ambuild);
    (*routine).aminsert = Some(lsm_hnsw_aminsert);
    (*routine).ambeginscan = Some(lsm_hnsw_ambeginscan);
    (*routine).amrescan = Some(lsm_hnsw_amrescan);
    (*routine).amgettuple = Some(lsm_hnsw_amgettuple);
    (*routine).amendscan = Some(lsm_hnsw_amendscan);
    (*routine).amcostestimate = Some(lsm_hnsw_amcostestimate);
    // ...
    routine
}
```

The critical flag is `amcanorderbyop = true` — this tells the planner "I can handle ORDER BY with an operator." Without this, PostgreSQL would never use the index for KNN queries.

### Cost estimation

```rust
unsafe extern "C" fn lsm_hnsw_amcostestimate(...) {
    *index_startup_cost = 0.0;
    *index_total_cost = 1.0;
    *index_selectivity = 1.0;
}
```

We report a very low cost so the planner prefers the HNSW index over a sequential scan + sort for `ORDER BY ... <-> ... LIMIT K` queries.

---

## 6. End-to-End: How a KNN Query Works

Let's trace what happens when you run:

```sql
SELECT id, title FROM documents
ORDER BY embedding <-> '[0.1, 0.2, 0.3]'::LsmVector
LIMIT 5;
```

### Step 1: Parse & Plan

PostgreSQL parses the SQL and recognizes `ORDER BY ... <-> ...`. The planner sees:
- There's an index on `embedding` using `lsm_hnsw`
- The index supports `amcanorderbyop`
- The operator `<->` matches the index's operator class
- Cost estimate says the index is cheap

Decision: **Use an index scan** (not a sequential scan + sort).

### Step 2: Begin index scan

PostgreSQL calls `lsm_hnsw_ambeginscan()`. We allocate scan state.

### Step 3: Rescan with query vector

PostgreSQL evaluates the expression `'[0.1, 0.2, 0.3]'::LsmVector` (calling our `input()` function to parse the text). It passes the resulting datum to `lsm_hnsw_amrescan()` via the `orderbys` scan keys.

Our code:
1. Deserializes the datum to `LsmVector` via CBOR
2. Calls `index.search_with_keys(&[0.1, 0.2, 0.3], 100)`
3. The HNSW search traverses the graph (layers 3→2→1→0) using beam search
4. For each candidate, it loads the vector from `LsmVectorStorage` (which calls `LsmStore.get()`)
5. Returns the top 100 results as `(tid_string, distance)` pairs
6. We convert the TID strings back to `ItemPointerData`

### Step 4: Get tuples

PostgreSQL calls `lsm_hnsw_amgettuple()` up to 5 times (LIMIT 5). Each call:
1. Returns the next TID from our result list
2. Sets `xs_heaptid` on the scan descriptor

### Step 5: Heap fetch

For each TID, PostgreSQL calls the TAM's `lsm_index_fetch_tuple()`:
1. Converts the `ItemPointer` to our u64 TID id
2. Calls `storage.fetch_by_tid()` to get the key + value bytes
3. Deserializes the row (key → column 0, value → columns 1+)
4. Fills the TupleTableSlot

### Step 6: Return to user

PostgreSQL sends the 5 rows back to the client. Done!

```
 id   |     title      
------+----------------
 doc1 | Intro to ML
 doc5 | Neural Networks
 doc3 | Data Science
 doc8 | Feature Engineering
 doc2 | Deep Learning
(5 rows)
```
