//! HNSW Index Access Method for PostgreSQL.
//!
//! Registers `lsm_hnsw` as an index AM so users can:
//! ```sql
//! CREATE INDEX ON items USING lsm_hnsw (embedding);
//! SELECT * FROM items ORDER BY embedding <-> '[1,2,3]'::lsm_vector LIMIT 10;
//! ```
//!
//! The planner recognizes `ORDER BY col <-> query LIMIT K` and routes
//! through this AM's scan callbacks, which delegate to the HNSW graph.

use pgrx::pg_sys;
use pgrx::prelude::*;

use super::distance;
use super::hnsw::{DistanceMetric, GraphSnapshot, HnswConfig, HnswIndex};
use super::types::LsmVector;
use super::vector_storage::LsmVectorStorage;
use lsm_engine::LsmConfig;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────
// Global HNSW Index Registry (keyed by index relation OID)
// ─────────────────────────────────────────────────────────────────────

struct IndexEntry {
    index: Arc<HnswIndex<LsmVectorStorage>>,
    metric: DistanceMetric,
}

static HNSW_INDEXES: Lazy<RwLock<HashMap<pg_sys::Oid, IndexEntry>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Create an LSM-backed vector store for a given index OID.
fn make_vector_storage(oid: pg_sys::Oid) -> Arc<LsmVectorStorage> {
    let rt = crate::tam::storage::TableStorage::global().runtime();
    let path = format!("/indexes/hnsw_{}/vectors", oid.as_u32());
    let config = LsmConfig::in_memory(&path);
    Arc::new(
        LsmVectorStorage::new(config, rt)
            .expect("Failed to create vector storage for HNSW index"),
    )
}

fn save_index(oid: pg_sys::Oid) {
    let map = HNSW_INDEXES.read();
    if let Some(entry) = map.get(&oid) {
        let snapshot = entry.index.snapshot();

        let json = match serde_json::to_vec(&snapshot) {
            Ok(j) => j,
            Err(e) => {
                tracing::warn!("Failed to serialize HNSW index {}: {}", oid.as_u32(), e);
                return;
            }
        };

        let storage = crate::tam::storage::TableStorage::global();
        let key = format!("__hnsw_graph_{}", oid.as_u32());
        if let Err(e) = storage.insert("__hnsw_meta", key.as_bytes(), &json) {
            tracing::warn!("Failed to persist HNSW index {}: {}", oid.as_u32(), e);
        }
    }
}

fn load_index(
    oid: pg_sys::Oid,
    metric: DistanceMetric,
) -> Option<Arc<HnswIndex<LsmVectorStorage>>> {
    let storage_global = crate::tam::storage::TableStorage::global();
    let key = format!("__hnsw_graph_{}", oid.as_u32());

    match storage_global.get("__hnsw_meta", key.as_bytes()) {
        Ok(Some(json)) => {
            match serde_json::from_slice::<GraphSnapshot>(&json) {
                Ok(snapshot) => {
                    let vs = make_vector_storage(oid);
                    let index = Arc::new(HnswIndex::restore(snapshot, vs));
                    HNSW_INDEXES.write().insert(oid, IndexEntry {
                        index: index.clone(),
                        metric,
                    });
                    Some(index)
                }
                Err(e) => {
                    tracing::warn!("Failed to deserialize HNSW index {}: {}", oid.as_u32(), e);
                    None
                }
            }
        }
        _ => None,
    }
}

fn get_or_create_index(
    oid: pg_sys::Oid,
    metric: DistanceMetric,
) -> Arc<HnswIndex<LsmVectorStorage>> {
    {
        let map = HNSW_INDEXES.read();
        if let Some(entry) = map.get(&oid) {
            return entry.index.clone();
        }
    }
    if let Some(index) = load_index(oid, metric) {
        return index;
    }
    let config = HnswConfig {
        metric,
        ..HnswConfig::default()
    };
    let storage = make_vector_storage(oid);
    let index = Arc::new(HnswIndex::new(config, storage));
    HNSW_INDEXES.write().insert(oid, IndexEntry {
        index: index.clone(),
        metric,
    });
    index
}

/// Detect the distance metric from the operator family registered
/// on the first indexed column.  Falls back to L2 if lookup fails.
///
/// Uses the system catalog cache (no SPI) to avoid interfering with
/// concurrent scan states.
unsafe fn detect_metric(index_rel: pg_sys::Relation) -> DistanceMetric {
    if (*index_rel).rd_opfamily.is_null() {
        return DistanceMetric::L2;
    }
    let opfamily_oid = *(*index_rel).rd_opfamily.add(0);

    let cache_id = pg_sys::SysCacheIdentifier::OPFAMILYOID as i32;
    let tuple = pg_sys::SearchSysCache1(
        cache_id,
        pg_sys::Datum::from(opfamily_oid.as_u32() as usize),
    );
    if tuple.is_null() {
        return DistanceMetric::L2;
    }
    let form = pg_sys::GETSTRUCT(tuple) as *const pg_sys::FormData_pg_opfamily;
    let name = std::ffi::CStr::from_ptr((*form).opfname.data.as_ptr())
        .to_string_lossy();
    pg_sys::ReleaseSysCache(tuple);

    match name.as_ref() {
        "lsm_vector_cosine_ops" => DistanceMetric::Cosine,
        "lsm_vector_ip_ops" => DistanceMetric::InnerProduct,
        _ => DistanceMetric::L2,
    }
}

fn get_index(oid: pg_sys::Oid) -> Option<Arc<HnswIndex<LsmVectorStorage>>> {
    HNSW_INDEXES.read().get(&oid).map(|e| e.index.clone())
}

// ─────────────────────────────────────────────────────────────────────
// TID ↔ HNSW node ID mapping (per index)
//
// Reuses the same encoding as the TAM's TidManager so that TIDs
// round-trip correctly between the heap and the HNSW index.
// ─────────────────────────────────────────────────────────────────────

use crate::tam::storage::{tid_to_block_offset, block_offset_to_tid};

/// Encode an ItemPointerData (TID) as a unique u64 for HNSW.
fn tid_to_u64(tid: &pg_sys::ItemPointerData) -> u64 {
    let block = unsafe { pg_sys::ItemPointerGetBlockNumberNoCheck(tid as *const _) };
    let offset = unsafe { pg_sys::ItemPointerGetOffsetNumberNoCheck(tid as *const _) };
    block_offset_to_tid(block, offset)
}

/// Decode a u64 back into an ItemPointerData.
fn u64_to_tid(val: u64) -> pg_sys::ItemPointerData {
    let (block, offset) = tid_to_block_offset(val);
    let mut tid = pg_sys::ItemPointerData::default();
    unsafe { pg_sys::ItemPointerSet(&mut tid, block, offset) };
    tid
}

// ─────────────────────────────────────────────────────────────────────
// Helper: extract LsmVector from a datum
// ─────────────────────────────────────────────────────────────────────

/// Deserialize an LsmVector from a Postgres datum.
///
/// pgrx `PostgresType` stores the struct as CBOR inside a varlena.
/// The datum may be TOASTed (compressed / out-of-line), so we must
/// detoast before accessing the payload bytes.
unsafe fn datum_to_vector(datum: pg_sys::Datum) -> Option<LsmVector> {
    if datum.value() == 0 {
        return None;
    }
    let raw_ptr = datum.cast_mut_ptr::<pg_sys::varlena>();
    let detoasted = pg_sys::pg_detoast_datum(raw_ptr);
    let data_len = pgrx::varsize_any_exhdr(detoasted as *const pg_sys::varlena);
    let data_ptr = pgrx::vardata_any(detoasted as *const pg_sys::varlena) as *const u8;
    let cbor_bytes = std::slice::from_raw_parts(data_ptr, data_len);
    serde_cbor::from_slice(cbor_bytes).ok()
}

// ─────────────────────────────────────────────────────────────────────
// Custom scan state (stored in scan->opaque)
// ─────────────────────────────────────────────────────────────────────

struct HnswScanState {
    results: Vec<(pg_sys::ItemPointerData, f32)>,
    position: usize,
}

// ─────────────────────────────────────────────────────────────────────
// IAM Callback: ambuild
// ─────────────────────────────────────────────────────────────────────

/// Build the HNSW index from existing table data.
///
/// Called during CREATE INDEX. We scan the heap table (via TAM's
/// index_build_range_scan), extract vectors, and insert into HNSW.
unsafe extern "C" fn lsm_hnsw_ambuild(
    heap_rel: pg_sys::Relation,
    index_rel: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    let index_oid = (*index_rel).rd_id;
    let metric = detect_metric(index_rel);
    let index = get_or_create_index(index_oid, metric);

    struct BuildState {
        index: Arc<HnswIndex<LsmVectorStorage>>,
        count: f64,
    }

    let mut state = BuildState {
        index: index.clone(),
        count: 0.0,
    };

    // The callback invoked for each heap tuple
    unsafe extern "C" fn build_callback(
        _index_rel: pg_sys::Relation,
        tid: pg_sys::ItemPointer,
        values: *mut pg_sys::Datum,
        isnull: *mut bool,
        _tuple_is_alive: bool,
        callback_state: *mut std::ffi::c_void,
    ) {
        let state = &mut *(callback_state as *mut BuildState);

        // Skip NULL vectors
        if *isnull.add(0) {
            return;
        }

        let datum = *values.add(0);

        let vector = match datum_to_vector(datum) {
            Some(v) => v,
            None => return,
        };

        let tid_val = tid_to_u64(&*tid);
        let tid_key = format!("{}", tid_val);

        state.index.insert_with_key(Some(tid_key), vector.data);
        state.count += 1.0;
    }

    // Use the table AM's index_build_range_scan to iterate tuples
    let heap_am = (*heap_rel).rd_tableam;
    if !heap_am.is_null() {
        if let Some(build_scan) = (*heap_am).index_build_range_scan {
            build_scan(
                heap_rel,
                index_rel,
                index_info,
                true,  // allow_sync
                false, // anyvisible
                false, // progress
                0,     // start_blockno
                pg_sys::InvalidBlockNumber,
                Some(build_callback),
                &mut state as *mut BuildState as *mut std::ffi::c_void,
                std::ptr::null_mut(), // scan
            );
        }
    }

    // Persist the built graph to object storage
    save_index(index_oid);

    // Allocate and return the build result
    let result = pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBuildResult>())
        as *mut pg_sys::IndexBuildResult;
    (*result).heap_tuples = state.count;
    (*result).index_tuples = state.count;

    result
}

/// Build an empty index (for REINDEX of an empty table).
unsafe extern "C" fn lsm_hnsw_ambuildempty(index_rel: pg_sys::Relation) {
    let index_oid = (*index_rel).rd_id;
    let metric = detect_metric(index_rel);
    get_or_create_index(index_oid, metric);
}

// ─────────────────────────────────────────────────────────────────────
// IAM Callback: aminsert
// ─────────────────────────────────────────────────────────────────────

/// Insert a new vector into the HNSW index during DML.
unsafe extern "C" fn lsm_hnsw_aminsert(
    index_rel: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    _heap_rel: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    if *isnull.add(0) {
        return false;
    }

    let index_oid = (*index_rel).rd_id;
    let metric = detect_metric(index_rel);
    let index = get_or_create_index(index_oid, metric);

    let datum = *values.add(0);
    let vector = match datum_to_vector(datum) {
        Some(v) => v,
        None => return false,
    };

    let tid_val = tid_to_u64(&*heap_tid);
    let tid_key = format!("{}", tid_val);

    index.insert_with_key(Some(tid_key), vector.data);
    save_index(index_oid);
    true
}

// ─────────────────────────────────────────────────────────────────────
// IAM Callbacks: Vacuum
// ─────────────────────────────────────────────────────────────────────

unsafe extern "C" fn lsm_hnsw_ambulkdelete(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    _callback: pg_sys::IndexBulkDeleteCallback,
    _callback_state: *mut std::ffi::c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    if stats.is_null() {
        let s = pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBulkDeleteResult>())
            as *mut pg_sys::IndexBulkDeleteResult;
        if let Some(idx) = get_index((*(*info).index).rd_id) {
            (*s).num_index_tuples = idx.len() as f64;
        }
        return s;
    }
    stats
}

unsafe extern "C" fn lsm_hnsw_amvacuumcleanup(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    if stats.is_null() {
        let s = pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBulkDeleteResult>())
            as *mut pg_sys::IndexBulkDeleteResult;
        if let Some(idx) = get_index((*(*info).index).rd_id) {
            (*s).num_index_tuples = idx.len() as f64;
        }
        return s;
    }
    stats
}

// ─────────────────────────────────────────────────────────────────────
// IAM Callback: amcostestimate
// ─────────────────────────────────────────────────────────────────────

/// Tell the query planner how expensive an HNSW index scan is.
///
/// We give a low cost so the planner prefers the index for
/// ORDER BY <-> LIMIT K queries.
unsafe extern "C" fn lsm_hnsw_amcostestimate(
    _root: *mut pg_sys::PlannerInfo,
    path: *mut pg_sys::IndexPath,
    _loop_count: f64,
    index_startup_cost: *mut pg_sys::Cost,
    index_total_cost: *mut pg_sys::Cost,
    index_selectivity: *mut pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    // HNSW search is O(log N * ef_search) — very cheap
    *index_startup_cost = 0.0;
    *index_total_cost = 1.0;
    *index_selectivity = 1.0;
    *index_correlation = 0.0;
    *index_pages = 1.0;

    // If the query has a LIMIT, incorporate it
    if !path.is_null() {
        let indexinfo = (*path).indexinfo;
        if !indexinfo.is_null() {
            let tuples = (*indexinfo).tuples;
            if tuples > 0.0 {
                *index_total_cost = (tuples.ln() * 10.0).max(1.0);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// IAM Callback: amoptions
// ─────────────────────────────────────────────────────────────────────

unsafe extern "C" fn lsm_hnsw_amoptions(
    _reloptions: pg_sys::Datum,
    _validate: bool,
) -> *mut pg_sys::bytea {
    std::ptr::null_mut()
}

// ─────────────────────────────────────────────────────────────────────
// IAM Callback: amvalidate
// ─────────────────────────────────────────────────────────────────────

unsafe extern "C" fn lsm_hnsw_amvalidate(_opclassoid: pg_sys::Oid) -> bool {
    true
}

// ─────────────────────────────────────────────────────────────────────
// IAM Callbacks: Index Scan
// ─────────────────────────────────────────────────────────────────────

/// Begin an index scan — allocate the scan descriptor.
unsafe extern "C" fn lsm_hnsw_ambeginscan(
    index_rel: pg_sys::Relation,
    nkeys: std::ffi::c_int,
    norderbys: std::ffi::c_int,
) -> pg_sys::IndexScanDesc {
    let scan = pg_sys::RelationGetIndexScan(index_rel, nkeys, norderbys);

    let state = Box::new(HnswScanState {
        results: Vec::new(),
        position: 0,
    });
    (*scan).opaque = Box::into_raw(state) as *mut std::ffi::c_void;

    scan
}

/// Rescan — receive the ORDER BY scan keys and perform the HNSW search.
///
/// For `ORDER BY embedding <-> '[1,2,3]'::lsm_vector LIMIT K`:
///   - orderbys[0].sk_argument = the query vector datum
///   - The distance operator strategy is used to identify the metric
unsafe extern "C" fn lsm_hnsw_amrescan(
    scan: pg_sys::IndexScanDesc,
    _keys: pg_sys::ScanKey,
    _nkeys: std::ffi::c_int,
    orderbys: pg_sys::ScanKey,
    norderbys: std::ffi::c_int,
) {
    let state = &mut *((*scan).opaque as *mut HnswScanState);
    state.results.clear();
    state.position = 0;

    if norderbys < 1 || orderbys.is_null() {
        return;
    }

    if !orderbys.is_null() && norderbys > 0 {
        std::ptr::copy_nonoverlapping(
            orderbys,
            (*scan).orderByData,
            norderbys as usize,
        );
    }

    let orderby = &*orderbys.add(0);
    let datum = orderby.sk_argument;

    let query = match datum_to_vector(datum) {
        Some(v) => v,
        None => return,
    };

    let index_oid = (*(*scan).indexRelation).rd_id;
    let index = match get_index(index_oid) {
        Some(idx) => idx,
        None => return,
    };

    let k = 100;
    let search_results = index.search_with_keys(&query.data, k);

    for (tid_key, dist) in search_results {
        if let Ok(tid_val) = tid_key.parse::<u64>() {
            let tid = u64_to_tid(tid_val);
            state.results.push((tid, dist));
        }
    }
}

/// Get the next tuple from the index scan.
///
/// Sets xs_heaptid (the TID for heap fetch) and xs_orderbyvals
/// (the distance for the planner to verify ordering).
unsafe extern "C" fn lsm_hnsw_amgettuple(
    scan: pg_sys::IndexScanDesc,
    _direction: pg_sys::ScanDirection::Type,
) -> bool {
    let state = &mut *((*scan).opaque as *mut HnswScanState);

    if state.position >= state.results.len() {
        return false;
    }

    let (ref tid, _dist) = state.results[state.position];
    state.position += 1;

    (*scan).xs_heaptid = *tid;
    (*scan).xs_recheck = false;
    (*scan).xs_recheckorderby = false;

    true
}

/// End the index scan — free our state.
unsafe extern "C" fn lsm_hnsw_amendscan(scan: pg_sys::IndexScanDesc) {
    if !(*scan).opaque.is_null() {
        let _ = Box::from_raw((*scan).opaque as *mut HnswScanState);
        (*scan).opaque = std::ptr::null_mut();
    }
}

// ─────────────────────────────────────────────────────────────────────
// IAM Stubs
// ─────────────────────────────────────────────────────────────────────

unsafe extern "C" fn lsm_hnsw_amcanreturn(
    _index_rel: pg_sys::Relation,
    _attno: std::ffi::c_int,
) -> bool {
    false // No index-only scans (must always fetch from heap)
}

// ─────────────────────────────────────────────────────────────────────
// IndexAmRoutine: The complete method table
// ─────────────────────────────────────────────────────────────────────

/// Construct the IndexAmRoutine at runtime (Postgres requires a heap-allocated node).
///
/// Because IndexAmRoutine contains Option types (not simple booleans),
/// we build it in the handler function rather than as a static const.
unsafe fn make_index_am_routine() -> *mut pg_sys::IndexAmRoutine {
    let routine = pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexAmRoutine>())
        as *mut pg_sys::IndexAmRoutine;

    (*routine).type_ = pg_sys::NodeTag::T_IndexAmRoutine;

    // AM capabilities
    (*routine).amstrategies = 1;        // 1 ordering strategy: distance
    (*routine).amsupport = 1;           // 1 support function: distance calc
    (*routine).amoptsprocnum = 0;
    (*routine).amcanorder = false;      // can't do btree-style ordering
    (*routine).amcanorderbyop = true;   // CAN do ORDER BY operator (<->)
    (*routine).amcanbackward = false;
    (*routine).amcanunique = false;
    (*routine).amcanmulticol = false;
    (*routine).amoptionalkey = true;    // scan key is optional (ORDER BY only)
    (*routine).amsearcharray = false;
    (*routine).amsearchnulls = false;
    (*routine).amstorage = false;
    (*routine).amclusterable = false;
    (*routine).ampredlocks = false;
    (*routine).amcanparallel = false;
    (*routine).amcanbuildparallel = false;
    (*routine).amcaninclude = false;
    (*routine).amusemaintenanceworkmem = false;
    (*routine).amsummarizing = false;
    (*routine).amkeytype = pg_sys::InvalidOid;

    // Required callbacks
    (*routine).ambuild = Some(lsm_hnsw_ambuild);
    (*routine).ambuildempty = Some(lsm_hnsw_ambuildempty);
    (*routine).aminsert = Some(lsm_hnsw_aminsert);
    (*routine).aminsertcleanup = None;
    (*routine).ambulkdelete = Some(lsm_hnsw_ambulkdelete);
    (*routine).amvacuumcleanup = Some(lsm_hnsw_amvacuumcleanup);
    (*routine).amcanreturn = Some(lsm_hnsw_amcanreturn);
    (*routine).amcostestimate = Some(lsm_hnsw_amcostestimate);
    (*routine).amoptions = Some(lsm_hnsw_amoptions);
    (*routine).amproperty = None;
    (*routine).ambuildphasename = None;
    (*routine).amvalidate = Some(lsm_hnsw_amvalidate);
    (*routine).amadjustmembers = None;

    // Scan callbacks
    (*routine).ambeginscan = Some(lsm_hnsw_ambeginscan);
    (*routine).amrescan = Some(lsm_hnsw_amrescan);
    (*routine).amgettuple = Some(lsm_hnsw_amgettuple);
    (*routine).amgetbitmap = None;
    (*routine).amendscan = Some(lsm_hnsw_amendscan);
    (*routine).ammarkpos = None;
    (*routine).amrestrpos = None;

    // Parallel scan (not supported)
    (*routine).amestimateparallelscan = None;
    (*routine).aminitparallelscan = None;
    (*routine).amparallelrescan = None;

    routine
}

// ─────────────────────────────────────────────────────────────────────
// Handler Function + SQL Registration
// ─────────────────────────────────────────────────────────────────────

// Part 1: Register the HNSW access method (no type dependency).
pgrx::extension_sql!(
    r#"
    CREATE OR REPLACE FUNCTION lsm_hnsw_handler(internal)
    RETURNS index_am_handler
    AS 'MODULE_PATHNAME', 'lsm_hnsw_handler'
    LANGUAGE C STRICT;

    CREATE ACCESS METHOD lsm_hnsw TYPE INDEX HANDLER lsm_hnsw_handler;
    "#,
    name = "lsm_hnsw_am_sql",
    requires = ["lsm_s3_tam_handler_sql"]
);

// Part 2: Operator families + operator classes.
// L2 is the DEFAULT.  Cosine and inner product are opt-in.
pgrx::extension_sql!(
    r#"
    -- L2 (Euclidean) — default operator class
    CREATE OPERATOR FAMILY lsm_vector_l2_ops USING lsm_hnsw;
    CREATE OPERATOR CLASS lsm_vector_l2_ops
    DEFAULT FOR TYPE LsmVector USING lsm_hnsw
    FAMILY lsm_vector_l2_ops AS
        OPERATOR 1 <-> (LsmVector, LsmVector) FOR ORDER BY float_ops,
        FUNCTION 1 lsm_vector_l2_distance(LsmVector, LsmVector);

    -- Cosine distance
    CREATE OPERATOR FAMILY lsm_vector_cosine_ops USING lsm_hnsw;
    CREATE OPERATOR CLASS lsm_vector_cosine_ops
    FOR TYPE LsmVector USING lsm_hnsw
    FAMILY lsm_vector_cosine_ops AS
        OPERATOR 1 <=> (LsmVector, LsmVector) FOR ORDER BY float_ops,
        FUNCTION 1 lsm_vector_cosine_distance(LsmVector, LsmVector);

    -- (Negative) inner product
    CREATE OPERATOR FAMILY lsm_vector_ip_ops USING lsm_hnsw;
    CREATE OPERATOR CLASS lsm_vector_ip_ops
    FOR TYPE LsmVector USING lsm_hnsw
    FAMILY lsm_vector_ip_ops AS
        OPERATOR 1 <#> (LsmVector, LsmVector) FOR ORDER BY float_ops,
        FUNCTION 1 lsm_vector_inner_product(LsmVector, LsmVector);
    "#,
    name = "lsm_hnsw_opclass_sql",
    requires = ["lsm_hnsw_am_sql", "lsm_vector_operators"]
);

#[no_mangle]
pub extern "C" fn pg_finfo_lsm_hnsw_handler() -> *const pg_sys::Pg_finfo_record {
    const MY_FINFO: pg_sys::Pg_finfo_record = pg_sys::Pg_finfo_record { api_version: 1 };
    &MY_FINFO
}

#[no_mangle]
pub extern "C" fn lsm_hnsw_handler(_fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    unsafe {
        let routine = make_index_am_routine();
        pg_sys::Datum::from(routine as usize)
    }
}

// ─────────────────────────────────────────────────────────────────────
// SQL helper: distance function that the operator class references
// ─────────────────────────────────────────────────────────────────────

/// L2 distance support function for the HNSW operator class.
/// (Re-exported so the operator class FUNCTION 1 reference works.)
#[pg_extern(immutable, parallel_safe)]
fn lsm_hnsw_l2_distance(a: LsmVector, b: LsmVector) -> f32 {
    distance::l2_distance(&a.data, &b.data)
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_create_hnsw_index() {
        Spi::run("CREATE TABLE hnsw_test (id TEXT, embedding LsmVector) USING lsm_s3;").unwrap();
        Spi::run("CREATE INDEX hnsw_idx ON hnsw_test USING lsm_hnsw (embedding);").unwrap();

        let exists = Spi::get_one::<bool>(
            "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname = 'hnsw_idx')"
        ).unwrap();
        assert_eq!(exists, Some(true));
    }

    #[pg_test]
    fn test_hnsw_insert_and_search() {
        Spi::run("CREATE TABLE vec_items (id TEXT, embedding LsmVector) USING lsm_s3;").unwrap();
        Spi::run("CREATE INDEX vec_idx ON vec_items USING lsm_hnsw (embedding);").unwrap();

        Spi::run("INSERT INTO vec_items VALUES ('a', '[1.0, 0.0, 0.0]');").unwrap();
        Spi::run("INSERT INTO vec_items VALUES ('b', '[0.0, 1.0, 0.0]');").unwrap();
        Spi::run("INSERT INTO vec_items VALUES ('c', '[0.0, 0.0, 1.0]');").unwrap();
        Spi::run("INSERT INTO vec_items VALUES ('d', '[1.0, 1.0, 0.0]');").unwrap();

        let nearest_id = Spi::get_one::<String>(
            "SELECT id FROM vec_items ORDER BY embedding <-> '[1.0, 0.0, 0.0]'::LsmVector LIMIT 1"
        ).unwrap();
        assert_eq!(nearest_id, Some("a".to_string()), "Nearest to [1,0,0] should be 'a'");
    }

    #[pg_test]
    fn test_hnsw_knn_ordering() {
        Spi::run("CREATE TABLE knn_test (id TEXT, vec LsmVector) USING lsm_s3;").unwrap();
        Spi::run("CREATE INDEX knn_idx ON knn_test USING lsm_hnsw (vec);").unwrap();

        Spi::run("INSERT INTO knn_test VALUES ('near',  '[0.1, 0.1]');").unwrap();
        Spi::run("INSERT INTO knn_test VALUES ('mid',   '[5.0, 5.0]');").unwrap();
        Spi::run("INSERT INTO knn_test VALUES ('far',   '[10.0, 10.0]');").unwrap();

        let ids: Vec<String> = Spi::connect(|client| {
            let mut results = Vec::new();
            let tup_table = client.select(
                "SELECT id FROM knn_test ORDER BY vec <-> '[0.0, 0.0]'::LsmVector LIMIT 3",
                None, None,
            ).unwrap();
            for row in tup_table {
                if let Ok(Some(id)) = row.get_by_name::<String, _>("id") {
                    results.push(id);
                }
            }
            results
        });

        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0], "near");
        assert_eq!(ids[1], "mid");
        assert_eq!(ids[2], "far");
    }

    #[pg_test]
    fn test_hnsw_empty_table() {
        Spi::run("CREATE TABLE empty_vec (id TEXT, v LsmVector) USING lsm_s3;").unwrap();
        Spi::run("CREATE INDEX empty_idx ON empty_vec USING lsm_hnsw (v);").unwrap();

        let count = Spi::get_one::<i64>(
            "SELECT COUNT(*) FROM (SELECT id FROM empty_vec ORDER BY v <-> '[1,2,3]'::LsmVector LIMIT 5) t"
        ).unwrap();
        assert_eq!(count, Some(0));
    }

    #[pg_test]
    fn test_hnsw_index_on_heap_table() {
        Spi::run("CREATE TABLE heap_vec (id SERIAL PRIMARY KEY, v LsmVector);").unwrap();
        Spi::run("CREATE INDEX heap_vec_idx ON heap_vec USING lsm_hnsw (v);").unwrap();

        Spi::run("INSERT INTO heap_vec (v) VALUES ('[1,0]'), ('[0,1]'), ('[1,1]');").unwrap();

        let count = Spi::get_one::<i64>(
            "SELECT COUNT(*) FROM heap_vec"
        ).unwrap();
        assert_eq!(count, Some(3));
    }
}
