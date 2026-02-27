//! TAM Handler: PostgreSQL Table Access Method implemented in Rust.
//!
//! This module registers LSM-S3 as a table storage engine so users can:
//! ```sql
//! CREATE TABLE items (id TEXT, data TEXT) USING lsm_s3;
//! INSERT INTO items VALUES ('key1', 'hello');
//! SELECT * FROM items;
//! DELETE FROM items WHERE id = 'key1';
//! ```
//!
//! The TAM routes through `TableStorage` to the LSM engine (same as
//! the function-based API, but with standard SQL syntax).

use pgrx::prelude::*;
use pgrx::pg_sys;
use super::storage::{TableStorage, tid_to_block_offset, block_offset_to_tid};
use super::txn;
use std::ffi::CStr;
use std::os::raw::c_int;

// ─────────────────────────────────────────────────────────────────────
// Helpers: generic datum serialization for any PostgreSQL type
//
// We serialize/deserialize individual datums based on their type
// info from the tuple descriptor (attbyval, attlen).  This allows
// the TAM to store tables with any column types.
// ─────────────────────────────────────────────────────────────────────

/// Extract the payload bytes from a varlena-typed datum (detoasted).
unsafe fn datum_to_varlena_bytes(datum: pg_sys::Datum) -> Vec<u8> {
    let varlena_ptr = datum.cast_mut_ptr::<pg_sys::varlena>();
    let detoasted = pg_sys::pg_detoast_datum(varlena_ptr);
    let len = pgrx::varsize_any_exhdr(detoasted as *const pg_sys::varlena);
    let ptr = pgrx::vardata_any(detoasted as *const pg_sys::varlena) as *const u8;
    std::slice::from_raw_parts(ptr, len).to_vec()
}

/// Recreate a varlena datum from stored payload bytes.
unsafe fn varlena_bytes_to_datum(bytes: &[u8]) -> pg_sys::Datum {
    let bytea_ptr = pgrx::rust_byte_slice_to_bytea(bytes);
    pg_sys::Datum::from(bytea_ptr.into_pg() as usize)
}

// ─────────────────────────────────────────────────────────────────────
// Multi-column row serialization
//
// Format (columns start_col .. natts-1):
//   [num_cols: u16]
//   For each column:
//     [is_null: u8]                (1 = null, 0 = has data)
//     If not null:
//       [byval: u8]               (1 = pass-by-value, 0 = varlena)
//       [data_len: u32]
//       [data: raw bytes]
// ─────────────────────────────────────────────────────────────────────

unsafe fn serialize_row(
    tts_values: *mut pg_sys::Datum,
    tts_isnull: *mut bool,
    tupdesc: pg_sys::TupleDesc,
    start_col: usize,
) -> Vec<u8> {
    let natts = (*tupdesc).natts as usize;
    let num_cols = natts.saturating_sub(start_col);
    let mut buf = Vec::with_capacity(2 + num_cols * 8);
    buf.extend_from_slice(&(num_cols as u16).to_le_bytes());

    let attrs_base = (*tupdesc).attrs.as_ptr();

    for i in start_col..natts {
        if *tts_isnull.add(i) {
            buf.push(1);
            continue;
        }
        buf.push(0); // not null

        let att = &*attrs_base.add(i);
        let datum = *tts_values.add(i);

        if att.attbyval {
            buf.push(1); // byval
            let len = att.attlen as usize;
            buf.extend_from_slice(&(len as u32).to_le_bytes());
            let raw = datum.value().to_le_bytes();
            buf.extend_from_slice(&raw[..len.min(8)]);
        } else if att.attlen == -1 {
            buf.push(0); // varlena
            let payload = datum_to_varlena_bytes(datum);
            buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            buf.extend_from_slice(&payload);
        } else if att.attlen == -2 {
            buf.push(0); // cstring (treat as varlena-like)
            let cstr = CStr::from_ptr(datum.cast_mut_ptr());
            let bytes = cstr.to_bytes_with_nul();
            buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(bytes);
        } else {
            buf.push(0); // fixed-length pass-by-ref
            let len = att.attlen as usize;
            buf.extend_from_slice(&(len as u32).to_le_bytes());
            let ptr = datum.cast_mut_ptr::<u8>();
            buf.extend_from_slice(std::slice::from_raw_parts(ptr, len));
        }
    }
    buf
}

unsafe fn deserialize_row(
    bytes: &[u8],
    tts_values: *mut pg_sys::Datum,
    tts_isnull: *mut bool,
    tupdesc: pg_sys::TupleDesc,
    start_col: usize,
) {
    let natts = (*tupdesc).natts as usize;
    if bytes.len() < 2 {
        for i in start_col..natts {
            *tts_isnull.add(i) = true;
        }
        return;
    }

    let attrs_base = (*tupdesc).attrs.as_ptr();
    let mut pos = 0usize;
    let num_cols = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
    pos += 2;

    for ci in 0..num_cols {
        let col = start_col + ci;
        if col >= natts || pos >= bytes.len() {
            break;
        }

        let is_null = bytes[pos];
        pos += 1;
        if is_null != 0 {
            *tts_isnull.add(col) = true;
            continue;
        }

        if pos >= bytes.len() { break; }
        let byval = bytes[pos] != 0;
        pos += 1;

        if pos + 4 > bytes.len() { break; }
        let data_len = u32::from_le_bytes([
            bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3],
        ]) as usize;
        pos += 4;

        if pos + data_len > bytes.len() { break; }
        let data = &bytes[pos..pos + data_len];
        pos += data_len;

        let att = &*attrs_base.add(col);

        if byval {
            let mut val: u64 = 0;
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                &mut val as *mut u64 as *mut u8,
                data_len.min(8),
            );
            *tts_values.add(col) = pg_sys::Datum::from(val as usize);
        } else if att.attlen == -2 {
            let ptr = pg_sys::palloc(data_len) as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data_len);
            *tts_values.add(col) = pg_sys::Datum::from(ptr as usize);
        } else {
            *tts_values.add(col) = varlena_bytes_to_datum(data);
        }
        *tts_isnull.add(col) = false;
    }

    for i in (start_col + num_cols)..natts {
        *tts_isnull.add(i) = true;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Custom scan state (extends TableScanDescData)
// ─────────────────────────────────────────────────────────────────────

#[repr(C)]
struct LsmScanDesc {
    base: pg_sys::TableScanDescData,
    table_name: String,
    /// Keys and TID ids only — values fetched on demand in getnextslot.
    keys: Vec<(Vec<u8>, u64)>,
    position: usize,
}

// ─────────────────────────────────────────────────────────────────────
// Custom index fetch state (extends IndexFetchTableData)
// ─────────────────────────────────────────────────────────────────────

#[repr(C)]
struct LsmIndexFetchData {
    base: pg_sys::IndexFetchTableData,
    table_name: String,
}

// ─────────────────────────────────────────────────────────────────────
// Helper: extract table name from a Relation
// ─────────────────────────────────────────────────────────────────────

unsafe fn relation_name(rel: pg_sys::Relation) -> String {
    let rd_rel = (*rel).rd_rel;
    let name_data = (*rd_rel).relname.data;
    let c_str = CStr::from_ptr(name_data.as_ptr());
    c_str.to_string_lossy().into_owned()
}

// ─────────────────────────────────────────────────────────────────────
// TAM Callback: slot_callbacks
// ─────────────────────────────────────────────────────────────────────

unsafe extern "C" fn lsm_slot_callbacks(
    _rel: pg_sys::Relation,
) -> *const pg_sys::TupleTableSlotOps {
    &pg_sys::TTSOpsVirtual
}

// ─────────────────────────────────────────────────────────────────────
// TAM Callbacks: Sequential Scan
// ─────────────────────────────────────────────────────────────────────

unsafe extern "C" fn lsm_scan_begin(
    rel: pg_sys::Relation,
    snapshot: pg_sys::Snapshot,
    nkeys: c_int,
    key: *mut pg_sys::ScanKeyData,
    pscan: pg_sys::ParallelTableScanDesc,
    flags: pg_sys::uint32,
) -> pg_sys::TableScanDesc {
    let table_name = relation_name(rel);

    let storage = TableStorage::global();
    let keys = match storage.scan_keys_with_tids(&table_name) {
        Ok(rows) => rows,
        Err(e) => {
            pgrx::warning!("LSM scan_begin error: {}", e);
            vec![]
        }
    };

    let scan = pg_sys::palloc0(std::mem::size_of::<LsmScanDesc>()) as *mut LsmScanDesc;

    (*scan).base.rs_rd = rel;
    (*scan).base.rs_snapshot = snapshot;
    (*scan).base.rs_nkeys = nkeys;
    (*scan).base.rs_key = key;
    (*scan).base.rs_flags = flags;
    (*scan).base.rs_parallel = pscan;

    std::ptr::write(&mut (*scan).table_name, table_name);
    std::ptr::write(&mut (*scan).keys, keys);
    (*scan).position = 0;

    scan as pg_sys::TableScanDesc
}

unsafe extern "C" fn lsm_scan_getnextslot(
    scan: pg_sys::TableScanDesc,
    _direction: pg_sys::ScanDirection::Type,
    slot: *mut pg_sys::TupleTableSlot,
) -> bool {
    let lsm_scan = scan as *mut LsmScanDesc;

    pg_sys::ExecClearTuple(slot);

    if (*lsm_scan).position >= (*lsm_scan).keys.len() {
        return false;
    }

    let pos = (*lsm_scan).position;
    let keys = &(*lsm_scan).keys;
    let (ref key_bytes, tid_id) = keys[pos];
    (*lsm_scan).position = pos + 1;

    let storage = TableStorage::global();
    let val_bytes = match storage.get(&(*lsm_scan).table_name, key_bytes) {
        Ok(Some(v)) => v,
        _ => return false,
    };

    let tupdesc = (*slot).tts_tupleDescriptor;
    let natts = (*tupdesc).natts as usize;
    let tts_values = (*slot).tts_values;
    let tts_isnull = (*slot).tts_isnull;

    if natts >= 1 {
        let key_str = String::from_utf8_lossy(key_bytes);
        if let Some(d) = key_str.as_ref().into_datum() {
            *tts_values.add(0) = d;
            *tts_isnull.add(0) = false;
        } else {
            *tts_isnull.add(0) = true;
        }
    }

    deserialize_row(&val_bytes, tts_values, tts_isnull, tupdesc, 1);

    pg_sys::ExecStoreVirtualTuple(slot);

    let (block, offset) = tid_to_block_offset(tid_id);
    pg_sys::ItemPointerSet(&mut (*slot).tts_tid, block, offset);

    true
}

unsafe extern "C" fn lsm_scan_end(scan: pg_sys::TableScanDesc) {
    let lsm_scan = scan as *mut LsmScanDesc;

    std::ptr::drop_in_place(&mut (*lsm_scan).table_name);
    std::ptr::drop_in_place(&mut (*lsm_scan).keys);

    pg_sys::pfree(scan as *mut std::ffi::c_void);
}

unsafe extern "C" fn lsm_scan_rescan(
    scan: pg_sys::TableScanDesc,
    _key: *mut pg_sys::ScanKeyData,
    _set_params: bool,
    _allow_strat: bool,
    _allow_sync: bool,
    _allow_pagemode: bool,
) {
    let lsm_scan = scan as *mut LsmScanDesc;
    (*lsm_scan).position = 0;
}

// ─────────────────────────────────────────────────────────────────────
// TAM Callbacks: Tuple DML
// ─────────────────────────────────────────────────────────────────────

unsafe extern "C" fn lsm_tuple_insert(
    rel: pg_sys::Relation,
    slot: *mut pg_sys::TupleTableSlot,
    _cid: pg_sys::CommandId,
    _options: c_int,
    _bistate: *mut pg_sys::BulkInsertStateData,
) {
    let table_name = relation_name(rel);
    let storage = TableStorage::global();

    pg_sys::slot_getallattrs(slot);

    let tupdesc = (*slot).tts_tupleDescriptor;
    let natts = (*tupdesc).natts as usize;
    let tts_values = (*slot).tts_values;
    let tts_isnull = (*slot).tts_isnull;

    let key = if natts >= 1 && !*tts_isnull.add(0) {
        let datum = *tts_values.add(0);
        let text_ptr = pg_sys::text_to_cstring(datum.cast_mut_ptr());
        let s = CStr::from_ptr(text_ptr).to_string_lossy().into_owned();
        pg_sys::pfree(text_ptr as *mut _);
        s
    } else {
        uuid::Uuid::new_v4().to_string()
    };

    let value_bytes = serialize_row(tts_values, tts_isnull, tupdesc, 1);

    txn::ensure_xact_callback();
    match storage.insert_with_tid(&table_name, key.as_bytes(), &value_bytes) {
        Ok(tid_id) => {
            txn::record_insert(&table_name, tid_id);
            let (block, offset) = tid_to_block_offset(tid_id);
            pg_sys::ItemPointerSet(&mut (*slot).tts_tid, block, offset);
        }
        Err(e) => {
            pgrx::warning!("LSM insert error: {}", e);
            pg_sys::ItemPointerSet(&mut (*slot).tts_tid, 0, 1);
        }
    }
}

/// Delete a tuple by TID.
unsafe extern "C" fn lsm_tuple_delete(
    rel: pg_sys::Relation,
    tid: pg_sys::ItemPointer,
    _cid: pg_sys::CommandId,
    _snapshot: pg_sys::Snapshot,
    _crosscheck: pg_sys::Snapshot,
    _wait: bool,
    _tmfd: *mut pg_sys::TM_FailureData,
    _changing_part: bool,
) -> pg_sys::TM_Result::Type {
    let table_name = relation_name(rel);
    let storage = TableStorage::global();

    let block = pg_sys::ItemPointerGetBlockNumberNoCheck(tid);
    let offset = pg_sys::ItemPointerGetOffsetNumberNoCheck(tid);
    let tid_id = block_offset_to_tid(block, offset);

    txn::ensure_xact_callback();
    // Save old data before deleting so we can undo on rollback
    let old_kv = storage.fetch_by_tid(&table_name, tid_id).unwrap_or(None);

    match storage.delete_by_tid(&table_name, tid_id) {
        Ok(Some(key_bytes)) => {
            if let Some((_, old_val)) = old_kv {
                txn::record_delete(&table_name, &key_bytes, &old_val, tid_id);
            }
            pg_sys::TM_Result::TM_Ok
        }
        Ok(None) => {
            pgrx::warning!("LSM DELETE: TID ({}, {}) not found in mapping", block, offset);
            pg_sys::TM_Result::TM_Ok
        }
        Err(e) => {
            pgrx::warning!("LSM DELETE error: {}", e);
            pg_sys::TM_Result::TM_Ok
        }
    }
}

/// Update a tuple (delete old + insert new).
unsafe extern "C" fn lsm_tuple_update(
    rel: pg_sys::Relation,
    otid: pg_sys::ItemPointer,
    slot: *mut pg_sys::TupleTableSlot,
    _cid: pg_sys::CommandId,
    _snapshot: pg_sys::Snapshot,
    _crosscheck: pg_sys::Snapshot,
    _wait: bool,
    _tmfd: *mut pg_sys::TM_FailureData,
    _lockmode: *mut pg_sys::LockTupleMode::Type,
    _update_indexes: *mut pg_sys::TU_UpdateIndexes::Type,
) -> pg_sys::TM_Result::Type {
    let table_name = relation_name(rel);
    let storage = TableStorage::global();

    txn::ensure_xact_callback();

    let old_block = pg_sys::ItemPointerGetBlockNumberNoCheck(otid);
    let old_offset = pg_sys::ItemPointerGetOffsetNumberNoCheck(otid);
    let old_tid_id = block_offset_to_tid(old_block, old_offset);

    // Save old row for undo before deleting
    let old_kv = storage.fetch_by_tid(&table_name, old_tid_id).unwrap_or(None);

    match storage.delete_by_tid(&table_name, old_tid_id) {
        Ok(Some(key_bytes)) => {
            if let Some((_, old_val)) = old_kv {
                txn::record_delete(&table_name, &key_bytes, &old_val, old_tid_id);
            }
        }
        Ok(None) => {}
        Err(e) => {
            pgrx::warning!("LSM UPDATE delete-phase error: {}", e);
        }
    }

    pg_sys::slot_getallattrs(slot);
    let tupdesc = (*slot).tts_tupleDescriptor;
    let natts = (*tupdesc).natts as usize;
    let tts_values = (*slot).tts_values;
    let tts_isnull = (*slot).tts_isnull;

    let key = if natts >= 1 && !*tts_isnull.add(0) {
        let datum = *tts_values.add(0);
        let text_ptr = pg_sys::text_to_cstring(datum.cast_mut_ptr());
        let s = CStr::from_ptr(text_ptr).to_string_lossy().into_owned();
        pg_sys::pfree(text_ptr as *mut _);
        s
    } else {
        uuid::Uuid::new_v4().to_string()
    };

    let value_bytes = serialize_row(tts_values, tts_isnull, tupdesc, 1);

    match storage.insert_with_tid(&table_name, key.as_bytes(), &value_bytes) {
        Ok(tid_id) => {
            txn::record_insert(&table_name, tid_id);
            let (block, offset) = tid_to_block_offset(tid_id);
            pg_sys::ItemPointerSet(&mut (*slot).tts_tid, block, offset);
        }
        Err(e) => {
            pgrx::warning!("LSM UPDATE insert-phase error: {}", e);
        }
    }

    if !_update_indexes.is_null() {
        *_update_indexes = pg_sys::TU_UpdateIndexes::TU_All;
    }

    pg_sys::TM_Result::TM_Ok
}

/// Lock a tuple — no-op for LSM (no row-level locking).
unsafe extern "C" fn lsm_tuple_lock(
    _rel: pg_sys::Relation,
    _tid: pg_sys::ItemPointer,
    _snapshot: pg_sys::Snapshot,
    _slot: *mut pg_sys::TupleTableSlot,
    _cid: pg_sys::CommandId,
    _mode: pg_sys::LockTupleMode::Type,
    _wait_policy: pg_sys::LockWaitPolicy::Type,
    _flags: pg_sys::uint8,
    _tmfd: *mut pg_sys::TM_FailureData,
) -> pg_sys::TM_Result::Type {
    pg_sys::TM_Result::TM_Ok
}

// ─────────────────────────────────────────────────────────────────────
// TAM Callbacks: Relation / DDL
// ─────────────────────────────────────────────────────────────────────

unsafe extern "C" fn lsm_relation_set_new_filelocator(
    rel: pg_sys::Relation,
    _newrlocator: *const pg_sys::RelFileLocator,
    _persistence: std::os::raw::c_char,
    freeze_xid: *mut pg_sys::TransactionId,
    min_multi: *mut pg_sys::MultiXactId,
) {
    if !freeze_xid.is_null() {
        *freeze_xid = pg_sys::InvalidTransactionId;
    }
    if !min_multi.is_null() {
        *min_multi = 0;
    }

    let table_name = relation_name(rel);
    let storage = TableStorage::global();
    if let Err(e) = storage.get_or_create(&table_name) {
        pgrx::warning!("LSM relation_set_new_filelocator error: {}", e);
    }
}

/// Truncate — clear all data and reset TID mapping.
unsafe extern "C" fn lsm_relation_nontransactional_truncate(
    rel: pg_sys::Relation,
) {
    let table_name = relation_name(rel);
    let storage = TableStorage::global();
    if let Err(e) = storage.truncate(&table_name) {
        pgrx::warning!("LSM TRUNCATE error: {}", e);
    }
}

unsafe extern "C" fn lsm_relation_size(
    _rel: pg_sys::Relation,
    _fork_number: pg_sys::ForkNumber::Type,
) -> pg_sys::uint64 {
    0
}

unsafe extern "C" fn lsm_relation_estimate_size(
    _rel: pg_sys::Relation,
    _attr_widths: *mut pg_sys::int32,
    pages: *mut pg_sys::BlockNumber,
    tuples: *mut f64,
    allvisfrac: *mut f64,
) {
    *pages = 1;
    *tuples = 100.0;
    *allvisfrac = 1.0;
}

// ─────────────────────────────────────────────────────────────────────
// TAM Callbacks: Index Fetch (for index scan → heap fetch path)
// ─────────────────────────────────────────────────────────────────────

unsafe extern "C" fn lsm_index_fetch_begin(
    rel: pg_sys::Relation,
) -> *mut pg_sys::IndexFetchTableData {
    let table_name = relation_name(rel);
    let fetch = pg_sys::palloc0(std::mem::size_of::<LsmIndexFetchData>())
        as *mut LsmIndexFetchData;

    (*fetch).base.rel = rel;
    std::ptr::write(&mut (*fetch).table_name, table_name);

    fetch as *mut pg_sys::IndexFetchTableData
}

unsafe extern "C" fn lsm_index_fetch_reset(_data: *mut pg_sys::IndexFetchTableData) {}

unsafe extern "C" fn lsm_index_fetch_end(data: *mut pg_sys::IndexFetchTableData) {
    let fetch = data as *mut LsmIndexFetchData;
    std::ptr::drop_in_place(&mut (*fetch).table_name);
    pg_sys::pfree(data as *mut std::ffi::c_void);
}

/// Fetch a single tuple by TID for index scan results.
///
/// This is the critical bridge between the HNSW index scan and the heap:
/// the index returns a TID, and this function fetches the actual row data.
unsafe extern "C" fn lsm_index_fetch_tuple(
    data: *mut pg_sys::IndexFetchTableData,
    tid: pg_sys::ItemPointer,
    _snapshot: pg_sys::Snapshot,
    slot: *mut pg_sys::TupleTableSlot,
    call_again: *mut bool,
    _all_dead: *mut bool,
) -> bool {
    if !call_again.is_null() {
        *call_again = false;
    }

    let fetch = data as *mut LsmIndexFetchData;
    let table_name = &(*fetch).table_name;
    let block = pg_sys::ItemPointerGetBlockNumberNoCheck(tid);
    let offset = pg_sys::ItemPointerGetOffsetNumberNoCheck(tid);
    let tid_id = block_offset_to_tid(block, offset);

    let storage = TableStorage::global();
    let result = match storage.fetch_by_tid(table_name, tid_id) {
        Ok(Some((key, value))) => Some((key, value)),
        Ok(None) => None,
        Err(e) => {
            pgrx::warning!("LSM index_fetch_tuple error: {}", e);
            None
        }
    };

    let (key_bytes, val_bytes) = match result {
        Some(kv) => kv,
        None => return false,
    };

    pg_sys::ExecClearTuple(slot);

    let tupdesc = (*slot).tts_tupleDescriptor;
    let natts = (*tupdesc).natts as usize;
    let tts_values = (*slot).tts_values;
    let tts_isnull = (*slot).tts_isnull;

    if natts >= 1 {
        let key_str = String::from_utf8_lossy(&key_bytes);
        if let Some(d) = key_str.as_ref().into_datum() {
            *tts_values.add(0) = d;
            *tts_isnull.add(0) = false;
        } else {
            *tts_isnull.add(0) = true;
        }
    }

    deserialize_row(&val_bytes, tts_values, tts_isnull, tupdesc, 1);

    pg_sys::ExecStoreVirtualTuple(slot);
    (*slot).tts_tid = *tid;

    true
}

// ─────────────────────────────────────────────────────────────────────
// TAM Callbacks: Stubs for required but unimplemented features
// ─────────────────────────────────────────────────────────────────────

unsafe extern "C" fn lsm_scan_set_tidrange(
    _scan: pg_sys::TableScanDesc,
    _mintid: pg_sys::ItemPointer,
    _maxtid: pg_sys::ItemPointer,
) {}

unsafe extern "C" fn lsm_scan_getnextslot_tidrange(
    _scan: pg_sys::TableScanDesc,
    _direction: pg_sys::ScanDirection::Type,
    _slot: *mut pg_sys::TupleTableSlot,
) -> bool {
    false
}

unsafe extern "C" fn lsm_parallelscan_estimate(_rel: pg_sys::Relation) -> pg_sys::Size {
    0
}

unsafe extern "C" fn lsm_parallelscan_initialize(
    _rel: pg_sys::Relation,
    _pscan: pg_sys::ParallelTableScanDesc,
) -> pg_sys::Size {
    0
}

unsafe extern "C" fn lsm_parallelscan_reinitialize(
    _rel: pg_sys::Relation,
    _pscan: pg_sys::ParallelTableScanDesc,
) {}

unsafe extern "C" fn lsm_tuple_fetch_row_version(
    rel: pg_sys::Relation,
    tid: pg_sys::ItemPointer,
    _snapshot: pg_sys::Snapshot,
    slot: *mut pg_sys::TupleTableSlot,
) -> bool {
    let table_name = relation_name(rel);
    let block = pg_sys::ItemPointerGetBlockNumberNoCheck(tid);
    let offset = pg_sys::ItemPointerGetOffsetNumberNoCheck(tid);
    let tid_id = block_offset_to_tid(block, offset);

    let storage = TableStorage::global();
    let result = match storage.fetch_by_tid(&table_name, tid_id) {
        Ok(Some((key, value))) => Some((key, value)),
        _ => None,
    };

    let (key_bytes, val_bytes) = match result {
        Some(kv) => kv,
        None => return false,
    };

    pg_sys::ExecClearTuple(slot);

    let tupdesc = (*slot).tts_tupleDescriptor;
    let natts = (*tupdesc).natts as usize;
    let tts_values = (*slot).tts_values;
    let tts_isnull = (*slot).tts_isnull;

    if natts >= 1 {
        let key_str = String::from_utf8_lossy(&key_bytes);
        if let Some(d) = key_str.as_ref().into_datum() {
            *tts_values.add(0) = d;
            *tts_isnull.add(0) = false;
        } else {
            *tts_isnull.add(0) = true;
        }
    }

    deserialize_row(&val_bytes, tts_values, tts_isnull, tupdesc, 1);

    pg_sys::ExecStoreVirtualTuple(slot);
    (*slot).tts_tid = *tid;

    true
}

unsafe extern "C" fn lsm_tuple_tid_valid(
    _scan: pg_sys::TableScanDesc,
    _tid: pg_sys::ItemPointer,
) -> bool {
    false
}

unsafe extern "C" fn lsm_tuple_get_latest_tid(
    _scan: pg_sys::TableScanDesc,
    _tid: pg_sys::ItemPointer,
) {}

unsafe extern "C" fn lsm_tuple_satisfies_snapshot(
    _rel: pg_sys::Relation,
    _slot: *mut pg_sys::TupleTableSlot,
    _snapshot: pg_sys::Snapshot,
) -> bool {
    true
}

unsafe extern "C" fn lsm_index_delete_tuples(
    _rel: pg_sys::Relation,
    _delstate: *mut pg_sys::TM_IndexDeleteOp,
) -> pg_sys::TransactionId {
    pg_sys::InvalidTransactionId
}

unsafe extern "C" fn lsm_tuple_insert_speculative(
    _rel: pg_sys::Relation,
    _slot: *mut pg_sys::TupleTableSlot,
    _cid: pg_sys::CommandId,
    _options: c_int,
    _bistate: *mut pg_sys::BulkInsertStateData,
    _spec_token: pg_sys::uint32,
) {}

unsafe extern "C" fn lsm_tuple_complete_speculative(
    _rel: pg_sys::Relation,
    _slot: *mut pg_sys::TupleTableSlot,
    _spec_token: pg_sys::uint32,
    _succeeded: bool,
) {}

unsafe extern "C" fn lsm_multi_insert(
    rel: pg_sys::Relation,
    slots: *mut *mut pg_sys::TupleTableSlot,
    nslots: c_int,
    cid: pg_sys::CommandId,
    options: c_int,
    bistate: *mut pg_sys::BulkInsertStateData,
) {
    for i in 0..nslots {
        lsm_tuple_insert(rel, *slots.add(i as usize), cid, options, bistate);
    }
}

unsafe extern "C" fn lsm_finish_bulk_insert(
    _rel: pg_sys::Relation,
    _options: c_int,
) {}

unsafe extern "C" fn lsm_relation_copy_data(
    _rel: pg_sys::Relation,
    _newrlocator: *const pg_sys::RelFileLocator,
) {}

unsafe extern "C" fn lsm_relation_copy_for_cluster(
    _old_table: pg_sys::Relation,
    _new_table: pg_sys::Relation,
    _old_index: pg_sys::Relation,
    _use_sort: bool,
    _oldest_xmin: pg_sys::TransactionId,
    _xid_cutoff: *mut pg_sys::TransactionId,
    _multi_cutoff: *mut pg_sys::MultiXactId,
    _num_tuples: *mut f64,
    _tups_vacuumed: *mut f64,
    _tups_recently_dead: *mut f64,
) {}

unsafe extern "C" fn lsm_relation_vacuum(
    _rel: pg_sys::Relation,
    _params: *mut pg_sys::VacuumParams,
    _bstrategy: pg_sys::BufferAccessStrategy,
) {}

unsafe extern "C" fn lsm_scan_analyze_next_block(
    _scan: pg_sys::TableScanDesc,
    _stream: *mut pg_sys::ReadStream,
) -> bool {
    false
}

unsafe extern "C" fn lsm_scan_analyze_next_tuple(
    _scan: pg_sys::TableScanDesc,
    _oldest_xmin: pg_sys::TransactionId,
    _liverows: *mut f64,
    _deadrows: *mut f64,
    _slot: *mut pg_sys::TupleTableSlot,
) -> bool {
    false
}

unsafe extern "C" fn lsm_relation_needs_toast_table(_rel: pg_sys::Relation) -> bool {
    false
}

unsafe extern "C" fn lsm_relation_toast_am(_rel: pg_sys::Relation) -> pg_sys::Oid {
    pg_sys::InvalidOid
}

// ─────────────────────────────────────────────────────────────────────
// TAM Callback: index_build_range_scan
//
// Called by CREATE INDEX to iterate all heap tuples and invoke the
// index build callback for each one. This is what makes
// CREATE INDEX ... USING lsm_hnsw work on an lsm_s3 table.
// ─────────────────────────────────────────────────────────────────────

unsafe extern "C" fn lsm_index_build_range_scan(
    heap_rel: pg_sys::Relation,
    index_rel: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
    _allow_sync: bool,
    _anyvisible: bool,
    _progress: bool,
    _start_blockno: pg_sys::BlockNumber,
    _numblocks: pg_sys::BlockNumber,
    callback: pg_sys::IndexBuildCallback,
    callback_state: *mut std::ffi::c_void,
    _scan: pg_sys::TableScanDesc,
) -> f64 {
    let table_name = relation_name(heap_rel);
    let storage = TableStorage::global();

    let rows = match storage.scan_all_with_tids(&table_name) {
        Ok(r) => r,
        Err(e) => {
            pgrx::warning!("LSM index_build_range_scan error: {}", e);
            return 0.0;
        }
    };

    let cb = match callback {
        Some(f) => f,
        None => return rows.len() as f64,
    };

    // We need to deform each row into index column values.
    // Create a temporary virtual tuple slot for this purpose.
    let tupdesc = (*heap_rel).rd_att;
    let slot = pg_sys::MakeSingleTupleTableSlot(tupdesc, &pg_sys::TTSOpsVirtual);

    let natts = (*tupdesc).natts as usize;
    let num_index_attrs = (*index_info).ii_NumIndexAttrs as usize;

    // Allocate arrays for the indexed column values
    let index_values = pg_sys::palloc0(
        (num_index_attrs * std::mem::size_of::<pg_sys::Datum>()) as pg_sys::Size,
    ) as *mut pg_sys::Datum;
    let index_isnull = pg_sys::palloc0(
        (num_index_attrs * std::mem::size_of::<bool>()) as pg_sys::Size,
    ) as *mut bool;

    let mut count = 0.0_f64;

    for (key_bytes, val_bytes, tid_offset) in &rows {
        pg_sys::ExecClearTuple(slot);

        let tts_values = (*slot).tts_values;
        let tts_isnull = (*slot).tts_isnull;

        if natts >= 1 {
            let key_str = String::from_utf8_lossy(key_bytes);
            if let Some(d) = key_str.as_ref().into_datum() {
                *tts_values.add(0) = d;
                *tts_isnull.add(0) = false;
            } else {
                *tts_isnull.add(0) = true;
            }
        }

        deserialize_row(val_bytes, tts_values, tts_isnull, tupdesc, 1);

        pg_sys::ExecStoreVirtualTuple(slot);

        let mut tid = pg_sys::ItemPointerData::default();
        let (blk, off) = tid_to_block_offset(*tid_offset);
        pg_sys::ItemPointerSet(&mut tid, blk, off);
        (*slot).tts_tid = tid;

        // Extract the indexed column values using FormIndexDatum
        pg_sys::FormIndexDatum(index_info, slot, std::ptr::null_mut(), index_values, index_isnull);

        // Invoke the callback
        cb(index_rel, &mut tid, index_values, index_isnull, true, callback_state);
        count += 1.0;
    }

    pg_sys::ExecDropSingleTupleTableSlot(slot);
    pg_sys::pfree(index_values as *mut _);
    pg_sys::pfree(index_isnull as *mut _);

    count
}

unsafe extern "C" fn lsm_index_validate_scan(
    _heap_rel: pg_sys::Relation,
    _index_rel: pg_sys::Relation,
    _index_info: *mut pg_sys::IndexInfo,
    _snapshot: pg_sys::Snapshot,
    _state: *mut pg_sys::ValidateIndexState,
) {}

// ─────────────────────────────────────────────────────────────────────
// TableAmRoutine: The complete method table
// ─────────────────────────────────────────────────────────────────────

static LSM_TAM_ROUTINE: pg_sys::TableAmRoutine = pg_sys::TableAmRoutine {
    type_: pg_sys::NodeTag::T_TableAmRoutine,
    slot_callbacks: Some(lsm_slot_callbacks),
    scan_begin: Some(lsm_scan_begin),
    scan_end: Some(lsm_scan_end),
    scan_rescan: Some(lsm_scan_rescan),
    scan_getnextslot: Some(lsm_scan_getnextslot),
    scan_set_tidrange: Some(lsm_scan_set_tidrange),
    scan_getnextslot_tidrange: Some(lsm_scan_getnextslot_tidrange),
    parallelscan_estimate: Some(lsm_parallelscan_estimate),
    parallelscan_initialize: Some(lsm_parallelscan_initialize),
    parallelscan_reinitialize: Some(lsm_parallelscan_reinitialize),
    index_fetch_begin: Some(lsm_index_fetch_begin),
    index_fetch_reset: Some(lsm_index_fetch_reset),
    index_fetch_end: Some(lsm_index_fetch_end),
    index_fetch_tuple: Some(lsm_index_fetch_tuple),
    tuple_fetch_row_version: Some(lsm_tuple_fetch_row_version),
    tuple_tid_valid: Some(lsm_tuple_tid_valid),
    tuple_get_latest_tid: Some(lsm_tuple_get_latest_tid),
    tuple_satisfies_snapshot: Some(lsm_tuple_satisfies_snapshot),
    index_delete_tuples: Some(lsm_index_delete_tuples),
    tuple_insert: Some(lsm_tuple_insert),
    tuple_insert_speculative: Some(lsm_tuple_insert_speculative),
    tuple_complete_speculative: Some(lsm_tuple_complete_speculative),
    multi_insert: Some(lsm_multi_insert),
    tuple_delete: Some(lsm_tuple_delete),
    tuple_update: Some(lsm_tuple_update),
    tuple_lock: Some(lsm_tuple_lock),
    finish_bulk_insert: Some(lsm_finish_bulk_insert),
    relation_set_new_filelocator: Some(lsm_relation_set_new_filelocator),
    relation_nontransactional_truncate: Some(lsm_relation_nontransactional_truncate),
    relation_copy_data: Some(lsm_relation_copy_data),
    relation_copy_for_cluster: Some(lsm_relation_copy_for_cluster),
    relation_vacuum: Some(lsm_relation_vacuum),
    scan_analyze_next_block: Some(lsm_scan_analyze_next_block),
    scan_analyze_next_tuple: Some(lsm_scan_analyze_next_tuple),
    relation_size: Some(lsm_relation_size),
    relation_needs_toast_table: Some(lsm_relation_needs_toast_table),
    relation_toast_am: Some(lsm_relation_toast_am),
    relation_fetch_toast_slice: None,
    relation_estimate_size: Some(lsm_relation_estimate_size),
    scan_bitmap_next_block: None,
    scan_bitmap_next_tuple: None,
    scan_sample_next_block: None,
    scan_sample_next_tuple: None,
    index_build_range_scan: Some(lsm_index_build_range_scan),
    index_validate_scan: Some(lsm_index_validate_scan),
};

// ─────────────────────────────────────────────────────────────────────
// Handler Function (SQL-callable)
// ─────────────────────────────────────────────────────────────────────

pgrx::extension_sql!(
    r#"
    CREATE OR REPLACE FUNCTION lsm_s3_tam_handler_wrapper(internal)
    RETURNS table_am_handler
    AS 'MODULE_PATHNAME', 'lsm_s3_tam_handler_wrapper'
    LANGUAGE C STRICT;

    CREATE ACCESS METHOD lsm_s3 TYPE TABLE HANDLER lsm_s3_tam_handler_wrapper;
    "#,
    name = "lsm_s3_tam_handler_sql"
);

#[no_mangle]
pub extern "C" fn pg_finfo_lsm_s3_tam_handler_wrapper() -> *const pg_sys::Pg_finfo_record {
    const MY_FINFO: pg_sys::Pg_finfo_record = pg_sys::Pg_finfo_record { api_version: 1 };
    &MY_FINFO
}

#[no_mangle]
pub extern "C" fn lsm_s3_tam_handler_wrapper(_fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    pg_sys::Datum::from(&LSM_TAM_ROUTINE as *const pg_sys::TableAmRoutine as usize)
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_create_table_using_lsm() {
        Spi::run("CREATE TABLE test_lsm_tam (id TEXT, val TEXT) USING lsm_s3;").expect("Failed to create table");
        let count = Spi::get_one::<i64>("SELECT COUNT(*) FROM test_lsm_tam").expect("Failed to count rows");
        assert_eq!(count, Some(0));
    }

    #[pg_test]
    fn test_insert_select_lsm_tam() {
        Spi::run("CREATE TABLE test_lsm_insert (id TEXT, val TEXT) USING lsm_s3;").unwrap();
        Spi::run("INSERT INTO test_lsm_insert VALUES ('key1', 'value1');").unwrap();

        let val = Spi::get_one::<String>("SELECT val FROM test_lsm_insert WHERE id = 'key1'").unwrap();
        assert_eq!(val, Some("value1".to_string()));
    }

    #[pg_test]
    fn test_multiple_insert_lsm_tam() {
        Spi::run("CREATE TABLE test_lsm_multi (id TEXT, data TEXT) USING lsm_s3;").unwrap();
        Spi::run("INSERT INTO test_lsm_multi VALUES ('a', '1'), ('b', '2'), ('c', '3');").unwrap();

        let count = Spi::get_one::<i64>("SELECT COUNT(*) FROM test_lsm_multi").unwrap();
        assert_eq!(count, Some(3));

        let val = Spi::get_one::<String>("SELECT data FROM test_lsm_multi WHERE id = 'b'").unwrap();
        assert_eq!(val, Some("2".to_string()));
    }

    #[pg_test]
    fn test_delete_lsm_tam() {
        Spi::run("CREATE TABLE test_lsm_del (id TEXT, val TEXT) USING lsm_s3;").unwrap();
        Spi::run("INSERT INTO test_lsm_del VALUES ('k1', 'v1'), ('k2', 'v2');").unwrap();
        Spi::run("DELETE FROM test_lsm_del WHERE id = 'k1';").unwrap();

        let count = Spi::get_one::<i64>("SELECT COUNT(*) FROM test_lsm_del").unwrap();
        assert_eq!(count, Some(1));

        let val = Spi::get_one::<String>("SELECT val FROM test_lsm_del WHERE id = 'k2'").unwrap();
        assert_eq!(val, Some("v2".to_string()));
    }

    #[pg_test]
    fn test_update_lsm_tam() {
        Spi::run("CREATE TABLE test_lsm_upd (id TEXT, val TEXT) USING lsm_s3;").unwrap();
        Spi::run("INSERT INTO test_lsm_upd VALUES ('k1', 'old');").unwrap();
        Spi::run("UPDATE test_lsm_upd SET val = 'new' WHERE id = 'k1';").unwrap();

        let val = Spi::get_one::<String>("SELECT val FROM test_lsm_upd WHERE id = 'k1'").unwrap();
        assert_eq!(val, Some("new".to_string()));
    }

    #[pg_test]
    fn test_truncate_lsm_tam() {
        Spi::run("CREATE TABLE test_lsm_trunc (id TEXT, val TEXT) USING lsm_s3;").unwrap();
        Spi::run("INSERT INTO test_lsm_trunc VALUES ('a', '1'), ('b', '2');").unwrap();
        Spi::run("TRUNCATE test_lsm_trunc;").unwrap();

        let count = Spi::get_one::<i64>("SELECT COUNT(*) FROM test_lsm_trunc").unwrap();
        assert_eq!(count, Some(0));
    }
}
