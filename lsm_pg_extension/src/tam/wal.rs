//! Custom WAL Resource Manager for LSM-Postgres.
//!
//! Integrates with PostgreSQL's Write-Ahead Log so that LSM mutations
//! survive crashes and propagate via streaming replication.
//!
//! ## Record types
//!
//! | Code   | Type            | Payload                                    |
//! |--------|-----------------|--------------------------------------------|
//! | `0x10` | INSERT          | table_name + key + value                   |
//! | `0x20` | DELETE          | table_name + key                           |
//! | `0x30` | TRUNCATE        | table_name                                 |
//! | `0x40` | COMMIT          | (empty — XID is in the WAL record header)  |
//! | `0x50` | VECTOR_INSERT   | index_name + vector_id + f32[] dims        |
//! | `0x60` | VECTOR_DELETE   | index_name + vector_id                     |
//!
//! ## Redo strategy
//!
//! During crash recovery, operations are **buffered per-XID**.  Only when a
//! matching COMMIT marker is encountered are the buffered operations applied
//! to the LSM stores.  At the end of recovery any remaining (uncommitted)
//! buffers are discarded — this prevents rolled-back data from being
//! re-introduced.

use pgrx::pg_sys;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

const LSM_WAL_INSERT: u8 = 0x10;
const LSM_WAL_DELETE: u8 = 0x20;
const LSM_WAL_TRUNCATE: u8 = 0x30;
const LSM_WAL_COMMIT: u8 = 0x40;
const LSM_WAL_VECTOR_INSERT: u8 = 0x50;
const LSM_WAL_VECTOR_DELETE: u8 = 0x60;

const LSM_RM_ID: pg_sys::RmgrId = pg_sys::RM_EXPERIMENTAL_ID as pg_sys::RmgrId;
const LSM_RM_NAME: &std::ffi::CStr = c"lsm_s3";

const XLR_INFO_MASK: u8 = 0x0F;

// ─────────────────────────────────────────────────────────────────────
// Redo buffer — collects operations per XID during recovery
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
enum BufferedOp {
    Insert {
        table: String,
        key: Vec<u8>,
        value: Vec<u8>,
    },
    Delete {
        table: String,
        key: Vec<u8>,
    },
    Truncate {
        table: String,
    },
    VectorInsert {
        index: String,
        id: String,
        vector: Vec<f32>,
    },
    VectorDelete {
        index: String,
        id: String,
    },
}

static REDO_BUFFER: Mutex<Option<HashMap<pg_sys::TransactionId, Vec<BufferedOp>>>> =
    Mutex::new(None);

static HAD_RECOVERY: AtomicBool = AtomicBool::new(false);

fn init_redo_buffer() {
    let mut buf = REDO_BUFFER.lock().unwrap();
    *buf = Some(HashMap::new());
}

fn push_buffered_op(xid: pg_sys::TransactionId, op: BufferedOp) {
    let mut buf = REDO_BUFFER.lock().unwrap();
    if let Some(ref mut map) = *buf {
        map.entry(xid).or_default().push(op);
    }
}

fn commit_buffered_ops(xid: pg_sys::TransactionId) {
    let ops = {
        let mut buf = REDO_BUFFER.lock().unwrap();
        buf.as_mut().and_then(|map| map.remove(&xid))
    };
    if let Some(ops) = ops {
        apply_ops(ops);
    }
}

fn apply_ops(ops: Vec<BufferedOp>) {
    let storage = super::storage::TableStorage::global();
    for op in ops {
        match op {
            BufferedOp::Insert { ref table, ref key, ref value } => {
                if let Err(e) = storage.insert(table, key, value) {
                    eprintln!("lsm_wal redo: INSERT {}: {}", table, e);
                }
            }
            BufferedOp::Delete { ref table, ref key } => {
                if let Err(e) = storage.delete(table, key) {
                    eprintln!("lsm_wal redo: DELETE {}: {}", table, e);
                }
            }
            BufferedOp::Truncate { ref table } => {
                if let Err(e) = storage.truncate(table) {
                    eprintln!("lsm_wal redo: TRUNCATE {}: {}", table, e);
                }
            }
            BufferedOp::VectorInsert { ref index, ref id, ref vector } => {
                replay_vector_insert(index, id, vector);
            }
            BufferedOp::VectorDelete { ref index, ref id } => {
                replay_vector_delete(index, id);
            }
        }
    }
}

fn replay_vector_insert(index_name: &str, key: &str, vector: &[f32]) {
    use crate::vector::vector_search::replay_index_insert;
    if let Err(e) = replay_index_insert(index_name, key, vector) {
        eprintln!("lsm_wal redo: VECTOR_INSERT {}: {}", index_name, e);
    }
}

fn replay_vector_delete(index_name: &str, key: &str) {
    use crate::vector::vector_search::replay_index_delete;
    if let Err(e) = replay_index_delete(index_name, key) {
        eprintln!("lsm_wal redo: VECTOR_DELETE {}: {}", index_name, e);
    }
}

/// Drain the redo buffer after recovery completes.
/// Discards operations from transactions that never committed.
pub fn drain_recovery_buffer() {
    let mut buf = REDO_BUFFER.lock().unwrap();
    if let Some(map) = buf.take() {
        if !map.is_empty() {
            let total_ops: usize = map.values().map(|v| v.len()).sum();
            eprintln!(
                "lsm_wal: discarding {} ops from {} uncommitted transactions after recovery",
                total_ops,
                map.len()
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// WAL Record Serialization
// ─────────────────────────────────────────────────────────────────────

fn encode_table_key_value(table_name: &str, key: &[u8], value: &[u8]) -> Vec<u8> {
    let tname = table_name.as_bytes();
    let mut buf = Vec::with_capacity(2 + tname.len() + 4 + key.len() + 4 + value.len());
    buf.extend_from_slice(&(tname.len() as u16).to_be_bytes());
    buf.extend_from_slice(tname);
    buf.extend_from_slice(&(key.len() as u32).to_be_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&(value.len() as u32).to_be_bytes());
    buf.extend_from_slice(value);
    buf
}

fn encode_table_key(table_name: &str, key: &[u8]) -> Vec<u8> {
    let tname = table_name.as_bytes();
    let mut buf = Vec::with_capacity(2 + tname.len() + 4 + key.len());
    buf.extend_from_slice(&(tname.len() as u16).to_be_bytes());
    buf.extend_from_slice(tname);
    buf.extend_from_slice(&(key.len() as u32).to_be_bytes());
    buf.extend_from_slice(key);
    buf
}

fn encode_table_only(table_name: &str) -> Vec<u8> {
    let tname = table_name.as_bytes();
    let mut buf = Vec::with_capacity(2 + tname.len());
    buf.extend_from_slice(&(tname.len() as u16).to_be_bytes());
    buf.extend_from_slice(tname);
    buf
}

fn encode_vector_insert(index_name: &str, key: &str, vector: &[f32]) -> Vec<u8> {
    let iname = index_name.as_bytes();
    let kbytes = key.as_bytes();
    let dim = vector.len();
    let mut buf = Vec::with_capacity(2 + iname.len() + 4 + kbytes.len() + 4 + dim * 4);
    buf.extend_from_slice(&(iname.len() as u16).to_be_bytes());
    buf.extend_from_slice(iname);
    buf.extend_from_slice(&(kbytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(kbytes);
    buf.extend_from_slice(&(dim as u32).to_be_bytes());
    for &f in vector {
        buf.extend_from_slice(&f.to_le_bytes());
    }
    buf
}

fn encode_vector_delete(index_name: &str, key: &str) -> Vec<u8> {
    let iname = index_name.as_bytes();
    let kbytes = key.as_bytes();
    let mut buf = Vec::with_capacity(2 + iname.len() + 4 + kbytes.len());
    buf.extend_from_slice(&(iname.len() as u16).to_be_bytes());
    buf.extend_from_slice(iname);
    buf.extend_from_slice(&(kbytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(kbytes);
    buf
}

// ─────────────────────────────────────────────────────────────────────
// Deserialization helpers
// ─────────────────────────────────────────────────────────────────────

struct RecordReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> RecordReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_u16(&mut self) -> Option<u16> {
        if self.pos + 2 > self.data.len() {
            return None;
        }
        let val = u16::from_be_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Some(val)
    }

    fn read_u32(&mut self) -> Option<u32> {
        if self.pos + 4 > self.data.len() {
            return None;
        }
        let val = u32::from_be_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Some(val)
    }

    fn read_bytes(&mut self, len: usize) -> Option<&'a [u8]> {
        if self.pos + len > self.data.len() {
            return None;
        }
        let slice = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Some(slice)
    }

    fn read_string(&mut self) -> Option<&'a str> {
        let len = self.read_u16()? as usize;
        let bytes = self.read_bytes(len)?;
        std::str::from_utf8(bytes).ok()
    }

    fn read_blob(&mut self) -> Option<&'a [u8]> {
        let len = self.read_u32()? as usize;
        self.read_bytes(len)
    }

    fn read_string_blob(&mut self) -> Option<&'a str> {
        let bytes = self.read_blob()?;
        std::str::from_utf8(bytes).ok()
    }

    fn read_f32_vec(&mut self) -> Option<Vec<f32>> {
        let dim = self.read_u32()? as usize;
        let byte_len = dim * 4;
        let bytes = self.read_bytes(byte_len)?;
        let mut vec = Vec::with_capacity(dim);
        for i in 0..dim {
            let off = i * 4;
            let f = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            vec.push(f);
        }
        Some(vec)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Public API: write WAL records from TAM / SQL callbacks
// ─────────────────────────────────────────────────────────────────────

/// Write a WAL record for an LSM INSERT. No-op during recovery.
pub unsafe fn wal_log_insert(table_name: &str, key: &[u8], value: &[u8]) {
    if pg_sys::RecoveryInProgress() {
        return;
    }
    maybe_drain_recovery_buffer();
    let data = encode_table_key_value(table_name, key, value);
    pg_sys::XLogBeginInsert();
    pg_sys::XLogRegisterData(data.as_ptr() as *mut i8, data.len() as u32);
    pg_sys::XLogInsert(LSM_RM_ID, LSM_WAL_INSERT);
}

/// Write a WAL record for an LSM DELETE. No-op during recovery.
pub unsafe fn wal_log_delete(table_name: &str, key: &[u8]) {
    if pg_sys::RecoveryInProgress() {
        return;
    }
    maybe_drain_recovery_buffer();
    let data = encode_table_key(table_name, key);
    pg_sys::XLogBeginInsert();
    pg_sys::XLogRegisterData(data.as_ptr() as *mut i8, data.len() as u32);
    pg_sys::XLogInsert(LSM_RM_ID, LSM_WAL_DELETE);
}

/// Write a WAL record for a TRUNCATE. No-op during recovery.
pub unsafe fn wal_log_truncate(table_name: &str) {
    if pg_sys::RecoveryInProgress() {
        return;
    }
    maybe_drain_recovery_buffer();
    let data = encode_table_only(table_name);
    pg_sys::XLogBeginInsert();
    pg_sys::XLogRegisterData(data.as_ptr() as *mut i8, data.len() as u32);
    pg_sys::XLogInsert(LSM_RM_ID, LSM_WAL_TRUNCATE);
}

/// Write a WAL COMMIT marker for the current transaction.
/// Called from the transaction callback at PRE_COMMIT.
pub unsafe fn wal_log_commit() {
    if pg_sys::RecoveryInProgress() {
        return;
    }
    pg_sys::XLogBeginInsert();
    pg_sys::XLogSetRecordFlags(pg_sys::XLOG_MARK_UNIMPORTANT as u8);
    pg_sys::XLogRegisterData(std::ptr::null_mut(), 0);
    pg_sys::XLogInsert(LSM_RM_ID, LSM_WAL_COMMIT);
}

/// Write a WAL record for a vector insert. No-op during recovery.
pub unsafe fn wal_log_vector_insert(index_name: &str, key: &str, vector: &[f32]) {
    if pg_sys::RecoveryInProgress() {
        return;
    }
    maybe_drain_recovery_buffer();
    let data = encode_vector_insert(index_name, key, vector);
    pg_sys::XLogBeginInsert();
    pg_sys::XLogRegisterData(data.as_ptr() as *mut i8, data.len() as u32);
    pg_sys::XLogInsert(LSM_RM_ID, LSM_WAL_VECTOR_INSERT);
}

/// Write a WAL record for a vector delete. No-op during recovery.
pub unsafe fn wal_log_vector_delete(index_name: &str, key: &str) {
    if pg_sys::RecoveryInProgress() {
        return;
    }
    maybe_drain_recovery_buffer();
    let data = encode_vector_delete(index_name, key);
    pg_sys::XLogBeginInsert();
    pg_sys::XLogRegisterData(data.as_ptr() as *mut i8, data.len() as u32);
    pg_sys::XLogInsert(LSM_RM_ID, LSM_WAL_VECTOR_DELETE);
}

/// If we went through recovery and this is the first normal-mode
/// WAL write, drain any leftover uncommitted redo buffers.
fn maybe_drain_recovery_buffer() {
    if HAD_RECOVERY.swap(false, Ordering::Relaxed) {
        drain_recovery_buffer();
    }
}

// ─────────────────────────────────────────────────────────────────────
// Resource Manager Callbacks
// ─────────────────────────────────────────────────────────────────────

/// Redo callback: buffers operations per-XID and applies on COMMIT.
unsafe extern "C" fn lsm_wal_redo(record: *mut pg_sys::XLogReaderState) {
    let decoded = (*record).record;
    if decoded.is_null() {
        return;
    }
    let main_data = (*decoded).main_data;
    let main_data_len = (*decoded).main_data_len as usize;
    let info = (*decoded).header.xl_info & !XLR_INFO_MASK;
    let xid = (*decoded).header.xl_xid;

    let data = if main_data.is_null() || main_data_len == 0 {
        &[] as &[u8]
    } else {
        std::slice::from_raw_parts(main_data as *const u8, main_data_len)
    };

    match info {
        LSM_WAL_INSERT => {
            if let Some(op) = parse_insert_op(data) {
                push_buffered_op(xid, op);
            }
        }
        LSM_WAL_DELETE => {
            if let Some(op) = parse_delete_op(data) {
                push_buffered_op(xid, op);
            }
        }
        LSM_WAL_TRUNCATE => {
            if let Some(op) = parse_truncate_op(data) {
                push_buffered_op(xid, op);
            }
        }
        LSM_WAL_COMMIT => {
            commit_buffered_ops(xid);
        }
        LSM_WAL_VECTOR_INSERT => {
            if let Some(op) = parse_vector_insert_op(data) {
                push_buffered_op(xid, op);
            }
        }
        LSM_WAL_VECTOR_DELETE => {
            if let Some(op) = parse_vector_delete_op(data) {
                push_buffered_op(xid, op);
            }
        }
        _ => {
            eprintln!("lsm_wal: unknown WAL record type 0x{:02x}", info);
        }
    }
}

fn parse_insert_op(data: &[u8]) -> Option<BufferedOp> {
    let mut r = RecordReader::new(data);
    let table = r.read_string()?.to_string();
    let key = r.read_blob()?.to_vec();
    let value = r.read_blob()?.to_vec();
    Some(BufferedOp::Insert { table, key, value })
}

fn parse_delete_op(data: &[u8]) -> Option<BufferedOp> {
    let mut r = RecordReader::new(data);
    let table = r.read_string()?.to_string();
    let key = r.read_blob()?.to_vec();
    Some(BufferedOp::Delete { table, key })
}

fn parse_truncate_op(data: &[u8]) -> Option<BufferedOp> {
    let mut r = RecordReader::new(data);
    let table = r.read_string()?.to_string();
    Some(BufferedOp::Truncate { table })
}

fn parse_vector_insert_op(data: &[u8]) -> Option<BufferedOp> {
    let mut r = RecordReader::new(data);
    let index = r.read_string()?.to_string();
    let id = r.read_string_blob()?.to_string();
    let vector = r.read_f32_vec()?;
    Some(BufferedOp::VectorInsert { index, id, vector })
}

fn parse_vector_delete_op(data: &[u8]) -> Option<BufferedOp> {
    let mut r = RecordReader::new(data);
    let index = r.read_string()?.to_string();
    let id = r.read_string_blob()?.to_string();
    Some(BufferedOp::VectorDelete { index, id })
}

/// Describe a WAL record for pg_waldump.
unsafe extern "C" fn lsm_wal_desc(buf: pg_sys::StringInfo, record: *mut pg_sys::XLogReaderState) {
    let decoded = (*record).record;
    if decoded.is_null() {
        return;
    }
    let main_data = (*decoded).main_data;
    let main_data_len = (*decoded).main_data_len as usize;
    let info = (*decoded).header.xl_info & !XLR_INFO_MASK;

    let desc = if main_data.is_null() || main_data_len == 0 {
        match info {
            LSM_WAL_COMMIT => "COMMIT".to_string(),
            _ => format!("op=0x{:02x} (no data)", info),
        }
    } else {
        let data = std::slice::from_raw_parts(main_data as *const u8, main_data_len);
        let mut r = RecordReader::new(data);
        match info {
            LSM_WAL_INSERT => {
                let table = r.read_string().unwrap_or("<?>");
                let key_len = r.read_u32().unwrap_or(0);
                format!("INSERT table={} key_len={}", table, key_len)
            }
            LSM_WAL_DELETE => {
                let table = r.read_string().unwrap_or("<?>");
                let key_len = r.read_u32().unwrap_or(0);
                format!("DELETE table={} key_len={}", table, key_len)
            }
            LSM_WAL_TRUNCATE => {
                let table = r.read_string().unwrap_or("<?>");
                format!("TRUNCATE table={}", table)
            }
            LSM_WAL_COMMIT => "COMMIT".to_string(),
            LSM_WAL_VECTOR_INSERT => {
                let index = r.read_string().unwrap_or("<?>");
                format!("VECTOR_INSERT index={}", index)
            }
            LSM_WAL_VECTOR_DELETE => {
                let index = r.read_string().unwrap_or("<?>");
                format!("VECTOR_DELETE index={}", index)
            }
            _ => format!("UNKNOWN op=0x{:02x}", info),
        }
    };

    let c_desc = std::ffi::CString::new(desc).unwrap_or_default();
    pg_sys::appendStringInfoString(buf, c_desc.as_ptr());
}

/// Identify a WAL record type by name.
unsafe extern "C" fn lsm_wal_identify(info: pg_sys::uint8) -> *const std::ffi::c_char {
    let masked = info & !XLR_INFO_MASK;
    match masked {
        LSM_WAL_INSERT => c"INSERT".as_ptr(),
        LSM_WAL_DELETE => c"DELETE".as_ptr(),
        LSM_WAL_TRUNCATE => c"TRUNCATE".as_ptr(),
        LSM_WAL_COMMIT => c"COMMIT".as_ptr(),
        LSM_WAL_VECTOR_INSERT => c"VECTOR_INSERT".as_ptr(),
        LSM_WAL_VECTOR_DELETE => c"VECTOR_DELETE".as_ptr(),
        _ => c"UNKNOWN".as_ptr(),
    }
}

/// Called once at startup. Initialises the redo buffer before recovery begins.
unsafe extern "C" fn lsm_wal_startup() {
    init_redo_buffer();
    HAD_RECOVERY.store(true, Ordering::Relaxed);
    let _ = super::storage::TableStorage::global();
}

/// Called during shutdown. Flushes all dirty LSM memtables to S3.
unsafe extern "C" fn lsm_wal_cleanup() {
    drain_recovery_buffer();
    let storage = super::storage::TableStorage::global();
    if let Err(e) = storage.flush_all() {
        eprintln!("lsm_wal: cleanup flush_all failed: {}", e);
    }
}

/// Flush all LSM stores to S3 — called from the CHECKPOINT hook.
pub fn checkpoint_flush() {
    let storage = super::storage::TableStorage::global();
    if let Err(e) = storage.flush_all() {
        eprintln!("lsm_wal: checkpoint flush failed: {}", e);
    }
}

// ─────────────────────────────────────────────────────────────────────
// ProcessUtility hook — intercepts CHECKPOINT for flush
// ─────────────────────────────────────────────────────────────────────

static mut PREV_PROCESS_UTILITY: pg_sys::ProcessUtility_hook_type = None;

unsafe extern "C" fn lsm_process_utility(
    pstmt: *mut pg_sys::PlannedStmt,
    query_string: *const std::ffi::c_char,
    read_only_tree: bool,
    context: pg_sys::ProcessUtilityContext::Type,
    params: pg_sys::ParamListInfo,
    query_env: *mut pg_sys::QueryEnvironment,
    dest: *mut pg_sys::DestReceiver,
    qc: *mut pg_sys::QueryCompletion,
) {
    let is_checkpoint = if !pstmt.is_null() {
        let utility = (*pstmt).utilityStmt;
        !utility.is_null() && (*(utility as *const pg_sys::Node)).type_ == pg_sys::NodeTag::T_CheckPointStmt
    } else {
        false
    };

    if is_checkpoint {
        checkpoint_flush();
    }

    if let Some(prev) = PREV_PROCESS_UTILITY {
        prev(
            pstmt, query_string, read_only_tree, context,
            params, query_env, dest, qc,
        );
    } else {
        pg_sys::standard_ProcessUtility(
            pstmt, query_string, read_only_tree, context,
            params, query_env, dest, qc,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Registration
// ─────────────────────────────────────────────────────────────────────

struct SyncRmgrData(pg_sys::RmgrData);
unsafe impl Sync for SyncRmgrData {}

static LSM_RMGR: SyncRmgrData = SyncRmgrData(pg_sys::RmgrData {
    rm_name: LSM_RM_NAME.as_ptr(),
    rm_redo: Some(lsm_wal_redo),
    rm_desc: Some(lsm_wal_desc),
    rm_identify: Some(lsm_wal_identify),
    rm_startup: Some(lsm_wal_startup),
    rm_cleanup: Some(lsm_wal_cleanup),
    rm_mask: None,
    rm_decode: None,
});

/// Register the custom WAL resource manager and the ProcessUtility hook.
/// Call from `_PG_init`.
pub fn register_rmgr() {
    unsafe {
        pg_sys::RegisterCustomRmgr(LSM_RM_ID, &LSM_RMGR.0);

        PREV_PROCESS_UTILITY = pg_sys::ProcessUtility_hook;
        pg_sys::ProcessUtility_hook = Some(lsm_process_utility);
    }
}
