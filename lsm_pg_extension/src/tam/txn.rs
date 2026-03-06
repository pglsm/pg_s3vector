//! Transaction-aware undo log for the LSM TAM.
//!
//! Tracks INSERT/DELETE/UPDATE operations within a PostgreSQL transaction.
//! On PRE_COMMIT: writes a WAL COMMIT marker (so redo knows this XID committed).
//! On COMMIT: clears the undo log (data already in LSM).
//! On ABORT:  replays the undo log in reverse to restore pre-txn state.

use super::storage::TableStorage;
use pgrx::pg_sys;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
enum UndoOp {
    Insert { table: String, tid_id: u64 },
    Delete { table: String, key: Vec<u8>, value: Vec<u8>, tid_id: u64 },
    Update {
        table: String,
        old_key: Vec<u8>,
        old_value: Vec<u8>,
        old_tid_id: u64,
        new_tid_id: u64,
    },
}

thread_local! {
    static UNDO_LOG: RefCell<Vec<UndoOp>> = RefCell::new(Vec::new());
    /// Tracks whether the current transaction wrote any LSM WAL records,
    /// so we only emit a COMMIT marker when there is something to commit.
    static TXN_HAS_LSM_WAL: RefCell<bool> = RefCell::new(false);
}

static CALLBACK_REGISTERED: AtomicBool = AtomicBool::new(false);

/// Ensure the PostgreSQL XactCallback is registered (idempotent).
pub fn ensure_xact_callback() {
    if !CALLBACK_REGISTERED.swap(true, Ordering::SeqCst) {
        unsafe {
            pg_sys::RegisterXactCallback(Some(lsm_xact_callback), std::ptr::null_mut());
        }
    }
}

/// Mark that this transaction has written at least one LSM WAL record.
pub fn mark_txn_has_wal() {
    TXN_HAS_LSM_WAL.with(|f| *f.borrow_mut() = true);
}

/// Record an INSERT so it can be rolled back.
pub fn record_insert(table: &str, tid_id: u64) {
    UNDO_LOG.with(|log| {
        log.borrow_mut().push(UndoOp::Insert {
            table: table.to_string(),
            tid_id,
        });
    });
}

/// Record a DELETE (with original data) so it can be rolled back.
pub fn record_delete(table: &str, key: &[u8], value: &[u8], tid_id: u64) {
    UNDO_LOG.with(|log| {
        log.borrow_mut().push(UndoOp::Delete {
            table: table.to_string(),
            key: key.to_vec(),
            value: value.to_vec(),
            tid_id,
        });
    });
}

/// Record an UPDATE (delete+insert pair) so it can be rolled back atomically.
pub fn record_update(
    table: &str,
    old_key: &[u8],
    old_value: &[u8],
    old_tid_id: u64,
    new_tid_id: u64,
) {
    UNDO_LOG.with(|log| {
        log.borrow_mut().push(UndoOp::Update {
            table: table.to_string(),
            old_key: old_key.to_vec(),
            old_value: old_value.to_vec(),
            old_tid_id,
            new_tid_id,
        });
    });
}

unsafe extern "C" fn lsm_xact_callback(
    event: pg_sys::XactEvent::Type,
    _arg: *mut std::ffi::c_void,
) {
    if event == pg_sys::XactEvent::XACT_EVENT_PRE_COMMIT {
        let has_wal = TXN_HAS_LSM_WAL.with(|f| *f.borrow());
        if has_wal {
            super::wal::wal_log_commit();
        }
        return;
    }

    let is_abort = event == pg_sys::XactEvent::XACT_EVENT_ABORT;

    if is_abort {
        UNDO_LOG.with(|log| {
            let mut ops = log.borrow_mut();
            let storage = TableStorage::global();
            for op in ops.drain(..).rev() {
                match op {
                    UndoOp::Insert { ref table, tid_id } => {
                        if let Err(e) = storage.delete_by_tid(table, tid_id) {
                            eprintln!("lsm_pg: undo INSERT failed for {}: {}", table, e);
                        }
                    }
                    UndoOp::Delete { ref table, ref key, ref value, tid_id } => {
                        if let Err(e) = storage.insert(table, key, value) {
                            eprintln!("lsm_pg: undo DELETE re-insert failed for {}: {}", table, e);
                        }
                        storage.restore_tid(table, tid_id, key);
                    }
                    UndoOp::Update {
                        ref table,
                        ref old_key,
                        ref old_value,
                        old_tid_id,
                        new_tid_id,
                    } => {
                        if let Err(e) = storage.delete_by_tid(table, new_tid_id) {
                            eprintln!("lsm_pg: undo UPDATE delete-new failed for {}: {}", table, e);
                        }
                        if let Err(e) = storage.insert(table, old_key, old_value) {
                            eprintln!("lsm_pg: undo UPDATE re-insert failed for {}: {}", table, e);
                        }
                        storage.restore_tid(table, old_tid_id, old_key);
                    }
                }
            }
        });
    } else {
        UNDO_LOG.with(|log| log.borrow_mut().clear());
    }

    TXN_HAS_LSM_WAL.with(|f| *f.borrow_mut() = false);
}
