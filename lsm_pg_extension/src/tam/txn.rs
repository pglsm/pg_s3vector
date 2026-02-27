//! Transaction-aware undo log for the LSM TAM.
//!
//! Tracks INSERT/DELETE operations within a PostgreSQL transaction.
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
}

thread_local! {
    static UNDO_LOG: RefCell<Vec<UndoOp>> = RefCell::new(Vec::new());
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

unsafe extern "C" fn lsm_xact_callback(
    event: pg_sys::XactEvent::Type,
    _arg: *mut std::ffi::c_void,
) {
    let is_abort = event == pg_sys::XactEvent::XACT_EVENT_ABORT;

    if is_abort {
        UNDO_LOG.with(|log| {
            let mut ops = log.borrow_mut();
            let storage = TableStorage::global();
            for op in ops.drain(..).rev() {
                match op {
                    UndoOp::Insert { ref table, tid_id } => {
                        let _ = storage.delete_by_tid(table, tid_id);
                    }
                    UndoOp::Delete { ref table, ref key, ref value, tid_id } => {
                        let _ = storage.insert(table, key, value);
                        storage.restore_tid(table, tid_id, key);
                    }
                }
            }
        });
    } else {
        UNDO_LOG.with(|log| log.borrow_mut().clear());
    }
}
