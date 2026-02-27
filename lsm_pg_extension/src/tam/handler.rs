//! TAM Handler: The core Table Access Method handler for LSM-S3 storage.
//!
//! This module implements the PostgreSQL Table Access Method interface,
//! routing SQL operations (INSERT, SELECT, UPDATE, DELETE) to the LSM engine.
//!
//! The handler is registered as a C extern function since pgrx doesn't have
//! full TAM abstractions. The SQL-callable functions provide the MVP interface.

use pgrx::prelude::*;
use super::storage::TableStorage;

/// Insert a row into an LSM-S3 backed table.
///
/// Usage: SELECT lsm_s3_insert('table_name', 'key', 'value');
#[pg_extern]
fn lsm_s3_insert(table_name: &str, key: &str, value: &str) -> String {
    let storage = TableStorage::global();

    match storage.insert(table_name, key.as_bytes(), value.as_bytes()) {
        Ok(()) => format!("INSERT 0 1"),
        Err(e) => {
            pgrx::warning!("lsm_s3_insert error: {}", e);
            format!("ERROR: {}", e)
        }
    }
}

/// Select a row from an LSM-S3 backed table by key.
///
/// Usage: SELECT lsm_s3_select('table_name', 'key');
#[pg_extern]
fn lsm_s3_select(table_name: &str, key: &str) -> Option<String> {
    let storage = TableStorage::global();

    match storage.get(table_name, key.as_bytes()) {
        Ok(Some(val)) => Some(String::from_utf8_lossy(&val).to_string()),
        Ok(None) => None,
        Err(e) => {
            pgrx::warning!("lsm_s3_select error: {}", e);
            None
        }
    }
}

/// Delete a row from an LSM-S3 backed table.
///
/// Usage: SELECT lsm_s3_delete('table_name', 'key');
#[pg_extern]
fn lsm_s3_delete(table_name: &str, key: &str) -> String {
    let storage = TableStorage::global();

    match storage.delete(table_name, key.as_bytes()) {
        Ok(()) => format!("DELETE 0 1"),
        Err(e) => {
            pgrx::warning!("lsm_s3_delete error: {}", e);
            format!("ERROR: {}", e)
        }
    }
}

/// Scan all rows in an LSM-S3 backed table.
///
/// Returns rows as a set of (key, value) tuples.
/// Usage: SELECT * FROM lsm_s3_scan('table_name');
#[pg_extern]
fn lsm_s3_scan(
    table_name: &str,
) -> TableIterator<'static, (name!(key, String), name!(value, String))> {
    let storage = TableStorage::global();

    let entries = match storage.scan_all(table_name) {
        Ok(entries) => entries,
        Err(e) => {
            pgrx::warning!("lsm_s3_scan error: {}", e);
            vec![]
        }
    };

    TableIterator::new(entries.into_iter().map(|(k, v)| {
        (
            String::from_utf8_lossy(&k).to_string(),
            String::from_utf8_lossy(&v).to_string(),
        )
    }))
}

/// Scan rows in an LSM-S3 backed table within a key range.
///
/// Usage: SELECT * FROM lsm_s3_scan_range('table_name', 'start_key', 'end_key');
#[pg_extern]
fn lsm_s3_scan_range(
    table_name: &str,
    start_key: &str,
    end_key: &str,
) -> TableIterator<'static, (name!(key, String), name!(value, String))> {
    let storage = TableStorage::global();

    let entries = match storage.scan_range(table_name, start_key.as_bytes(), end_key.as_bytes()) {
        Ok(entries) => entries,
        Err(e) => {
            pgrx::warning!("lsm_s3_scan_range error: {}", e);
            vec![]
        }
    };

    TableIterator::new(entries.into_iter().map(|(k, v)| {
        (
            String::from_utf8_lossy(&k).to_string(),
            String::from_utf8_lossy(&v).to_string(),
        )
    }))
}

/// Get statistics for a table's LSM store.
#[pg_extern]
fn lsm_s3_table_stats(table_name: &str) -> String {
    let storage = TableStorage::global();
    match storage.stats(table_name) {
        Ok(stats) => stats,
        Err(e) => format!("ERROR: {}", e),
    }
}

/// Force a flush of a table's MemTable to S3.
#[pg_extern]
fn lsm_s3_flush_table(table_name: &str) -> String {
    let storage = TableStorage::global();
    match storage.flush(table_name) {
        Ok(()) => "Flush complete".to_string(),
        Err(e) => format!("ERROR: {}", e),
    }
}
