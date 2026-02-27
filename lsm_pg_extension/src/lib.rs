//! LSM-Postgres: S3-Native Vector Storage Extension for PostgreSQL
//!
//! This extension provides a Table Access Method (TAM) that stores data on S3
//! using an LSM-tree architecture, optimized for high-dimensional vector storage.

use pgrx::prelude::*;
use std::ffi::CStr;

mod tam;
mod vector;
mod fence;

pgrx::pg_module_magic!();

// ─────────────────────────────────────────────────────────────────────
// GUC (Grand Unified Configuration) Variables
// ─────────────────────────────────────────────────────────────────────

/// S3 endpoint URL (or "memory" for in-memory testing).
static LSM_S3_ENDPOINT: pgrx::GucSetting<Option<&'static CStr>> =
    pgrx::GucSetting::<Option<&'static CStr>>::new(Some(c"memory"));

/// S3 bucket name.
static LSM_S3_BUCKET: pgrx::GucSetting<Option<&'static CStr>> =
    pgrx::GucSetting::<Option<&'static CStr>>::new(Some(c"lsm-postgres"));

/// S3 region.
static LSM_S3_REGION: pgrx::GucSetting<Option<&'static CStr>> =
    pgrx::GucSetting::<Option<&'static CStr>>::new(Some(c"us-east-1"));

/// Flush interval in milliseconds.
static LSM_S3_FLUSH_INTERVAL_MS: pgrx::GucSetting<i32> =
    pgrx::GucSetting::<i32>::new(5000);

/// MemTable size limit in MB.
static LSM_S3_MEMTABLE_SIZE_MB: pgrx::GucSetting<i32> =
    pgrx::GucSetting::<i32>::new(64);

/// Block cache size in MB.
static LSM_S3_CACHE_SIZE_MB: pgrx::GucSetting<i32> =
    pgrx::GucSetting::<i32>::new(256);

#[pg_extern]
fn lsm_postgres_version() -> &'static str {
    "0.1.0"
}

/// Initialize GUC variables when the extension loads.
fn init_gucs() {
    pgrx::GucRegistry::define_string_guc(
        "lsm_s3.endpoint",
        "S3 endpoint URL (or 'memory' for in-memory testing)",
        "S3 endpoint URL",
        &LSM_S3_ENDPOINT,
        pgrx::GucContext::Suset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_string_guc(
        "lsm_s3.bucket",
        "S3 bucket name for data storage",
        "S3 bucket name",
        &LSM_S3_BUCKET,
        pgrx::GucContext::Suset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_string_guc(
        "lsm_s3.region",
        "AWS region for S3",
        "AWS region",
        &LSM_S3_REGION,
        pgrx::GucContext::Suset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_int_guc(
        "lsm_s3.flush_interval_ms",
        "MemTable flush interval in milliseconds",
        "Flush interval",
        &LSM_S3_FLUSH_INTERVAL_MS,
        1,
        60000,
        pgrx::GucContext::Suset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_int_guc(
        "lsm_s3.memtable_size_mb",
        "MemTable size limit in MB before rotation",
        "MemTable size",
        &LSM_S3_MEMTABLE_SIZE_MB,
        1,
        4096,
        pgrx::GucContext::Suset,
        pgrx::GucFlags::default(),
    );

    pgrx::GucRegistry::define_int_guc(
        "lsm_s3.cache_size_mb",
        "Block cache size in MB",
        "Cache size",
        &LSM_S3_CACHE_SIZE_MB,
        1,
        65536,
        pgrx::GucContext::Suset,
        pgrx::GucFlags::default(),
    );
}

// ─────────────────────────────────────────────────────────────────────
// SQL Functions for Management & Monitoring
// ─────────────────────────────────────────────────────────────────────

/// Get the current status of the LSM store.
#[pg_extern]
fn lsm_s3_status() -> String {
    let endpoint = LSM_S3_ENDPOINT
        .get()
        .map(|c| c.to_str().unwrap_or("invalid"))
        .unwrap_or("not set");
    let bucket = LSM_S3_BUCKET
        .get()
        .map(|c| c.to_str().unwrap_or("invalid"))
        .unwrap_or("not set");

    format!(
        "LSM-Postgres v{}\nEndpoint: {}\nBucket: {}\nFlush Interval: {}ms\nMemTable Limit: {}MB\nCache Size: {}MB",
        lsm_postgres_version(),
        endpoint,
        bucket,
        LSM_S3_FLUSH_INTERVAL_MS.get(),
        LSM_S3_MEMTABLE_SIZE_MB.get(),
        LSM_S3_CACHE_SIZE_MB.get(),
    )
}

/// Force a flush of all in-memory MemTables to S3.
#[pg_extern]
fn lsm_s3_flush() -> String {
    let storage = tam::storage::TableStorage::global();
    match storage.flush_all() {
        Ok(count) => format!("Flushed {} table(s)", count),
        Err(e) => format!("ERROR: {}", e),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Extension Initialization
// ─────────────────────────────────────────────────────────────────────

/// Called when the shared library is loaded.
#[pg_guard]
pub extern "C" fn _PG_init() {
    init_gucs();
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_version() {
        assert_eq!(crate::lsm_postgres_version(), "0.1.0");
    }

    #[pg_test]
    fn test_status() {
        let status = crate::lsm_s3_status();
        assert!(status.contains("LSM-Postgres v0.1.0"));
    }

    #[pg_test]
    fn test_insert_and_select() {
        let result = Spi::get_one::<String>(
            "SELECT lsm_s3_insert('test_table', 'hello', 'world')"
        ).unwrap().unwrap();
        assert!(result.contains("INSERT"), "Insert should return INSERT message");

        let value = Spi::get_one::<String>(
            "SELECT lsm_s3_select('test_table', 'hello')"
        ).unwrap();
        assert_eq!(value, Some("world".to_string()));
    }

    #[pg_test]
    fn test_select_missing() {
        let value = Spi::get_one::<String>(
            "SELECT lsm_s3_select('empty_table', 'nonexistent')"
        ).unwrap();
        assert_eq!(value, None, "Missing key should return NULL");
    }

    #[pg_test]
    fn test_delete() {
        Spi::run("SELECT lsm_s3_insert('del_table', 'k1', 'v1')").unwrap();

        let before = Spi::get_one::<String>(
            "SELECT lsm_s3_select('del_table', 'k1')"
        ).unwrap();
        assert_eq!(before, Some("v1".to_string()));

        Spi::run("SELECT lsm_s3_delete('del_table', 'k1')").unwrap();

        let after = Spi::get_one::<String>(
            "SELECT lsm_s3_select('del_table', 'k1')"
        ).unwrap();
        assert_eq!(after, None, "Deleted key should return NULL");
    }

    #[pg_test]
    fn test_overwrite() {
        Spi::run("SELECT lsm_s3_insert('ow_table', 'key', 'v1')").unwrap();
        Spi::run("SELECT lsm_s3_insert('ow_table', 'key', 'v2')").unwrap();

        let value = Spi::get_one::<String>(
            "SELECT lsm_s3_select('ow_table', 'key')"
        ).unwrap();
        assert_eq!(value, Some("v2".to_string()), "Should return latest value");
    }

    #[pg_test]
    fn test_scan() {
        Spi::run("SELECT lsm_s3_insert('scan_t', 'alpha', '1')").unwrap();
        Spi::run("SELECT lsm_s3_insert('scan_t', 'beta', '2')").unwrap();
        Spi::run("SELECT lsm_s3_insert('scan_t', 'gamma', '3')").unwrap();

        let count = Spi::get_one::<i64>(
            "SELECT count(*) FROM lsm_s3_scan('scan_t')"
        ).unwrap().unwrap();
        assert_eq!(count, 3, "Scan should return all 3 rows");
    }

    #[pg_test]
    fn test_scan_range() {
        Spi::run("SELECT lsm_s3_insert('range_t', 'a', '1')").unwrap();
        Spi::run("SELECT lsm_s3_insert('range_t', 'b', '2')").unwrap();
        Spi::run("SELECT lsm_s3_insert('range_t', 'c', '3')").unwrap();
        Spi::run("SELECT lsm_s3_insert('range_t', 'd', '4')").unwrap();

        let count = Spi::get_one::<i64>(
            "SELECT count(*) FROM lsm_s3_scan_range('range_t', 'b', 'd')"
        ).unwrap().unwrap();
        assert_eq!(count, 2, "Range scan [b,d) should return b,c");
    }

    #[pg_test]
    fn test_table_isolation() {
        Spi::run("SELECT lsm_s3_insert('iso_t1', 'key', 'table1')").unwrap();
        Spi::run("SELECT lsm_s3_insert('iso_t2', 'key', 'table2')").unwrap();

        let v1 = Spi::get_one::<String>(
            "SELECT lsm_s3_select('iso_t1', 'key')"
        ).unwrap();
        let v2 = Spi::get_one::<String>(
            "SELECT lsm_s3_select('iso_t2', 'key')"
        ).unwrap();

        assert_eq!(v1, Some("table1".to_string()));
        assert_eq!(v2, Some("table2".to_string()));
    }

    #[pg_test]
    fn test_table_stats() {
        Spi::run("SELECT lsm_s3_insert('stats_t', 'k', 'v')").unwrap();

        let stats = Spi::get_one::<String>(
            "SELECT lsm_s3_table_stats('stats_t')"
        ).unwrap().unwrap();
        assert!(stats.contains("MemTable"), "Stats should contain MemTable info");
    }

    #[pg_test]
    fn test_flush_table() {
        Spi::run("SELECT lsm_s3_insert('flush_t', 'k', 'v')").unwrap();

        let result = Spi::get_one::<String>(
            "SELECT lsm_s3_flush_table('flush_t')"
        ).unwrap().unwrap();
        assert!(result.contains("Flush"), "Flush should succeed");

        // Data should persist after flush
        let value = Spi::get_one::<String>(
            "SELECT lsm_s3_select('flush_t', 'k')"
        ).unwrap();
        assert_eq!(value, Some("v".to_string()));
    }
}

#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {
        // pre-test setup goes here
    }

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec![]
    }
}
