//! SQL-callable functions for vector index management and search.
//!
//! Provides a global registry of HNSW indexes backed by LSM vector storage,
//! with SQL functions for creating indexes, inserting vectors, and KNN search.

use pgrx::prelude::*;
use super::hnsw::{DistanceMetric, HnswConfig, HnswIndex};
use super::vector_storage::LsmVectorStorage;
use super::types::LsmVector;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────
// Global Index Registry
// ─────────────────────────────────────────────────────────────────────

struct IndexRegistry {
    indexes: RwLock<HashMap<String, Arc<HnswIndex<LsmVectorStorage>>>>,
}

impl IndexRegistry {
    fn new() -> Self {
        Self {
            indexes: RwLock::new(HashMap::new()),
        }
    }

    fn global() -> &'static IndexRegistry {
        static REGISTRY: once_cell::sync::Lazy<IndexRegistry> =
            once_cell::sync::Lazy::new(IndexRegistry::new);
        &REGISTRY
    }

    fn get(&self, name: &str) -> Option<Arc<HnswIndex<LsmVectorStorage>>> {
        self.indexes.read().get(name).cloned()
    }

    fn flush_all(&self) -> Result<usize, String> {
        let indexes = self.indexes.read();
        let mut count = 0;
        for (name, index) in indexes.iter() {
            index.storage_ref().flush().map_err(|e| {
                format!("Flush failed for vector index '{}': {}", name, e)
            })?;
            count += 1;
        }
        Ok(count)
    }

    fn create(
        &self,
        name: &str,
        config: HnswConfig,
    ) -> Result<Arc<HnswIndex<LsmVectorStorage>>, String> {
        let mut indexes = self.indexes.write();
        if let Some(existing) = indexes.get(name) {
            return Ok(existing.clone());
        }
        let global_storage = crate::tam::storage::TableStorage::global();
        let path = format!("/indexes/sql_{}/vectors", name);
        let lsm_config = global_storage.build_config_for(&path)?;
        let rt = global_storage.runtime();
        let cache_bytes = crate::vector_cache_bytes();
        let storage = Arc::new(LsmVectorStorage::with_cache(lsm_config, rt, cache_bytes)?);
        let index = Arc::new(HnswIndex::new(config, storage));
        indexes.insert(name.to_string(), index.clone());
        Ok(index)
    }
}

fn parse_metric(metric_str: &str) -> DistanceMetric {
    match metric_str.to_lowercase().as_str() {
        "cosine" | "cos" => DistanceMetric::Cosine,
        "ip" | "inner_product" | "dot" => DistanceMetric::InnerProduct,
        _ => DistanceMetric::L2,
    }
}

// ─────────────────────────────────────────────────────────────────────
// SQL Functions
// ─────────────────────────────────────────────────────────────────────

/// Create a new HNSW vector index.
///
/// Parameters:
/// - `index_name`: Name of the index (used as identifier)
/// - `m`: Max connections per node (default: 16)
/// - `ef_construction`: Construction beam width (default: 200)
/// - `ef_search`: Search beam width (default: 64)
/// - `metric`: Distance metric: 'l2', 'cosine', or 'ip' (default: 'l2')
/// - `max_layers`: Maximum graph layers (default: 16)
///
/// Usage: `SELECT lsm_s3_create_vector_index('my_index', 16, 200, 64, 'cosine', 24);`
#[pg_extern]
fn lsm_s3_create_vector_index(
    index_name: &str,
    m: default!(i32, 16),
    ef_construction: default!(i32, 200),
    ef_search: default!(i32, 64),
    metric: default!(&str, "'l2'"),
    max_layers: default!(i32, 16),
) -> String {
    let m = m.max(2) as usize;
    let max_layers = max_layers.max(1) as usize;
    let config = HnswConfig {
        m,
        m_max_0: m * 2,
        ef_construction: ef_construction.max(1) as usize,
        ef_search: ef_search.max(1) as usize,
        ml: 1.0 / (m as f64).ln(),
        max_layers,
        metric: parse_metric(metric),
    };

    let registry = IndexRegistry::global();
    match registry.create(index_name, config) {
        Ok(_) => format!("CREATED INDEX {}", index_name),
        Err(e) => {
            pgrx::warning!("lsm_s3_create_vector_index error: {}", e);
            format!("ERROR creating index {}: {}", index_name, e)
        }
    }
}

/// Insert a vector into an HNSW index with an associated key.
///
/// The index must already exist (created via `lsm_s3_create_vector_index`).
///
/// Usage: `SELECT lsm_s3_index_insert('my_index', 'doc:42', '[1.0, 2.0, 3.0]'::lsm_vector);`
#[pg_extern]
fn lsm_s3_index_insert(
    index_name: &str,
    key: &str,
    vector: LsmVector,
) -> String {
    let registry = IndexRegistry::global();
    let index = match registry.get(index_name) {
        Some(idx) => idx,
        None => {
            pgrx::warning!("Index '{}' not found. Create it first with lsm_s3_create_vector_index.", index_name);
            return format!("ERROR: index '{}' not found", index_name);
        }
    };

    let id = index.insert_with_key(Some(key.to_string()), vector.data.clone());
    unsafe {
        crate::tam::wal::wal_log_vector_insert(index_name, key, &vector.data);
        crate::tam::txn::mark_txn_has_wal();
    }
    format!("INSERT {} (id={})", index_name, id)
}

/// Search for K nearest neighbors in an HNSW index.
///
/// Returns a table of (key, distance) ordered by distance ascending.
///
/// Usage: `SELECT * FROM lsm_s3_vector_search('my_index', '[1.0, 2.0, 3.0]'::lsm_vector, 10);`
#[pg_extern]
fn lsm_s3_vector_search(
    index_name: &str,
    query: LsmVector,
    k: default!(i32, 10),
) -> TableIterator<
    'static,
    (
        name!(key, String),
        name!(distance, f32),
    ),
> {
    let registry = IndexRegistry::global();
    let k = k.max(1) as usize;

    let results = match registry.get(index_name) {
        Some(index) => index.search_with_keys(&query.data, k),
        None => {
            pgrx::warning!("Index '{}' not found", index_name);
            vec![]
        }
    };

    TableIterator::new(results.into_iter().map(|(key, dist)| (key, dist)))
}

// ─────────────────────────────────────────────────────────────────────
// WAL redo helpers — called from tam::wal during crash recovery
// ─────────────────────────────────────────────────────────────────────

/// Replay a vector insert during WAL redo.
pub fn replay_index_insert(index_name: &str, key: &str, vector: &[f32]) -> Result<(), String> {
    let registry = IndexRegistry::global();
    let index = registry
        .get(index_name)
        .ok_or_else(|| format!("index '{}' not found during redo", index_name))?;
    index.insert_with_key(Some(key.to_string()), vector.to_vec());
    Ok(())
}

/// Replay a vector delete during WAL redo.
pub fn replay_index_delete(index_name: &str, _key: &str) -> Result<(), String> {
    let registry = IndexRegistry::global();
    let _index = registry
        .get(index_name)
        .ok_or_else(|| format!("index '{}' not found during redo", index_name))?;
    // HNSW does not currently support delete-by-key; the vector is
    // effectively orphaned.  Full delete support requires an HNSW
    // tombstone mechanism (future work).
    Ok(())
}

/// Flush all vector index stores to object storage.
///
/// Returns the number of indexes flushed.
pub fn flush_all_vector_indexes() -> Result<usize, String> {
    IndexRegistry::global().flush_all()
}

/// Get information about a vector index including cache stats.
///
/// Usage: `SELECT lsm_s3_vector_index_info('my_index');`
#[pg_extern]
fn lsm_s3_vector_index_info(index_name: &str) -> String {
    let registry = IndexRegistry::global();

    match registry.get(index_name) {
        Some(index) => {
            let (hits, misses) = index.storage_ref().cache_stats();
            let total = hits + misses;
            let hit_rate = if total > 0 {
                (hits as f64 / total as f64) * 100.0
            } else {
                0.0
            };
            format!(
                "Index '{}': {} vectors, cache: {}/{} hits ({:.1}%)",
                index_name,
                index.len(),
                hits,
                total,
                hit_rate,
            )
        }
        None => format!("Index '{}' not found", index_name),
    }
}
