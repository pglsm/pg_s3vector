//! SQL-callable functions for vector index management and search.
//!
//! Provides a global registry of HNSW indexes backed by LSM vector storage,
//! with SQL functions for creating indexes, inserting vectors, and KNN search.

use pgrx::prelude::*;
use super::hnsw::{HnswConfig, HnswIndex};
use super::vector_storage::InMemoryVectorStorage;
use super::types::LsmVector;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────
// Global Index Registry
// ─────────────────────────────────────────────────────────────────────

/// Global registry of HNSW indexes.
///
/// For the MVP, uses InMemoryVectorStorage. Future: LsmVectorStorage.
struct IndexRegistry {
    indexes: RwLock<HashMap<String, Arc<HnswIndex<InMemoryVectorStorage>>>>,
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

    fn get(&self, name: &str) -> Option<Arc<HnswIndex<InMemoryVectorStorage>>> {
        self.indexes.read().get(name).cloned()
    }

    fn create(
        &self,
        name: &str,
        config: HnswConfig,
    ) -> Arc<HnswIndex<InMemoryVectorStorage>> {
        let mut indexes = self.indexes.write();
        if let Some(existing) = indexes.get(name) {
            return existing.clone();
        }
        let storage = Arc::new(InMemoryVectorStorage::new());
        let index = Arc::new(HnswIndex::new(config, storage));
        indexes.insert(name.to_string(), index.clone());
        index
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
///
/// Usage: `SELECT lsm_s3_create_vector_index('my_index', 16, 200, 64);`
#[pg_extern]
fn lsm_s3_create_vector_index(
    index_name: &str,
    m: default!(i32, 16),
    ef_construction: default!(i32, 200),
    ef_search: default!(i32, 64),
) -> String {
    let m = m.max(2) as usize;
    let config = HnswConfig {
        m,
        m_max_0: m * 2,
        ef_construction: ef_construction.max(1) as usize,
        ef_search: ef_search.max(1) as usize,
        ml: 1.0 / (m as f64).ln(),
        max_layers: 16,
        metric: super::hnsw::DistanceMetric::L2,
    };

    let registry = IndexRegistry::global();
    registry.create(index_name, config);
    format!("CREATED INDEX {}", index_name)
}

/// Insert a vector into an HNSW index with an associated key.
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
            // Auto-create with defaults if not exists
            registry.create(index_name, HnswConfig::default())
        }
    };

    let id = index.insert_with_key(Some(key.to_string()), vector.data.clone());
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

/// Get information about a vector index.
///
/// Usage: `SELECT lsm_s3_vector_index_info('my_index');`
#[pg_extern]
fn lsm_s3_vector_index_info(index_name: &str) -> String {
    let registry = IndexRegistry::global();

    match registry.get(index_name) {
        Some(index) => {
            format!(
                "Index '{}': {} vectors",
                index_name,
                index.len()
            )
        }
        None => format!("Index '{}' not found", index_name),
    }
}
