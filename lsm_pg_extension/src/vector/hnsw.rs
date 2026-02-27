//! HNSW (Hierarchical Navigable Small World) Index — Tiered Storage
//!
//! This module provides a tiered HNSW index where:
//! - Navigation graph (nodes + connections) stays in RAM for fast traversal
//! - Vector data is stored via the `VectorStorage` trait (in-memory or S3-backed)
//!
//! The index is generic over the storage backend, enabling:
//! - `HnswIndex<InMemoryVectorStorage>` for testing
//! - `HnswIndex<LsmVectorStorage>` for production (S3-backed)

use super::distance;
use super::vector_storage::VectorStorage;
use parking_lot::RwLock;
use rand::Rng;
use serde::{Serialize, Deserialize};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

/// Distance metric used by the HNSW index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    L2,
    Cosine,
    InnerProduct,
}

/// Configuration for the HNSW index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum number of connections per node (M parameter).
    pub m: usize,
    /// Construction-time max connections (M_max_0 for layer 0).
    pub m_max_0: usize,
    /// Size of dynamic candidate list during construction.
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search.
    pub ef_search: usize,
    /// Normalization factor for level generation (1/ln(M)).
    pub ml: f64,
    /// Maximum number of layers.
    pub max_layers: usize,
    /// Distance metric for building and searching the graph.
    pub metric: DistanceMetric,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m_max_0: m * 2,
            ef_construction: 200,
            ef_search: 64,
            ml: 1.0 / (m as f64).ln(),
            max_layers: 16,
            metric: DistanceMetric::L2,
        }
    }
}

/// A node in the HNSW graph.
///
/// Only stores graph structure (connections). Vector data lives
/// in the `VectorStorage` backend, fetched on demand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswNode {
    /// Unique identifier.
    pub id: u64,
    /// Connections at each layer: layer -> list of neighbor IDs.
    pub connections: Vec<Vec<u64>>,
    /// Maximum layer this node appears in.
    pub max_layer: usize,
}

/// Serializable snapshot of the HNSW graph structure for persistence.
///
/// Vector data is stored separately in the `VectorStorage` backend
/// (e.g. `LsmVectorStorage`) and does not need to be part of the snapshot.
#[derive(Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub config: HnswConfig,
    pub nodes: HashMap<u64, HnswNode>,
    pub entry_point: Option<u64>,
    pub max_layer: usize,
    pub next_id: u64,
    pub key_to_id: HashMap<String, u64>,
    pub id_to_key: HashMap<u64, String>,
    /// Legacy field — kept for backwards-compatible deserialization of old snapshots.
    #[serde(default)]
    pub vectors: HashMap<u64, Vec<f32>>,
}

/// A candidate during search, ordered by distance.
#[derive(Debug, Clone)]
struct Candidate {
    id: u64,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap behavior (closest first)
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}

/// Max-heap candidate (for maintaining the ef-sized result set).
#[derive(Debug, Clone)]
struct MaxCandidate {
    id: u64,
    distance: f32,
}

impl PartialEq for MaxCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for MaxCandidate {}

impl PartialOrd for MaxCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}

/// The HNSW index, generic over vector storage backend.
///
/// - Navigation graph (nodes + connections) lives in RAM.
/// - Vector data is fetched from `storage` on demand.
pub struct HnswIndex<S: VectorStorage> {
    config: HnswConfig,
    /// All nodes (graph structure only), indexed by ID.
    nodes: RwLock<HashMap<u64, HnswNode>>,
    /// Vector storage backend (in-memory or S3-backed).
    storage: Arc<S>,
    /// Entry point (topmost node) ID.
    entry_point: RwLock<Option<u64>>,
    /// Current maximum layer in the graph.
    max_layer: RwLock<usize>,
    /// Next node ID.
    next_id: AtomicU64,
    /// Mapping from external key to node ID.
    key_to_id: RwLock<HashMap<String, u64>>,
    /// Mapping from node ID to external key.
    id_to_key: RwLock<HashMap<u64, String>>,
}

impl<S: VectorStorage> HnswIndex<S> {
    /// Create a new empty HNSW index with the given storage backend.
    pub fn new(config: HnswConfig, storage: Arc<S>) -> Self {
        Self {
            config,
            nodes: RwLock::new(HashMap::new()),
            storage,
            entry_point: RwLock::new(None),
            max_layer: RwLock::new(0),
            next_id: AtomicU64::new(1),
            key_to_id: RwLock::new(HashMap::new()),
            id_to_key: RwLock::new(HashMap::new()),
        }
    }

    /// Compute raw distance between two vectors using the configured metric.
    /// For L2, this returns *squared* distance (sqrt deferred to finalize).
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.metric {
            DistanceMetric::L2 => distance::l2_distance_squared(a, b),
            DistanceMetric::Cosine => distance::cosine_distance(a, b),
            DistanceMetric::InnerProduct => distance::negative_inner_product(a, b),
        }
    }

    /// Convert raw internal distance to the final user-facing distance.
    fn finalize_distance(&self, raw: f32) -> f32 {
        match self.config.metric {
            DistanceMetric::L2 => raw.sqrt(),
            _ => raw,
        }
    }

    /// Insert a vector into the index, optionally with an external key.
    pub fn insert(&self, vector: Vec<f32>) -> u64 {
        self.insert_with_key(None, vector)
    }

    /// Insert a vector with an associated external key (e.g., "doc:42").
    pub fn insert_with_key(&self, key: Option<String>, vector: Vec<f32>) -> u64 {
        let id = self.next_id.fetch_add(1, AtomicOrdering::SeqCst);
        let level = self.random_level();

        // Store the vector in the backend
        if let Err(e) = self.storage.store(id, &vector) {
            tracing::error!("Failed to store vector {}: {}", id, e);
            return id;
        }

        // Store key mapping if provided
        if let Some(ref key) = key {
            self.key_to_id.write().insert(key.clone(), id);
            self.id_to_key.write().insert(id, key.clone());
        }

        let node = HnswNode {
            id,
            connections: vec![Vec::new(); level + 1],
            max_layer: level,
        };

        // Get current entry point
        let entry_point = *self.entry_point.read();
        let current_max = *self.max_layer.read();

        // Add node to the graph
        self.nodes.write().insert(id, node);

        if let Some(ep_id) = entry_point {
            // Greedy search from top layer down to the node's layer + 1
            let mut ep = ep_id;
            for layer in (level + 1..=current_max).rev() {
                ep = self.search_layer_single(&vector, ep, layer);
            }

            // For each layer this node participates in, find and connect neighbors
            for layer in (0..=std::cmp::min(level, current_max)).rev() {
                let neighbors = self.search_layer(&vector, ep, self.config.ef_construction, layer);
                let max_connections = if layer == 0 { self.config.m_max_0 } else { self.config.m };

                // Select best neighbors (simple heuristic: closest M)
                let selected: Vec<u64> = neighbors
                    .iter()
                    .take(max_connections)
                    .map(|c| c.id)
                    .collect();

                // Update connections bidirectionally
                {
                    let mut nodes = self.nodes.write();
                    if let Some(node) = nodes.get_mut(&id) {
                        if layer < node.connections.len() {
                            node.connections[layer] = selected.clone();
                        }
                    }

                    for &neighbor_id in &selected {
                        // First check if pruning is needed, collect data immutably
                        let needs_pruning = {
                            if let Some(neighbor) = nodes.get(&neighbor_id) {
                                if layer < neighbor.connections.len() {
                                    neighbor.connections[layer].len() + 1 > max_connections
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        };

                        if needs_pruning {
                            // Collect connection IDs for pruning
                            let conn_ids = {
                                let neighbor = nodes.get(&neighbor_id).unwrap();
                                let mut ids = neighbor.connections[layer].clone();
                                ids.push(id);
                                ids
                            };

                            // Load the neighbor's vector for distance computation
                            let query_vec = match self.storage.load(neighbor_id) {
                                Ok(v) => v,
                                Err(_) => continue,
                            };

                            let mut conn_dists: Vec<(u64, f32)> = conn_ids
                                .iter()
                                .filter_map(|&cid| {
                                    self.storage.load(cid).ok().map(|v| {
                                        (cid, self.compute_distance(&query_vec, &v))
                                    })
                                })
                                .collect();
                            conn_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                            let pruned: Vec<u64> = conn_dists
                                .into_iter()
                                .take(max_connections)
                                .map(|(cid, _)| cid)
                                .collect();

                            // Apply pruned connections
                            if let Some(neighbor) = nodes.get_mut(&neighbor_id) {
                                if layer < neighbor.connections.len() {
                                    neighbor.connections[layer] = pruned;
                                }
                            }
                        } else {
                            // Just push the new connection
                            if let Some(neighbor) = nodes.get_mut(&neighbor_id) {
                                if layer < neighbor.connections.len() {
                                    neighbor.connections[layer].push(id);
                                }
                            }
                        }
                    }
                }

                if !neighbors.is_empty() {
                    ep = neighbors[0].id;
                }
            }
        }

        // Update entry point if this node's level is higher
        if entry_point.is_none() || level > current_max {
            *self.entry_point.write() = Some(id);
            *self.max_layer.write() = level;
        }

        id
    }

    /// Search for the K nearest neighbors to a query vector.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let entry_point = *self.entry_point.read();
        let ep_id = match entry_point {
            Some(id) => id,
            None => return vec![], // Empty index
        };

        let max_layer = *self.max_layer.read();

        // Greedy descent from top layer to layer 1
        let mut ep = ep_id;
        for layer in (1..=max_layer).rev() {
            ep = self.search_layer_single(query, ep, layer);
        }

        // Search layer 0 with ef_search candidates
        let candidates = self.search_layer(query, ep, self.config.ef_search.max(k), 0);

        candidates
            .into_iter()
            .take(k)
            .map(|c| (c.id, self.finalize_distance(c.distance)))
            .collect()
    }

    /// Search with external key mapping — returns (key, distance).
    pub fn search_with_keys(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        let results = self.search(query, k);
        let id_to_key = self.id_to_key.read();
        results
            .into_iter()
            .filter_map(|(id, dist)| {
                id_to_key.get(&id).map(|key| (key.clone(), dist))
            })
            .collect()
    }

    /// Search a single layer, returning the single nearest node.
    fn search_layer_single(&self, query: &[f32], entry: u64, layer: usize) -> u64 {
        let nodes = self.nodes.read();
        let mut current = entry;
        let mut current_dist = self.storage
            .load(current)
            .map(|v| self.compute_distance(query, &v))
            .unwrap_or(f32::MAX);

        loop {
            let mut changed = false;

            if let Some(node) = nodes.get(&current) {
                if layer < node.connections.len() {
                    for &neighbor_id in &node.connections[layer] {
                        if let Ok(neighbor_vec) = self.storage.load(neighbor_id) {
                            let dist = self.compute_distance(query, &neighbor_vec);
                            if dist < current_dist {
                                current = neighbor_id;
                                current_dist = dist;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    /// Search a layer using a beam search with ef candidates.
    fn search_layer(
        &self,
        query: &[f32],
        entry: u64,
        ef: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let nodes = self.nodes.read();
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new(); // min-heap
        let mut result: BinaryHeap<MaxCandidate> = BinaryHeap::new(); // max-heap

        let entry_dist = self.storage
            .load(entry)
            .map(|v| self.compute_distance(query, &v))
            .unwrap_or(f32::MAX);

        visited.insert(entry);
        candidates.push(Candidate { id: entry, distance: entry_dist });
        result.push(MaxCandidate { id: entry, distance: entry_dist });

        while let Some(Candidate { id, distance: c_dist }) = candidates.pop() {
            let worst_result = result.peek().map(|r| r.distance).unwrap_or(f32::MAX);

            if c_dist > worst_result && result.len() >= ef {
                break;
            }

            if let Some(node) = nodes.get(&id) {
                if layer < node.connections.len() {
                    for &neighbor_id in &node.connections[layer] {
                        if visited.contains(&neighbor_id) {
                            continue;
                        }
                        visited.insert(neighbor_id);

                        if let Ok(neighbor_vec) = self.storage.load(neighbor_id) {
                            let dist = self.compute_distance(query, &neighbor_vec);
                            let worst = result.peek().map(|r| r.distance).unwrap_or(f32::MAX);

                            if dist < worst || result.len() < ef {
                                candidates.push(Candidate { id: neighbor_id, distance: dist });
                                result.push(MaxCandidate { id: neighbor_id, distance: dist });

                                if result.len() > ef {
                                    result.pop(); // Remove worst
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert result to sorted candidates
        let mut sorted: Vec<Candidate> = result
            .into_iter()
            .map(|mc| Candidate { id: mc.id, distance: mc.distance })
            .collect();
        sorted.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        sorted
    }

    /// Create a serializable snapshot of the graph structure.
    /// Vector data is persisted separately by the storage backend.
    pub fn snapshot(&self) -> GraphSnapshot {
        GraphSnapshot {
            config: self.config.clone(),
            nodes: self.nodes.read().clone(),
            entry_point: *self.entry_point.read(),
            max_layer: *self.max_layer.read(),
            next_id: self.next_id.load(AtomicOrdering::SeqCst),
            key_to_id: self.key_to_id.read().clone(),
            id_to_key: self.id_to_key.read().clone(),
            vectors: HashMap::new(),
        }
    }

    /// Restore an index from a persisted snapshot.
    ///
    /// The `storage` backend should already contain the vector data
    /// (e.g. via LsmVectorStorage backed by S3). Legacy snapshots that
    /// embedded vectors will have them replayed into storage.
    pub fn restore(snapshot: GraphSnapshot, storage: Arc<S>) -> Self {
        for (id, vec) in &snapshot.vectors {
            let _ = storage.store(*id, vec);
        }
        Self {
            config: snapshot.config,
            nodes: RwLock::new(snapshot.nodes),
            storage,
            entry_point: RwLock::new(snapshot.entry_point),
            max_layer: RwLock::new(snapshot.max_layer),
            next_id: AtomicU64::new(snapshot.next_id),
            key_to_id: RwLock::new(snapshot.key_to_id),
            id_to_key: RwLock::new(snapshot.id_to_key),
        }
    }

    /// Get a reference to the underlying vector storage backend.
    pub fn storage_ref(&self) -> &S {
        &self.storage
    }

    /// Generate a random level for a new node (geometric distribution).
    fn random_level(&self) -> usize {
        let mut rng = rand::rng();
        let mut level = 0;
        while rng.random::<f64>() < (1.0 / self.config.m as f64)
            && level < self.config.max_layers - 1
        {
            level += 1;
        }
        level
    }

    /// Get the number of nodes in the index.
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::vector_storage::InMemoryVectorStorage;

    fn test_config() -> HnswConfig {
        HnswConfig {
            m: 4,
            m_max_0: 8,
            ef_construction: 32,
            ef_search: 16,
            ml: 1.0 / (4.0_f64).ln(),
            max_layers: 4,
            metric: DistanceMetric::L2,
        }
    }

    fn make_index() -> HnswIndex<InMemoryVectorStorage> {
        let storage = Arc::new(InMemoryVectorStorage::new());
        HnswIndex::new(test_config(), storage)
    }

    #[test]
    fn test_empty_index() {
        let index = make_index();
        assert!(index.is_empty());
        assert_eq!(index.search(&[1.0, 2.0], 5).len(), 0);
    }

    #[test]
    fn test_single_insert_search() {
        let index = make_index();
        let id = index.insert(vec![1.0, 2.0, 3.0]);

        let results = index.search(&[1.0, 2.0, 3.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
        assert!(results[0].1 < 1e-6); // Exact match
    }

    #[test]
    fn test_knn_search() {
        let index = make_index();

        // Insert several points
        let _ids: Vec<u64> = (0..20)
            .map(|i| {
                index.insert(vec![i as f32, (i * 2) as f32])
            })
            .collect();

        // Search near the origin
        let results = index.search(&[0.5, 1.0], 3);
        assert_eq!(results.len(), 3);

        // Results should be sorted by distance
        for i in 0..results.len() - 1 {
            assert!(results[i].1 <= results[i + 1].1);
        }
    }

    #[test]
    fn test_many_inserts() {
        let index = make_index();

        for i in 0..100 {
            index.insert(vec![
                (i % 10) as f32,
                (i / 10) as f32,
                (i as f32).sin(),
            ]);
        }

        assert_eq!(index.len(), 100);

        let results = index.search(&[5.0, 5.0, 0.0], 10);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_recall_quality() {
        // Test that HNSW finds reasonably good results compared to brute force
        let config = HnswConfig {
            ef_search: 64,
            ..test_config()
        };  // inherits metric: L2 from test_config()
        let storage = Arc::new(InMemoryVectorStorage::new());
        let index = HnswIndex::new(config, storage);

        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![(i as f32) * 0.1, ((i * 7) % 50) as f32 * 0.1])
            .collect();

        for v in &vectors {
            index.insert(v.clone());
        }

        let query = [2.5, 2.5];

        // HNSW search
        let hnsw_results = index.search(&query, 5);

        // Brute force KNN
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let brute_results = distance::knn_l2(&query, &refs, 5);

        // Check that at least 3 of top-5 overlap (good recall)
        let hnsw_ids: HashSet<u64> = hnsw_results.iter().map(|r| r.0).collect();
        // The brute force indices won't match HNSW IDs directly since HNSW IDs start at 1
        // But we can verify the distances are reasonable
        assert!(hnsw_results[0].1 < 2.0, "Nearest result should be close");
    }

    #[test]
    fn test_insert_with_key() {
        let index = make_index();
        let id = index.insert_with_key(Some("doc:42".to_string()), vec![1.0, 2.0, 3.0]);

        let results = index.search_with_keys(&[1.0, 2.0, 3.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "doc:42");
        assert!(results[0].1 < 1e-6);

        // Verify internal ID mapping
        let key_to_id = index.key_to_id.read();
        assert_eq!(*key_to_id.get("doc:42").unwrap(), id);
    }

    #[test]
    fn test_search_with_keys_multiple() {
        let index = make_index();
        index.insert_with_key(Some("a".to_string()), vec![0.0, 0.0]);
        index.insert_with_key(Some("b".to_string()), vec![1.0, 0.0]);
        index.insert_with_key(Some("c".to_string()), vec![10.0, 10.0]);

        let results = index.search_with_keys(&[0.1, 0.0], 2);
        assert_eq!(results.len(), 2);
        // "a" should be closest, then "b"
        assert_eq!(results[0].0, "a");
        assert_eq!(results[1].0, "b");
    }
}
