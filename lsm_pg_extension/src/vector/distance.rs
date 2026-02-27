//! Distance functions for vector similarity search.
//!
//! Supports L2 (Euclidean), cosine, and inner product distances.
//! These map to pgvector's `<->`, `<=>`, and `<#>` operators.

use super::types::LsmVector;
use pgrx::prelude::*;

/// Compute the L2 (Euclidean) distance between two vectors.
///
/// This is the `<->` operator in pgvector.
/// d(a, b) = sqrt(sum((a_i - b_i)^2))
#[pg_extern(immutable, parallel_safe)]
fn lsm_vector_l2_distance(a: LsmVector, b: LsmVector) -> f32 {
    l2_distance(&a.data, &b.data)
}

/// Compute the cosine distance between two vectors.
///
/// This is the `<=>` operator in pgvector.
/// d(a, b) = 1 - (a · b) / (|a| * |b|)
#[pg_extern(immutable, parallel_safe)]
fn lsm_vector_cosine_distance(a: LsmVector, b: LsmVector) -> f32 {
    cosine_distance(&a.data, &b.data)
}

/// Compute the negative inner product between two vectors.
///
/// This is the `<#>` operator in pgvector.
/// d(a, b) = -(a · b)
#[pg_extern(immutable, parallel_safe)]
fn lsm_vector_inner_product(a: LsmVector, b: LsmVector) -> f32 {
    negative_inner_product(&a.data, &b.data)
}

// Register distance operators. These must come AFTER the distance
// functions (which depend on the LsmVector type).
pgrx::extension_sql!(
    r#"
    CREATE OPERATOR <-> (
        LEFTARG = LsmVector,
        RIGHTARG = LsmVector,
        FUNCTION = lsm_vector_l2_distance,
        COMMUTATOR = <->
    );

    CREATE OPERATOR <=> (
        LEFTARG = LsmVector,
        RIGHTARG = LsmVector,
        FUNCTION = lsm_vector_cosine_distance,
        COMMUTATOR = <=>
    );

    CREATE OPERATOR <#> (
        LEFTARG = LsmVector,
        RIGHTARG = LsmVector,
        FUNCTION = lsm_vector_inner_product,
        COMMUTATOR = <#>
    );
    "#,
    name = "lsm_vector_operators",
    requires = [lsm_vector_l2_distance, lsm_vector_cosine_distance, lsm_vector_inner_product]
);

// ─────────────────────────────────────────────────────────────────────
// Core Distance Functions (optimized, no allocation)
// ─────────────────────────────────────────────────────────────────────

/// L2 (Euclidean) distance.
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

/// L2 squared distance (avoids sqrt for comparison-only use cases).
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
}

/// Dot product (inner product).
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

/// Cosine distance = 1 - cosine_similarity.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Maximum distance if either vector is zero
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// Negative inner product (for ORDER BY compatibility).
pub fn negative_inner_product(a: &[f32], b: &[f32]) -> f32 {
    -dot_product(a, b)
}

// ─────────────────────────────────────────────────────────────────────
// Batch Operations (for HNSW search)
// ─────────────────────────────────────────────────────────────────────

/// Compute distances from a query vector to multiple candidate vectors.
/// Returns (index, distance) pairs sorted by distance (ascending).
pub fn batch_l2_distances(query: &[f32], candidates: &[&[f32]]) -> Vec<(usize, f32)> {
    let mut results: Vec<(usize, f32)> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2_distance_squared(query, c)))
        .collect();

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Find the K nearest neighbors by L2 distance.
pub fn knn_l2(query: &[f32], candidates: &[&[f32]], k: usize) -> Vec<(usize, f32)> {
    let mut distances = batch_l2_distances(query, candidates);
    distances.truncate(k);

    // Convert squared distances to actual distances
    for (_, d) in &mut distances {
        *d = d.sqrt();
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let d = l2_distance(&a, &b);
        assert!((d - std::f32::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_l2_same_vector() {
        let a = [1.0, 2.0, 3.0];
        assert!((l2_distance(&a, &a) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        // Orthogonal vectors have cosine distance = 1.0
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_same_direction() {
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 4.0, 6.0];
        // Same direction => cosine distance = 0.0
        assert!((cosine_distance(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_knn() {
        let query = [0.0, 0.0];
        let c1 = [1.0, 0.0];
        let c2 = [0.0, 3.0];
        let c3 = [0.5, 0.5];

        let candidates: Vec<&[f32]> = vec![&c1, &c2, &c3];
        let results = knn_l2(&query, &candidates, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 2); // c3 is closest
        assert_eq!(results[1].0, 0); // c1 is next
    }

    #[test]
    fn test_batch_distances() {
        let query = [0.0, 0.0];
        let c1 = [3.0, 4.0]; // distance = 5
        let c2 = [1.0, 0.0]; // distance = 1

        let candidates: Vec<&[f32]> = vec![&c1, &c2];
        let results = batch_l2_distances(&query, &candidates);

        // Should be sorted: c2 first (closer)
        assert_eq!(results[0].0, 1);
        assert_eq!(results[1].0, 0);
    }
}
