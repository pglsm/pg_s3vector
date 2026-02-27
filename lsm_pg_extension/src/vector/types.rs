//! LsmVector: A custom vector type for PostgreSQL.
//!
//! This type represents a fixed-dimension floating-point vector,
//! designed to be compatible with pgvector's `vector` type.

use pgrx::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A fixed-dimension vector of f32 values.
///
/// Stored as a compact binary format:
/// - 4 bytes: dimension count (u32)
/// - N * 4 bytes: f32 values
#[derive(Clone, Debug, Serialize, Deserialize, PostgresType)]
#[inoutfuncs]
pub struct LsmVector {
    /// The vector dimensions.
    pub data: Vec<f32>,
}

impl LsmVector {
    /// Create a new vector from a slice of f32 values.
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Get the number of dimensions.
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Get the raw bytes for storage in the LSM engine.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4 + self.data.len() * 4);
        bytes.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        for &val in &self.data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Reconstruct a vector from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 4 {
            return Err("Vector bytes too short".to_string());
        }

        let dim = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let expected_len = 4 + dim * 4;

        if bytes.len() < expected_len {
            return Err(format!(
                "Expected {} bytes for {} dimensions, got {}",
                expected_len, dim, bytes.len()
            ));
        }

        let mut data = Vec::with_capacity(dim);
        for i in 0..dim {
            let offset = 4 + i * 4;
            let val = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            data.push(val);
        }

        Ok(Self { data })
    }

    /// Compute the magnitude (L2 norm) of the vector.
    pub fn magnitude(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize the vector to unit length.
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag == 0.0 {
            return self.clone();
        }
        Self {
            data: self.data.iter().map(|x| x / mag).collect(),
        }
    }
}

impl InOutFuncs for LsmVector {
    /// Parse a vector from text representation: '[1.0, 2.0, 3.0]'
    fn input(input: &core::ffi::CStr) -> Self
    where
        Self: Sized,
    {
        let s = input.to_str().unwrap_or("");
        let s = s.trim();

        // Strip brackets
        let inner = if s.starts_with('[') && s.ends_with(']') {
            &s[1..s.len() - 1]
        } else {
            s
        };

        let data: Vec<f32> = inner
            .split(',')
            .filter_map(|part| part.trim().parse::<f32>().ok())
            .collect();

        if data.is_empty() {
            pgrx::error!("Invalid vector format. Expected '[1.0, 2.0, 3.0]'");
        }

        Self { data }
    }

    /// Output vector as text: '[1.0, 2.0, 3.0]'
    fn output(&self, buffer: &mut pgrx::StringInfo) {
        buffer.push('[');
        for (i, val) in self.data.iter().enumerate() {
            if i > 0 {
                buffer.push_str(", ");
            }
            buffer.push_str(&format!("{}", val));
        }
        buffer.push(']');
    }
}

impl fmt::Display for LsmVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, val) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", val)?;
        }
        write!(f, "]")
    }
}

// ─────────────────────────────────────────────────────────────────────
// SQL Functions for Vector Operations
// ─────────────────────────────────────────────────────────────────────

/// Get the number of dimensions in a vector.
#[pg_extern(immutable, parallel_safe)]
fn lsm_vector_dim(v: LsmVector) -> i32 {
    v.dim() as i32
}

/// Compute the magnitude of a vector.
#[pg_extern(immutable, parallel_safe)]
fn lsm_vector_magnitude(v: LsmVector) -> f32 {
    v.magnitude()
}

/// Normalize a vector to unit length.
#[pg_extern(immutable, parallel_safe)]
fn lsm_vector_normalize(v: LsmVector) -> LsmVector {
    v.normalize()
}

/// Insert a vector into an LSM-S3 table.
#[pg_extern]
fn lsm_s3_insert_vector(table_name: &str, key: &str, vector: LsmVector) -> String {
    let storage = super::super::tam::storage::TableStorage::global();
    let bytes = vector.to_bytes();

    match storage.insert(table_name, key.as_bytes(), &bytes) {
        Ok(()) => format!("INSERT 0 1 ({}D vector)", vector.dim()),
        Err(e) => format!("ERROR: {}", e),
    }
}

/// Retrieve a vector from an LSM-S3 table.
#[pg_extern]
fn lsm_s3_get_vector(table_name: &str, key: &str) -> Option<LsmVector> {
    let storage = super::super::tam::storage::TableStorage::global();

    match storage.get(table_name, key.as_bytes()) {
        Ok(Some(bytes)) => match LsmVector::from_bytes(&bytes) {
            Ok(v) => Some(v),
            Err(e) => {
                pgrx::warning!("Failed to decode vector: {}", e);
                None
            }
        },
        Ok(None) => None,
        Err(e) => {
            pgrx::warning!("lsm_s3_get_vector error: {}", e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_new() {
        let v = LsmVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.dim(), 3);
    }

    #[test]
    fn test_vector_bytes_roundtrip() {
        let v = LsmVector::new(vec![1.0, -2.5, 3.14, 0.0]);
        let bytes = v.to_bytes();
        let v2 = LsmVector::from_bytes(&bytes).unwrap();
        assert_eq!(v.data, v2.data);
    }

    #[test]
    fn test_vector_magnitude() {
        let v = LsmVector::new(vec![3.0, 4.0]);
        assert!((v.magnitude() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_normalize() {
        let v = LsmVector::new(vec![3.0, 4.0]);
        let n = v.normalize();
        assert!((n.magnitude() - 1.0).abs() < 1e-6);
        assert!((n.data[0] - 0.6).abs() < 1e-6);
        assert!((n.data[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_vector_display() {
        let v = LsmVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(format!("{}", v), "[1, 2, 3]");
    }
}
