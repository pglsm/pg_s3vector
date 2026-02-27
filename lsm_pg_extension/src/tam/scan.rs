//! Sequential scan implementation for LSM-S3 tables.
//!
//! Supports full table scans and filtered scans by merging data
//! from MemTable + SSTables.

/// Scan state for tracking position during a sequential scan.
#[allow(dead_code)]
pub struct LsmScanState {
    /// Table name being scanned.
    pub table_name: String,
    /// Current position in the result set.
    pub position: usize,
    /// Cached results from the LSM scan.
    pub results: Vec<(Vec<u8>, Vec<u8>)>,
    /// Whether the scan has been initialized.
    pub initialized: bool,
}

impl LsmScanState {
    pub fn new(table_name: String) -> Self {
        Self {
            table_name,
            position: 0,
            results: Vec::new(),
            initialized: false,
        }
    }

    /// Get the next entry in the scan, or None if exhausted.
    pub fn next(&mut self) -> Option<(&[u8], &[u8])> {
        if self.position < self.results.len() {
            let (k, v) = &self.results[self.position];
            self.position += 1;
            Some((k.as_slice(), v.as_slice()))
        } else {
            None
        }
    }

    /// Reset the scan to the beginning.
    pub fn reset(&mut self) {
        self.position = 0;
    }
}
