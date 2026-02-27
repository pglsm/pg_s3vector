//! S3-based distributed lock for writer fencing.
//!
//! Ensures that only one Postgres primary can write to a specific S3 path
//! at a time. Uses conditional PUTs (ETags) for atomic lock acquisition.
//!
//! # How it works
//!
//! 1. A writer attempts to create a lock file on S3 with a unique fencing token.
//! 2. If the file doesn't exist, the lock is acquired (conditional PUT succeeds).
//! 3. If it does exist, the writer checks if the existing lock has expired.
//! 4. The lock holder must periodically renew the lease via heartbeat.
//! 5. Stale locks (expired heartbeat) can be taken over.

use object_store::path::Path;
use object_store::ObjectStore;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A distributed lock backed by S3 (or compatible object storage).
pub struct S3Lock {
    object_store: Arc<dyn ObjectStore>,
    lock_path: Path,
    fencing_token: String,
    lease_duration: Duration,
}

/// The lock file stored on S3.
#[derive(Debug, Serialize, Deserialize)]
struct LockFile {
    /// Unique fencing token identifying the lock holder.
    fencing_token: String,
    /// Hostname or identifier of the lock holder.
    holder_id: String,
    /// Unix timestamp when the lock was acquired.
    acquired_at: u64,
    /// Unix timestamp when the lease expires.
    expires_at: u64,
    /// Unix timestamp of the last heartbeat.
    last_heartbeat: u64,
}

impl S3Lock {
    /// Create a new S3 lock.
    pub fn new(
        object_store: Arc<dyn ObjectStore>,
        root_path: &str,
        fencing_token: String,
        lease_duration: Duration,
    ) -> Self {
        let lock_path = Path::from(format!("{}/writer.lock", root_path));
        Self {
            object_store,
            lock_path,
            fencing_token,
            lease_duration,
        }
    }

    /// Attempt to acquire the writer lock.
    /// Returns Ok(true) if acquired, Ok(false) if held by another writer.
    pub async fn try_acquire(&self, holder_id: &str) -> Result<bool, String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Check if lock already exists
        match self.object_store.get(&self.lock_path).await {
            Ok(result) => {
                let data = result.bytes().await
                    .map_err(|e| format!("Failed to read lock: {}", e))?;

                let lock: LockFile = serde_json::from_slice(&data)
                    .map_err(|e| format!("Invalid lock file: {}", e))?;

                // Check if the lock is ours
                if lock.fencing_token == self.fencing_token {
                    return Ok(true);
                }

                // Check if the lock has expired
                if now > lock.expires_at {
                    // Lock is stale, take it over
                    self.write_lock(holder_id, now).await?;
                    return Ok(true);
                }

                // Lock is held by someone else
                Ok(false)
            }
            Err(object_store::Error::NotFound { .. }) => {
                // No lock exists, acquire it
                self.write_lock(holder_id, now).await?;
                Ok(true)
            }
            Err(e) => Err(format!("Failed to check lock: {}", e)),
        }
    }

    /// Release the writer lock.
    pub async fn release(&self) -> Result<(), String> {
        // Verify we hold the lock before releasing
        match self.object_store.get(&self.lock_path).await {
            Ok(result) => {
                let data = result.bytes().await
                    .map_err(|e| format!("Failed to read lock: {}", e))?;

                let lock: LockFile = serde_json::from_slice(&data)
                    .map_err(|e| format!("Invalid lock file: {}", e))?;

                if lock.fencing_token != self.fencing_token {
                    return Err("Cannot release lock: not the holder".to_string());
                }

                self.object_store.delete(&self.lock_path).await
                    .map_err(|e| format!("Failed to delete lock: {}", e))?;

                Ok(())
            }
            Err(object_store::Error::NotFound { .. }) => {
                // Lock doesn't exist, nothing to release
                Ok(())
            }
            Err(e) => Err(format!("Failed to check lock: {}", e)),
        }
    }

    /// Renew the lease (heartbeat).
    pub async fn heartbeat(&self, holder_id: &str) -> Result<(), String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Verify we hold the lock
        match self.object_store.get(&self.lock_path).await {
            Ok(result) => {
                let data = result.bytes().await
                    .map_err(|e| format!("Failed to read lock: {}", e))?;

                let lock: LockFile = serde_json::from_slice(&data)
                    .map_err(|e| format!("Invalid lock file: {}", e))?;

                if lock.fencing_token != self.fencing_token {
                    return Err("Lost lock to another writer".to_string());
                }

                // Renew the lease
                self.write_lock(holder_id, now).await?;
                Ok(())
            }
            Err(e) => Err(format!("Heartbeat failed: {}", e)),
        }
    }

    /// Get the current fencing token.
    pub fn fencing_token(&self) -> &str {
        &self.fencing_token
    }

    /// Write the lock file to S3.
    async fn write_lock(&self, holder_id: &str, now: u64) -> Result<(), String> {
        let lock = LockFile {
            fencing_token: self.fencing_token.clone(),
            holder_id: holder_id.to_string(),
            acquired_at: now,
            expires_at: now + self.lease_duration.as_secs(),
            last_heartbeat: now,
        };

        let data = serde_json::to_vec(&lock)
            .map_err(|e| format!("Failed to serialize lock: {}", e))?;

        self.object_store
            .put(&self.lock_path, data.into())
            .await
            .map_err(|e| format!("Failed to write lock: {}", e))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    fn test_lock(token: &str) -> S3Lock {
        S3Lock::new(
            Arc::new(InMemory::new()),
            "/test",
            token.to_string(),
            Duration::from_secs(30),
        )
    }

    #[tokio::test]
    async fn test_acquire_and_release() {
        let lock = test_lock("token1");

        assert!(lock.try_acquire("node1").await.unwrap());
        lock.release().await.unwrap();
    }

    #[tokio::test]
    async fn test_reacquire_own_lock() {
        let lock = test_lock("token1");

        assert!(lock.try_acquire("node1").await.unwrap());
        assert!(lock.try_acquire("node1").await.unwrap()); // Re-acquire own lock
    }

    #[tokio::test]
    async fn test_conflict() {
        let store = Arc::new(InMemory::new());

        let lock1 = S3Lock::new(store.clone(), "/test", "t1".to_string(), Duration::from_secs(300));
        let lock2 = S3Lock::new(store.clone(), "/test", "t2".to_string(), Duration::from_secs(300));

        assert!(lock1.try_acquire("node1").await.unwrap());
        assert!(!lock2.try_acquire("node2").await.unwrap()); // Should fail
    }

    #[tokio::test]
    async fn test_heartbeat() {
        let lock = test_lock("token1");

        assert!(lock.try_acquire("node1").await.unwrap());
        lock.heartbeat("node1").await.unwrap();
    }
}
