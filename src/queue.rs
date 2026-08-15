use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

#[derive(Clone, Default)]
pub struct UserQueueManager {
    user_locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
    dedup: Arc<Mutex<HashMap<String, Instant>>>,
    rate_limiter: Arc<Mutex<HashMap<String, Vec<Instant>>>>,
}

impl UserQueueManager {
    pub fn new() -> Self {
        Self {
            user_locks: Arc::new(Mutex::new(HashMap::new())),
            dedup: Arc::new(Mutex::new(HashMap::new())),
            rate_limiter: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Gets an async lock dedicated to a specific user to ensure sequential processing for that user.
    pub async fn get_user_lock(&self, user_id: &str) -> Arc<Mutex<()>> {
        let mut locks = self.user_locks.lock().await;
        locks
            .entry(user_id.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// Checks if an identical action/button-tap was sent by this user within the last 3 seconds.
    pub async fn is_duplicate_action(&self, user_id: &str, action: &str) -> bool {
        // Never drop slash commands or regular typed messages
        if action.starts_with('/') || (!action.contains(':') && !action.starts_with("DEPOSIT_") && !action.starts_with("Q") && !action.starts_with("TOGGLE_")) {
            return false;
        }

        let mut dedup = self.dedup.lock().await;
        let now = Instant::now();

        // Cleanup entries older than 10 seconds
        dedup.retain(|_, &mut time| now.duration_since(time) < Duration::from_secs(10));

        let key = format!("{}:{}", user_id, action);
        if let Some(&last_time) = dedup.get(&key) {
            if now.duration_since(last_time) < Duration::from_secs(3) {
                return true;
            }
        }

        dedup.insert(key, now);
        false
    }

    /// Checks if a user has exceeded 30 messages per minute.
    /// Returns `true` if allowed, `false` if rate-limited.
    pub async fn check_rate_limit(&self, user_id: &str) -> bool {
        let mut rate_map = self.rate_limiter.lock().await;
        let now = Instant::now();
        let window = Duration::from_secs(60);

        let timestamps = rate_map.entry(user_id.to_string()).or_default();
        timestamps.retain(|&time| now.duration_since(time) < window);

        if timestamps.len() >= 30 {
            return false;
        }

        timestamps.push(now);
        true
    }
}
