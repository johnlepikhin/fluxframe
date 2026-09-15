//! Backoff for automatic reconnects to the daemon.

use std::time::Duration;

/// Delay before the first reconnect attempt.
const FIRST_DELAY: Duration = Duration::from_secs(1);

/// Longest delay between attempts.
const MAX_DELAY: Duration = Duration::from_secs(30);

/// Delay before reconnect attempt `attempt` (0-based): doubles from
/// [`FIRST_DELAY`] up to [`MAX_DELAY`], so a daemon restart is picked up
/// within a second or two while a daemon that stays down costs one
/// attempt every half minute.
#[must_use]
pub(crate) fn delay(attempt: u32) -> Duration {
    let factor = 2_u32.checked_pow(attempt).unwrap_or(u32::MAX);
    FIRST_DELAY
        .checked_mul(factor)
        .map_or(MAX_DELAY, |d| d.min(MAX_DELAY))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_doubles_from_one_second_up_to_the_cap() {
        let secs: Vec<u64> = (0..7).map(|attempt| delay(attempt).as_secs()).collect();
        assert_eq!(secs, vec![1, 2, 4, 8, 16, 30, 30]);
    }

    #[test]
    fn delay_stays_capped_for_absurd_attempt_counts() {
        assert_eq!(delay(64), MAX_DELAY);
        assert_eq!(delay(u32::MAX), MAX_DELAY);
    }
}
