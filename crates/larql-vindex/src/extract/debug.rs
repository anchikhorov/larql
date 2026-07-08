//! Debug/timing flags for extraction stages.
//!
//! Enable via env vars following the `LARQL_EXTRACT_*` convention.

use std::sync::OnceLock;

/// Env var: `LARQL_EXTRACT_DOWN_META_TIMING=1` — print per-layer timing
/// and in-flight concurrency during the parallel down-meta matmul stage.
pub const ENV_EXTRACT_DOWN_META_TIMING: &str = "LARQL_EXTRACT_DOWN_META_TIMING";

/// Returns `true` when `LARQL_EXTRACT_DOWN_META_TIMING` is set (to any value).
pub fn extract_down_meta_timing() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var_os(ENV_EXTRACT_DOWN_META_TIMING).is_some())
}
