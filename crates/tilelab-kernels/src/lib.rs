//! cuTile Rust kernels, one module per kernel family.
//!
//! Each kernel has a GPU test under `tests/` that compares it against the
//! CPU reference in `tilelab-ref`, and a benchmark binary under `src/bin/`.

pub mod elementwise;

/// cuTile's JIT compiler recurses deeply, so run GPU work on a thread with a
/// large stack. The upstream test suite does the same.
pub fn with_big_stack<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(f)
        .expect("spawn GPU thread")
        .join()
        .expect("GPU thread panicked")
}
