//! Shared data-parallel primitives for effect kernels.
//!
//! Image effects run over independent pixels/rows/tiles. Instead of each
//! effect hand-rolling its own rayon plumbing (chunk sizing, the serial
//! cutoff, error propagation), they express *what* to compute per chunk
//! and let these helpers own *how*.
//!
//! All work runs on the process-wide rayon pool that `cap_rayon_pool`
//! (in `fluxframe-cli`) sizes deliberately low — saturating every core
//! starves the kernel's USB-isoc soft-IRQ and the UVC camera drops
//! packets *before* frames reach the pipeline. These helpers never spawn
//! their own pool; they submit to that global one.

use fluxframe_core::error::EffectError;
use rayon::prelude::*;

/// Serial-vs-parallel cutoff, in buffer elements.
///
/// Below this, a chunked kernel runs sequentially: rayon's per-dispatch
/// spawn/join (~µs) would otherwise swamp the arithmetic. Frame-res
/// buffers (e.g. `800×448×3 ≈ 1.1M` bytes) cross the threshold and
/// parallelise; model-res masks (`256² = 65_536` f32) mostly stay
/// serial. Tuned empirically in the benchmark stage.
pub const MIN_PARALLEL_ELEMS: usize = 50_000;

/// Whether a buffer of `len` elements is worth dispatching to rayon.
#[must_use]
pub fn should_parallelize(len: usize) -> bool {
    len >= MIN_PARALLEL_ELEMS
}

/// Split `data` into contiguous `chunk_len`-element chunks and apply
/// `f(offset, chunk)` to each, where `offset` is the chunk's start index
/// in `data`. Runs on the global rayon pool above [`MIN_PARALLEL_ELEMS`],
/// serially below it.
///
/// Chunk boundaries are identical in both paths, so a kernel that treats
/// each element independently produces byte-identical output regardless
/// of thread count — the property the golden tests rely on.
pub fn for_each_chunk_mut<T, F>(data: &mut [T], chunk_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    if chunk_len == 0 {
        return;
    }
    if should_parallelize(data.len()) {
        data.par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(i, chunk)| f(i * chunk_len, chunk));
    } else {
        data.chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(i, chunk)| f(i * chunk_len, chunk));
    }
}

/// Fallible variant of [`for_each_chunk_mut`]: `f` returns a `Result` and
/// the first error aborts the remaining work (via rayon's
/// `try_for_each`) and is returned.
///
/// Use for kernels that can genuinely fail; prefer the infallible
/// variant otherwise so no error path is threaded through the hot loop.
/// Panicking inside `f` would unwind into the rayon join and abort the
/// worker — return an [`EffectError`] instead.
///
/// # Errors
///
/// Returns the first [`EffectError`] produced by any chunk.
pub fn try_for_each_chunk_mut<T, F>(
    data: &mut [T],
    chunk_len: usize,
    f: F,
) -> Result<(), EffectError>
where
    T: Send,
    F: Fn(usize, &mut [T]) -> Result<(), EffectError> + Sync,
{
    if chunk_len == 0 {
        return Ok(());
    }
    if should_parallelize(data.len()) {
        data.par_chunks_mut(chunk_len)
            .enumerate()
            .try_for_each(|(i, chunk)| f(i * chunk_len, chunk))
    } else {
        data.chunks_mut(chunk_len)
            .enumerate()
            .try_for_each(|(i, chunk)| f(i * chunk_len, chunk))
    }
}

/// Apply `f(y, row)` to each row of a row-major buffer, dispatched
/// parallel or serial per the cutoff (like [`for_each_chunk_mut`], the
/// `for_each_` prefix does not promise parallelism). `row_len` is the
/// per-row element count (`width * bytes_per_pixel` for RGB, `width` for
/// a mask); `f` receives the row index and a mutable row slice.
///
/// Convenience over [`for_each_chunk_mut`] for the common row-independent
/// kernel (separable horizontal pass, per-scanline LUTs).
///
/// # Panics
///
/// Panics in debug builds if `data.len()` is not a multiple of `row_len`.
pub fn for_each_row_mut<T, F>(data: &mut [T], row_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    debug_assert!(row_len == 0 || data.len() % row_len == 0);
    for_each_chunk_mut(data, row_len, move |offset, row| {
        let y = if row_len == 0 { 0 } else { offset / row_len };
        f(y, row);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The global chunk offset must be correct in both the serial and
    /// the parallel path — `len` below the cutoff exercises the former,
    /// above it the latter. Writing each element to its own global index
    /// and reading it back proves offsets never drift with thread count.
    #[test]
    fn chunk_offset_is_global_in_both_paths() {
        for len in [10_usize, MIN_PARALLEL_ELEMS * 2 + 3] {
            let mut data = vec![0_u32; len];
            for_each_chunk_mut(&mut data, 7, |off, chunk| {
                for (i, x) in chunk.iter_mut().enumerate() {
                    *x = u32::try_from(off + i).expect("index fits u32");
                }
            });
            for (i, &x) in data.iter().enumerate() {
                assert_eq!(x, u32::try_from(i).unwrap(), "len={len} idx={i}");
            }
        }
    }

    #[test]
    fn zero_chunk_len_is_a_noop() {
        let mut data = vec![1_u8; 4];
        for_each_chunk_mut(&mut data, 0, |_, chunk| chunk.fill(9));
        assert_eq!(data, vec![1_u8; 4]);
    }

    #[test]
    fn try_variant_surfaces_an_error() {
        let mut data = vec![0_u8; 30];
        let result = try_for_each_chunk_mut(&mut data, 3, |off, _chunk| {
            if off == 9 {
                Err(EffectError::ProcessFailed {
                    name: "test".into(),
                    reason: "boom".into(),
                })
            } else {
                Ok(())
            }
        });
        assert!(matches!(result, Err(EffectError::ProcessFailed { .. })));
    }

    #[test]
    fn try_variant_ok_when_no_error() {
        let mut data = vec![0_u8; 30];
        let result = try_for_each_chunk_mut(&mut data, 3, |off, chunk| {
            chunk.fill(u8::try_from(off % 256).unwrap());
            Ok(())
        });
        assert!(result.is_ok());
    }

    #[test]
    fn for_each_row_supplies_row_index() {
        // 4 rows of 3 elements; write the row index into every cell.
        let mut data = vec![0_u32; 12];
        for_each_row_mut(&mut data, 3, |y, row| {
            row.fill(u32::try_from(y).unwrap());
        });
        assert_eq!(data, vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
    }
}
