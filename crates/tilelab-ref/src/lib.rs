//! CPU reference implementations that every GPU kernel is checked against.
//!
//! This crate has no CUDA dependency, so it builds and tests anywhere.

/// Elementwise `x + y`.
pub fn vector_add(x: &[f32], y: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), y.len(), "length mismatch");
    x.iter().zip(y).map(|(a, b)| a + b).collect()
}

/// Deterministic pseudo-random values in `[-1, 1)` (SplitMix64), so tests
/// are reproducible without pulling in a rand crate.
pub fn seeded_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Largest absolute difference and the index where it occurs.
pub fn max_abs_diff(got: &[f32], want: &[f32]) -> (f32, usize) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .enumerate()
        .fold(
            (0.0, 0),
            |(m, mi), (i, d)| if d > m || d.is_nan() { (d, i) } else { (m, mi) },
        )
}

/// Panics with a useful message unless every element satisfies
/// `|got - want| <= atol + rtol * |want|` (the numpy/torch `allclose` rule).
pub fn assert_close(got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let ok = (g - w).abs() <= atol + rtol * w.abs();
        assert!(
            ok,
            "mismatch at index {i}: got {g}, want {w} (atol={atol}, rtol={rtol}); max_abs_diff={:?}",
            max_abs_diff(got, want)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_add_matches_by_hand() {
        assert_eq!(vector_add(&[1.0, 2.0], &[3.0, -2.0]), vec![4.0, 0.0]);
    }

    #[test]
    fn seeded_vec_is_deterministic_and_in_range() {
        let a = seeded_vec(1000, 7);
        assert_eq!(a, seeded_vec(1000, 7));
        assert_ne!(a, seeded_vec(1000, 8));
        assert!(a.iter().all(|v| (-1.0..1.0).contains(v)));
    }

    #[test]
    #[should_panic(expected = "mismatch at index 1")]
    fn assert_close_reports_index() {
        assert_close(&[1.0, 2.5], &[1.0, 2.0], 1e-6, 1e-6);
    }

    #[test]
    fn max_abs_diff_finds_worst() {
        assert_eq!(max_abs_diff(&[0.0, 1.0, 5.0], &[0.0, 0.0, 4.5]), (1.0, 1));
    }
}
