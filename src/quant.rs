//! Per-vector 8-bit scalar quantisation for index blocks.
//!
//! Each stored vector keeps its own `min` and `scale`: `x ≈ min + code * scale`
//! with `code` in `0..=255`. Dequantising and requantising a vector gives the
//! same codes (the two endpoints are exact), so repeated checkpoints do not
//! drift. The log keeps full `f32`, so records written since the last
//! checkpoint are exact; `get` on an indexed record returns the dequantised
//! vector.

/// Quantise `v` into `out` (same length). Returns `(min, scale)`.
pub(crate) fn quantize(v: &[f32], out: &mut [u8]) -> (f32, f32) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in v {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    if !lo.is_finite() {
        lo = 0.0;
        hi = 0.0;
    }
    let scale = if hi > lo { (hi - lo) / 255.0 } else { 1.0 };
    let inv = 1.0 / scale;
    for (o, &x) in out.iter_mut().zip(v) {
        *o = ((x - lo) * inv).round().clamp(0.0, 255.0) as u8;
    }
    (lo, scale)
}

/// Inverse of `quantize`.
pub(crate) fn dequantize(codes: &[u8], min: f32, scale: f32, out: &mut [f32]) {
    for (o, &c) in out.iter_mut().zip(codes) {
        *o = min + c as f32 * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_error_is_within_half_a_step_and_stable() {
        let v: Vec<f32> = (0..128)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.731)
            .collect();
        let mut codes = vec![0u8; 128];
        let (min, scale) = quantize(&v, &mut codes);
        let mut back = vec![0f32; 128];
        dequantize(&codes, min, scale, &mut back);
        for (a, b) in v.iter().zip(&back) {
            assert!((a - b).abs() <= scale / 2.0 + 1e-5, "{a} vs {b}");
        }
        // Requantising the dequantised vector reproduces the codes exactly.
        let mut again = vec![0u8; 128];
        let (min2, scale2) = quantize(&back, &mut again);
        assert_eq!(codes, again);
        assert!((min - min2).abs() < 1e-6 && (scale - scale2).abs() < 1e-6);
        // Constant and empty vectors do not divide by zero.
        let (m, s) = quantize(&[3.0; 4], &mut [0u8; 4]);
        assert!(m == 3.0 && s == 1.0);
        quantize(&[], &mut []);
    }
}
