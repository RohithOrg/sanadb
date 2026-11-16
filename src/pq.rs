//! Product quantization: each vector is split into `m` sub-vectors and each
//! sub-vector replaced by the index (one byte) of its nearest centroid among
//! 256 learned per subspace. A query builds one lookup table per subspace and
//! an approximate distance to any stored vector is `m` table reads and adds,
//! with no access to the full vector.

use rayon::prelude::*;

use crate::Metric;

pub(crate) const K: usize = 256;
/// Below this many vectors k-means has too few points per centroid to be useful.
pub(crate) const MIN_N: usize = 1024;

/// Number of subspaces for `dim`: sub-vectors of ~4 floats, at most 128 of them.
pub(crate) fn choose_m(dim: usize) -> usize {
    let range = || (1..=dim.min(128)).rev();
    range().find(|&m| dim.is_multiple_of(m) && dim / m >= 4).or_else(|| range().find(|&m| dim.is_multiple_of(m))).unwrap_or(1)
}

#[derive(Debug, Clone)]
pub(crate) struct Codebook {
    pub dim: usize,
    pub m: usize,
    /// `m * K * sub` floats, centroid `c` of subspace `s` at `(s*K + c) * sub`.
    pub centroids: Vec<f32>,
    /// The same centroids transposed per subspace, dimension-major:
    /// `by_dim[(s*sub + d) * K + c]`, so `lut` runs over the 256 centroids of
    /// one dimension contiguously and vectorises.
    by_dim: Vec<f32>,
}

impl Codebook {
    pub fn new(dim: usize, m: usize, centroids: Vec<f32>) -> Codebook {
        let sub = dim / m;
        let mut by_dim = vec![0f32; centroids.len()];
        for s in 0..m {
            for c in 0..K {
                for d in 0..sub {
                    by_dim[(s * sub + d) * K + c] = centroids[(s * K + c) * sub + d];
                }
            }
        }
        Codebook { dim, m, centroids, by_dim }
    }

    pub fn sub(&self) -> usize {
        self.dim / self.m
    }

    /// e-means per subspace over a sample of `vectors` (row-major).
    pub fn train(vectors: &[f32], dim: usize, m: usize, seed: u64) -> Codebook {
        let n = vectors.len() / dim;
        let sub = dim / m;
        let sample: Vec<usize> = {
            let take = n.min(65_536);
            let mut rng = XorShift(seed | 1);
            let mut idx: Vec<usize> = (0..n).collect();
            for i in 0..take {
                let j = i + (rng.next() % (n - i) as u64) as usize;
                idx.swap(i, j);
            }
            idx.truncate(take);
            idx
        };
        let centroids: Vec<Vec<f32>> = (0..m)
            .into_par_iter()
            .map(|s| {
                let point = |i: usize| -> &[f32] {
                    let row = sample[i];
                    &vectors[row * dim + s * sub..row * dim + (s + 1) * sub]
                };
                crate::kmeans::emeans(&point, sample.len(), sub, K, 3, seed ^ (s as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15))
            })
            .collect();
        Codebook::new(dim, m, centroids.concat())
    }

    /// Encode every row of `vectors` to `m` bytes each.
    pub fn encode_all(&self, vectors: &[f32]) -> Vec<u8> {
        let (dim, m, sub) = (self.dim, self.m, self.sub());
        let n = vectors.len() / dim;
        let mut codes = vec![0u8; n * m];
        codes.par_chunks_mut(m).enumerate().for_each(|(row, out)| {
            let v = &vectors[row * dim..(row + 1) * dim];
            for s in 0..m {
                out[s] = nearest(&self.centroids[s * K * sub..(s + 1) * K * sub], sub, &v[s * sub..(s + 1) * sub]) as u8;
            }
        });
        codes
    }

    /// Per-query table: `lut[s*K + c]` is the metric's contribution from
    /// subspace `s` if the stored vector's code there is `c`.
    pub fn lut(&self, q: &[f32], metric: Metric) -> Vec<f32> {
        let sub = self.sub();
        let mut lut = vec![0f32; self.m * K];
        for s in 0..self.m {
            let row = &mut lut[s * K..(s + 1) * K];
            for d in 0..sub {
                let qd = q[s * sub + d];
                let cents = &self.by_dim[(s * sub + d) * K..(s * sub + d + 1) * K];
                match metric {
                    Metric::L2 | Metric::Cosine => row.iter_mut().zip(cents).for_each(|(l, &c)| *l += (qd - c) * (qd - c)),
                    Metric::Dot => row.iter_mut().zip(cents).for_each(|(l, &c)| *l -= qd * c),
                }
            }
        }
        lut
    }
}

/// Approximate distance from a query (via its `lut`) to one stored code.
#[inline]
pub(crate) fn lut_distance(lut: &[f32], code: &[u8]) -> f32 {
    // Four accumulators: a single running sum is a serial chain of dependent
    // adds, and the table gathers cannot hide it (30% slower in isolation).
    let (chunks, rest) = code.as_chunks::<4>();
    let mut acc = [0f32; 4];
    for (j, c) in chunks.iter().enumerate() {
        let base = j * 4 * K;
        for i in 0..4 {
            acc[i] += lut[base + i * K + c[i] as usize];
        }
    }
    let done = chunks.len() * 4;
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + rest.iter().enumerate().map(|(i, &c)| lut[(done + i) * K + c as usize]).sum::<f32>()
}

use crate::kmeans::{XorShift, nearest};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose_m_divides_dim() {
        for dim in [1usize, 8, 16, 32, 100, 128, 384, 768, 1536] {
            let m = choose_m(dim);
            assert_eq!(dim % m, 0, "dim {dim} m {m}");
            assert!(m <= 128);
        }
        assert_eq!(choose_m(128), 32);
        assert_eq!(choose_m(768), 128);
    }

    #[test]
    fn pq_distance_tracks_exact() {
        let (dim, n) = (16usize, 4000usize);
        let mut rng = XorShift(7);
        let vectors: Vec<f32> = (0..n * dim).map(|_| (rng.next() >> 40) as f32 / (1u64 << 24) as f32).collect();
        let cb = Codebook::train(&vectors, dim, choose_m(dim), 1);
        let codes = cb.encode_all(&vectors);
        let q = &vectors[..dim];
        let lut = cb.lut(q, Metric::L2);
        let mut err = 0f32;
        for i in 0..n {
            let exact: f32 = q.iter().zip(&vectors[i * dim..(i + 1) * dim]).map(|(a, b)| (a - b) * (a - b)).sum();
            let approx = lut_distance(&lut, &codes[i * cb.m..(i + 1) * cb.m]);
            err += (exact - approx).abs() / exact.max(1e-3);
        }
        let mean_rel = err / n as f32;
        assert!(mean_rel < 0.25, "mean relative error {mean_rel}");
    }
}
