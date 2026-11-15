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
}

impl Codebook {
    pub fn sub(&self) -> usize {
        self.dim / self.m
    }

    fn centroid(&self, s: usize, c: usize) -> &[f32] {
        let sub = self.sub();
        let o = (s * K + c) * sub;
        &self.centroids[o..o + sub]
    }

    /// Lloyd's k-means per subspace over a sample of `vectors` (row-major).
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
                let mut rng = XorShift(seed ^ (s as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
                let point = |row: usize| &vectors[row * dim + s * sub..row * dim + (s + 1) * sub];
                let mut cents: Vec<f32> = (0..K).flat_map(|_| point(sample[(rng.next() % sample.len() as u64) as usize]).to_vec()).collect();
                let mut assign = vec![0usize; sample.len()];
                for _ in 0..12 {
                    for (a, &row) in assign.iter_mut().zip(&sample) {
                        *a = nearest(&cents, sub, point(row));
                    }
                    let mut sums = vec![0f32; K * sub];
                    let mut counts = vec![0u32; K];
                    for (&a, &row) in assign.iter().zip(&sample) {
                        counts[a] += 1;
                        for (acc, x) in sums[a * sub..(a + 1) * sub].iter_mut().zip(point(row)) {
                            *acc += x;
                        }
                    }
                    for c in 0..K {
                        if counts[c] == 0 {
                            // Empty cluster: reseed on a random sample point.
                            cents[c * sub..(c + 1) * sub].copy_from_slice(point(sample[(rng.next() % sample.len() as u64) as usize]));
                        } else {
                            for (dst, acc) in cents[c * sub..(c + 1) * sub].iter_mut().zip(&sums[c * sub..(c + 1) * sub]) {
                                *dst = acc / counts[c] as f32;
                            }
                        }
                    }
                }
                cents
            })
            .collect();
        Codebook { dim, m, centroids: centroids.concat() }
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
            let qs = &q[s * sub..(s + 1) * sub];
            for c in 0..K {
                let cent = self.centroid(s, c);
                lut[s * K + c] = match metric {
                    Metric::L2 => qs.iter().zip(cent).map(|(a, b)| (a - b) * (a - b)).sum(),
                    Metric::Dot => -qs.iter().zip(cent).map(|(a, b)| a * b).sum::<f32>(),
                };
            }
        }
        lut
    }
}

/// Approximate distance from a query (via its `lut`) to one stored code.
#[inline]
pub(crate) fn lut_distance(lut: &[f32], code: &[u8]) -> f32 {
    code.iter().enumerate().map(|(s, &c)| lut[s * K + c as usize]).sum()
}

fn nearest(cents: &[f32], sub: usize, x: &[f32]) -> usize {
    let mut best = (f32::INFINITY, 0);
    for (c, cent) in cents.chunks_exact(sub).enumerate() {
        let d: f32 = x.iter().zip(cent).map(|(a, b)| (a - b) * (a - b)).sum();
        if d < best.0 {
            best = (d, c);
        }
    }
    best.1
}

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

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
