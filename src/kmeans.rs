//! e-means: mini-batch k-means with exponential-moving-average centroids,
//! after Meta FAIR's `faiss/contrib/e_means` (MIT). Reaches Lloyd's quality
//! in a few passes over the data instead of tens. Used for PQ codebooks and
//! for partitioning a large build into shards.
//!
//! Recipe, from the reference implementation:
//! - batch size `B = round_pow2(0.035134 * k^0.1347 * n^0.8765 * T^0.8133)`,
//!   capped at `n/2`;
//! - initial step `a0 = 0.8244 * k^-0.4617 * B^0.9003 * n^-0.4498 * T^-0.1057`,
//!   clamped to `[1e-4, 1]`;
//! - per step, `a = a0 * (0.1 + 0.9 * 0.5 * (1 + cos(pi * t / T_steps)))`;
//! - for each cluster hit by the batch, `N <- (1-a) N + a n_j`,
//!   `c <- c + (a n_j / N) (mean_j - c)`;
//! - every 5 steps, clusters with mass below `0.1 * mean mass` are revived by
//!   splitting a mass-weighted random donor with a `1e-4` relative nudge;
//! - the last epoch also accumulates exact per-cluster sums, and the final
//!   centroids are those exact means.

use rayon::prelude::*;

/// Train `k` centroids over `n` points of dimension `dim`, reading point `i`
/// through `point(i)`. Returns `k * dim` floats.
pub(crate) fn emeans<'a>(point: &(dyn Fn(usize) -> &'a [f32] + Sync + 'a), n: usize, dim: usize, k: usize, epochs: usize, seed: u64) -> Vec<f32> {
    let mut rng = XorShift(seed | 1);
    if n == 0 || k == 0 {
        return vec![0.0; k * dim];
    }
    // Degenerate: at most one point per centroid.
    if n <= k {
        let mut c = Vec::with_capacity(k * dim);
        for i in 0..k {
            c.extend_from_slice(point(i % n));
        }
        return c;
    }
    let t = epochs.max(1) as f64;
    let (nf, kf) = (n as f64, k as f64);
    // The reference's "auto" batch law: the "quality" coefficient (batches
    // eight times smaller, so more steps) for dim <= 16, "throughput" above.
    // PQ subspaces are a few floats wide, so they take the quality law.
    let coefficient = if dim <= 16 { 0.00439175 } else { 0.035134 };
    let raw_b = coefficient * kf.powf(0.134749) * nf.powf(0.876473) * t.powf(0.813344);
    let batch = if dim <= 16 { round_pow2(raw_b).min(n / 2) } else { round_pow2(raw_b).min(round_pow2(nf / 2.0)) }.clamp(1, n);
    let alpha0 = (0.8244 * kf.powf(-0.4617) * (batch as f64).powf(0.9003) * nf.powf(-0.4498) * t.powf(-0.1057)).clamp(1e-4, 1.0);

    // Init: k-means++ (D^2 sampling) over a subsample. The reference uses a
    // random subset; ++ seeding avoids the midpoint-between-two-clusters
    // optimum that random seeding falls into on well-separated data.
    let mut perm: Vec<usize> = (0..n).collect();
    let pool = n.min(20_000.max(4 * k));
    for i in 0..pool {
        let j = i + (rng.next() % (n - i) as u64) as usize;
        perm.swap(i, j);
    }
    let mut cents: Vec<f32> = Vec::with_capacity(k * dim);
    cents.extend_from_slice(point(perm[0]));
    let mut d2: Vec<f32> = perm[..pool].iter().map(|&i| sq_dist(point(i), &cents[..dim])).collect();
    for _ in 1..k {
        let total: f64 = d2.iter().map(|&d| d as f64).sum();
        let pick = if total <= 0.0 {
            (rng.next() % pool as u64) as usize
        } else {
            let mut r = ((rng.next() >> 11) as f64 / (1u64 << 53) as f64) * total;
            let mut idx = pool - 1;
            for (i, &d) in d2.iter().enumerate() {
                r -= d as f64;
                if r <= 0.0 {
                    idx = i;
                    break;
                }
            }
            idx
        };
        let p = point(perm[pick]);
        cents.extend_from_slice(p);
        for (dd, &i) in d2.iter_mut().zip(&perm[..pool]) {
            *dd = dd.min(sq_dist(point(i), p));
        }
    }
    let mut mass = vec![2f64.powi(-16); k];

    let steps_per_epoch = n.div_ceil(batch);
    let total_steps = steps_per_epoch * epochs.max(1);
    let mut step = 0usize;
    let mut pass_sums = vec![0f64; k * dim];
    let mut pass_counts = vec![0u64; k];
    let mut sums = vec![0f32; k * dim];
    let mut counts = vec![0u32; k];

    for epoch in 0..epochs.max(1) {
        let last = epoch + 1 == epochs.max(1);
        // Shuffle the whole index set once per epoch.
        for i in (1..n).rev() {
            perm.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        for chunk in perm.chunks(batch) {
            let frac = step.min(total_steps) as f64 / total_steps.max(1) as f64;
            let alpha = alpha0 * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * frac).cos()));
            let assign: Vec<usize> = chunk.par_iter().map(|&i| nearest(&cents, dim, point(i))).collect();
            sums.iter_mut().for_each(|s| *s = 0.0);
            counts.iter_mut().for_each(|c| *c = 0);
            for (&i, &a) in chunk.iter().zip(&assign) {
                counts[a] += 1;
                for (acc, x) in sums[a * dim..(a + 1) * dim].iter_mut().zip(point(i)) {
                    *acc += x;
                }
            }
            for j in 0..k {
                let nj = counts[j];
                if nj == 0 {
                    continue;
                }
                let new_mass = (1.0 - alpha) * mass[j] + alpha * nj as f64;
                let w = (alpha * nj as f64 / new_mass) as f32;
                let inv = 1.0 / nj as f32;
                for (c, s) in cents[j * dim..(j + 1) * dim].iter_mut().zip(&sums[j * dim..(j + 1) * dim]) {
                    *c += w * (s * inv - *c);
                }
                mass[j] = new_mass;
                if last {
                    pass_counts[j] += nj as u64;
                    for (p, s) in pass_sums[j * dim..(j + 1) * dim].iter_mut().zip(&sums[j * dim..(j + 1) * dim]) {
                        *p += *s as f64;
                    }
                }
            }
            step += 1;
            if step.is_multiple_of(5) {
                revive(&mut cents, &mut mass, dim, &mut rng);
            }
        }
    }
    // Finalise on exact last-epoch means where a cluster saw any point.
    for j in 0..k {
        if pass_counts[j] > 0 {
            for (c, p) in cents[j * dim..(j + 1) * dim].iter_mut().zip(&pass_sums[j * dim..(j + 1) * dim]) {
                *c = (p / pass_counts[j] as f64) as f32;
            }
        }
    }
    cents
}

#[inline]
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Index of the centroid nearest to `x` (squared L2).
pub(crate) fn nearest(cents: &[f32], dim: usize, x: &[f32]) -> usize {
    let mut best = (f32::INFINITY, 0);
    for (c, cent) in cents.chunks_exact(dim).enumerate() {
        let d: f32 = x.iter().zip(cent).map(|(a, b)| (a - b) * (a - b)).sum();
        if d < best.0 {
            best = (d, c);
        }
    }
    best.1
}

/// Dead clusters (mass below a tenth of the mean) take half of a
/// mass-weighted random donor, nudged apart so they separate.
fn revive(cents: &mut [f32], mass: &mut [f64], dim: usize, rng: &mut XorShift) {
    let k = mass.len();
    let threshold = 0.1 * mass.iter().sum::<f64>() / k as f64;
    let dead: Vec<usize> = (0..k).filter(|&j| mass[j] < threshold).collect();
    if dead.is_empty() {
        return;
    }
    // Donors: alive, sampled without replacement in proportion to mass
    // (Gumbel-top-k on log mass).
    let mut keyed: Vec<(f64, usize)> = (0..k)
        .filter(|&j| mass[j] >= threshold && mass[j] > 0.0)
        .map(|j| {
            let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
            (mass[j].ln() - (-(u.max(1e-300)).ln()).ln(), j)
        })
        .collect();
    keyed.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (&target, &(_, donor)) in dead.iter().zip(&keyed) {
        let half = mass[donor] / 2.0;
        mass[donor] = half;
        mass[target] = half;
        for i in 0..dim {
            let v = cents[donor * dim + i];
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            cents[donor * dim + i] = v * (1.0 - 1e-4 * sign);
            cents[target * dim + i] = v * (1.0 + 1e-4 * sign);
        }
        if cents[donor * dim..(donor + 1) * dim] == cents[target * dim..(target + 1) * dim] {
            cents[target * dim] = f32::from_bits(cents[target * dim].to_bits() + 1);
        }
    }
}

fn round_pow2(v: f64) -> usize {
    if v <= 1.0 {
        return 1;
    }
    let low = 2f64.powi(v.log2().floor() as i32);
    let high = 2.0 * low;
    (if v / low < high / v { low } else { high }) as usize
}

pub(crate) struct XorShift(pub u64);
impl XorShift {
    pub fn next(&mut self) -> u64 {
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
    fn recovers_well_separated_clusters() {
        // 8 tight blobs in 4 dims; k = 8 must land one centroid per blob.
        let dim = 4;
        let mut rng = XorShift(3);
        let blobs: Vec<Vec<f32>> = (0..8).map(|b| (0..dim).map(|d| ((b >> d) & 1) as f32 * 10.0).collect()).collect();
        let pts: Vec<Vec<f32>> = (0..4000)
            .map(|i| blobs[i % 8].iter().map(|c| c + ((rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5)).collect())
            .collect();
        let cents = emeans(&|i| pts[i].as_slice(), pts.len(), dim, 8, 3, 1);
        let mut hit = vec![false; 8];
        for c in cents.chunks_exact(dim) {
            let b = nearest(&blobs.concat(), dim, c);
            let d: f32 = c.iter().zip(&blobs[b]).map(|(a, x)| (a - x) * (a - x)).sum();
            assert!(d < 1.0, "centroid {c:?} far from blob {b}");
            hit[b] = true;
        }
        assert!(hit.iter().all(|&h| h), "every blob gets a centroid: {hit:?}");
    }

    /// Plain Lloyd's algorithm, the baseline e-means is supposed to match in
    /// far fewer passes.
    fn lloyd<'a>(point: &dyn Fn(usize) -> &'a [f32], n: usize, dim: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
        let mut rng = XorShift(seed | 1);
        let mut cents: Vec<f32> = (0..k).flat_map(|_| point((rng.next() % n as u64) as usize).to_vec()).collect();
        for _ in 0..iters {
            let mut sums = vec![0f64; k * dim];
            let mut counts = vec![0usize; k];
            for i in 0..n {
                let a = nearest(&cents, dim, point(i));
                counts[a] += 1;
                for (s, x) in sums[a * dim..(a + 1) * dim].iter_mut().zip(point(i)) {
                    *s += *x as f64;
                }
            }
            for c in 0..k {
                if counts[c] == 0 {
                    cents[c * dim..(c + 1) * dim].copy_from_slice(point((rng.next() % n as u64) as usize));
                } else {
                    for (dst, s) in cents[c * dim..(c + 1) * dim].iter_mut().zip(&sums[c * dim..(c + 1) * dim]) {
                        *dst = (s / counts[c] as f64) as f32;
                    }
                }
            }
        }
        cents
    }

    fn distortion<'a>(point: &dyn Fn(usize) -> &'a [f32], n: usize, dim: usize, cents: &[f32]) -> f64 {
        (0..n).map(|i| {
            let c = nearest(cents, dim, point(i));
            sq_dist(point(i), &cents[c * dim..(c + 1) * dim]) as f64
        }).sum::<f64>() / n as f64
    }

    /// The setting that matters for PQ: 256 centroids over 4-dimensional
    /// sub-vectors of a realistic, non-blobby distribution. e-means in 3
    /// epochs must land within a few percent of Lloyd in 20.
    #[test]
    fn pq_sized_codebook_matches_lloyd() {
        let (dim, n, k) = (4usize, 40_000usize, 256usize);
        let mut rng = XorShift(11);
        // Heavy-tailed-ish data: sum of two uniforms, scaled per dimension.
        let pts: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|d| {
                let u = |r: &mut XorShift| (r.next() >> 40) as f32 / (1u64 << 24) as f32;
                (u(&mut rng) + u(&mut rng)) * (1.0 + d as f32)
            }).collect())
            .collect();
        let point = |i: usize| pts[i].as_slice();
        let e = distortion(&point, n, dim, &emeans(&point, n, dim, k, 3, 1));
        let l = distortion(&point, n, dim, &lloyd(&point, n, dim, k, 20, 1));
        eprintln!("distortion e-means(3 epochs)={e:.5} lloyd(20 iters)={l:.5} ratio={:.3}", e / l);
        assert!(e <= l * 1.05, "e-means distortion {e} vs lloyd {l}");
    }

    #[test]
    fn tiny_inputs_do_not_panic() {
        let pts = [[1.0f32, 2.0], [3.0, 4.0]];
        let c = emeans(&|i| &pts[i][..], 2, 2, 4, 2, 9);
        assert_eq!(c.len(), 8);
        let c = emeans(&|_| &pts[0][..], 0, 2, 3, 2, 9);
        assert_eq!(c.len(), 6);
    }
}
