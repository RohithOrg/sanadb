//! Exact index over the records written since the last checkpoint. It holds
//! no vectors: each record is a byte offset into the memory-mapped log, and
//! the caller supplies the resolver. A scan therefore streams through the
//! page cache rather than through a RAM copy, so a bulk load costs 16 bytes
//! of RAM per record here instead of the vector.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::{Hit, Metric};

pub(crate) struct Flat {
    metric: Metric,
    ids: Vec<u64>,
    offs: Vec<u64>,
    slot_of: HashMap<u64, usize>,
}

impl Flat {
    pub fn new(metric: Metric) -> Self {
        Flat {
            metric,
            ids: Vec::new(),
            offs: Vec::new(),
            slot_of: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: u64, off: u64) {
        match self.slot_of.get(&id) {
            Some(&slot) => self.offs[slot] = off,
            None => {
                self.slot_of.insert(id, self.ids.len());
                self.ids.push(id);
                self.offs.push(off);
            }
        }
    }

    /// Swap-remove so the arrays stay dense.
    pub fn remove(&mut self, id: u64) {
        let Some(slot) = self.slot_of.remove(&id) else {
            return;
        };
        let last = self.ids.len() - 1;
        if slot != last {
            self.ids.swap(slot, last);
            self.offs.swap(slot, last);
            self.slot_of.insert(self.ids[slot], slot);
        }
        self.ids.pop();
        self.offs.pop();
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn offset(&self, id: u64) -> Option<u64> {
        self.slot_of.get(&id).map(|&s| self.offs[s])
    }

    pub fn iter(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.ids.iter().copied().zip(self.offs.iter().copied())
    }

    /// Exact k-NN over every record, resolving vectors through `vec`.
    pub fn search<'a>(
        &self,
        q: &[f32],
        k: usize,
        accept: impl Fn(u64) -> bool,
        vec: &dyn Fn(u64) -> &'a [f32],
    ) -> Vec<Hit> {
        if k == 0 {
            return Vec::new();
        }
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);
        for (id, off) in self.iter() {
            if !accept(id) {
                continue;
            }
            let score = distance(self.metric, q, vec(off));
            if heap.len() < k {
                heap.push(Candidate { score, id });
            } else if score < heap.peek().unwrap().score {
                heap.pop();
                heap.push(Candidate { score, id });
            }
        }
        let mut out: Vec<Hit> = heap
            .into_iter()
            .map(|c| Hit {
                id: c.id,
                score: c.score,
            })
            .collect();
        out.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.id.cmp(&b.id)));
        out
    }
}

/// Max-heap on score so the worst candidate sits on top and is cheap to evict.
struct Candidate {
    score: f32,
    id: u64,
}

impl PartialEq for Candidate {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Candidate {
    fn cmp(&self, o: &Self) -> Ordering {
        self.score.total_cmp(&o.score).then(self.id.cmp(&o.id))
    }
}

#[inline]
pub(crate) fn distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        // Cosine inputs are unit length, so squared L2 = 2 - 2cos: the same
        // ranking as negative cosine, but non-negative, which RobustPrune's
        // alpha test needs. The API converts the score back (see `Db::search`).
        Metric::L2 | Metric::Cosine => l2_sq(a, b),
        Metric::Dot => -dot(a, b),
    }
}

/// Squared L2. On x86-64 with AVX2 and FMA an explicit kernel is used: the
/// default x86-64 target is SSE2 without FMA, so the autovectoriser there
/// does a quarter of the work per instruction. Everywhere else (aarch64 has
/// NEON in its baseline) the eight-accumulator loop below already compiles
/// to packed FMA; a hand-written NEON kernel measured no faster.
#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if x86::avx2() {
        // Safety: the feature was detected at runtime.
        return unsafe { x86::l2_sq_avx2(a, b) };
    }
    l2_sq_scalar(a, b)
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if x86::avx2() {
        // Safety: the feature was detected at runtime.
        return unsafe { x86::dot_avx2(a, b) };
    }
    dot_scalar(a, b)
}

/// Eight independent accumulators keep the loop free of a serial dependency
/// chain so LLVM emits packed FMA.
#[inline]
fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            let d = x[i] - y[i];
            acc[i] += d * d;
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for (x, y) in ra.iter().zip(rb) {
        let d = x - y;
        s += d * d;
    }
    s
}

#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f32>()
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;
    use std::sync::atomic::{AtomicU8, Ordering};

    /// 0 unknown, 1 absent, 2 present. Detection is a few hundred cycles the
    /// first time; it must not sit on the per-distance path.
    static AVX2: AtomicU8 = AtomicU8::new(0);

    #[inline]
    pub(super) fn avx2() -> bool {
        match AVX2.load(Ordering::Relaxed) {
            2 => true,
            1 => false,
            _ => {
                let yes = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
                AVX2.store(if yes { 2 } else { 1 }, Ordering::Relaxed);
                yes
            }
        }
    }

    /// Four 8-lane accumulators, 32 floats per iteration, scalar tail.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn l2_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 32 <= n {
            for k in 0..4 {
                let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i + k * 8)), _mm256_loadu_ps(pb.add(i + k * 8)));
                acc[k] = _mm256_fmadd_ps(d, d, acc[k]);
            }
            i += 32;
        }
        while i + 8 <= n {
            let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            acc[0] = _mm256_fmadd_ps(d, d, acc[0]);
            i += 8;
        }
        let mut s = hsum(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])));
        while i < n {
            let d = a[i] - b[i];
            s += d * d;
            i += 1;
        }
        s
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 32 <= n {
            for k in 0..4 {
                acc[k] = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + k * 8)), _mm256_loadu_ps(pb.add(i + k * 8)), acc[k]);
            }
            i += 32;
        }
        while i + 8 <= n {
            acc[0] = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc[0]);
            i += 8;
        }
        let mut s = hsum(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])));
        while i < n {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }

    #[target_feature(enable = "avx2")]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }
}

#[cfg(test)]
mod kernel_tests {
    use super::*;

    /// The dispatched kernels must agree with the scalar reference on every
    /// length, including tails shorter than one vector register. On x86-64
    /// CI this exercises the AVX2 path; elsewhere it checks dispatch is a
    /// no-op.
    #[test]
    fn kernels_match_scalar_reference() {
        let mut x = 0x1234_5678u64;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / 16777216.0 - 0.5
        };
        for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100, 128, 129, 768, 1536] {
            let a: Vec<f32> = (0..len).map(|_| rnd()).collect();
            let b: Vec<f32> = (0..len).map(|_| rnd()).collect();
            let (l, ls) = (l2_sq(&a, &b), l2_sq_scalar(&a, &b));
            let (d, ds) = (dot(&a, &b), dot_scalar(&a, &b));
            let tol = 1e-4 * (1.0 + ls.abs());
            assert!((l - ls).abs() <= tol, "l2 len {len}: {l} vs {ls}");
            assert!((d - ds).abs() <= 1e-4 * (1.0 + ds.abs()), "dot len {len}: {d} vs {ds}");
        }
    }
}
