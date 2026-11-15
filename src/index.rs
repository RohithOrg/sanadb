//! Exact, in-memory vector index. Vectors live in one contiguous `Vec<f32>`
//! so a scan is a sequential sweep the compiler can vectorize.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::{Hit, Metric};

pub(crate) struct Flat {
    dim: usize,
    metric: Metric,
    data: Vec<f32>,
    ids: Vec<u64>,
    slot_of: HashMap<u64, usize>,
}

impl Flat {
    pub fn new(dim: usize, metric: Metric) -> Self {
        Flat { dim, metric, data: Vec::new(), ids: Vec::new(), slot_of: HashMap::new() }
    }

    pub fn insert(&mut self, id: u64, v: &[f32]) {
        debug_assert_eq!(v.len(), self.dim);
        match self.slot_of.get(&id) {
            Some(&slot) => self.row_mut(slot).copy_from_slice(v),
            None => {
                self.slot_of.insert(id, self.ids.len());
                self.ids.push(id);
                self.data.extend_from_slice(v);
            }
        }
    }

    /// Swap-remove so the data stays dense.
    pub fn remove(&mut self, id: u64) {
        let Some(slot) = self.slot_of.remove(&id) else { return };
        let last = self.ids.len() - 1;
        if slot != last {
            let (head, tail) = self.data.split_at_mut(last * self.dim);
            head[slot * self.dim..(slot + 1) * self.dim].copy_from_slice(&tail[..self.dim]);
            let moved = self.ids[last];
            self.ids[slot] = moved;
            self.slot_of.insert(moved, slot);
        }
        self.ids.pop();
        self.data.truncate(last * self.dim);
    }

    pub fn vector(&self, id: u64) -> Option<&[f32]> {
        self.slot_of.get(&id).map(|&s| self.row(s))
    }

    pub fn iter(&self) -> impl Iterator<Item = (u64, &[f32])> {
        self.ids.iter().copied().zip(self.data.chunks_exact(self.dim))
    }

    pub fn search(&self, q: &[f32], k: usize, accept: impl Fn(u64) -> bool) -> Vec<Hit> {
        if k == 0 {
            return Vec::new();
        }
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);
        for (slot, row) in self.data.chunks_exact(self.dim).enumerate() {
            let id = self.ids[slot];
            if !accept(id) {
                continue;
            }
            let score = distance(self.metric, q, row);
            if heap.len() < k {
                heap.push(Candidate { score, id });
            } else if score < heap.peek().unwrap().score {
                heap.pop();
                heap.push(Candidate { score, id });
            }
        }
        let mut out: Vec<Hit> = heap.into_iter().map(|c| Hit { id: c.id, score: c.score }).collect();
        out.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.id.cmp(&b.id)));
        out
    }

    fn row(&self, slot: usize) -> &[f32] {
        &self.data[slot * self.dim..(slot + 1) * self.dim]
    }

    fn row_mut(&mut self, slot: usize) -> &mut [f32] {
        &mut self.data[slot * self.dim..(slot + 1) * self.dim]
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
        Metric::L2 => l2_sq(a, b),
        Metric::Dot => -dot(a, b),
    }
}

/// Eight independent accumulators keep the loop free of a serial dependency
/// chain so LLVM emits packed FMA.
#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let ((ca, ra), (cb, rb)) = (ca, cb);
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
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let ((ca, ra), (cb, rb)) = (ca, cb);
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f32>()
}
