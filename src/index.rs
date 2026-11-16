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

/// Eight independent accumulators keep the loop free of a serial dependency
/// chain so LLVM emits packed FMA.
#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
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
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f32>()
}
