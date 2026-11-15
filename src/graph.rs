//! Disk-resident Vamana graph index (the DiskANN layout).
//!
//! One node per fixed-size block: `vector f32*dim | count u32 | neighbors u32*R`.
//! The file is memory-mapped, so a query touches only the blocks it visits and
//! the OS page cache holds the working set. The dataset no longer has to fit
//! in RAM; the per-node id table (8 bytes/node) still does.
//!
//! File: `SANAIDX1 | dim u32 | metric u8 | pad[3] | n u32 | r u32 | entry u32 |
//! block u32 | ids u64*n | pad to 64 | blocks`.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use memmap2::Mmap;

use crate::index::distance;
use crate::{Error, Metric, Result};

const MAGIC: &[u8; 8] = b"SANAIDX1";
const HEADER_LEN: usize = 8 + 4 + 1 + 3 + 4 + 4 + 4 + 4;
const ALIGN: usize = 64;

#[derive(Debug, Clone, Copy)]
pub struct GraphParams {
    /// Max out-degree R. Memory and search cost scale with it.
    pub max_degree: u32,
    /// Candidate list size L during build. Larger = better graph, slower build.
    pub build_beam: u32,
    /// Pruning slack. 1.0 is a pure RNG graph; 1.2 adds long edges that cut hops.
    pub alpha: f32,
}

impl Default for GraphParams {
    fn default() -> Self {
        GraphParams { max_degree: 32, build_beam: 100, alpha: 1.2 }
    }
}

/// Anything the greedy walk can traverse: the in-RAM graph under construction
/// or the memory-mapped file.
trait Walkable {
    fn vector(&self, slot: u32) -> &[f32];
    fn neighbors(&self, slot: u32) -> &[u32];
}

pub(crate) struct GraphIndex {
    mmap: Mmap,
    dim: usize,
    metric: Metric,
    r: usize,
    entry: u32,
    block: usize,
    blocks_off: usize,
    ids: Vec<u64>,
    slot_of: HashMap<u64, u32>,
}

impl GraphIndex {
    /// Returns `None` when no index file exists yet.
    pub fn open(path: &Path, dim: u32, metric: Metric) -> Result<Option<GraphIndex>> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        // Safety: the file is only ever replaced atomically by rename, never
        // written in place, so the mapping cannot observe a partial write.
        let mmap = unsafe { Mmap::map(&file)? };
        let h = mmap.get(..HEADER_LEN).ok_or(Error::Corrupt("index shorter than header"))?;
        if &h[..8] != MAGIC {
            return Err(Error::Corrupt("index bad magic"));
        }
        let u32_at = |o: usize| u32::from_le_bytes(h[o..o + 4].try_into().unwrap());
        let (fdim, fmetric) = (u32_at(8), Metric::from_u8(h[12]).ok_or(Error::Corrupt("index unknown metric"))?);
        if fdim != dim || fmetric != metric {
            return Err(Error::HeaderMismatch { on_disk: (fdim, fmetric), requested: (dim, metric) });
        }
        let (n, r, entry, block) = (u32_at(16) as usize, u32_at(20) as usize, u32_at(24), u32_at(28) as usize);
        let ids_off = HEADER_LEN;
        let blocks_off = (ids_off + n * 8).next_multiple_of(ALIGN);
        if block != block_size(dim as usize, r) || mmap.len() < blocks_off + n * block || (n > 0 && entry as usize >= n) {
            return Err(Error::Corrupt("index geometry inconsistent"));
        }
        let ids: Vec<u64> = mmap[ids_off..ids_off + n * 8].as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect();
        let slot_of = ids.iter().enumerate().map(|(s, &id)| (id, s as u32)).collect();
        Ok(Some(GraphIndex { mmap, dim: dim as usize, metric, r, entry, block, blocks_off, ids, slot_of }))
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn slot(&self, id: u64) -> Option<u32> {
        self.slot_of.get(&id).copied()
    }

    pub fn id(&self, slot: u32) -> u64 {
        self.ids[slot as usize]
    }

    pub fn vector_of(&self, id: u64) -> Option<&[f32]> {
        self.slot(id).map(|s| self.vector(s))
    }

    /// Beam search from the medoid. Returns up to `k` `(distance, slot)` pairs
    /// among visited nodes that pass `accept`, closest first. `beam` bounds the
    /// candidate list; raise it for better recall or very selective filters.
    pub fn search(&self, q: &[f32], k: usize, beam: usize, accept: impl Fn(u32) -> bool) -> Vec<(f32, u32)> {
        if self.ids.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut visited = greedy(self, self.metric, self.entry, q, beam.max(k));
        visited.retain(|&(_, s)| accept(s));
        visited.truncate(k);
        visited
    }
}

impl Walkable for GraphIndex {
    fn vector(&self, slot: u32) -> &[f32] {
        let o = self.blocks_off + slot as usize * self.block;
        bytemuck::cast_slice(&self.mmap[o..o + self.dim * 4])
    }
    fn neighbors(&self, slot: u32) -> &[u32] {
        let o = self.blocks_off + slot as usize * self.block + self.dim * 4;
        let count = u32::from_le_bytes(self.mmap[o..o + 4].try_into().unwrap()) as usize;
        bytemuck::cast_slice(&self.mmap[o + 4..o + 4 + count.min(self.r) * 4])
    }
}

fn block_size(dim: usize, r: usize) -> usize {
    (dim * 4 + 4 + r * 4).next_multiple_of(ALIGN)
}

struct MemGraph<'a> {
    dim: usize,
    vectors: &'a [f32],
    adj: Vec<Vec<u32>>,
}

impl Walkable for MemGraph<'_> {
    fn vector(&self, slot: u32) -> &[f32] {
        let s = slot as usize * self.dim;
        &self.vectors[s..s + self.dim]
    }
    fn neighbors(&self, slot: u32) -> &[u32] {
        &self.adj[slot as usize]
    }
}

/// Greedy best-first walk. Returns every expanded node as `(distance, slot)`,
/// closest first. The candidate list is capped at `beam`.
fn greedy<G: Walkable>(g: &G, metric: Metric, start: u32, q: &[f32], beam: usize) -> Vec<(f32, u32)> {
    // (dist, slot, expanded), kept sorted ascending by dist.
    let mut cand: Vec<(f32, u32, bool)> = vec![(distance(metric, q, g.vector(start)), start, false)];
    let mut seen: HashSet<u32> = HashSet::from([start]);
    let mut out = Vec::new();
    while let Some(i) = cand.iter().position(|c| !c.2) {
        cand[i].2 = true;
        let (d, s, _) = cand[i];
        out.push((d, s));
        for &nb in g.neighbors(s) {
            if !seen.insert(nb) {
                continue;
            }
            let nd = distance(metric, q, g.vector(nb));
            if cand.len() >= beam && nd >= cand[beam - 1].0 {
                continue;
            }
            let pos = cand.partition_point(|c| c.0 <= nd);
            cand.insert(pos, (nd, nb, false));
            cand.truncate(beam);
        }
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    out
}

/// RobustPrune: pick the closest candidate, then drop every candidate that is
/// closer to the pick than to `p` by a factor of `alpha`. Repeat until R edges.
fn prune(g: &MemGraph, metric: Metric, p: u32, mut cands: Vec<(f32, u32)>, r: usize, alpha: f32) -> Vec<u32> {
    cands.retain(|&(_, s)| s != p);
    cands.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    cands.dedup_by_key(|c| c.1);
    let mut out = Vec::with_capacity(r);
    while let Some(&(_, best)) = cands.first() {
        out.push(best);
        if out.len() == r {
            break;
        }
        let bv = g.vector(best);
        cands.retain(|&(d_pq, s)| s != best && alpha * distance(metric, bv, g.vector(s)) > d_pq);
    }
    out
}

/// Build a Vamana graph over `vectors` (row-major, `n * dim`) and write the
/// index file to `path`. `ids[i]` is the external id of row `i`.
pub(crate) fn build_and_write(path: &Path, dim: usize, metric: Metric, ids: &[u64], vectors: &[f32], params: GraphParams) -> Result<()> {
    let n = ids.len();
    assert_eq!(vectors.len(), n * dim);
    let r = params.max_degree as usize;
    let l = (params.build_beam as usize).max(r);
    let mut rng = XorShift(0x9E3779B97F4A7C15);

    let mut g = MemGraph { dim, vectors, adj: vec![Vec::new(); n] };
    let mut entry = 0u32;
    if n > 0 {
        // Random initial graph so every node is reachable from the start.
        for a in g.adj.iter_mut() {
            while a.len() < r.min(n - 1) {
                let t = (rng.next() % n as u64) as u32;
                if !a.contains(&t) {
                    a.push(t);
                }
            }
        }
        // Medoid: the row closest to the centroid.
        let mut centroid = vec![0f32; dim];
        for row in vectors.chunks_exact(dim) {
            for (c, x) in centroid.iter_mut().zip(row) {
                *c += x;
            }
        }
        centroid.iter_mut().for_each(|c| *c /= n as f32);
        entry = (0..n as u32).min_by(|&a, &b| distance(metric, &centroid, g.vector(a)).total_cmp(&distance(metric, &centroid, g.vector(b)))).unwrap();

        let mut order: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            order.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        for &p in &order {
            let pv = g.vector(p);
            let mut cands = greedy(&g, metric, entry, pv, l);
            cands.extend(g.adj[p as usize].iter().map(|&s| (distance(metric, pv, g.vector(s)), s)));
            let new = prune(&g, metric, p, cands, r, params.alpha);
            g.adj[p as usize] = new.clone();
            for q in new {
                g.adj[q as usize].push(p);
                if g.adj[q as usize].len() > r {
                    let qv = g.vector(q);
                    let c: Vec<(f32, u32)> = g.adj[q as usize].iter().map(|&s| (distance(metric, qv, g.vector(s)), s)).collect();
                    g.adj[q as usize] = prune(&g, metric, q, c, r, params.alpha);
                }
            }
        }
    }

    let block = block_size(dim, r);
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(MAGIC)?;
    w.write_all(&(dim as u32).to_le_bytes())?;
    w.write_all(&[metric as u8, 0, 0, 0])?;
    w.write_all(&(n as u32).to_le_bytes())?;
    w.write_all(&(r as u32).to_le_bytes())?;
    w.write_all(&entry.to_le_bytes())?;
    w.write_all(&(block as u32).to_le_bytes())?;
    for id in ids {
        w.write_all(&id.to_le_bytes())?;
    }
    let pad = (HEADER_LEN + n * 8).next_multiple_of(ALIGN) - (HEADER_LEN + n * 8);
    w.write_all(&vec![0u8; pad])?;
    let mut buf = vec![0u8; block];
    for s in 0..n {
        buf.fill(0);
        buf[..dim * 4].copy_from_slice(bytemuck::cast_slice(g.vector(s as u32)));
        let a = &g.adj[s];
        buf[dim * 4..dim * 4 + 4].copy_from_slice(&(a.len() as u32).to_le_bytes());
        buf[dim * 4 + 4..dim * 4 + 4 + a.len() * 4].copy_from_slice(bytemuck::cast_slice(a));
        w.write_all(&buf)?;
    }
    w.flush()?;
    w.get_ref().sync_all()?;
    Ok(())
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
