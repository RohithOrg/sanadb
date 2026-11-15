//! Disk-resident Vamana graph index (the DiskANN layout).
//!
//! One node per fixed-size block: `vector f32*dim | count u32 | neighbors u32*R`.
//! The file is memory-mapped, so a query touches only the blocks it visits and
//! the OS page cache holds the working set.
//!
//! For indexes of at least `pq::MIN_N` nodes the file also carries a product
//! quantization codebook and one `m`-byte code per node. The codes are loaded
//! into RAM on open, and the walk scores neighbours from them, so expanding a
//! node costs one block read (its adjacency) instead of `R` reads (each
//! neighbour's vector). Full-precision vectors are read only to re-rank the
//! final candidates.
//!
//! Building: up to `GraphParams::shard_size` vectors are indexed in one pass.
//! Larger sets are split by k-means into overlapping shards (each vector in
//! its two nearest), each shard gets its own Vamana graph in RAM, the shard
//! adjacencies are spilled to temp files, and a final pass merges each node's
//! edge lists and prunes them to R. Peak memory is one shard's adjacency plus
//! a few words per node, not the whole graph. The vectors themselves are read
//! through whatever slice the caller passes, which the store memory-maps.
//!
//! File: `SANAIDX2 | dim u32 | metric u8 | pad[3] | n u32 | r u32 | entry u32 |
//! block u32 | m u32 | ids u64*n | pad to 64 | blocks | codebook f32*(m*256*dim/m) | codes u8*(n*m)`.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use memmap2::Mmap;
use rayon::prelude::*;

use crate::index::distance;
use crate::pq::{self, Codebook};
use crate::{Error, Metric, Result};

const MAGIC: &[u8; 8] = b"SANAIDX2";
const HEADER_LEN: usize = 8 + 4 + 1 + 3 + 4 + 4 + 4 + 4 + 4;
const ALIGN: usize = 64;
const NONE: u32 = u32::MAX;

#[derive(Debug, Clone, Copy)]
pub struct GraphParams {
    /// Max out-degree R. Memory and search cost scale with it.
    pub max_degree: u32,
    /// Candidate list size L during build. Larger = better graph, slower build.
    pub build_beam: u32,
    /// Pruning slack. 1.0 is a pure RNG graph; 1.2 adds long edges that cut hops.
    pub alpha: f32,
    /// Store product-quantized codes and walk on them. Off means every hop
    /// reads R full vectors from the mapped file.
    pub use_pq: bool,
    /// Largest set built as a single graph. Above this the build is sharded;
    /// peak RAM is roughly `shard_size * (R*4 + 50)` bytes for the adjacency.
    pub shard_size: u32,
}

impl Default for GraphParams {
    fn default() -> Self {
        GraphParams { max_degree: 32, build_beam: 100, alpha: 1.2, use_pq: true, shard_size: 1_000_000 }
    }
}

/// Anything the greedy walk can traverse: the in-RAM graph under construction
/// or the memory-mapped file.
trait Walkable: Sync {
    /// Append `slot`'s neighbours to `out`. A callback rather than a slice so
    /// the in-RAM graph can hand out lock-guarded adjacency.
    fn neighbors_into(&self, slot: u32, out: &mut Vec<u32>);
    /// Hint that these slots' blocks are about to be read, so a disk-backed
    /// graph can issue the reads concurrently instead of faulting one at a time.
    fn prefetch(&self, _slots: &[u32]) {}
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
    pq: Option<(Codebook, Vec<u8>)>,
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
        let dim = dim as usize;
        let (n, r, entry, block, m) = (u32_at(16) as usize, u32_at(20) as usize, u32_at(24), u32_at(28) as usize, u32_at(32) as usize);
        let ids_off = HEADER_LEN;
        let blocks_off = (ids_off + n * 8).next_multiple_of(ALIGN);
        let pq_off = blocks_off + n * block;
        let pq_len = if m == 0 { 0 } else { m * pq::K * (dim / m) * 4 + n * m };
        if block != block_size(dim, r) || (m != 0 && !dim.is_multiple_of(m)) || mmap.len() < pq_off + pq_len || (n > 0 && entry as usize >= n) {
            return Err(Error::Corrupt("index geometry inconsistent"));
        }
        let ids: Vec<u64> = mmap[ids_off..ids_off + n * 8].as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect();
        let slot_of = ids.iter().enumerate().map(|(s, &id)| (id, s as u32)).collect();
        let pq = (m != 0).then(|| {
            let cb_len = m * pq::K * (dim / m) * 4;
            let centroids = mmap[pq_off..pq_off + cb_len].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            let codes = mmap[pq_off + cb_len..pq_off + pq_len].to_vec();
            (Codebook { dim, m, centroids }, codes)
        });
        Ok(Some(GraphIndex { mmap, dim, metric, r, entry, block, blocks_off, ids, slot_of, pq }))
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Number of PQ subspaces, 0 when the walk uses full vectors.
    pub fn pq_subspaces(&self) -> usize {
        self.pq.as_ref().map_or(0, |(cb, _)| cb.m)
    }

    pub fn slot(&self, id: u64) -> Option<u32> {
        self.slot_of.get(&id).copied()
    }

    pub fn id(&self, slot: u32) -> u64 {
        self.ids[slot as usize]
    }

    pub fn vector(&self, slot: u32) -> &[f32] {
        let o = self.blocks_off + slot as usize * self.block;
        bytemuck::cast_slice(&self.mmap[o..o + self.dim * 4])
    }

    pub fn vector_of(&self, id: u64) -> Option<&[f32]> {
        self.slot(id).map(|s| self.vector(s))
    }

    /// Beam search from the medoid. Returns up to `k` `(distance, slot)` pairs
    /// among visited nodes that pass `accept`, closest first, with exact
    /// distances.
    ///
    /// `beam` bounds the candidate list on the first attempt. When a selective
    /// filter leaves fewer than `k` accepted nodes, the walk is retried with
    /// the beam doubled, up to `max_beam`, so filtered queries still fill `k`
    /// without the caller tuning anything.
    ///
    /// `width` is how many candidates are expanded per step; their blocks are
    /// prefetched together, which is what makes a cold (disk-bound) query
    /// cheaper. 1 reproduces the classic one-at-a-time walk.
    pub fn search(&self, q: &[f32], k: usize, beam: usize, max_beam: usize, width: usize, accept: impl Fn(u32) -> bool) -> Vec<(f32, u32)> {
        if self.ids.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut beam = beam.max(k);
        let cap = max_beam.max(beam).min(self.ids.len());
        let lut = self.pq.as_ref().map(|(cb, _)| cb.lut(q, self.metric));
        loop {
            let mut visited = match (&self.pq, &lut) {
                (Some((cb, codes)), Some(lut)) => {
                    let m = cb.m;
                    greedy(self, self.entry, beam, width, |s| pq::lut_distance(lut, &codes[s as usize * m..(s as usize + 1) * m]))
                }
                _ => greedy(self, self.entry, beam, width, |s| distance(self.metric, q, self.vector(s))),
            };
            let exhausted = visited.len() >= self.ids.len();
            visited.retain(|&(_, s)| accept(s));
            if visited.len() >= k || beam >= cap || exhausted {
                if self.pq.is_some() {
                    // Re-rank the best `beam` approximate candidates exactly.
                    visited.truncate(beam.max(k));
                    for c in visited.iter_mut() {
                        c.0 = distance(self.metric, q, self.vector(c.1));
                    }
                    visited.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                }
                visited.truncate(k);
                return visited;
            }
            beam = (beam * 2).min(cap);
        }
    }
}

impl Walkable for GraphIndex {
    fn neighbors_into(&self, slot: u32, out: &mut Vec<u32>) {
        let o = self.blocks_off + slot as usize * self.block + self.dim * 4;
        let count = u32::from_le_bytes(self.mmap[o..o + 4].try_into().unwrap()) as usize;
        out.extend_from_slice(bytemuck::cast_slice(&self.mmap[o + 4..o + 4 + count.min(self.r) * 4]));
    }
    #[cfg(unix)]
    fn prefetch(&self, slots: &[u32]) {
        if slots.len() < 2 {
            return;
        }
        for &s in slots {
            let o = self.blocks_off + s as usize * self.block;
            // Advisory: the kernel starts the read; a failure only costs the hint.
            let _ = self.mmap.advise_range(memmap2::Advice::WillNeed, o, self.block);
        }
    }
    // Windows has no madvise; the walk still works, one page fault at a time.
}

fn block_size(dim: usize, r: usize) -> usize {
    (dim * 4 + 4 + r * 4).next_multiple_of(ALIGN)
}

/// Graph under construction over a subset (`rows`) of the vector set. Local
/// slot `i` is global row `rows[i]`. Per-node locks let many threads insert
/// at once; no thread ever holds two locks, so there is no deadlock.
struct MemGraph<'a> {
    dim: usize,
    vectors: &'a [f32],
    rows: &'a [u32],
    adj: Vec<RwLock<Vec<u32>>>,
}

impl MemGraph<'_> {
    fn vector(&self, local: u32) -> &[f32] {
        let g = self.rows[local as usize] as usize * self.dim;
        &self.vectors[g..g + self.dim]
    }
}

impl Walkable for MemGraph<'_> {
    fn neighbors_into(&self, slot: u32, out: &mut Vec<u32>) {
        out.extend_from_slice(&self.adj[slot as usize].read().unwrap());
    }
}

/// Greedy best-first walk scored by `dist`. Returns every expanded node as
/// `(distance, slot)`, closest first. The candidate list is capped at `beam`.
///
/// Up to `width` of the best unexpanded candidates are expanded per step,
/// after one `prefetch` call for all of them. With `width == 1` this is the
/// textbook walk; larger widths trade a few extra expansions for overlapping
/// disk reads.
fn greedy<G: Walkable>(g: &G, start: u32, beam: usize, width: usize, dist: impl Fn(u32) -> f32) -> Vec<(f32, u32)> {
    // (dist, slot, expanded), kept sorted ascending by dist.
    let mut cand: Vec<(f32, u32, bool)> = vec![(dist(start), start, false)];
    let mut seen: HashSet<u32> = HashSet::from([start]);
    let mut out = Vec::new();
    let mut nbrs = Vec::new();
    let mut picks: Vec<u32> = Vec::with_capacity(width.max(1));
    loop {
        picks.clear();
        for c in cand.iter_mut().filter(|c| !c.2).take(width.max(1)) {
            c.2 = true;
            out.push((c.0, c.1));
            picks.push(c.1);
        }
        if picks.is_empty() {
            break;
        }
        g.prefetch(&picks);
        for &s in &picks {
            nbrs.clear();
            g.neighbors_into(s, &mut nbrs);
            for &nb in &nbrs {
                if !seen.insert(nb) {
                    continue;
                }
                let nd = dist(nb);
                if cand.len() >= beam && nd >= cand[beam - 1].0 {
                    continue;
                }
                let pos = cand.partition_point(|c| c.0 <= nd);
                cand.insert(pos, (nd, nb, false));
                cand.truncate(beam);
            }
        }
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    out
}

/// RobustPrune: pick the closest candidate, then drop every candidate that is
/// closer to the pick than to `p` by a factor of `alpha`. Repeat until R edges.
/// `vec` resolves a slot (in whatever space `cands` use) to its vector.
fn prune<'a>(vec: &dyn Fn(u32) -> &'a [f32], metric: Metric, p: u32, mut cands: Vec<(f32, u32)>, r: usize, alpha: f32) -> Vec<u32> {
    cands.retain(|&(_, s)| s != p);
    cands.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    cands.dedup_by_key(|c| c.1);
    let mut out = Vec::with_capacity(r);
    while let Some(&(_, best)) = cands.first() {
        out.push(best);
        if out.len() == r {
            break;
        }
        let bv = vec(best);
        cands.retain(|&(d_pq, s)| s != best && alpha * distance(metric, bv, vec(s)) > d_pq);
    }
    out
}

/// Row closest to the mean of `rows`.
fn medoid(vectors: &[f32], dim: usize, metric: Metric, rows: &[u32]) -> u32 {
    let mut centroid = vec![0f32; dim];
    for &row in rows {
        let v = &vectors[row as usize * dim..(row as usize + 1) * dim];
        for (c, x) in centroid.iter_mut().zip(v) {
            *c += x;
        }
    }
    centroid.iter_mut().for_each(|c| *c /= rows.len() as f32);
    let d = |i: usize| distance(metric, &centroid, &vectors[rows[i] as usize * dim..(rows[i] as usize + 1) * dim]);
    (0..rows.len()).min_by(|&a, &b| d(a).total_cmp(&d(b))).unwrap() as u32
}

/// Vamana over the subset `rows`. Returns each local node's edges as local
/// slots. This is the whole algorithm for sets up to `shard_size`.
fn build_shard(vectors: &[f32], dim: usize, metric: Metric, rows: &[u32], params: GraphParams, seed: u64) -> Vec<Vec<u32>> {
    let n = rows.len();
    let r = params.max_degree as usize;
    let l = (params.build_beam as usize).max(r);
    let mut rng = XorShift(seed | 1);
    let mut g = MemGraph { dim, vectors, rows, adj: (0..n).map(|_| RwLock::new(Vec::new())).collect() };
    if n == 0 {
        return Vec::new();
    }
    // Random initial graph so every node is reachable from the start.
    for a in g.adj.iter_mut() {
        let a = a.get_mut().unwrap();
        while a.len() < r.min(n - 1) {
            let t = (rng.next() % n as u64) as u32;
            if !a.contains(&t) {
                a.push(t);
            }
        }
    }
    let entry = medoid(vectors, dim, metric, rows);
    let mut order: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        order.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
    // Each node's insertion is independent apart from the adjacency it
    // touches; per-node locks serialise only those.
    let g = &g;
    let vec = |s: u32| g.vector(s);
    order.par_iter().for_each(|&p| {
        let pv = g.vector(p);
        let mut cands = greedy(g, entry, l, 1, |s| distance(metric, pv, g.vector(s)));
        cands.extend(g.adj[p as usize].read().unwrap().iter().map(|&s| (distance(metric, pv, g.vector(s)), s)));
        let new = prune(&vec, metric, p, cands, r, params.alpha);
        *g.adj[p as usize].write().unwrap() = new.clone();
        for q in new {
            let mut aq = g.adj[q as usize].write().unwrap();
            if aq.contains(&p) {
                continue;
            }
            aq.push(p);
            if aq.len() > r {
                let qv = g.vector(q);
                let c: Vec<(f32, u32)> = aq.iter().map(|&s| (distance(metric, qv, g.vector(s)), s)).collect();
                *aq = prune(&vec, metric, q, c, r, params.alpha);
            }
        }
    });
    g.adj.iter().map(|a| a.read().unwrap().clone()).collect()
}

/// k-means on a sample, then every row's two nearest centroids. Returns the
/// shard membership lists (global rows) and, per row, its (shard, local)
/// position in each of its two shards.
fn partition(vectors: &[f32], dim: usize, metric: Metric, n: usize, k: usize, seed: u64) -> (Vec<Vec<u32>>, Vec<[u32; 4]>) {
    let mut rng = XorShift(seed | 1);
    let sample: Vec<u32> = (0..n.min(200_000)).map(|_| (rng.next() % n as u64) as u32).collect();
    let row = |i: u32| &vectors[i as usize * dim..(i as usize + 1) * dim];
    let mut cents: Vec<f32> = (0..k).flat_map(|_| row(sample[(rng.next() % sample.len() as u64) as usize]).to_vec()).collect();
    let nearest = |cents: &[f32], v: &[f32]| -> usize { (0..k).min_by(|&a, &b| distance(metric, v, &cents[a * dim..(a + 1) * dim]).total_cmp(&distance(metric, v, &cents[b * dim..(b + 1) * dim]))).unwrap() };
    for _ in 0..8 {
        let assign: Vec<usize> = sample.par_iter().map(|&i| nearest(&cents, row(i))).collect();
        let mut sums = vec![0f32; k * dim];
        let mut counts = vec![0u32; k];
        for (&a, &i) in assign.iter().zip(&sample) {
            counts[a] += 1;
            for (acc, x) in sums[a * dim..(a + 1) * dim].iter_mut().zip(row(i)) {
                *acc += x;
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                cents[c * dim..(c + 1) * dim].copy_from_slice(row(sample[(rng.next() % sample.len() as u64) as usize]));
            } else {
                for (dst, acc) in cents[c * dim..(c + 1) * dim].iter_mut().zip(&sums[c * dim..(c + 1) * dim]) {
                    *dst = acc / counts[c] as f32;
                }
            }
        }
    }
    // Two nearest centroids per row.
    let pair: Vec<(u32, u32)> = (0..n as u32)
        .into_par_iter()
        .map(|i| {
            let v = row(i);
            let (mut b0, mut b1) = ((f32::INFINITY, NONE), (f32::INFINITY, NONE));
            for c in 0..k {
                let d = distance(metric, v, &cents[c * dim..(c + 1) * dim]);
                if d < b0.0 {
                    b1 = b0;
                    b0 = (d, c as u32);
                } else if d < b1.0 {
                    b1 = (d, c as u32);
                }
            }
            (b0.1, b1.1)
        })
        .collect();
    let mut shards: Vec<Vec<u32>> = vec![Vec::new(); k];
    let mut pos = vec![[NONE; 4]; n];
    for (i, &(a, b)) in pair.iter().enumerate() {
        pos[i][0] = a;
        pos[i][1] = shards[a as usize].len() as u32;
        shards[a as usize].push(i as u32);
        if b != NONE && b != a {
            pos[i][2] = b;
            pos[i][3] = shards[b as usize].len() as u32;
            shards[b as usize].push(i as u32);
        }
    }
    (shards, pos)
}

/// Spilled shard adjacency: `rows.len() * R` u32 slots, `NONE`-padded, as
/// global rows.
struct ShardFile {
    mmap: Mmap,
    r: usize,
}

impl ShardFile {
    fn write(path: &Path, rows: &[u32], adj: &[Vec<u32>], r: usize) -> Result<ShardFile> {
        // Read + write: the mapping below needs read access on the same handle.
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path)?;
        let mut w = BufWriter::new(file);
        for a in adj {
            let mut i = 0;
            for &local in a.iter().take(r) {
                w.write_all(&rows[local as usize].to_le_bytes())?;
                i += 1;
            }
            for _ in i..r {
                w.write_all(&NONE.to_le_bytes())?;
            }
        }
        w.flush()?;
        let file = w.into_inner().map_err(|e| e.into_error())?;
        // Safety: written and flushed above, never modified again.
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(ShardFile { mmap, r })
    }
    fn edges(&self, local: u32) -> impl Iterator<Item = u32> + '_ {
        let o = local as usize * self.r * 4;
        self.mmap[o..o + self.r * 4].as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).filter(|&e| e != NONE)
    }
}

/// Build a Vamana graph over `vectors` (row-major, `n * dim`, typically a
/// memory-mapped file) and write the index to `path`. `ids[i]` is the external
/// id of row `i`. Temp files are created next to `path` and removed.
pub(crate) fn build_and_write(path: &Path, dim: usize, metric: Metric, ids: &[u64], vectors: &[f32], params: GraphParams) -> Result<()> {
    let n = ids.len();
    assert_eq!(vectors.len(), n * dim);
    let r = params.max_degree as usize;
    let all: Vec<u32> = (0..n as u32).collect();
    let entry = if n == 0 { 0 } else { medoid(vectors, dim, metric, &all) };
    let vec = |s: u32| &vectors[s as usize * dim..(s as usize + 1) * dim];

    if n <= params.shard_size.max(1) as usize {
        let adj = build_shard(vectors, dim, metric, &all, params, 0x9E3779B97F4A7C15);
        return write_index(path, dim, metric, ids, vectors, r, entry, params, &|s| adj[s as usize].clone());
    }

    // Sharded: k-means split with 2-way overlap, one Vamana per shard spilled
    // to disk, then a merge-and-prune pass per node.
    let k = n.div_ceil(params.shard_size as usize);
    let (shards, pos) = partition(vectors, dim, metric, n, k, 0xC0FFEE);
    let mut files: Vec<ShardFile> = Vec::with_capacity(k);
    let mut tmp_paths: Vec<PathBuf> = Vec::with_capacity(k);
    for (s, rows) in shards.iter().enumerate() {
        let adj = build_shard(vectors, dim, metric, rows, params, 0x5EED + s as u64);
        let p = PathBuf::from(format!("{}.shard{s}", path.display()));
        files.push(ShardFile::write(&p, rows, &adj, r)?);
        tmp_paths.push(p);
    }
    drop(shards);
    let merged = |g: u32| -> Vec<u32> {
        let p = pos[g as usize];
        let gv = vec(g);
        let mut cands: Vec<(f32, u32)> = Vec::with_capacity(2 * r);
        for (shard, local) in [(p[0], p[1]), (p[2], p[3])] {
            if shard != NONE {
                cands.extend(files[shard as usize].edges(local).map(|e| (distance(metric, gv, vec(e)), e)));
            }
        }
        prune(&vec, metric, g, cands, r, params.alpha)
    };
    let result = write_index(path, dim, metric, ids, vectors, r, entry, params, &merged);
    drop(files);
    for p in tmp_paths {
        let _ = std::fs::remove_file(p);
    }
    result
}

/// Serialise the index. `edges(slot)` yields a node's final adjacency and is
/// evaluated in parallel chunks ahead of the sequential write.
#[allow(clippy::too_many_arguments)]
fn write_index(path: &Path, dim: usize, metric: Metric, ids: &[u64], vectors: &[f32], r: usize, entry: u32, params: GraphParams, edges: &(dyn Fn(u32) -> Vec<u32> + Sync)) -> Result<()> {
    let n = ids.len();
    let pq = (params.use_pq && n >= pq::MIN_N).then(|| {
        let cb = Codebook::train(vectors, dim, pq::choose_m(dim), 0xC0DEB00C);
        let codes = cb.encode_all(vectors);
        (cb, codes)
    });
    let m = pq.as_ref().map_or(0, |(cb, _)| cb.m);

    let block = block_size(dim, r);
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(MAGIC)?;
    w.write_all(&(dim as u32).to_le_bytes())?;
    w.write_all(&[metric as u8, 0, 0, 0])?;
    w.write_all(&(n as u32).to_le_bytes())?;
    w.write_all(&(r as u32).to_le_bytes())?;
    w.write_all(&entry.to_le_bytes())?;
    w.write_all(&(block as u32).to_le_bytes())?;
    w.write_all(&(m as u32).to_le_bytes())?;
    for id in ids {
        w.write_all(&id.to_le_bytes())?;
    }
    let pad = (HEADER_LEN + n * 8).next_multiple_of(ALIGN) - (HEADER_LEN + n * 8);
    w.write_all(&vec![0u8; pad])?;
    let mut buf = vec![0u8; block];
    const CHUNK: usize = 1 << 15;
    for start in (0..n).step_by(CHUNK) {
        let end = (start + CHUNK).min(n);
        let chunk: Vec<Vec<u32>> = (start as u32..end as u32).into_par_iter().map(edges).collect();
        for (s, a) in (start..end).zip(&chunk) {
            buf.fill(0);
            buf[..dim * 4].copy_from_slice(bytemuck::cast_slice(&vectors[s * dim..(s + 1) * dim]));
            let a = &a[..a.len().min(r)];
            buf[dim * 4..dim * 4 + 4].copy_from_slice(&(a.len() as u32).to_le_bytes());
            buf[dim * 4 + 4..dim * 4 + 4 + a.len() * 4].copy_from_slice(bytemuck::cast_slice(a));
            w.write_all(&buf)?;
        }
    }
    if let Some((cb, codes)) = &pq {
        w.write_all(bytemuck::cast_slice(&cb.centroids))?;
        w.write_all(codes)?;
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
