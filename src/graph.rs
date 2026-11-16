//! Disk-resident Vamana graph index (the DiskANN layout).
//!
//! One node per fixed-size block: `vector f32*dim | count u32 | neighbors u32*R`.
//! The file is memory-mapped, so a query touches only the blocks it walks and
//! the OS page cache holds the working set.
//!
//! For indexes of at least `pq::MIN_N` nodes the file also carries a product
//! quantization codebook and one `m`-byte code per node. The codes are loaded
//! into RAM on open, and the walk scores neighbours from them, so expanding a
//! node costs one block read (its adjacency) instead of `R` reads (each
//! neighbour's vector). Full-precision vectors are read only to re-rank the
//! final candidates.
//!
//! Building from scratch: up to `GraphParams::shard_size` vectors are indexed
//! in one pass. Larger sets are split by k-means into overlapping shards (each
//! vector in its two nearest), each shard gets its own Vamana graph in RAM,
//! the shard adjacencies are spilled to temp files, and a final pass merges
//! each node's edge lists and prunes them to R.
//!
//! Updating in place (`update_and_write`): the existing graph's adjacency is
//! loaded, deleted nodes are consolidated out (their neighbours inherit their
//! edges and re-prune, as in FreshDiskANN), new vectors are inserted with the
//! same greedy-search-and-prune step the build uses, and a fresh file is
//! written without the deleted slots. Cost scales with the change, not the
//! store.
//!
//! File: `SANAIDX2 | dim u32 | metric u8 | reader_level u8 | pad[2] | n u32 |
//! r u32 | entry u32 | block u32 | m u32 | ids u64*n | pad to 64 | blocks |
//! codebook f32*(m*256*dim/m) | codes u8*(n*m)`.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;

use parking_lot::RwLock;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use rayon::prelude::*;

use std::sync::Arc;

use crate::attrs::{self, Attrs, Pred};
use crate::index::distance;
use crate::pq::{self, Codebook};
use crate::text::{self, TextIndex};
use crate::{Error, Filter, Metadata, Metric, Result};

/// Metadata lookup by record id, for writing attribute columns.
pub(crate) type MetaFn<'a> = dyn Fn(u64) -> Option<Arc<Metadata>> + Sync + 'a;

const MAGIC: &[u8; 8] = b"SANAIDX2";
/// Footer closing an index written by 0.2.0 or later:
/// `attrs_table_off u64 | idtab_off u64 | reserved u64 | "SANAFTR2"`. The
/// id table is `n` ids sorted ascending (`u64` each) followed by their slots
/// (`u32` each), so `slot(id)` is a binary search on the mapping.
const FOOTER_MAGIC: &[u8; 8] = b"SANAFTR2";
const FOOTER_LEN: usize = 32;
const HEADER_LEN: usize = 8 + 4 + 1 + 3 + 4 + 4 + 4 + 4 + 4;
const ALIGN: usize = 64;
/// Reader level of the index file (header byte 13), independent of the log's.
const IDX_READER_LEVEL: u8 = 1;
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
    /// Also the largest graph that is updated in place rather than rebuilt.
    pub shard_size: u32,
    /// How vectors are stored in the index blocks. The log always holds
    /// `f32`, so records since the last checkpoint are exact either way.
    pub storage: Storage,
    /// During insertion an adjacency list may grow to `prune_slack * R`
    /// before it is pruned back to R. 1.0 prunes on every overflow; 1.3 (the
    /// DiskANN default) cuts prune calls sharply on data with hub nodes.
    pub prune_slack: f32,
}

impl Default for GraphParams {
    fn default() -> Self {
        GraphParams {
            max_degree: 32,
            build_beam: 100,
            alpha: 1.2,
            use_pq: true,
            shard_size: 1_000_000,
            prune_slack: 1.3,
            storage: Storage::F32,
        }
    }
}

/// Vector storage in the index file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Storage {
    /// Full precision: `dim * 4` bytes per record.
    #[default]
    F32 = 0,
    /// 8-bit scalar quantisation with a per-record min and scale: `dim + 8`
    /// bytes per record, about a quarter of the index size for the vectors
    /// and so about a quarter of the disk reads per hop. Exact re-ranking
    /// and `get` use the dequantised vector; recall drops by a fraction of a
    /// percent on typical embeddings (measured in the tests). Builds that
    /// update an index in place first expand the stored vectors to `f32` in
    /// a temp file.
    I8 = 1,
}

impl Storage {
    fn from_u8(b: u8) -> Option<Storage> {
        match b {
            0 => Some(Storage::F32),
            1 => Some(Storage::I8),
            _ => None,
        }
    }
    /// Bytes of a block taken by the vector (and, for `I8`, its min and
    /// scale), padded to 4.
    fn vector_bytes(self, dim: usize) -> usize {
        match self {
            Storage::F32 => dim * 4,
            Storage::I8 => dim.next_multiple_of(4) + 8,
        }
    }
}

/// Anything the greedy walk can traverse: the in-RAM graph under construction
/// or the memory-mapped file.
trait Walkable: Sync {
    /// Number of slots, for sizing the visited bitmap.
    fn len(&self) -> usize;
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
    storage: Storage,
    n: usize,
    /// Offset of the sorted id table, when the file has one.
    idtab: Option<usize>,
    /// Built on open for files without an id table (written before 0.2.0).
    slot_of: Option<HashMap<u64, u32>>,
    pq: Option<(Codebook, Vec<u8>)>,
    /// Attribute columns, when the file has them (written since 0.1.4).
    attrs: Option<Attrs>,
    /// Inverted index over the text field, when the store has one.
    text: Option<TextIndex>,
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
        let h = mmap
            .get(..HEADER_LEN)
            .ok_or(Error::Corrupt("index shorter than header"))?;
        if &h[..8] != MAGIC {
            return Err(Error::Corrupt("index bad magic"));
        }
        if h[13] > IDX_READER_LEVEL {
            return Err(Error::NeedsNewerVersion {
                file_level: h[13],
                supported: IDX_READER_LEVEL,
            });
        }
        let u32_at = |o: usize| u32::from_le_bytes(h[o..o + 4].try_into().unwrap());
        let (fdim, fmetric) = (
            u32_at(8),
            Metric::from_u8(h[12]).ok_or(Error::Corrupt("index unknown metric"))?,
        );
        if fdim != dim || fmetric != metric {
            return Err(Error::HeaderMismatch {
                on_disk: (fdim, fmetric),
                requested: (dim, metric),
            });
        }
        let dim = dim as usize;
        let storage =
            Storage::from_u8(h[14]).ok_or(Error::Corrupt("index unknown vector storage"))?;
        let (n, r, entry, block, m) = (
            u32_at(16) as usize,
            u32_at(20) as usize,
            u32_at(24),
            u32_at(28) as usize,
            u32_at(32) as usize,
        );
        let ids_off = HEADER_LEN;
        let blocks_off = (ids_off + n * 8).next_multiple_of(ALIGN);
        let pq_off = blocks_off + n * block;
        let pq_len = if m == 0 {
            0
        } else {
            m * pq::K * (dim / m) * 4 + n * m
        };
        if block != block_size(dim, r, storage)
            || (m != 0 && !dim.is_multiple_of(m))
            || mmap.len() < pq_off + pq_len
            || (n > 0 && entry as usize >= n)
        {
            return Err(Error::Corrupt("index geometry inconsistent"));
        }
        let pq = (m != 0).then(|| {
            let cb_len = m * pq::K * (dim / m) * 4;
            let centroids = mmap[pq_off..pq_off + cb_len]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let codes = mmap[pq_off + cb_len..pq_off + pq_len].to_vec();
            (Codebook::new(dim, m, centroids), codes)
        });
        // Optional sections after the fixed ones: a 0.2.0 footer names the
        // attribute table and the id table; a 0.1.4 trailer names only the
        // attribute table; anything else is a 0.1.3 file with neither.
        let fixed_end = pq_off + pq_len;
        let tail = &mmap[fixed_end..];
        let (attrs_off, idtab, text_off) = if tail.len() >= FOOTER_LEN
            && &mmap[mmap.len() - 8..] == FOOTER_MAGIC
        {
            let f = mmap.len() - FOOTER_LEN;
            let at =
                |o: usize| u64::from_le_bytes(mmap[f + o..f + o + 8].try_into().unwrap()) as usize;
            let (a, t, x) = (at(0), at(8), at(16));
            if t < fixed_end || t + n * 12 > f {
                return Err(Error::Corrupt("index id table inconsistent"));
            }
            (Some(a), Some(t), (x != 0).then_some(x))
        } else if tail.len() >= attrs::LEGACY_TRAILER_LEN
            && &mmap[mmap.len() - 8..] == attrs::LEGACY_TRAILER_MAGIC
        {
            let o = mmap.len() - attrs::LEGACY_TRAILER_LEN;
            (
                Some(u64::from_le_bytes(mmap[o..o + 8].try_into().unwrap()) as usize),
                None,
                None,
            )
        } else {
            (None, None, None)
        };
        let text = text_off
            .map(|x| TextIndex::parse(&mmap, fixed_end, x, n))
            .transpose()?;
        let attrs = attrs_off
            .map(|a| Attrs::parse(&mmap, fixed_end, a, n))
            .transpose()?;
        let slot_of = idtab.is_none().then(|| {
            (0..n)
                .map(|s| {
                    (
                        u64::from_le_bytes(
                            mmap[ids_off + s * 8..ids_off + s * 8 + 8]
                                .try_into()
                                .unwrap(),
                        ),
                        s as u32,
                    )
                })
                .collect()
        });
        Ok(Some(GraphIndex {
            mmap,
            dim,
            metric,
            r,
            entry,
            block,
            blocks_off,
            storage,
            n,
            idtab,
            slot_of,
            pq,
            attrs,
            text,
        }))
    }

    pub fn len(&self) -> usize {
        self.n
    }

    /// Number of PQ subspaces, 0 when the walk uses full vectors.
    pub fn pq_subspaces(&self) -> usize {
        self.pq.as_ref().map_or(0, |(cb, _)| cb.m)
    }

    pub fn slot(&self, id: u64) -> Option<u32> {
        if let Some(map) = &self.slot_of {
            return map.get(&id).copied();
        }
        let t = self.idtab?;
        let n = self.n;
        let id_at =
            |i: usize| u64::from_le_bytes(self.mmap[t + i * 8..t + i * 8 + 8].try_into().unwrap());
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match id_at(mid).cmp(&id) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => {
                    let o = t + n * 8 + mid * 4;
                    let slot = u32::from_le_bytes(self.mmap[o..o + 4].try_into().unwrap());
                    return ((slot as usize) < n).then_some(slot);
                }
            }
        }
        None
    }

    pub fn id(&self, slot: u32) -> u64 {
        let o = HEADER_LEN + slot as usize * 8;
        u64::from_le_bytes(self.mmap[o..o + 8].try_into().unwrap())
    }

    /// The stored vector: borrowed from the mapping for `F32`, dequantised
    /// into a fresh buffer for `I8`.
    pub fn vector(&self, slot: u32) -> std::borrow::Cow<'_, [f32]> {
        let o = self.blocks_off + slot as usize * self.block;
        match self.storage {
            Storage::F32 => {
                std::borrow::Cow::Borrowed(bytemuck::cast_slice(&self.mmap[o..o + self.dim * 4]))
            }
            Storage::I8 => {
                let codes = &self.mmap[o..o + self.dim];
                let p = o + self.dim.next_multiple_of(4);
                let min = f32::from_le_bytes(self.mmap[p..p + 4].try_into().unwrap());
                let scale = f32::from_le_bytes(self.mmap[p + 4..p + 8].try_into().unwrap());
                let mut out = vec![0f32; self.dim];
                crate::quant::dequantize(codes, min, scale, &mut out);
                std::borrow::Cow::Owned(out)
            }
        }
    }

    pub fn storage(&self) -> Storage {
        self.storage
    }

    fn neighbors(&self, slot: u32) -> &[u32] {
        let o = self.blocks_off + slot as usize * self.block + self.storage.vector_bytes(self.dim);
        let count = u32::from_le_bytes(self.mmap[o..o + 4].try_into().unwrap()) as usize;
        bytemuck::cast_slice(&self.mmap[o + 4..o + 4 + count.min(self.r) * 4])
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
    ///
    /// With `filtered_walk`, `accept` also steers the walk: only accepted
    /// nodes occupy result slots, while rejected nodes are still expanded for
    /// navigation. Without it the walk is blind to the filter and `accept`
    /// is applied to the visited set afterwards (the right choice when
    /// `accept` is expensive, such as a Python callable).
    #[allow(clippy::too_many_arguments)]
    pub fn search(
        &self,
        q: &[f32],
        k: usize,
        beam: usize,
        max_beam: usize,
        width: usize,
        accept: impl Fn(u32) -> bool,
        filtered_walk: bool,
    ) -> Vec<(f32, u32)> {
        if self.n == 0 || k == 0 {
            return Vec::new();
        }
        let mut beam = beam.max(k);
        let cap = max_beam.max(beam).min(self.n);
        let lut = self.pq.as_ref().map(|(cb, _)| cb.lut(q, self.metric));
        let pass: Option<&dyn Fn(u32) -> bool> =
            filtered_walk.then_some(&accept as &dyn Fn(u32) -> bool);
        loop {
            let mut visited = match (&self.pq, &lut) {
                (Some((cb, codes)), Some(lut)) => {
                    let m = cb.m;
                    greedy(
                        self,
                        self.entry,
                        beam,
                        width,
                        |s| pq::lut_distance(lut, &codes[s as usize * m..(s as usize + 1) * m]),
                        pass,
                    )
                }
                _ => greedy(
                    self,
                    self.entry,
                    beam,
                    width,
                    |s| distance(self.metric, q, &self.vector(s)),
                    pass,
                ),
            };
            let exhausted = visited.len() >= self.n;
            visited.retain(|&(_, s)| accept(s));
            if visited.len() >= k || beam >= cap || exhausted {
                if self.pq.is_some() {
                    // Re-rank the best `beam` approximate candidates exactly.
                    visited.truncate(beam.max(k));
                    for c in visited.iter_mut() {
                        c.0 = distance(self.metric, q, &self.vector(c.1));
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

impl GraphIndex {
    /// Exact k-NN over an explicit slot list: the path for filters that
    /// leave few enough candidates that reading each one beats a walk.
    pub fn scan(&self, q: &[f32], k: usize, slots: impl Iterator<Item = u32>) -> Vec<(f32, u32)> {
        let mut out: Vec<(f32, u32)> = slots
            .map(|s| (distance(self.metric, q, &self.vector(s)), s))
            .collect();
        out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        out.truncate(k);
        out
    }

    /// Compile a filter against this index's attribute columns; `None` when
    /// the file has none (built before 0.1.4), so the caller falls back to
    /// the metadata map.
    pub fn prepare(&self, f: &Filter, text_field: Option<&str>) -> Option<Pred> {
        let tx = self.text.as_ref().zip(text_field);
        self.attrs.as_ref().map(|a| a.prepare(&self.mmap, f, tx))
    }
    pub fn pass(&self, p: &Pred, slot: u32) -> bool {
        self.attrs
            .as_ref()
            .is_some_and(|a| a.pass(&self.mmap, p, slot, self.text.as_ref()))
    }
    pub fn estimate(&self, p: &Pred) -> usize {
        self.attrs
            .as_ref()
            .map_or(self.n, |a| a.estimate(&self.mmap, p, self.text.as_ref()))
    }
    pub fn candidates(&self, p: &Pred) -> Option<Vec<u32>> {
        self.attrs
            .as_ref()?
            .candidates(&self.mmap, p, self.text.as_ref())
    }

    /// A slot's metadata decoded from the columns; `None` without columns.
    pub fn metadata(&self, slot: u32) -> Option<Metadata> {
        self.attrs.as_ref().map(|a| a.record(&self.mmap, slot))
    }

    /// The text index, if the store has a text field and has checkpointed.
    pub fn text(&self) -> Option<&TextIndex> {
        self.text.as_ref()
    }
    pub fn bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Every metadata key the columns hold, with its type.
    pub fn attr_kinds(&self) -> impl Iterator<Item = (&str, crate::Kind)> {
        self.attrs.iter().flat_map(|a| a.kinds())
    }
}

impl Walkable for GraphIndex {
    fn len(&self) -> usize {
        self.n
    }
    fn neighbors_into(&self, slot: u32, out: &mut Vec<u32>) {
        out.extend_from_slice(self.neighbors(slot));
    }
    #[cfg(unix)]
    fn prefetch(&self, slots: &[u32]) {
        if slots.len() < 2 {
            return;
        }
        // Always advise when the caller asked for a width above 1. A probe
        // gate (skip madvise once recent blocks read as resident) saved 15%
        // of a warm query but cost 45% on a cold one in a purge-then-query
        // A/B on the 5M index, because hub blocks are resident long before
        // the rest of a cold file. Warm-only deployments set `io_width(1)`.
        for &s in slots {
            let o = self.blocks_off + s as usize * self.block;
            // Advisory: the kernel starts the read; a failure only costs the hint.
            let _ = self
                .mmap
                .advise_range(memmap2::Advice::WillNeed, o, self.block);
        }
    }
    // Windows has no madvise; the walk still works, one page fault at a time.
}

fn block_size(dim: usize, r: usize, storage: Storage) -> usize {
    (storage.vector_bytes(dim) + 4 + r * 4).next_multiple_of(ALIGN)
}

/// Vector lookup by slot, shared by the build and update paths.
type VecFn<'a> = dyn Fn(u32) -> &'a [f32] + Sync + 'a;

/// Graph under construction or update. Per-node locks let many threads
/// insert at once; no thread ever holds two locks, so there is no deadlock.
struct MemGraph<'a> {
    vec: &'a VecFn<'a>,
    adj: Vec<RwLock<Vec<u32>>>,
}

impl<'a> MemGraph<'a> {
    fn vector(&self, slot: u32) -> &'a [f32] {
        (self.vec)(slot)
    }

    /// Vamana insertion of node `p`: walk from `entry`, prune the visited set
    /// to R, link back from each chosen neighbour and re-prune it once it
    /// overflows the slack. `live` filters candidates (a deleted node may be
    /// walked through but never linked to).
    ///
    /// The slack matters: hub nodes in skewed data (word embeddings, for
    /// instance) receive reverse edges constantly, and pruning on every
    /// overflow made prune, not distance, the dominant build cost. Lists are
    /// allowed to grow to `SLACK * R` and pruned back to R then, as DiskANN
    /// does; `finish` prunes whatever is still over R at the end.
    fn insert(
        &self,
        p: u32,
        entry: u32,
        metric: Metric,
        params: GraphParams,
        live: &(dyn Fn(u32) -> bool + Sync),
    ) {
        let r = params.max_degree as usize;
        let slack = ((r as f32) * params.prune_slack.max(1.0)).ceil() as usize;
        let l = (params.build_beam as usize).max(r);
        let pv = self.vector(p);
        let mut cands = greedy(
            self,
            entry,
            l,
            1,
            |s| distance(metric, pv, self.vector(s)),
            None,
        );
        cands.extend(
            self.adj[p as usize]
                .read()
                .iter()
                .map(|&s| (distance(metric, pv, self.vector(s)), s)),
        );
        cands.retain(|&(_, s)| live(s));
        let new = prune(self.vec, metric, p, cands, r, params.alpha);
        *self.adj[p as usize].write() = new.clone();
        for q in new {
            let mut aq = self.adj[q as usize].write();
            if aq.contains(&p) {
                continue;
            }
            aq.push(p);
            if aq.len() > slack {
                let qv = self.vector(q);
                let c: Vec<(f32, u32)> = aq
                    .iter()
                    .map(|&s| (distance(metric, qv, self.vector(s)), s))
                    .collect();
                *aq = prune(self.vec, metric, q, c, r, params.alpha);
            }
        }
    }

    /// Prune every list still longer than R (they were allowed slack during
    /// insertion) so the written graph has at most R edges per node.
    fn finish(&self, metric: Metric, params: GraphParams, live: &(dyn Fn(u32) -> bool + Sync)) {
        let r = params.max_degree as usize;
        (0..self.adj.len() as u32).into_par_iter().for_each(|p| {
            let over = self.adj[p as usize].read().len() > r;
            if !over || !live(p) {
                return;
            }
            let pv = self.vector(p);
            let c: Vec<(f32, u32)> = self.adj[p as usize]
                .read()
                .iter()
                .map(|&s| (distance(metric, pv, self.vector(s)), s))
                .collect();
            *self.adj[p as usize].write() = prune(self.vec, metric, p, c, r, params.alpha);
        });
    }
}

impl Walkable for MemGraph<'_> {
    fn len(&self) -> usize {
        self.adj.len()
    }
    fn neighbors_into(&self, slot: u32, out: &mut Vec<u32>) {
        out.extend_from_slice(&self.adj[slot as usize].read());
    }
}

/// Greedy best-first walk scored by `dist`. Returns every expanded node as
/// `(distance, slot)`, closest first. The candidate list is capped at `beam`.
///
/// Up to `width` of the best unexpanded candidates are expanded per step,
/// after one `prefetch` call for all of them. With `width == 1` this is the
/// textbook walk; larger widths trade a few extra expansions for overlapping
/// disk reads.
fn greedy<G: Walkable>(
    g: &G,
    start: u32,
    beam: usize,
    width: usize,
    dist: impl Fn(u32) -> f32,
    pass: Option<&dyn Fn(u32) -> bool>,
) -> Vec<(f32, u32)> {
    // Scratch buffers are thread-local and reused: a build runs one walk per
    // inserted node, and fresh allocations were a large share of its time.
    thread_local! {
        static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
    }
    SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        let Scratch {
            cand,
            top,
            seen,
            nbrs,
            picks,
        } = &mut *s;
        cand.clear();
        top.clear();
        seen.reset(g.len());
        let c = Cand {
            d: dist(start),
            s: start,
        };
        seen.insert(start);
        cand.push(Reverse(c));
        if pass.is_none_or(|p| p(start)) {
            top.push(c);
        }
        greedy_in(g, beam, width, &dist, pass, cand, top, seen, nbrs, picks)
    })
}

/// `(distance, slot)` with a total order, so it can sit in a heap.
#[derive(Clone, Copy, PartialEq)]
struct Cand {
    d: f32,
    s: u32,
}
impl Eq for Cand {}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.d.total_cmp(&o.d).then(self.s.cmp(&o.s))
    }
}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Visited set as a bitmap over slots. Clearing touches only the words a walk
/// set, so a reset costs the walk's size, not the graph's, and the bitmap is
/// n/8 bytes per thread: 125 KB for 1M nodes against the hash set it replaced,
/// whose insert was 14% of a warm query.
#[derive(Default)]
struct Visited {
    words: Vec<u64>,
    touched: Vec<u32>,
}

impl Visited {
    fn reset(&mut self, n: usize) {
        for &w in &self.touched {
            self.words[w as usize] = 0;
        }
        self.touched.clear();
        let need = n.div_ceil(64);
        if self.words.len() < need {
            self.words.resize(need, 0);
        }
    }

    /// True when `s` was not yet in the set.
    #[inline]
    fn insert(&mut self, s: u32) -> bool {
        let (w, bit) = ((s >> 6) as usize, 1u64 << (s & 63));
        if self.words[w] & bit != 0 {
            return false;
        }
        self.words[w] |= bit;
        self.touched.push(w as u32);
        true
    }
}

#[derive(Default)]
struct Scratch {
    /// Unexpanded candidates, closest first.
    cand: BinaryHeap<Reverse<Cand>>,
    /// The `beam` closest nodes seen so far, farthest first, so the bound is
    /// `top.peek()`.
    top: BinaryHeap<Cand>,
    seen: Visited,
    nbrs: Vec<u32>,
    picks: Vec<u32>,
}

/// The textbook best-first walk with two heaps: expand the closest unexpanded
/// candidate while it is within the `beam` best seen; a neighbour enters the
/// heaps only if it beats the current `beam`-th best. Replaces a sorted `Vec`
/// whose shifting inserts were 18% of a warm query.
#[allow(clippy::too_many_arguments)]
fn greedy_in<G: Walkable>(
    g: &G,
    beam: usize,
    width: usize,
    dist: &impl Fn(u32) -> f32,
    pass: Option<&dyn Fn(u32) -> bool>,
    cand: &mut BinaryHeap<Reverse<Cand>>,
    top: &mut BinaryHeap<Cand>,
    seen: &mut Visited,
    nbrs: &mut Vec<u32>,
    picks: &mut Vec<u32>,
) -> Vec<(f32, u32)> {
    let mut out = Vec::with_capacity(beam * 2);
    // A filtered walk expands rejected nodes to get past them, so its length
    // is no longer bounded by `beam`; cap it so a filter that matches almost
    // nothing cannot turn into a scan of the graph (the caller has a cheaper
    // path for that case).
    let max_expand = if pass.is_some() {
        beam * 64
    } else {
        usize::MAX
    };
    loop {
        picks.clear();
        if out.len() >= max_expand {
            break;
        }
        while picks.len() < width.max(1) {
            let Some(&Reverse(c)) = cand.peek() else {
                break;
            };
            if top.len() >= beam && c.d > top.peek().map_or(f32::INFINITY, |t| t.d) {
                break;
            }
            cand.pop();
            out.push((c.d, c.s));
            picks.push(c.s);
        }
        if picks.is_empty() {
            break;
        }
        g.prefetch(picks);
        for &s in picks.iter() {
            nbrs.clear();
            g.neighbors_into(s, nbrs);
            for &nb in nbrs.iter() {
                // A damaged index may name a slot past the end; skip it
                // rather than read out of bounds.
                if nb as usize >= g.len() || !seen.insert(nb) {
                    continue;
                }
                let nd = dist(nb);
                if top.len() >= beam && nd >= top.peek().map_or(f32::INFINITY, |t| t.d) {
                    continue;
                }
                let c = Cand { d: nd, s: nb };
                cand.push(Reverse(c));
                if pass.is_none_or(|p| p(nb)) {
                    top.push(c);
                    if top.len() > beam {
                        top.pop();
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    out
}

/// RobustPrune: pick the closest candidate, then drop every candidate that is
/// closer to the pick than to `p` by a factor of `alpha`. Repeat until R edges.
fn prune(
    vec: &VecFn<'_>,
    metric: Metric,
    p: u32,
    mut cands: Vec<(f32, u32)>,
    r: usize,
    alpha: f32,
) -> Vec<u32> {
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

/// Slot among `slots` closest to their mean.
fn medoid(
    vec: &VecFn<'_>,
    dim: usize,
    metric: Metric,
    slots: impl Iterator<Item = u32> + Clone,
) -> u32 {
    let mut centroid = vec![0f32; dim];
    let mut n = 0usize;
    for s in slots.clone() {
        for (c, x) in centroid.iter_mut().zip(vec(s)) {
            *c += x;
        }
        n += 1;
    }
    centroid.iter_mut().for_each(|c| *c /= n.max(1) as f32);
    slots
        .min_by(|&a, &b| {
            distance(metric, &centroid, vec(a)).total_cmp(&distance(metric, &centroid, vec(b)))
        })
        .unwrap_or(0)
}

fn shuffled(n: usize, rng: &mut XorShift) -> Vec<u32> {
    let mut order: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        order.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
    order
}

/// Vamana over the subset `rows` (global row ids). Returns each local node's
/// edges as local slots. This is the whole algorithm for sets up to
/// `shard_size`.
fn build_shard(
    vectors: &[f32],
    dim: usize,
    metric: Metric,
    rows: &[u32],
    params: GraphParams,
    seed: u64,
) -> Vec<Vec<u32>> {
    let n = rows.len();
    if n == 0 {
        return Vec::new();
    }
    let r = params.max_degree as usize;
    let mut rng = XorShift(seed | 1);
    let vec = move |local: u32| -> &[f32] {
        let g = rows[local as usize] as usize * dim;
        &vectors[g..g + dim]
    };
    let mut g = MemGraph {
        vec: &vec,
        adj: (0..n).map(|_| RwLock::new(Vec::new())).collect(),
    };
    // Random initial graph so every node is reachable from the start.
    for a in g.adj.iter_mut() {
        let a = a.get_mut();
        while a.len() < r.min(n - 1) {
            let t = (rng.next() % n as u64) as u32;
            if !a.contains(&t) {
                a.push(t);
            }
        }
    }
    let entry = medoid(&vec, dim, metric, 0..n as u32);
    let order = shuffled(n, &mut rng);
    let g = &g;
    order
        .par_iter()
        .for_each(|&p| g.insert(p, entry, metric, params, &|_| true));
    g.finish(metric, params, &|_| true);
    g.adj.iter().map(|a| a.read().clone()).collect()
}

/// k-means on a sample, then every row's two nearest centroids. Returns the
/// shard membership lists (global rows) and, per row, its (shard, local)
/// position in each of its two shards.
fn partition(
    vectors: &[f32],
    dim: usize,
    metric: Metric,
    n: usize,
    k: usize,
    seed: u64,
) -> (Vec<Vec<u32>>, Vec<[u32; 4]>) {
    let mut rng = XorShift(seed | 1);
    let sample: Vec<u32> = (0..n.min(200_000))
        .map(|_| (rng.next() % n as u64) as u32)
        .collect();
    let row = |i: u32| &vectors[i as usize * dim..(i as usize + 1) * dim];
    let cents = crate::kmeans::emeans(&|i| row(sample[i]), sample.len(), dim, k, 3, seed);
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
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
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
        self.mmap[o..o + self.r * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .filter(|&e| e != NONE)
    }
}

/// Build a Vamana graph over `vectors` (row-major, `n * dim`, typically a
/// memory-mapped file) and write the index to `path`. `ids[i]` is the external
/// id of row `i`. Temp files are created next to `path` and removed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_and_write(
    path: &Path,
    dim: usize,
    metric: Metric,
    ids: &[u64],
    vectors: &[f32],
    params: GraphParams,
    meta_of: &MetaFn<'_>,
    text_field: Option<&str>,
) -> Result<()> {
    let n = ids.len();
    assert_eq!(vectors.len(), n * dim);
    let r = params.max_degree as usize;
    let vec = move |s: u32| -> &[f32] { &vectors[s as usize * dim..(s as usize + 1) * dim] };
    let entry = if n == 0 {
        0
    } else {
        medoid(&vec, dim, metric, 0..n as u32)
    };
    let pq = (params.use_pq && n >= pq::MIN_N).then(|| {
        let cb = Codebook::train(vectors, dim, pq::choose_m(dim), 0xC0DEB00C);
        let codes = cb.encode_all(vectors);
        (cb, codes)
    });
    let pq_ref = pq.as_ref().map(|(cb, codes)| (cb, codes.as_slice()));

    if n <= params.shard_size.max(1) as usize {
        let all: Vec<u32> = (0..n as u32).collect();
        let adj = build_shard(vectors, dim, metric, &all, params, 0x9E3779B97F4A7C15);
        return write_index(
            path,
            dim,
            metric,
            ids,
            &vec,
            r,
            entry,
            pq_ref,
            &|s| adj[s as usize].clone(),
            meta_of,
            params.storage,
            text_field,
        );
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
                cands.extend(
                    files[shard as usize]
                        .edges(local)
                        .map(|e| (distance(metric, gv, vec(e)), e)),
                );
            }
        }
        prune(&vec, metric, g, cands, r, params.alpha)
    };
    let result = write_index(
        path,
        dim,
        metric,
        ids,
        &vec,
        r,
        entry,
        pq_ref,
        &merged,
        meta_of,
        params.storage,
        text_field,
    );
    drop(files);
    for p in tmp_paths {
        let _ = std::fs::remove_file(p);
    }
    result
}

/// Update `existing` in place: drop the `dead` slots, insert `new_vectors`
/// (row-major, one per `new_ids`), and write the result to `path`.
///
/// Slots are numbered `0..old` for the existing nodes and `old..old+new` for
/// the additions while working; the written file keeps only live nodes, in
/// that order. The existing PQ codebook is reused: old codes are copied, new
/// vectors are encoded against it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_and_write(
    path: &Path,
    existing: &GraphIndex,
    dead: &HashSet<u32>,
    new_ids: &[u64],
    new_vectors: &[f32],
    params: GraphParams,
    meta_of: &MetaFn<'_>,
    text_field: Option<&str>,
) -> Result<()> {
    let dim = existing.dim;
    let metric = existing.metric;
    let old = existing.len();
    let added = new_ids.len();
    assert_eq!(new_vectors.len(), added * dim);
    let total = old + added;
    let r = params.max_degree.max(existing.r as u32) as usize;
    // Old vectors come straight from the mapping when stored as f32; an
    // int8 index is expanded once into a temp file next to the new index.
    let expanded: Option<(memmap2::Mmap, PathBuf)> = match existing.storage {
        Storage::F32 => None,
        Storage::I8 => {
            let tmp = path.with_extension("f32.tmp");
            {
                let mut w = BufWriter::new(File::create(&tmp)?);
                for s in 0..old as u32 {
                    w.write_all(bytemuck::cast_slice(&existing.vector(s)))?;
                }
                w.flush()?;
            }
            let f = File::open(&tmp)?;
            // Safety: written above and never touched again while mapped.
            Some((unsafe { memmap2::Mmap::map(&f)? }, tmp))
        }
    };
    let old_f32: Option<&[f32]> = expanded.as_ref().map(|(m, _)| bytemuck::cast_slice(&m[..]));
    let vec = move |s: u32| -> &[f32] {
        let s = s as usize;
        if s < old {
            match old_f32 {
                Some(all) => &all[s * dim..(s + 1) * dim],
                None => match existing.vector(s as u32) {
                    std::borrow::Cow::Borrowed(v) => v,
                    std::borrow::Cow::Owned(_) => unreachable!("f32 storage borrows"),
                },
            }
        } else {
            &new_vectors[(s - old) * dim..(s - old + 1) * dim]
        }
    };
    let live = |s: u32| !dead.contains(&s);

    // 1. Load the adjacency; new nodes start empty.
    let adj: Vec<RwLock<Vec<u32>>> = (0..total as u32)
        .map(|s| {
            RwLock::new(if (s as usize) < old {
                existing.neighbors(s).to_vec()
            } else {
                Vec::new()
            })
        })
        .collect();
    let g = MemGraph { vec: &vec, adj };

    // 2. Delete consolidation (FreshDiskANN): a live node that points at a
    //    dead one inherits the dead node's live neighbours, then re-prunes.
    if !dead.is_empty() {
        (0..old as u32)
            .into_par_iter()
            .filter(|&p| live(p))
            .for_each(|p| {
                let mine: Vec<u32> = g.adj[p as usize].read().clone();
                if !mine.iter().any(|&q| dead.contains(&q)) {
                    return;
                }
                let pv = vec(p);
                let mut cands: Vec<(f32, u32)> = Vec::new();
                for &q in &mine {
                    if live(q) {
                        cands.push((distance(metric, pv, vec(q)), q));
                    } else {
                        cands.extend(
                            g.adj[q as usize]
                                .read()
                                .iter()
                                .copied()
                                .filter(|&x| live(x) && x != p)
                                .map(|x| (distance(metric, pv, vec(x)), x)),
                        );
                    }
                }
                *g.adj[p as usize].write() = prune(&vec, metric, p, cands, r, params.alpha);
            });
    }

    // 3. Insert the new nodes, walking from a live entry point.
    let entry = {
        let e = existing.entry;
        if old > 0 && live(e) {
            e
        } else {
            (0..total as u32).find(|&s| live(s)).unwrap_or(0)
        }
    };
    if added > 0 {
        let mut rng = XorShift(0xADD5 | 1);
        let order: Vec<u32> = shuffled(added, &mut rng)
            .into_iter()
            .map(|i| i + old as u32)
            .collect();
        if old == 0 || dead.len() == old {
            // Nothing live to walk from: seed like a fresh build.
            for &p in &order {
                let a = &mut *g.adj[p as usize].write();
                while a.len() < r.min(added - 1) {
                    let t = old as u32 + (rng.next() % added as u64) as u32;
                    if t != p && !a.contains(&t) {
                        a.push(t);
                    }
                }
            }
        }
        let g = &g;
        order
            .par_iter()
            .for_each(|&p| g.insert(p, entry, metric, params, &live));
        g.finish(metric, params, &live);
    }

    // 4. Compact: live old slots keep their order, new slots follow.
    let mut new_slot = vec![NONE; total];
    let mut kept: Vec<u32> = Vec::with_capacity(total - dead.len());
    for s in 0..total as u32 {
        if live(s) {
            new_slot[s as usize] = kept.len() as u32;
            kept.push(s);
        }
    }
    let ids: Vec<u64> = kept
        .iter()
        .map(|&s| {
            if (s as usize) < old {
                existing.id(s)
            } else {
                new_ids[s as usize - old]
            }
        })
        .collect();
    let kept_ref = &kept;
    let vec_new = move |ns: u32| -> &[f32] { vec(kept_ref[ns as usize]) };
    let entry_new = if kept.is_empty() {
        0
    } else {
        medoid(&vec_new, dim, metric, 0..kept.len() as u32)
    };
    let pq = existing.pq.as_ref().map(|(cb, codes)| {
        let m = cb.m;
        let fresh = cb.encode_all(new_vectors);
        let mut out = Vec::with_capacity(kept.len() * m);
        for &s in &kept {
            let s = s as usize;
            if s < old {
                out.extend_from_slice(&codes[s * m..(s + 1) * m])
            } else {
                out.extend_from_slice(&fresh[(s - old) * m..(s - old + 1) * m])
            }
        }
        (cb, out)
    });
    let pq_ref = pq.as_ref().map(|(cb, codes)| (*cb, codes.as_slice()));
    let edges = |ns: u32| -> Vec<u32> {
        let s = kept[ns as usize];
        g.adj[s as usize]
            .read()
            .iter()
            .filter(|&&q| live(q))
            .map(|&q| new_slot[q as usize])
            .take(r)
            .collect()
    };
    let result = write_index(
        path,
        dim,
        metric,
        &ids,
        &vec_new,
        r,
        entry_new,
        pq_ref,
        &edges,
        meta_of,
        params.storage,
        text_field,
    );
    if let Some((m, tmp)) = expanded {
        drop(m);
        let _ = std::fs::remove_file(tmp);
    }
    result
}

/// Serialise the index. `vec(slot)` and `edges(slot)` are in final slot
/// order; `edges` is evaluated in parallel chunks ahead of the sequential write.
#[allow(clippy::too_many_arguments)]
fn write_index(
    path: &Path,
    dim: usize,
    metric: Metric,
    ids: &[u64],
    vec: &VecFn<'_>,
    r: usize,
    entry: u32,
    pq: Option<(&Codebook, &[u8])>,
    edges: &(dyn Fn(u32) -> Vec<u32> + Sync),
    meta_of: &MetaFn<'_>,
    storage: Storage,
    text_field: Option<&str>,
) -> Result<()> {
    let n = ids.len();
    let m = pq.map_or(0, |(cb, _)| cb.m);
    let block = block_size(dim, r, storage);
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(MAGIC)?;
    w.write_all(&(dim as u32).to_le_bytes())?;
    w.write_all(&[metric as u8, IDX_READER_LEVEL, storage as u8, 0])?;
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
        let chunk: Vec<Vec<u32>> = (start as u32..end as u32)
            .into_par_iter()
            .map(edges)
            .collect();
        let vb = storage.vector_bytes(dim);
        for (s, a) in (start..end).zip(&chunk) {
            buf.fill(0);
            match storage {
                Storage::F32 => buf[..dim * 4].copy_from_slice(bytemuck::cast_slice(vec(s as u32))),
                Storage::I8 => {
                    let (min, scale) = crate::quant::quantize(vec(s as u32), &mut buf[..dim]);
                    let p = dim.next_multiple_of(4);
                    buf[p..p + 4].copy_from_slice(&min.to_le_bytes());
                    buf[p + 4..p + 8].copy_from_slice(&scale.to_le_bytes());
                }
            }
            let a = &a[..a.len().min(r)];
            buf[vb..vb + 4].copy_from_slice(&(a.len() as u32).to_le_bytes());
            buf[vb + 4..vb + 4 + a.len() * 4].copy_from_slice(bytemuck::cast_slice(a));
            w.write_all(&buf)?;
        }
    }
    let mut pos = HEADER_LEN + n * 8 + pad + n * block;
    if let Some((cb, codes)) = pq {
        debug_assert_eq!(codes.len(), n * m);
        w.write_all(bytemuck::cast_slice(&cb.centroids))?;
        w.write_all(codes)?;
        pos += cb.centroids.len() * 4 + codes.len();
    }
    // Attribute columns, 8-aligned, found by readers through their trailer.
    let aligned = pos.next_multiple_of(8);
    w.write_all(&vec![0u8; aligned - pos])?;
    let (attr_len, attrs_table) =
        attrs::write(&mut w, aligned as u64, n, &|s| meta_of(ids[s as usize]))?;
    // Sorted id table: ids ascending, then the matching slots.
    let idtab = (aligned as u64 + attr_len).next_multiple_of(8);
    w.write_all(&vec![0u8; (idtab - aligned as u64 - attr_len) as usize])?;
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by_key(|&s| ids[s as usize]);
    for &s in &order {
        w.write_all(&ids[s as usize].to_le_bytes())?;
    }
    w.write_all(bytemuck::cast_slice(&order))?;
    let after = idtab + n as u64 * 12;
    w.write_all(&vec![0u8; (after.next_multiple_of(8) - after) as usize])?;
    // Text section, when the store indexes a text field.
    let mut text_table = 0u64;
    if let Some(field) = text_field {
        let start = after.next_multiple_of(8);
        let text_of = |s: u32| {
            meta_of(ids[s as usize])
                .and_then(|m| m.get(field).and_then(|v| v.as_str().map(str::to_string)))
        };
        let (len, table) = text::write(&mut w, start, n, &text_of)?;
        text_table = table;
        let end = start + len;
        w.write_all(&vec![0u8; (end.next_multiple_of(8) - end) as usize])?;
    }
    for v in [attrs_table, idtab, text_table] {
        w.write_all(&v.to_le_bytes())?;
    }
    w.write_all(FOOTER_MAGIC)?;
    w.flush()?;
    w.get_ref().sync_all()?;
    Ok(())
}

use crate::kmeans::XorShift;
