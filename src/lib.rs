//! sanadb: a disk-first embedded vector + metadata store.
//!
//! Two files: `<path>` is a write-ahead log of inserts, deletes and metadata;
//! `<path>.idx` is a memory-mapped Vamana graph over every vector present at
//! the last `checkpoint`. Writes since then live in a small exact in-RAM
//! "delta" index. A search walks the graph and scans the delta, then merges.
//!
//! Records in the graph that were later overwritten or deleted are hidden by
//! an in-RAM tombstone set until the next checkpoint rebuilds the graph.
//!
//! `Db` is `Send + Sync`. Reads take a shared lock, writes an exclusive one,
//! and a checkpoint builds the new graph outside both so readers and writers
//! keep going while it runs.

mod graph;
mod index;
mod kmeans;
mod pq;
mod wal;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// parking_lot locks do not poison: a panic on one thread (for instance from
// a filter closure) leaves the store usable by the others.
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

pub use graph::GraphParams;
use graph::GraphIndex;
use index::Flat;
use wal::{Entry, Header, Replayed, Wal};

pub type Metadata = HashMap<String, String>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Corrupt(&'static str),
    HeaderMismatch { on_disk: (u32, Metric), requested: (u32, Metric) },
    DimMismatch { expected: u32, got: usize },
    MetaTooLarge,
    /// A vector contained NaN or infinity.
    InvalidVector,
    /// Another `Db` (in this or another process) has the store open.
    Locked,
    /// The file was written by a newer sanadb whose layout this build cannot
    /// read. Upgrade sanadb; the data is intact.
    NeedsNewerVersion { file_level: u8, supported: u8 },
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Corrupt(why) => write!(f, "corrupt file: {why}"),
            Error::HeaderMismatch { on_disk, requested } => {
                write!(f, "file has dim={} metric={:?}, opened with dim={} metric={:?}", on_disk.0, on_disk.1, requested.0, requested.1)
            }
            Error::DimMismatch { expected, got } => write!(f, "vector has {got} dims, store expects {expected}"),
            Error::MetaTooLarge => write!(f, "metadata key/value longer than 65535 bytes or more than 65535 pairs"),
            Error::InvalidVector => write!(f, "vector contains NaN or infinity, or is all zeros under the cosine metric"),
            Error::Locked => write!(f, "store is open elsewhere (lock file held)"),
            Error::NeedsNewerVersion { file_level, supported } => {
                write!(f, "file needs reader level {file_level}, this sanadb supports {supported}: upgrade sanadb")
            }
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Metric {
    /// Squared Euclidean distance. Lower is closer.
    L2 = 0,
    /// Negative dot product on vectors stored as given. Lower is closer.
    /// Meant for inner-product scores that are not cosine; graph quality is
    /// lower than with `Cosine` because inner product is not a metric.
    Dot = 1,
    /// Negative cosine similarity, so scores lie in `[-1, 1]` and lower is
    /// closer. Vectors are L2-normalised on insert and queries on search;
    /// `get` returns the normalised vector. All-zero vectors are rejected
    /// (`Error::InvalidVector`) because their cosine is undefined.
    Cosine = 2,
}

impl Metric {
    fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(Metric::L2),
            1 => Some(Metric::Dot),
            2 => Some(Metric::Cosine),
            _ => None,
        }
    }
}

/// Reject NaN/inf (they poison every distance) and normalise for cosine.
fn prepare(metric: Metric, v: &[f32]) -> Result<std::borrow::Cow<'_, [f32]>> {
    if v.iter().any(|x| !x.is_finite()) {
        return Err(Error::InvalidVector);
    }
    if metric != Metric::Cosine {
        return Ok(std::borrow::Cow::Borrowed(v));
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return Err(Error::InvalidVector);
    }
    Ok(std::borrow::Cow::Owned(v.iter().map(|x| x / norm).collect()))
}

/// Internal distances for `Cosine` are squared L2 on unit vectors; present
/// them as negative cosine similarity, which is what the metric promises.
fn present(metric: Metric, hits: &mut [Hit]) {
    if metric == Metric::Cosine {
        for h in hits {
            h.score = h.score / 2.0 - 1.0;
        }
    }
}

#[derive(Debug, Clone)]
pub struct Options {
    pub dim: u32,
    pub metric: Metric,
    /// fsync after every write. Off by default: writes reach the OS page
    /// cache immediately and survive a process crash; call `sync` for power-loss safety.
    pub sync_every_write: bool,
    /// Graph construction parameters used by `checkpoint`.
    pub graph: GraphParams,
    /// Candidate list size during graph search. Larger = higher recall, slower.
    pub search_beam: u32,
    /// Upper bound the beam may grow to when a filter leaves fewer than `k`
    /// results. Set equal to `search_beam` to disable the retry.
    pub max_search_beam: u32,
    /// Run `checkpoint` automatically once the log holds this many entries
    /// more than there are live records. 0 disables.
    pub auto_checkpoint_garbage: usize,
    /// Run `checkpoint` automatically once at least this many records sit in
    /// the delta (whose search is an exact scan) and the delta is at least as
    /// large as the indexed graph. The second condition keeps a bulk load to
    /// a geometric series of builds, at most about twice the work of one
    /// build at the end, instead of a rebuild every N records. 0 disables.
    pub auto_checkpoint_delta: usize,
    /// Graph nodes expanded per walk step, with their disk blocks prefetched
    /// together. 1 is the classic walk; 4–8 overlaps reads when the index is
    /// not in the page cache, at the cost of a few extra expansions.
    pub io_width: u32,
}

impl Options {
    pub fn new(dim: u32) -> Self {
        Options {
            dim,
            metric: Metric::L2,
            sync_every_write: false,
            graph: GraphParams::default(),
            search_beam: 64,
            max_search_beam: 4096,
            auto_checkpoint_garbage: 100_000,
            auto_checkpoint_delta: 250_000,
            io_width: 4,
        }
    }
    pub fn auto_checkpoint_delta(mut self, records: usize) -> Self {
        self.auto_checkpoint_delta = records;
        self
    }
    pub fn io_width(mut self, width: u32) -> Self {
        self.io_width = width.max(1);
        self
    }
    pub fn metric(mut self, m: Metric) -> Self {
        self.metric = m;
        self
    }
    pub fn sync_every_write(mut self, on: bool) -> Self {
        self.sync_every_write = on;
        self
    }
    pub fn graph(mut self, g: GraphParams) -> Self {
        self.graph = g;
        self
    }
    pub fn search_beam(mut self, beam: u32) -> Self {
        self.search_beam = beam;
        self
    }
    pub fn max_search_beam(mut self, beam: u32) -> Self {
        self.max_search_beam = beam;
        self
    }
    pub fn auto_checkpoint_garbage(mut self, entries: usize) -> Self {
        self.auto_checkpoint_garbage = entries;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexStats {
    /// Records in the graph, including tombstoned ones.
    pub nodes: usize,
    /// Product-quantization subspaces (bytes per node held in RAM). 0 means the
    /// walk reads full vectors from the mapped file.
    pub pq_subspaces: usize,
    /// Graph records hidden by a later delete or overwrite.
    pub tombstones: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub id: u64,
    /// Distance under the store's metric. Lower is closer.
    pub score: f32,
}

/// Everything behind the lock.
struct Inner {
    wal: Wal,
    /// Shared so a checkpoint can read the current graph while it builds
    /// the next one without holding the store lock.
    graph: Option<Arc<GraphIndex>>,
    /// Graph slots whose record was deleted or overwritten since the last checkpoint.
    tombstones: HashSet<u32>,
    /// Records written since the last checkpoint: ids and their vectors'
    /// offsets in the log. The vectors themselves stay in the mapped log.
    delta: Flat,
    /// Sequence number of the write that produced each delta record.
    delta_seq: HashMap<u64, u64>,
    meta: HashMap<u64, Metadata>,
    /// Monotonic count of writes, used to tell which delta records a running
    /// checkpoint has already captured.
    write_seq: u64,
    /// Ids deleted while a checkpoint is building; `None` when none is.
    deletes_since_snapshot: Option<HashSet<u64>>,
    /// Log entries since the last checkpoint. Drives automatic compaction.
    log_entries: usize,
}

pub struct Db {
    path: PathBuf,
    idx_path: PathBuf,
    header: Header,
    opts: Options,
    inner: RwLock<Inner>,
    /// Serialises checkpoints; held for the whole build.
    checkpoint_lock: Mutex<()>,
    /// `<path>.lock`, exclusively flock'ed for as long as this `Db` lives.
    /// A separate file because the log and index are replaced by rename.
    _file_lock: std::fs::File,
}

impl Db {
    /// Open or create a store. Fails with `Error::Locked` if another `Db`
    /// has the same path open, in this process or any other.
    pub fn open(path: impl AsRef<Path>, opts: Options) -> Result<Db> {
        let path = path.as_ref().to_path_buf();
        let idx_path = sidecar(&path, ".idx");
        let file_lock = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(sidecar(&path, ".lock"))?;
        match file_lock.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => return Err(Error::Locked),
            Err(std::fs::TryLockError::Error(e)) => return Err(e.into()),
        }
        let header = Header { dim: opts.dim, metric: opts.metric };
        let graph = GraphIndex::open(&idx_path, opts.dim, opts.metric)?.map(Arc::new);
        let (wal, entries) = Wal::open(&path, header)?;
        let mut inner = Inner {
            wal,
            graph,
            tombstones: HashSet::new(),
            delta: Flat::new(opts.metric),
            delta_seq: HashMap::new(),
            meta: HashMap::new(),
            write_seq: 0,
            deletes_since_snapshot: None,
            log_entries: entries.len(),
        };
        for e in entries {
            match e {
                Replayed::Insert { id, vector_off, meta } => inner.apply_insert(id, vector_off, meta),
                Replayed::Delete { id } => inner.apply_delete(id),
                Replayed::Meta { id, meta } => {
                    inner.meta.insert(id, meta);
                }
            }
        }
        // Every live record must have a vector somewhere.
        for &id in inner.meta.keys() {
            if inner.delta.offset(id).is_none() && inner.graph.as_ref().and_then(|g| g.slot(id)).is_none() {
                return Err(Error::Corrupt("record has metadata but its vector is missing from the index"));
            }
        }
        Ok(Db { path, idx_path, header, opts, inner: RwLock::new(inner), checkpoint_lock: Mutex::new(()), _file_lock: file_lock })
    }

    fn read(&self) -> Result<RwLockReadGuard<'_, Inner>> {
        Ok(self.inner.read())
    }

    fn write(&self) -> Result<RwLockWriteGuard<'_, Inner>> {
        Ok(self.inner.write())
    }

    pub fn len(&self) -> usize {
        self.inner.read().meta.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dim(&self) -> u32 {
        self.header.dim
    }

    /// Number of records served from the in-RAM delta rather than the graph.
    pub fn delta_len(&self) -> usize {
        self.inner.read().delta.len()
    }

    /// Shape of the on-disk graph index, or `None` before the first checkpoint.
    pub fn index_stats(&self) -> Option<IndexStats> {
        let i = self.inner.read();
        i.graph.as_ref().map(|g| IndexStats { nodes: g.len(), pq_subspaces: g.pq_subspaces(), tombstones: i.tombstones.len() })
    }

    /// Insert or replace the record with this id.
    pub fn insert(&self, id: u64, vector: &[f32], meta: Metadata) -> Result<()> {
        if vector.len() != self.header.dim as usize {
            return Err(Error::DimMismatch { expected: self.header.dim, got: vector.len() });
        }
        if meta.len() > u16::MAX as usize || meta.iter().any(|(k, v)| k.len() > u16::MAX as usize || v.len() > u16::MAX as usize) {
            return Err(Error::MetaTooLarge);
        }
        let vector = prepare(self.header.metric, vector)?;
        let garbage = {
            let mut i = self.write()?;
            let off = i.append(&Entry::Insert { id, vector: vector.to_vec(), meta: meta.clone() }, self.opts.sync_every_write)?;
            i.apply_insert(id, off, meta);
            i.garbage()
        };
        self.maybe_auto_checkpoint(garbage)
    }

    /// Remove a record. Returns whether it existed.
    pub fn delete(&self, id: u64) -> Result<bool> {
        let garbage = {
            let mut i = self.write()?;
            if !i.meta.contains_key(&id) {
                return Ok(false);
            }
            i.append(&Entry::Delete { id }, self.opts.sync_every_write)?;
            i.apply_delete(id);
            i.garbage()
        };
        self.maybe_auto_checkpoint(garbage)?;
        Ok(true)
    }

    pub fn get(&self, id: u64) -> Option<(Vec<f32>, Metadata)> {
        let i = self.inner.read();
        let meta = i.meta.get(&id)?;
        let vector = i.delta.offset(id).map(|o| i.wal.vector_at(o)).or_else(|| i.graph.as_ref()?.vector_of(id))?;
        Some((vector.to_vec(), meta.clone()))
    }

    /// k-nearest-neighbour search. Exact over records written since the last
    /// checkpoint, approximate (graph walk) over the rest.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>> {
        self.search_filtered(query, k, |_| true)
    }

    /// k-NN restricted to records whose metadata passes `filter`.
    pub fn search_filtered(&self, query: &[f32], k: usize, filter: impl Fn(&Metadata) -> bool) -> Result<Vec<Hit>> {
        if query.len() != self.header.dim as usize {
            return Err(Error::DimMismatch { expected: self.header.dim, got: query.len() });
        }
        let query = &*prepare(self.header.metric, query)?;
        let i = self.read()?;
        let passes = |id: u64| i.meta.get(&id).is_some_and(&filter);
        let mut hits = i.delta.search(query, k, passes, &|off| i.wal.vector_at(off));
        if let Some(g) = &i.graph {
            let (beam, max_beam, width) = (self.opts.search_beam as usize, self.opts.max_search_beam as usize, self.opts.io_width as usize);
            hits.extend(
                g.search(query, k, beam, max_beam, width, |slot| !i.tombstones.contains(&slot) && passes(g.id(slot)))
                    .into_iter()
                    .map(|(score, slot)| Hit { id: g.id(slot), score }),
            );
            hits.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.id.cmp(&b.id)));
            hits.truncate(k);
        }
        present(self.header.metric, &mut hits);
        Ok(hits)
    }

    /// fsync the log so acknowledged writes survive power loss.
    pub fn sync(&self) -> Result<()> {
        self.write()?.wal.sync()
    }

    /// Bring the graph index up to date with every live record and shrink
    /// the log.
    ///
    /// When a graph exists and the change since it was built is small (fewer
    /// pending writes and deletes than live graph nodes, and the result fits
    /// `shard_size`), the existing graph is updated in place: deleted nodes
    /// are consolidated out and new vectors inserted, so the cost follows the
    /// change, not the store. Otherwise the graph is rebuilt from scratch.
    ///
    /// The build runs without holding the store lock, so reads and writes
    /// proceed meanwhile; writes that land during the build stay in the delta
    /// afterwards. Each file is replaced atomically, index first, so the log
    /// never references a vector the index lacks. Only one checkpoint runs at
    /// a time; a second caller waits for the first.
    pub fn checkpoint(&self) -> Result<()> {
        let _one_at_a_time = self.checkpoint_lock.lock();
        let dim = self.header.dim as usize;

        // 1. Snapshot under a brief exclusive lock. For a rebuild, every live
        //    vector is written to a temp file; for an update, only the delta.
        //    The temp file is memory-mapped, so the build's working set, not
        //    the dataset, has to fit in RAM.
        let vec_tmp = self.idx_path.with_extension("vec.tmp");
        let (ids, snap_seq, base) = {
            let mut i = self.write()?;
            i.wal.sync()?;
            let live_in_graph = i.graph.as_ref().map_or(0, |g| g.len() - i.tombstones.len());
            let change = i.delta.len() + i.tombstones.len();
            let total = live_in_graph + i.delta.len();
            let incremental = i.graph.as_ref().is_some_and(|g| {
                change < live_in_graph
                    && total <= self.opts.graph.shard_size as usize
                    // Keep PQ consistent: an index without codes stays without
                    // until a rebuild adds them.
                    && (g.pq_subspaces() > 0 || !self.opts.graph.use_pq || total < pq::MIN_N)
            });
            let mut ids: Vec<u64> = Vec::with_capacity(if incremental { i.delta.len() } else { i.meta.len() });
            let mut w = std::io::BufWriter::new(std::fs::File::create(&vec_tmp)?);
            use std::io::Write as _;
            if let Some(g) = i.graph.as_ref().filter(|_| !incremental) {
                for slot in 0..g.len() as u32 {
                    if !i.tombstones.contains(&slot) {
                        ids.push(g.id(slot));
                        w.write_all(bytemuck::cast_slice(g.vector(slot)))?;
                    }
                }
            }
            for (id, off) in i.delta.iter() {
                ids.push(id);
                w.write_all(bytemuck::cast_slice(i.wal.vector_at(off)))?;
            }
            w.flush()?;
            i.deletes_since_snapshot = Some(HashSet::new());
            let base = incremental.then(|| (i.graph.clone().unwrap(), i.tombstones.clone()));
            (ids, i.write_seq, base)
        };

        // 2. Build outside the lock. This is the slow part.
        let idx_tmp = self.idx_path.with_extension("idx.tmp");
        let built = (|| -> Result<()> {
            let file = std::fs::File::open(&vec_tmp)?;
            // Safety: the snapshot file is complete and never written again.
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            let vectors: &[f32] = if ids.is_empty() { &[] } else { bytemuck::cast_slice(&mmap) };
            match &base {
                Some((graph, dead)) => graph::update_and_write(&idx_tmp, graph, dead, &ids, vectors, self.opts.graph),
                None => graph::build_and_write(&idx_tmp, dim, self.header.metric, &ids, vectors, self.opts.graph),
            }
        })();
        drop(base);
        let _ = std::fs::remove_file(&vec_tmp);
        if let Err(e) = built {
            let _ = std::fs::remove_file(&idx_tmp);
            self.inner.write().deletes_since_snapshot = None;
            return Err(e);
        }

        // 3. Swap files and reconcile with writes that happened meanwhile.
        let mut i = self.write()?;
        // Windows cannot replace a file that is still mapped, so the old
        // index mapping goes first. We hold the write lock, so no reader has
        // a reference, and `base` was dropped above. If the rename fails the
        // old file is still there; reopen it so the store stays usable.
        i.graph = None;
        if let Err(e) = std::fs::rename(&idx_tmp, &self.idx_path) {
            i.graph = GraphIndex::open(&self.idx_path, self.header.dim, self.header.metric).ok().flatten().map(Arc::new);
            let _ = std::fs::remove_file(&idx_tmp);
            return Err(e.into());
        }
        // Map the new file before the log swap so the log we write only names
        // vectors the new index holds.
        let graph = GraphIndex::open(&self.idx_path, self.header.dim, self.header.metric)?.ok_or(Error::Corrupt("index vanished after rename"))?;

        let deleted = i.deletes_since_snapshot.take().unwrap_or_default();
        // Records written during the build survive in the delta; they are
        // rewritten into the new log first, in this order, so their new
        // offsets can be matched up afterwards.
        let survivors: Vec<(u64, u64)> = i.delta.iter().filter(|(id, _)| i.delta_seq.get(id).is_some_and(|&s| s > snap_seq)).collect();
        let mut tombstones = HashSet::new();
        for (id, _) in &survivors {
            if let Some(slot) = graph.slot(*id) {
                tombstones.insert(slot);
            }
        }
        for id in deleted {
            // Deleted during the build: hide its graph slot unless it was
            // re-inserted (then the survivor loop already tombstoned it).
            if let Some(slot) = graph.slot(id).filter(|_| !i.meta.contains_key(&id)) {
                tombstones.insert(slot);
            }
        }

        let log_tmp = self.path.with_extension("tmp");
        let survivor_ids: HashSet<u64> = survivors.iter().map(|(id, _)| *id).collect();
        let inserts = survivors.iter().map(|&(id, off)| Entry::Insert { id, vector: i.wal.vector_at(off).to_vec(), meta: i.meta[&id].clone() });
        let metas = i.meta.iter().filter(|(id, _)| !survivor_ids.contains(id)).map(|(&id, m)| Entry::Meta { id, meta: m.clone() });
        let new_offs = wal::write_snapshot(&log_tmp, self.header, inserts.chain(metas))?;
        // Same Windows rule for the log: close the old one before replacing
        // it. If the rename fails, reopen the old log (its contents are
        // unchanged, so the delta's offsets stay valid) and report the error.
        i.wal.release();
        if let Err(e) = std::fs::rename(&log_tmp, &self.path) {
            let _ = std::fs::remove_file(&log_tmp);
            i.wal = Wal::open(&self.path, self.header)?.0;
            i.graph = Some(Arc::new(graph));
            return Err(e.into());
        }
        if let Some(dir) = self.path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::File::open(dir)?.sync_all()?;
        }

        let (wal, _) = Wal::open(&self.path, self.header)?;
        let mut delta = Flat::new(self.header.metric);
        let mut delta_seq = HashMap::new();
        for ((id, _), off) in survivors.iter().zip(new_offs) {
            delta.insert(*id, off);
            delta_seq.insert(*id, i.delta_seq[id]);
        }
        i.wal = wal;
        i.graph = Some(Arc::new(graph));
        i.tombstones = tombstones;
        i.delta = delta;
        i.delta_seq = delta_seq;
        i.log_entries = i.meta.len();
        Ok(())
    }

    fn maybe_auto_checkpoint(&self, (garbage, delta, indexed): (usize, usize, usize)) -> Result<()> {
        let g = self.opts.auto_checkpoint_garbage;
        let d = self.opts.auto_checkpoint_delta;
        let due = (g != 0 && garbage >= g) || (d != 0 && delta >= d && delta >= indexed);
        if !due {
            return Ok(());
        }
        // Skip if a checkpoint is already running; it will pick these up.
        match self.checkpoint_lock.try_lock() {
            Some(guard) => {
                drop(guard);
                self.checkpoint()
            }
            None => Ok(()),
        }
    }
}

impl Inner {
    fn apply_insert(&mut self, id: u64, vector_off: u64, meta: Metadata) {
        self.write_seq += 1;
        if let Some(slot) = self.graph.as_ref().and_then(|g| g.slot(id)) {
            self.tombstones.insert(slot);
        }
        self.delta.insert(id, vector_off);
        self.delta_seq.insert(id, self.write_seq);
        self.meta.insert(id, meta);
    }

    fn apply_delete(&mut self, id: u64) {
        self.write_seq += 1;
        if let Some(slot) = self.graph.as_ref().and_then(|g| g.slot(id)) {
            self.tombstones.insert(slot);
        }
        self.delta.remove(id);
        self.delta_seq.remove(&id);
        self.meta.remove(&id);
        if let Some(d) = &mut self.deletes_since_snapshot {
            d.insert(id);
        }
    }

    /// Returns the vector offset for inserts (meaningless for other kinds).
    fn append(&mut self, e: &Entry, sync: bool) -> Result<u64> {
        let off = self.wal.append(e)?;
        if sync {
            self.wal.sync()?;
        } else {
            self.wal.flush()?;
        }
        self.log_entries += 1;
        Ok(off)
    }

    /// (log entries that no longer describe a live record, delta size, live
    /// records in the graph).
    fn garbage(&self) -> (usize, usize, usize) {
        let indexed = self.graph.as_ref().map_or(0, |g| g.len() - self.tombstones.len());
        (self.log_entries.saturating_sub(self.meta.len()), self.delta.len(), indexed)
    }
}

fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}
