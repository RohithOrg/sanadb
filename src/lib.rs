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
mod pq;
mod wal;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

pub use graph::GraphParams;
use graph::GraphIndex;
use index::Flat;
use wal::{Entry, Header, Wal};

pub type Metadata = HashMap<String, String>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Corrupt(&'static str),
    HeaderMismatch { on_disk: (u32, Metric), requested: (u32, Metric) },
    DimMismatch { expected: u32, got: usize },
    MetaTooLarge,
    /// A thread panicked while holding the store's lock; in-memory state may
    /// be inconsistent. Reopen the store to rebuild it from disk.
    Poisoned,
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl<T> From<std::sync::PoisonError<T>> for Error {
    fn from(_: std::sync::PoisonError<T>) -> Self {
        Error::Poisoned
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
            Error::Poisoned => write!(f, "store lock poisoned by a panic; reopen the store"),
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
    /// Negative dot product. Lower is closer. Normalize inputs for cosine.
    Dot = 1,
}

impl Metric {
    fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(Metric::L2),
            1 => Some(Metric::Dot),
            _ => None,
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
        }
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
    graph: Option<GraphIndex>,
    /// Graph slots whose record was deleted or overwritten since the last checkpoint.
    tombstones: HashSet<u32>,
    /// Records written since the last checkpoint.
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
}

impl Db {
    pub fn open(path: impl AsRef<Path>, opts: Options) -> Result<Db> {
        let path = path.as_ref().to_path_buf();
        let idx_path = sidecar(&path);
        let header = Header { dim: opts.dim, metric: opts.metric };
        let graph = GraphIndex::open(&idx_path, opts.dim, opts.metric)?;
        let (wal, entries) = Wal::open(&path, header)?;
        let mut inner = Inner {
            wal,
            graph,
            tombstones: HashSet::new(),
            delta: Flat::new(opts.dim as usize, opts.metric),
            delta_seq: HashMap::new(),
            meta: HashMap::new(),
            write_seq: 0,
            deletes_since_snapshot: None,
            log_entries: entries.len(),
        };
        for e in entries {
            match e {
                Entry::Insert { id, vector, meta } => inner.apply_insert(id, &vector, meta),
                Entry::Delete { id } => inner.apply_delete(id),
                Entry::Meta { id, meta } => {
                    inner.meta.insert(id, meta);
                }
            }
        }
        // Every live record must have a vector somewhere.
        for &id in inner.meta.keys() {
            if inner.delta.vector(id).is_none() && inner.graph.as_ref().and_then(|g| g.slot(id)).is_none() {
                return Err(Error::Corrupt("record has metadata but its vector is missing from the index"));
            }
        }
        Ok(Db { path, idx_path, header, opts, inner: RwLock::new(inner), checkpoint_lock: Mutex::new(()) })
    }

    fn read(&self) -> Result<RwLockReadGuard<'_, Inner>> {
        Ok(self.inner.read()?)
    }

    fn write(&self) -> Result<RwLockWriteGuard<'_, Inner>> {
        Ok(self.inner.write()?)
    }

    pub fn len(&self) -> usize {
        self.inner.read().map(|i| i.meta.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dim(&self) -> u32 {
        self.header.dim
    }

    /// Number of records served from the in-RAM delta rather than the graph.
    pub fn delta_len(&self) -> usize {
        self.inner.read().map(|i| i.delta.len()).unwrap_or(0)
    }

    /// Shape of the on-disk graph index, or `None` before the first checkpoint.
    pub fn index_stats(&self) -> Option<IndexStats> {
        let i = self.inner.read().ok()?;
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
        let garbage = {
            let mut i = self.write()?;
            i.append(&Entry::Insert { id, vector: vector.to_vec(), meta: meta.clone() }, self.opts.sync_every_write)?;
            i.apply_insert(id, vector, meta);
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
        let i = self.inner.read().ok()?;
        let meta = i.meta.get(&id)?;
        let vector = i.delta.vector(id).or_else(|| i.graph.as_ref()?.vector_of(id))?;
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
        let i = self.read()?;
        let passes = |id: u64| i.meta.get(&id).is_some_and(&filter);
        let mut hits = i.delta.search(query, k, passes);
        if let Some(g) = &i.graph {
            let (beam, max_beam) = (self.opts.search_beam as usize, self.opts.max_search_beam as usize);
            hits.extend(
                g.search(query, k, beam, max_beam, |slot| !i.tombstones.contains(&slot) && passes(g.id(slot)))
                    .into_iter()
                    .map(|(score, slot)| Hit { id: g.id(slot), score }),
            );
            hits.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.id.cmp(&b.id)));
            hits.truncate(k);
        }
        Ok(hits)
    }

    /// fsync the log so acknowledged writes survive power loss.
    pub fn sync(&self) -> Result<()> {
        self.write()?.wal.sync()
    }

    /// Rebuild the graph index over every live record and shrink the log.
    ///
    /// The build runs without holding the store lock, so reads and writes
    /// proceed meanwhile; writes that land during the build stay in the delta
    /// afterwards. Each file is replaced atomically, index first, so the log
    /// never references a vector the index lacks. Only one checkpoint runs at
    /// a time; a second caller waits for the first.
    pub fn checkpoint(&self) -> Result<()> {
        let _one_at_a_time = self.checkpoint_lock.lock()?;
        let dim = self.header.dim as usize;

        // 1. Snapshot live vectors. Brief exclusive lock: a memcpy of the data.
        let (ids, vectors, snap_seq) = {
            let mut i = self.write()?;
            i.wal.sync()?;
            let mut ids: Vec<u64> = Vec::with_capacity(i.meta.len());
            let mut vectors: Vec<f32> = Vec::with_capacity(i.meta.len() * dim);
            if let Some(g) = &i.graph {
                for slot in 0..g.len() as u32 {
                    if !i.tombstones.contains(&slot) {
                        ids.push(g.id(slot));
                        vectors.extend_from_slice(g.vector(slot));
                    }
                }
            }
            for (id, v) in i.delta.iter() {
                ids.push(id);
                vectors.extend_from_slice(v);
            }
            debug_assert_eq!(ids.len(), i.meta.len());
            i.deletes_since_snapshot = Some(HashSet::new());
            (ids, vectors, i.write_seq)
        };

        // 2. Build outside the lock. This is the slow part.
        let idx_tmp = self.idx_path.with_extension("idx.tmp");
        let built = graph::build_and_write(&idx_tmp, dim, self.header.metric, &ids, &vectors, self.opts.graph);
        drop(vectors);
        if let Err(e) = built {
            let _ = std::fs::remove_file(&idx_tmp);
            if let Ok(mut i) = self.inner.write() {
                i.deletes_since_snapshot = None;
            }
            return Err(e);
        }

        // 3. Swap files and reconcile with writes that happened meanwhile.
        let mut i = self.write()?;
        std::fs::rename(&idx_tmp, &self.idx_path)?;
        // Map the new file before the log swap so the log we write only names
        // vectors the new index holds.
        let graph = GraphIndex::open(&self.idx_path, self.header.dim, self.header.metric)?.ok_or(Error::Corrupt("index vanished after rename"))?;

        let deleted = i.deletes_since_snapshot.take().unwrap_or_default();
        let mut delta = Flat::new(dim, self.header.metric);
        let mut delta_seq = HashMap::new();
        let mut tombstones = HashSet::new();
        for (id, v) in i.delta.iter() {
            if i.delta_seq.get(&id).is_some_and(|&s| s > snap_seq) {
                delta.insert(id, v);
                delta_seq.insert(id, i.delta_seq[&id]);
                if let Some(slot) = graph.slot(id) {
                    tombstones.insert(slot);
                }
            }
        }
        for id in deleted {
            // Deleted during the build: hide its graph slot unless it was
            // re-inserted (then the delta already tombstoned it above).
            if let Some(slot) = graph.slot(id).filter(|_| !i.meta.contains_key(&id)) {
                tombstones.insert(slot);
            }
        }

        let log_tmp = self.path.with_extension("tmp");
        let entries = i.meta.iter().map(|(&id, m)| match delta.vector(id) {
            Some(v) => Entry::Insert { id, vector: v.to_vec(), meta: m.clone() },
            None => Entry::Meta { id, meta: m.clone() },
        });
        wal::write_snapshot(&log_tmp, self.header, entries)?;
        std::fs::rename(&log_tmp, &self.path)?;
        if let Some(dir) = self.path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::File::open(dir)?.sync_all()?;
        }

        let (wal, _) = Wal::open(&self.path, self.header)?;
        i.wal = wal;
        i.graph = Some(graph);
        i.tombstones = tombstones;
        i.delta = delta;
        i.delta_seq = delta_seq;
        i.log_entries = i.meta.len();
        Ok(())
    }

    fn maybe_auto_checkpoint(&self, garbage: usize) -> Result<()> {
        let limit = self.opts.auto_checkpoint_garbage;
        if limit == 0 || garbage < limit {
            return Ok(());
        }
        // Skip if a checkpoint is already running; it will pick these up.
        match self.checkpoint_lock.try_lock() {
            Ok(guard) => {
                drop(guard);
                self.checkpoint()
            }
            Err(std::sync::TryLockError::WouldBlock) => Ok(()),
            Err(std::sync::TryLockError::Poisoned(_)) => Err(Error::Poisoned),
        }
    }
}

impl Inner {
    fn apply_insert(&mut self, id: u64, vector: &[f32], meta: Metadata) {
        self.write_seq += 1;
        if let Some(slot) = self.graph.as_ref().and_then(|g| g.slot(id)) {
            self.tombstones.insert(slot);
        }
        self.delta.insert(id, vector);
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

    fn append(&mut self, e: &Entry, sync: bool) -> Result<()> {
        self.wal.append(e)?;
        if sync {
            self.wal.sync()?;
        } else {
            self.wal.flush()?;
        }
        self.log_entries += 1;
        Ok(())
    }

    /// Log entries that no longer describe a live record.
    fn garbage(&self) -> usize {
        self.log_entries.saturating_sub(self.meta.len())
    }
}

fn sidecar(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".idx");
    PathBuf::from(s)
}
