//! sanadb: a disk-first embedded vector + metadata store.
//!
//! Two files: `<path>` is a write-ahead log of inserts, deletes and metadata;
//! `<path>.idx` is a memory-mapped Vamana graph over every vector present at
//! the last `checkpoint`. Writes since then live in a small exact in-RAM
//! "delta" index. A search walks the graph and scans the delta, then merges.
//!
//! Records in the graph that were later overwritten or deleted are hidden by
//! an in-RAM tombstone set until the next checkpoint rebuilds the graph.

mod graph;
mod index;
mod wal;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

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
}

impl Options {
    pub fn new(dim: u32) -> Self {
        Options { dim, metric: Metric::L2, sync_every_write: false, graph: GraphParams::default(), search_beam: 64 }
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub id: u64,
    /// Distance under the store's metric. Lower is closer.
    pub score: f32,
}

pub struct Db {
    path: PathBuf,
    idx_path: PathBuf,
    header: Header,
    opts: Options,
    wal: Wal,
    graph: Option<GraphIndex>,
    /// Graph slots whose record was deleted or overwritten since the last checkpoint.
    tombstones: HashSet<u32>,
    /// Records written since the last checkpoint.
    delta: Flat,
    meta: HashMap<u64, Metadata>,
    /// Log entries since the last checkpoint. Drives automatic compaction.
    log_entries: usize,
}

impl Db {
    pub fn open(path: impl AsRef<Path>, opts: Options) -> Result<Db> {
        let path = path.as_ref().to_path_buf();
        let idx_path = sidecar(&path);
        let header = Header { dim: opts.dim, metric: opts.metric };
        let graph = GraphIndex::open(&idx_path, opts.dim, opts.metric)?;
        let (wal, entries) = Wal::open(&path, header)?;
        let mut db = Db {
            path,
            idx_path,
            header,
            wal,
            graph,
            tombstones: HashSet::new(),
            delta: Flat::new(opts.dim as usize, opts.metric),
            meta: HashMap::new(),
            log_entries: entries.len(),
            opts,
        };
        for e in entries {
            match e {
                Entry::Insert { id, vector, meta } => {
                    db.tombstone(id);
                    db.delta.insert(id, &vector);
                    db.meta.insert(id, meta);
                }
                Entry::Delete { id } => {
                    db.tombstone(id);
                    db.delta.remove(id);
                    db.meta.remove(&id);
                }
                Entry::Meta { id, meta } => {
                    db.meta.insert(id, meta);
                }
            }
        }
        // Every live record must have a vector somewhere.
        for &id in db.meta.keys() {
            if db.delta.vector(id).is_none() && db.graph.as_ref().and_then(|g| g.slot(id)).is_none() {
                return Err(Error::Corrupt("record has metadata but its vector is missing from the index"));
            }
        }
        Ok(db)
    }

    pub fn len(&self) -> usize {
        self.meta.len()
    }

    pub fn is_empty(&self) -> bool {
        self.meta.is_empty()
    }

    pub fn dim(&self) -> u32 {
        self.header.dim
    }

    /// Number of records served from the in-RAM delta rather than the graph.
    pub fn delta_len(&self) -> usize {
        self.delta.len()
    }

    /// Insert or replace the record with this id.
    pub fn insert(&mut self, id: u64, vector: &[f32], meta: Metadata) -> Result<()> {
        if vector.len() != self.header.dim as usize {
            return Err(Error::DimMismatch { expected: self.header.dim, got: vector.len() });
        }
        if meta.len() > u16::MAX as usize || meta.iter().any(|(k, v)| k.len() > u16::MAX as usize || v.len() > u16::MAX as usize) {
            return Err(Error::MetaTooLarge);
        }
        self.append(&Entry::Insert { id, vector: vector.to_vec(), meta: meta.clone() })?;
        self.tombstone(id);
        self.delta.insert(id, vector);
        self.meta.insert(id, meta);
        Ok(())
    }

    /// Remove a record. Returns whether it existed.
    pub fn delete(&mut self, id: u64) -> Result<bool> {
        if !self.meta.contains_key(&id) {
            return Ok(false);
        }
        self.append(&Entry::Delete { id })?;
        self.tombstone(id);
        self.delta.remove(id);
        self.meta.remove(&id);
        Ok(true)
    }

    pub fn get(&self, id: u64) -> Option<(&[f32], &Metadata)> {
        let meta = self.meta.get(&id)?;
        let vector = self.delta.vector(id).or_else(|| self.graph.as_ref()?.vector_of(id))?;
        Some((vector, meta))
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
        let passes = |id: u64| self.meta.get(&id).is_some_and(&filter);
        let mut hits = self.delta.search(query, k, passes);
        if let Some(g) = &self.graph {
            let beam = self.opts.search_beam as usize;
            hits.extend(
                g.search(query, k, beam, |slot| !self.tombstones.contains(&slot) && passes(g.id(slot)))
                    .into_iter()
                    .map(|(score, slot)| Hit { id: g.id(slot), score }),
            );
            hits.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.id.cmp(&b.id)));
            hits.truncate(k);
        }
        Ok(hits)
    }

    /// fsync the log so acknowledged writes survive power loss.
    pub fn sync(&mut self) -> Result<()> {
        self.wal.sync()
    }

    /// Rebuild the graph index over every live record and shrink the log to
    /// metadata only. Each file is replaced atomically; the index is swapped
    /// first so the log never references a vector the index lacks.
    pub fn checkpoint(&mut self) -> Result<()> {
        self.wal.sync()?;
        let dim = self.header.dim as usize;
        let mut ids: Vec<u64> = Vec::with_capacity(self.meta.len());
        let mut vectors: Vec<f32> = Vec::with_capacity(self.meta.len() * dim);
        if let Some(g) = &self.graph {
            for slot in 0..g.len() as u32 {
                if !self.tombstones.contains(&slot) {
                    ids.push(g.id(slot));
                    vectors.extend_from_slice(g.vector_of(g.id(slot)).unwrap());
                }
            }
        }
        for (id, v) in self.delta.iter() {
            ids.push(id);
            vectors.extend_from_slice(v);
        }
        debug_assert_eq!(ids.len(), self.meta.len());

        let idx_tmp = self.idx_path.with_extension("idx.tmp");
        graph::build_and_write(&idx_tmp, dim, self.header.metric, &ids, &vectors, self.opts.graph)?;
        drop(vectors);
        std::fs::rename(&idx_tmp, &self.idx_path)?;
        // Mapping the new file before the log swap guarantees the log we are
        // about to write only names vectors the new index holds.
        let graph = GraphIndex::open(&self.idx_path, self.header.dim, self.header.metric)?;

        let log_tmp = self.path.with_extension("tmp");
        wal::write_snapshot(&log_tmp, self.header, self.meta.iter().map(|(&id, m)| Entry::Meta { id, meta: m.clone() }))?;
        std::fs::rename(&log_tmp, &self.path)?;
        if let Some(dir) = self.path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::File::open(dir)?.sync_all()?;
        }

        let (wal, _) = Wal::open(&self.path, self.header)?;
        self.wal = wal;
        self.graph = graph;
        self.tombstones.clear();
        self.delta = Flat::new(dim, self.header.metric);
        self.log_entries = self.meta.len();
        Ok(())
    }

    fn tombstone(&mut self, id: u64) {
        if let Some(slot) = self.graph.as_ref().and_then(|g| g.slot(id)) {
            self.tombstones.insert(slot);
        }
    }

    fn append(&mut self, e: &Entry) -> Result<()> {
        self.wal.append(e)?;
        if self.opts.sync_every_write {
            self.wal.sync()?;
        } else {
            self.wal.flush()?;
        }
        self.log_entries += 1;
        // Compact once garbage exceeds live data by a wide margin.
        if self.log_entries > 1024 && self.log_entries > 4 * self.meta.len().max(1) {
            self.checkpoint()?;
        }
        Ok(())
    }
}

fn sidecar(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".idx");
    PathBuf::from(s)
}
