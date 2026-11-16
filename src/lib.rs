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

mod attrs;
#[doc(hidden)]
pub mod fault;
mod filter;
mod graph;
mod index;
mod kmeans;
mod pq;
mod quant;
mod sparse;
mod text;
mod wal;

pub use filter::Filter;
pub use graph::Storage;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// parking_lot locks do not poison: a panic on one thread (for instance from
// a filter closure) leaves the store usable by the others.
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use rayon::prelude::*;

/// A `tracing` event when the feature is on; nothing otherwise.
macro_rules! event {
    (warn, $($t:tt)*) => {
        #[cfg(feature = "tracing")]
        tracing::warn!(target: "sanadb", $($t)*);
    };
    ($($t:tt)*) => {
        #[cfg(feature = "tracing")]
        tracing::info!(target: "sanadb", $($t)*);
    };
}

use graph::GraphIndex;
pub use graph::GraphParams;
use index::Flat;
use wal::{Entry, Header, Replayed, Wal};

mod value;
pub use value::{Kind, Metadata, Value};

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Corrupt(&'static str),
    HeaderMismatch {
        on_disk: (u32, Metric),
        requested: (u32, Metric),
    },
    DimMismatch {
        expected: u32,
        got: usize,
    },
    MetaTooLarge,
    /// A key already holds values of another type in this store.
    TypeMismatch {
        key: String,
        expected: Kind,
        got: Kind,
    },
    /// A second metadata key holds `Value::Sparse` values; a store indexes
    /// one sparse key.
    SecondSparseKey {
        have: String,
        got: String,
    },
    /// A vector contained NaN or infinity.
    InvalidVector,
    /// Another `Db` (in this or another process) has the store open.
    Locked,
    /// The store was opened with `open_read_only`.
    ReadOnly,
    /// The file was written by a newer sanadb whose layout this build cannot
    /// read. Upgrade sanadb; the data is intact.
    NeedsNewerVersion {
        file_level: u8,
        supported: u8,
    },
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// Attach the failing operation and path to an I/O error, so a report from a
/// platform we cannot reproduce on says which step broke.
fn io_ctx<T>(r: std::io::Result<T>, what: &str, path: &Path) -> Result<T> {
    r.map_err(|e| {
        Error::Io(std::io::Error::new(
            e.kind(),
            format!("{what} {}: {e}", path.display()),
        ))
    })
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Corrupt(why) => write!(f, "corrupt file: {why}"),
            Error::HeaderMismatch { on_disk, requested } => {
                write!(
                    f,
                    "file has dim={} metric={:?}, opened with dim={} metric={:?}",
                    on_disk.0, on_disk.1, requested.0, requested.1
                )
            }
            Error::DimMismatch { expected, got } => {
                write!(f, "vector has {got} dims, store expects {expected}")
            }
            Error::MetaTooLarge => write!(
                f,
                "metadata key/value longer than 65535 bytes or more than 65535 pairs"
            ),
            Error::SecondSparseKey { have, got } => write!(
                f,
                "a store holds one sparse key: {got:?} conflicts with {have:?}"
            ),
            Error::TypeMismatch { key, expected, got } => write!(
                f,
                "metadata key {key:?} holds {expected:?} values in this store, got {got:?}"
            ),
            Error::InvalidVector => write!(
                f,
                "vector contains NaN or infinity, or is all zeros under the cosine metric"
            ),
            Error::Locked => write!(f, "store is open elsewhere (lock file held)"),
            Error::ReadOnly => write!(f, "store is open read-only"),
            Error::NeedsNewerVersion {
                file_level,
                supported,
            } => {
                write!(
                    f,
                    "file needs reader level {file_level}, this sanadb supports {supported}: upgrade sanadb"
                )
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
    Ok(std::borrow::Cow::Owned(
        v.iter().map(|x| x / norm).collect(),
    ))
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
    /// `search_where` answers a filter exactly, by reading every matching
    /// record, when the index's posting lists put the match count at or
    /// below this. Above it the graph is walked with the filter steering it.
    pub exact_filter_scan: u32,
    /// Metadata key whose string value is indexed for keyword search
    /// (`search_text`, `search_hybrid`). `None` builds no text index.
    pub text_field: Option<String>,
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
            exact_filter_scan: 16_384,
            text_field: None,
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
    /// Vector storage in the index: `Storage::F32` (default) or `Storage::I8`.
    pub fn storage(mut self, s: Storage) -> Self {
        self.graph.storage = s;
        self
    }
    pub fn exact_filter_scan(mut self, records: u32) -> Self {
        self.exact_filter_scan = records;
        self
    }
    /// Index this metadata key's string values for `search_text` and
    /// `search_hybrid`. Takes effect at the next checkpoint; records written
    /// since the last one are scored from their metadata directly.
    pub fn text_field(mut self, key: impl Into<String>) -> Self {
        self.text_field = Some(key.into());
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
    /// How the index stores vectors.
    pub storage: Storage,
}

/// Counters since this `Db` was opened, from `Db::stats`. Everything is a
/// monotonic count except `last_checkpoint_ms`. Cheap enough to poll.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    /// Vector searches, by the regime that answered the graph part.
    pub searches: u64,
    /// Plain or closure-filtered graph walks.
    pub graph_walks: u64,
    /// `search_where` answered by walking the graph with the filter
    /// steering it (large matching set).
    pub filtered_walks: u64,
    /// `search_where` answered by scanning the posting lists' candidates
    /// (small matching set).
    pub exact_scans: u64,
    /// Searches that had no graph yet (delta only).
    pub delta_only: u64,
    pub text_searches: u64,
    pub sparse_searches: u64,
    pub inserts: u64,
    pub deletes: u64,
    pub batches: u64,
    pub checkpoints: u64,
    /// Checkpoints that updated the graph in place instead of rebuilding.
    pub incremental_checkpoints: u64,
    pub failed_checkpoints: u64,
    pub last_checkpoint_ms: u64,
    /// Read-only refreshes that replayed appended log frames.
    pub refresh_appended: u64,
    /// Read-only refreshes that reloaded after the writer's checkpoint.
    pub refresh_reloaded: u64,
}

#[derive(Default)]
struct Counters {
    searches: AtomicU64,
    graph_walks: AtomicU64,
    filtered_walks: AtomicU64,
    exact_scans: AtomicU64,
    delta_only: AtomicU64,
    text_searches: AtomicU64,
    sparse_searches: AtomicU64,
    inserts: AtomicU64,
    deletes: AtomicU64,
    batches: AtomicU64,
    checkpoints: AtomicU64,
    incremental_checkpoints: AtomicU64,
    failed_checkpoints: AtomicU64,
    last_checkpoint_ms: AtomicU64,
    refresh_appended: AtomicU64,
    refresh_reloaded: AtomicU64,
}

fn bump(c: &AtomicU64) {
    c.fetch_add(1, Ordering::Relaxed);
}

impl Counters {
    fn snapshot(&self) -> Stats {
        let g = |c: &AtomicU64| c.load(Ordering::Relaxed);
        Stats {
            searches: g(&self.searches),
            graph_walks: g(&self.graph_walks),
            filtered_walks: g(&self.filtered_walks),
            exact_scans: g(&self.exact_scans),
            delta_only: g(&self.delta_only),
            text_searches: g(&self.text_searches),
            sparse_searches: g(&self.sparse_searches),
            inserts: g(&self.inserts),
            deletes: g(&self.deletes),
            batches: g(&self.batches),
            checkpoints: g(&self.checkpoints),
            incremental_checkpoints: g(&self.incremental_checkpoints),
            failed_checkpoints: g(&self.failed_checkpoints),
            last_checkpoint_ms: g(&self.last_checkpoint_ms),
            refresh_appended: g(&self.refresh_appended),
            refresh_reloaded: g(&self.refresh_reloaded),
        }
    }
}

/// Inserts and deletes to apply together with `Db::write_batch`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Batch {
    ops: Vec<BatchOp>,
}

#[derive(Debug, Clone, PartialEq)]
enum BatchOp {
    Insert {
        id: u64,
        vector: Vec<f32>,
        meta: Metadata,
    },
    Delete {
        id: u64,
    },
}

impl Batch {
    pub fn new() -> Batch {
        Batch::default()
    }
    pub fn insert(mut self, id: u64, vector: &[f32], meta: Metadata) -> Batch {
        self.ops.push(BatchOp::Insert {
            id,
            vector: vector.to_vec(),
            meta,
        });
        self
    }
    pub fn delete(mut self, id: u64) -> Batch {
        self.ops.push(BatchOp::Delete { id });
        self
    }
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// How a search restricts its results.
#[derive(Clone, Copy)]
enum Spec<'a> {
    All,
    Closure(&'a dyn Fn(&Metadata) -> bool),
    Expr(&'a Filter),
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
    /// Metadata of records whose only copy is in the log: everything in the
    /// delta, plus graph records of an index without columns (built before
    /// 0.1.4) until the next checkpoint. Graph records otherwise read their
    /// metadata from the index's attribute columns, so this map is small.
    meta: HashMap<u64, Arc<Metadata>>,
    /// Type fixed for each metadata key by the first value written under it.
    types: HashMap<String, Kind>,
    /// Monotonic count of writes, used to tell which delta records a running
    /// checkpoint has already captured.
    write_seq: u64,
    /// Ids deleted while a checkpoint is building; `None` when none is.
    deletes_since_snapshot: Option<HashSet<u64>>,
    /// Log entries since the last checkpoint. Drives automatic compaction.
    log_entries: usize,
    /// Frames of a batch whose `Commit` has not been seen yet (a reader
    /// mid-refresh, or a writer replaying); applied together on `Commit`.
    pending: Option<Vec<Replayed>>,
    /// Id of the index file this log pairs with (`<path>.idx.<id>`); `None`
    /// for the unnumbered `<path>.idx` of stores last checkpointed before
    /// 0.2.0, or for a store that has no index yet.
    snapshot: Option<u64>,
}

pub struct Db {
    path: PathBuf,
    header: Header,
    opts: Options,
    inner: RwLock<Inner>,
    /// Serialises checkpoints; held for the whole build.
    checkpoint_lock: Mutex<()>,
    /// `<path>.lock`, exclusively flock'ed for as long as a writer lives.
    /// A separate file because the log is replaced by rename. Readers
    /// (`open_read_only`) hold no lock at all.
    _file_lock: Option<std::fs::File>,
    read_only: bool,
    counters: Counters,
}

impl Db {
    /// Open or create a store for reading and writing. Fails with
    /// `Error::Locked` if another writer has the same path open, in this
    /// process or any other. Readers (`open_read_only`) do not conflict.
    pub fn open(path: impl AsRef<Path>, opts: Options) -> Result<Db> {
        let path = path.as_ref().to_path_buf();
        let file_lock = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(sidecar(&path, ".lock"))?;
        // A store that was just closed can read as locked for a few
        // milliseconds (seen about once in 25 parallel test-suite runs under
        // load on macOS, holder never identified), so a held lock is retried
        // briefly, as SQLite's busy handler does, before it is an error.
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(200);
        loop {
            match file_lock.try_lock() {
                Ok(()) => break,
                Err(std::fs::TryLockError::WouldBlock) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(std::fs::TryLockError::WouldBlock) => return Err(Error::Locked),
                Err(std::fs::TryLockError::Error(e)) => return Err(e.into()),
            }
        }
        let header = Header {
            dim: opts.dim,
            metric: opts.metric,
        };
        let inner = Self::load(&path, header, false)?;
        // Index files no longer referenced: left by a crash between writing
        // a new index and swapping the log, or by a checkpoint whose delete
        // of the old index failed because a reader still mapped it.
        Self::collect_orphans(&path, inner.snapshot);
        Ok(Db {
            path,
            header,
            opts,
            inner: RwLock::new(inner),
            checkpoint_lock: Mutex::new(()),
            _file_lock: Some(file_lock),
            read_only: false,
            counters: Counters::default(),
        })
    }

    /// Open an existing store for reading while another process (or this
    /// one) writes it. Takes no lock and writes no file. The view is a
    /// prefix of the writer's history, brought up to date at the start of
    /// every read (`refresh` forces it); writes fail with `Error::ReadOnly`.
    pub fn open_read_only(path: impl AsRef<Path>, opts: Options) -> Result<Db> {
        let path = path.as_ref().to_path_buf();
        let header = Header {
            dim: opts.dim,
            metric: opts.metric,
        };
        let inner = Self::load(&path, header, true)?;
        Ok(Db {
            path,
            header,
            opts,
            inner: RwLock::new(inner),
            checkpoint_lock: Mutex::new(()),
            _file_lock: None,
            read_only: true,
            counters: Counters::default(),
        })
    }

    /// Read the log, find the index it names, replay the tail.
    fn load(path: &Path, header: Header, read_only: bool) -> Result<Inner> {
        let (wal, entries) = if read_only {
            Wal::open_read_only(path, header)?
        } else {
            Wal::open(path, header)?
        };
        let snapshot = entries.iter().find_map(|e| match e {
            Replayed::Snapshot { id } => Some(*id),
            _ => None,
        });
        let graph = GraphIndex::open(&index_path_for(path, snapshot), header.dim, header.metric)?
            .map(Arc::new);
        let mut inner = Inner {
            wal,
            graph,
            tombstones: HashSet::new(),
            delta: Flat::new(header.metric),
            delta_seq: HashMap::new(),
            meta: HashMap::new(),
            types: HashMap::new(),
            write_seq: 0,
            deletes_since_snapshot: None,
            log_entries: entries.len(),
            pending: None,
            snapshot,
        };
        if let Some(g) = &inner.graph {
            for (k, kind) in g.attr_kinds() {
                inner.types.entry(k.to_string()).or_insert(kind);
            }
            if let Some(sp) = g.sparse() {
                inner
                    .types
                    .entry(sp.name(g.bytes()).to_string())
                    .or_insert(Kind::Sparse);
            }
        }
        for e in entries {
            inner.replay(e);
        }
        // Every live record must have a vector somewhere.
        for &id in inner.meta.keys() {
            if inner.delta.offset(id).is_none()
                && inner.graph.as_ref().and_then(|g| g.slot(id)).is_none()
            {
                return Err(Error::Corrupt(
                    "record has metadata but its vector is missing from the index",
                ));
            }
        }
        Ok(inner)
    }

    /// Delete index files other than the current one. Errors are ignored: a
    /// reader in another process may still have one mapped (Windows refuses
    /// the delete then), and the next writer open tries again.
    fn collect_orphans(path: &Path, current: Option<u64>) {
        let keep = index_path_for(path, current);
        let Some(dir) = path.parent() else { return };
        let Some(stem) = path.file_name().and_then(|n| n.to_str()) else {
            return;
        };
        let prefix = format!("{stem}.idx");
        let Ok(entries) = std::fs::read_dir(if dir.as_os_str().is_empty() {
            Path::new(".")
        } else {
            dir
        }) else {
            return;
        };
        for e in entries.flatten() {
            let name = e.file_name();
            let Some(name) = name.to_str() else { continue };
            let is_index = name == prefix
                || (name.starts_with(&prefix) && name[prefix.len()..].starts_with('.'));
            if is_index && e.path() != keep {
                event!(warn, orphan = %e.path().display(), "removing index file no log names");
                let _ = std::fs::remove_file(e.path());
            }
        }
        let _ = std::fs::remove_file(sidecar(path, ".vec.tmp"));
    }

    /// Bring a read-only view up to date with the writer: pick up appended
    /// frames, or reload entirely after a checkpoint replaced the files.
    /// Called by every read; a no-op for a writer.
    pub fn refresh(&self) -> Result<()> {
        if !self.read_only {
            return Ok(());
        }
        let mut i = self.inner.write();
        match i.wal.refresh(&self.path)? {
            wal::Refresh::Appended(entries) => {
                bump(&self.counters.refresh_appended);
                i.log_entries += entries.len();
                for e in entries {
                    i.replay(e);
                }
            }
            wal::Refresh::Replaced => {
                // The writer swaps the index before the log, so the log we
                // see now names an index that exists.
                *i = Self::load(&self.path, self.header, true)?;
                bump(&self.counters.refresh_reloaded);
                event!(path = %self.path.display(), records = i.len(), "reader reloaded after checkpoint");
            }
        }
        Ok(())
    }

    /// Copy the store into `dir` in a consistent state and return the path of
    /// the copy's log; `Db::open` on that path opens the backup. The index
    /// file is copied first (no checkpoint can replace it meanwhile: this
    /// holds the checkpoint lock), then the log's complete frames are copied
    /// under the write lock, so writers pause for the time the log tail
    /// takes, not the index. Works from a read-only `Db` too. The result is
    /// a point-in-time copy with the same guarantees as the original; ship
    /// it anywhere, or open it read-only beside the original.
    pub fn backup(&self, dir: impl AsRef<Path>) -> Result<PathBuf> {
        let _no_checkpoint = self.checkpoint_lock.lock();
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        let name = self
            .path
            .file_name()
            .ok_or(Error::Corrupt("store path has no file name"))?;
        let dest = dir.join(name);
        let index = {
            let i = self.inner.read();
            i.graph.as_ref().map(|_| {
                (
                    index_path_for(&self.path, i.snapshot),
                    index_path_for(&dest, i.snapshot),
                )
            })
        };
        if let Some((src, dst)) = &index {
            io_ctx(std::fs::copy(src, dst).map(|_| ()), "copy index", src)?;
            // Windows needs a writable handle to flush; a read handle is refused.
            io_ctx(
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(dst)
                    .and_then(|f| f.sync_all()),
                "sync index copy",
                dst,
            )?;
        }
        {
            let mut i = self.inner.write();
            let mut out = std::io::BufWriter::new(io_ctx(
                std::fs::File::create(&dest),
                "create log copy",
                &dest,
            )?);
            i.wal.copy_prefix(&mut out)?;
            use std::io::Write as _;
            out.flush()?;
            out.get_ref().sync_all()?;
        }
        Ok(dest)
    }

    /// Path of the current index file, if a checkpoint has produced one.
    pub fn index_path(&self) -> Option<PathBuf> {
        let i = self.inner.read();
        i.graph
            .as_ref()
            .map(|_| index_path_for(&self.path, i.snapshot))
    }

    fn read(&self) -> Result<RwLockReadGuard<'_, Inner>> {
        Ok(self.inner.read())
    }

    fn write(&self) -> Result<RwLockWriteGuard<'_, Inner>> {
        if self.read_only {
            return Err(Error::ReadOnly);
        }
        Ok(self.inner.write())
    }

    pub fn len(&self) -> usize {
        let _ = self.refresh();
        self.inner.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dim(&self) -> u32 {
        self.header.dim
    }

    /// Number of records served from the in-RAM delta rather than the graph.
    pub fn delta_len(&self) -> usize {
        let _ = self.refresh();
        self.inner.read().delta.len()
    }

    /// The dimension and metric a store was created with, read from the
    /// file's header without opening the store. Tools use it to open a
    /// store they were only given the path of.
    pub fn peek(path: impl AsRef<Path>) -> Result<(u32, Metric)> {
        let h = wal::peek(path.as_ref())?;
        Ok((h.dim, h.metric))
    }

    /// Every live record id, ascending. Allocates one `u64` per record.
    pub fn ids(&self) -> Vec<u64> {
        let _ = self.refresh();
        let i = self.inner.read();
        let mut ids: Vec<u64> = i.delta.iter().map(|(id, _)| id).collect();
        if let Some(g) = &i.graph {
            ids.extend(
                (0..g.len() as u32)
                    .filter(|s| !i.tombstones.contains(s))
                    .map(|s| g.id(s))
                    .filter(|id| i.delta.offset(*id).is_none()),
            );
        }
        ids.sort_unstable();
        ids
    }

    /// Every metadata key written to the store with its type.
    pub fn keys(&self) -> Vec<(String, Kind)> {
        let _ = self.refresh();
        let i = self.inner.read();
        let mut keys: Vec<(String, Kind)> = i.types.iter().map(|(k, v)| (k.clone(), *v)).collect();
        keys.sort_by(|a, b| a.0.cmp(&b.0));
        keys
    }

    /// `search` with the graph walk's trace: `(hits, walk)`, where `walk`
    /// lists the indexed records in the order the walk expanded them with
    /// the distance the walk used (PQ-approximate when the index has codes).
    /// Unfiltered, one pass at `search_beam`. The delta is scanned as usual
    /// but does not appear in the walk. For tooling that shows a search.
    pub fn search_traced(&self, query: &[f32], k: usize) -> Result<(Vec<Hit>, Vec<Hit>)> {
        if query.len() != self.header.dim as usize {
            return Err(Error::DimMismatch {
                expected: self.header.dim,
                got: query.len(),
            });
        }
        let query = &*prepare(self.header.metric, query)?;
        self.refresh()?;
        let i = self.read()?;
        bump(&self.counters.searches);
        let mut hits = i
            .delta
            .search(query, k, |_| true, &|off| i.wal.vector_at(off));
        let mut walk = Vec::new();
        if let Some(g) = &i.graph {
            bump(&self.counters.graph_walks);
            let (gh, expanded) = g.search_traced(
                query,
                k,
                self.opts.search_beam as usize,
                self.opts.io_width as usize,
            );
            let dead = !i.tombstones.is_empty();
            hits.extend(
                gh.into_iter()
                    .filter(|(_, s)| !dead || !i.tombstones.contains(s))
                    .map(|(score, slot)| Hit {
                        id: g.id(slot),
                        score,
                    }),
            );
            hits.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.id.cmp(&b.id)));
            hits.truncate(k);
            walk = expanded
                .into_iter()
                .map(|(score, slot)| Hit {
                    id: g.id(slot),
                    score,
                })
                .collect();
        }
        present(self.header.metric, &mut hits);
        present(self.header.metric, &mut walk);
        Ok((hits, walk))
    }

    /// Operation counters since open; see [`Stats`].
    pub fn stats(&self) -> Stats {
        self.counters.snapshot()
    }

    /// Shape of the on-disk graph index, or `None` before the first checkpoint.
    pub fn index_stats(&self) -> Option<IndexStats> {
        let _ = self.refresh();
        let i = self.inner.read();
        i.graph.as_ref().map(|g| IndexStats {
            nodes: g.len(),
            pq_subspaces: g.pq_subspaces(),
            tombstones: i.tombstones.len(),
            storage: g.storage(),
        })
    }

    /// Insert or replace the record with this id.
    pub fn insert(&self, id: u64, vector: &[f32], meta: Metadata) -> Result<()> {
        if vector.len() != self.header.dim as usize {
            return Err(Error::DimMismatch {
                expected: self.header.dim,
                got: vector.len(),
            });
        }
        let meta = check_meta(meta)?;
        let vector = prepare(self.header.metric, vector)?;
        let garbage = {
            let mut i = self.write()?;
            i.check_types(&meta)?;
            let off = i.append(
                &Entry::Insert {
                    id,
                    vector: vector.to_vec(),
                    meta: meta.clone(),
                },
                self.opts.sync_every_write,
            )?;
            i.apply_insert(id, off, meta);
            i.garbage()
        };
        bump(&self.counters.inserts);
        self.maybe_auto_checkpoint(garbage)
    }

    /// Remove a record. Returns whether it existed.
    pub fn delete(&self, id: u64) -> Result<bool> {
        let garbage = {
            let mut i = self.write()?;
            if !i.is_live(id) {
                return Ok(false);
            }
            i.append(&Entry::Delete { id }, self.opts.sync_every_write)?;
            i.apply_delete(id);
            i.garbage()
        };
        bump(&self.counters.deletes);
        self.maybe_auto_checkpoint(garbage)?;
        Ok(true)
    }

    /// Apply a batch of inserts and deletes atomically: after a crash either
    /// all of them are in the store or none is, and a reader never sees a
    /// prefix. Everything is validated before anything is written, so a bad
    /// vector or a type conflict anywhere in the batch leaves the store
    /// unchanged. The whole batch is written under one lock; a batch of a
    /// million records holds writers and readers off for its duration.
    pub fn write_batch(&self, batch: Batch) -> Result<()> {
        if batch.ops.is_empty() {
            return Ok(());
        }
        // Validate shape and values first; types need the lock.
        let mut prepared = Vec::with_capacity(batch.ops.len());
        for op in &batch.ops {
            match op {
                BatchOp::Insert { id, vector, meta } => {
                    if vector.len() != self.header.dim as usize {
                        return Err(Error::DimMismatch {
                            expected: self.header.dim,
                            got: vector.len(),
                        });
                    }
                    let meta = check_meta(meta.clone())?;
                    let v = prepare(self.header.metric, vector)?.into_owned();
                    prepared.push(Entry::Insert {
                        id: *id,
                        vector: v,
                        meta,
                    });
                }
                BatchOp::Delete { id } => prepared.push(Entry::Delete { id: *id }),
            }
        }
        let garbage = {
            let mut i = self.write()?;
            // Types: against the store, and consistent within the batch.
            let mut batch_types: HashMap<&str, Kind> = HashMap::new();
            for e in &prepared {
                if let Entry::Insert { meta, .. } = e {
                    i.check_types(meta)?;
                    for (k, v) in meta {
                        match batch_types.get(k.as_str()) {
                            Some(&t) if t != v.kind() => {
                                return Err(Error::TypeMismatch {
                                    key: k.clone(),
                                    expected: t,
                                    got: v.kind(),
                                });
                            }
                            Some(_) => {}
                            None => {
                                batch_types.insert(k, v.kind());
                            }
                        }
                    }
                }
            }
            let mut frames = Vec::with_capacity(prepared.len() + 2);
            frames.push(Entry::Begin);
            frames.extend(prepared);
            frames.push(Entry::Commit);
            let offs = i.append_many(&frames, self.opts.sync_every_write)?;
            for (e, off) in frames.into_iter().zip(offs) {
                match e {
                    Entry::Insert { id, meta, .. } => i.apply_insert(id, off, meta),
                    Entry::Delete { id } => i.apply_delete(id),
                    _ => {}
                }
            }
            i.garbage()
        };
        bump(&self.counters.batches);
        self.maybe_auto_checkpoint(garbage)
    }

    pub fn get(&self, id: u64) -> Option<(Vec<f32>, Metadata)> {
        let _ = self.refresh();
        let i = self.inner.read();
        if let Some(off) = i.delta.offset(id) {
            let meta = i.meta.get(&id).map(|m| (**m).clone()).unwrap_or_default();
            return Some((i.wal.vector_at(off).to_vec(), meta));
        }
        let g = i.graph.as_ref()?;
        let slot = g.slot(id).filter(|s| !i.tombstones.contains(s))?;
        let meta = i
            .meta
            .get(&id)
            .map(|m| (**m).clone())
            .or_else(|| g.metadata(slot))?;
        Some((g.vector(slot).into_owned(), meta))
    }

    /// k-nearest-neighbour search. Exact over records written since the last
    /// checkpoint, approximate (graph walk) over the rest.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>> {
        self.search_impl(query, k, Spec::All)
    }

    /// k-NN restricted to records whose metadata passes `filter`. The
    /// closure is opaque to the engine, so it is applied to the nodes the
    /// walk visits (with the beam widened until `k` pass); prefer
    /// `search_where` for anything a `Filter` can express.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: impl Fn(&Metadata) -> bool,
    ) -> Result<Vec<Hit>> {
        self.search_impl(query, k, Spec::Closure(&filter))
    }

    /// k-NN restricted to records matching `filter`. Against an index built
    /// by 0.1.4 or later the filter runs on attribute columns: a visited node
    /// is checked with an array read, and a filter that matches few records
    /// (at most `Options::exact_filter_scan` by the posting lists) is
    /// answered exactly by scanning those records instead of walking.
    pub fn search_where(&self, query: &[f32], k: usize, filter: &Filter) -> Result<Vec<Hit>> {
        self.search_impl(query, k, Spec::Expr(filter))
    }

    /// `search_many` restricted to records matching `filter`.
    /// Keyword search with BM25 over the text field named by
    /// `Options::text_field`. Scores are negated BM25 (lower is better, as
    /// every `Hit`), so they sort with the rest of the API. Indexed records
    /// are scored from the inverted index; records written since the last
    /// checkpoint are tokenised on the fly. `filter` restricts the result.
    pub fn search_text(&self, query: &str, k: usize, filter: Option<&Filter>) -> Result<Vec<Hit>> {
        let field = self
            .opts
            .text_field
            .as_deref()
            .ok_or(Error::Corrupt("search_text needs Options::text_field"))?;
        bump(&self.counters.text_searches);
        let terms = text::tokenize(query);
        if terms.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        self.refresh()?;
        let i = self.read()?;
        let graph = i.graph.as_ref();
        let ix = graph.and_then(|g| g.text());
        // Corpus statistics: index plus tail. Tail records are tokenised here;
        // they are few by construction.
        let mut tail: Vec<(u64, HashMap<String, u16>, u32)> = Vec::new();
        for (id, _) in i.delta.iter() {
            if let Some(t) = i
                .meta
                .get(&id)
                .and_then(|m| m.get(field))
                .and_then(|v| v.as_str())
            {
                let toks = text::tokenize(t);
                if !toks.is_empty() {
                    let len = toks.len() as u32;
                    let tf: HashMap<String, u16> = text::term_freqs(&toks)
                        .into_iter()
                        .map(|(k, v)| (k.to_string(), v))
                        .collect();
                    tail.push((id, tf, len));
                }
            }
        }
        let (mut n_docs, mut total_len) = ix.map_or((0u64, 0u64), |x| (x.n_docs, x.total_len));
        n_docs += tail.len() as u64;
        total_len += tail.iter().map(|(_, _, l)| *l as u64).sum::<u64>();
        if n_docs == 0 {
            return Ok(Vec::new());
        }
        let avg = total_len as f32 / n_docs as f32;
        let dead = !i.tombstones.is_empty();
        let passes = |id: u64| filter.is_none_or(|f| i.metadata_of(id).is_some_and(|m| f.eval(&m)));
        // Term at a time; the per-term postings are short for all but stop
        // words, and the query has a handful of terms.
        let mut scores: HashMap<u32, f32> = HashMap::new();
        let mut tail_scores: HashMap<u64, f32> = HashMap::new();
        let mut seen_terms: HashSet<&str> = HashSet::new();
        for term in &terms {
            if !seen_terms.insert(term.as_str()) {
                continue;
            }
            let df_tail = tail
                .iter()
                .filter(|(_, tf, _)| tf.contains_key(term))
                .count();
            let df = ix.map_or(0, |x| x.df(graph.unwrap().bytes(), term)) + df_tail;
            if df == 0 {
                continue;
            }
            if let (Some(g), Some(x)) = (graph, ix) {
                for (slot, tf) in x.postings(g.bytes(), term) {
                    if dead && i.tombstones.contains(&slot) {
                        continue;
                    }
                    let len = x.len_of(g.bytes(), slot) as f32;
                    *scores.entry(slot).or_insert(0.0) +=
                        text::bm25(tf as f32, df as f32, n_docs as f32, len, avg);
                }
            }
            for (id, tf, len) in &tail {
                if let Some(&t) = tf.get(term) {
                    *tail_scores.entry(*id).or_insert(0.0) +=
                        text::bm25(t as f32, df as f32, n_docs as f32, *len as f32, avg);
                }
            }
        }
        let mut hits: Vec<Hit> = scores
            .into_iter()
            .map(|(slot, s)| Hit {
                id: graph.unwrap().id(slot),
                score: -s,
            })
            .chain(tail_scores.into_iter().map(|(id, s)| Hit { id, score: -s }))
            .filter(|h| passes(h.id))
            .collect();
        hits.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.id.cmp(&b.id)));
        hits.truncate(k);
        Ok(hits)
    }

    /// Dense and keyword search fused by reciprocal rank: each list is
    /// fetched with `3k` results and a record scores `sum(1 / (60 + rank))`
    /// over the lists it appears in. `Hit::score` is the negated fused score.
    /// Reciprocal rank fusion needs no calibration between a distance and a
    /// BM25 score, which is why it is the default in every published
    /// comparison of hybrid retrieval on small corpora.
    pub fn search_hybrid(
        &self,
        query: &[f32],
        text: &str,
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<Hit>> {
        let wide = k.saturating_mul(3).max(k);
        let dense = match filter {
            Some(f) => self.search_where(query, wide, f)?,
            None => self.search(query, wide)?,
        };
        let keyword = self.search_text(text, wide, filter)?;
        Ok(fuse(&[dense, keyword], k))
    }

    /// Dot-product search over the store's sparse key (the one metadata key
    /// holding `Value::Sparse` values): indexed records through the inverted
    /// index in the index file, records written since the last checkpoint by
    /// brute force over their metadata. `Hit::score` is the negated dot
    /// product. The query's pairs are normalised like stored ones (sorted,
    /// duplicates summed). Empty before any sparse value was written.
    pub fn search_sparse(
        &self,
        query: &[(u32, f32)],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<Hit>> {
        bump(&self.counters.sparse_searches);
        let mut q = query.to_vec();
        if !sparse::normalise(&mut q) {
            return Err(Error::InvalidVector);
        }
        if q.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        self.refresh()?;
        let i = self.read()?;
        let Some(field) = i.sparse_key() else {
            return Ok(Vec::new());
        };
        let dead = !i.tombstones.is_empty();
        let passes = |id: u64| filter.is_none_or(|f| i.metadata_of(id).is_some_and(|m| f.eval(&m)));
        let mut hits: Vec<Hit> = Vec::new();
        if let Some(g) = &i.graph
            && let Some(sp) = g.sparse()
        {
            let mut scores: HashMap<u32, f32> = HashMap::new();
            for &(term, qw) in &q {
                for (slot, w) in sp.postings(g.bytes(), term) {
                    if dead && i.tombstones.contains(&slot) {
                        continue;
                    }
                    *scores.entry(slot).or_insert(0.0) += qw * w;
                }
            }
            hits.extend(scores.into_iter().map(|(slot, s)| Hit {
                id: g.id(slot),
                score: -s,
            }));
        }
        for (id, _) in i.delta.iter() {
            if let Some(Value::Sparse(p)) = i.meta.get(&id).and_then(|m| m.get(field)) {
                let s = sparse::dot(&q, p);
                if s != 0.0 {
                    hits.push(Hit { id, score: -s });
                }
            }
        }
        hits.retain(|h| passes(h.id));
        hits.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.id.cmp(&b.id)));
        hits.truncate(k);
        Ok(hits)
    }

    /// Dense and sparse search fused by reciprocal rank, as `search_hybrid`
    /// does for dense and keyword search.
    pub fn search_hybrid_sparse(
        &self,
        query: &[f32],
        sparse: &[(u32, f32)],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<Hit>> {
        let wide = k.saturating_mul(3).max(k);
        let dense = match filter {
            Some(f) => self.search_where(query, wide, f)?,
            None => self.search(query, wide)?,
        };
        let sparse = self.search_sparse(sparse, wide, filter)?;
        Ok(fuse(&[dense, sparse], k))
    }

    pub fn search_many_where(
        &self,
        queries: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<Vec<Hit>>> {
        let dim = self.header.dim as usize;
        if !queries.len().is_multiple_of(dim) {
            return Err(Error::DimMismatch {
                expected: self.header.dim,
                got: queries.len() % dim,
            });
        }
        queries
            .par_chunks(dim)
            .map(|q| self.search_where(q, k, filter))
            .collect()
    }

    /// k-NN for many queries at once, spread over all cores. `queries` is
    /// row-major, `n * dim` floats; the result has one `Vec<Hit>` per row in
    /// the same order. Each row sees the same store state as a single
    /// `search` would, but rows may interleave with concurrent writes.
    pub fn search_many(&self, queries: &[f32], k: usize) -> Result<Vec<Vec<Hit>>> {
        self.search_many_filtered(queries, k, |_| true)
    }

    /// `search_many` restricted to records whose metadata passes `filter`.
    pub fn search_many_filtered(
        &self,
        queries: &[f32],
        k: usize,
        filter: impl Fn(&Metadata) -> bool + Sync,
    ) -> Result<Vec<Vec<Hit>>> {
        let dim = self.header.dim as usize;
        if !queries.len().is_multiple_of(dim) {
            return Err(Error::DimMismatch {
                expected: self.header.dim,
                got: queries.len() % dim,
            });
        }
        queries
            .par_chunks(dim)
            .map(|q| self.search_filtered(q, k, &filter))
            .collect()
    }

    fn search_impl(&self, query: &[f32], k: usize, spec: Spec<'_>) -> Result<Vec<Hit>> {
        if query.len() != self.header.dim as usize {
            return Err(Error::DimMismatch {
                expected: self.header.dim,
                got: query.len(),
            });
        }
        let query = &*prepare(self.header.metric, query)?;
        self.refresh()?;
        let i = self.read()?;
        // Without a filter the metadata map is never consulted: that lookup
        // ran once per visited node and was 6% of a warm query.
        let passes_meta = |m: &Metadata| match spec {
            Spec::All => true,
            Spec::Closure(f) => f(m),
            Spec::Expr(f) => f.eval(m),
        };
        let passes = |id: u64| {
            matches!(spec, Spec::All) || i.metadata_of(id).is_some_and(|m| passes_meta(&m))
        };
        let mut hits = i
            .delta
            .search(query, k, passes, &|off| i.wal.vector_at(off));
        bump(&self.counters.searches);
        if i.graph.is_none() {
            bump(&self.counters.delta_only);
        }
        if let Some(g) = &i.graph {
            let (beam, max_beam, width) = (
                self.opts.search_beam as usize,
                self.opts.max_search_beam as usize,
                self.opts.io_width as usize,
            );
            let dead = !i.tombstones.is_empty();
            let live = |slot: u32| !dead || !i.tombstones.contains(&slot);
            let prepared = match spec {
                Spec::Expr(f) => g.prepare(f, self.opts.text_field.as_deref()),
                _ => None,
            };
            let graph_hits = match &prepared {
                Some(p) => {
                    let max_scan = self.opts.exact_filter_scan as usize;
                    // Few enough matches by the posting lists: read them all.
                    let scan = (g.estimate(p) <= max_scan * 2)
                        .then(|| g.candidates(p))
                        .flatten()
                        .filter(|c| c.len() <= max_scan);
                    match scan {
                        Some(c) => {
                            bump(&self.counters.exact_scans);
                            g.scan(query, k, c.into_iter().filter(|&s| live(s)))
                        }
                        None => {
                            bump(&self.counters.filtered_walks);
                            g.search(
                                query,
                                k,
                                beam,
                                max_beam,
                                width,
                                |slot| live(slot) && g.pass(p, slot),
                                true,
                            )
                        }
                    }
                }
                None => {
                    bump(&self.counters.graph_walks);
                    g.search(
                        query,
                        k,
                        beam,
                        max_beam,
                        width,
                        |slot| live(slot) && passes(g.id(slot)),
                        false,
                    )
                }
            };
            hits.extend(graph_hits.into_iter().map(|(score, slot)| Hit {
                id: g.id(slot),
                score,
            }));
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
        if self.read_only {
            return Err(Error::ReadOnly);
        }
        let t = std::time::Instant::now();
        match self.checkpoint_inner() {
            Ok(incremental) => {
                let ms = t.elapsed().as_millis() as u64;
                self.counters
                    .last_checkpoint_ms
                    .store(ms, Ordering::Relaxed);
                bump(&self.counters.checkpoints);
                if incremental {
                    bump(&self.counters.incremental_checkpoints);
                }
                event!(path = %self.path.display(), incremental, ms, records = self.len(), "checkpoint");
                Ok(())
            }
            Err(e) => {
                bump(&self.counters.failed_checkpoints);
                event!(warn, path = %self.path.display(), error = %e, "checkpoint failed");
                Err(e)
            }
        }
    }

    /// Returns whether the graph was updated in place.
    fn checkpoint_inner(&self) -> Result<bool> {
        let _one_at_a_time = self.checkpoint_lock.lock();
        let dim = self.header.dim as usize;
        // The new index gets its own file name; nothing is ever renamed over
        // an index, so readers in other processes can keep the old one
        // mapped (Windows refuses to replace a mapped file).
        let snap_id = new_snapshot_id();
        let idx_new = index_path_for(&self.path, Some(snap_id));

        // 1. Snapshot under a brief exclusive lock. For a rebuild, every live
        //    vector is written to a temp file; for an update, only the delta.
        //    The temp file is memory-mapped, so the build's working set, not
        //    the dataset, has to fit in RAM.
        let vec_tmp = sidecar(&self.path, ".vec.tmp");
        let (ids, snap_seq, base, metas, old_graph, sparse_key) = {
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
            let mut ids: Vec<u64> =
                Vec::with_capacity(if incremental { i.delta.len() } else { i.len() });
            let mut w = std::io::BufWriter::new(io_ctx(
                std::fs::File::create(&vec_tmp),
                "create snapshot",
                &vec_tmp,
            )?);
            use std::io::Write as _;
            if let Some(g) = i.graph.as_ref().filter(|_| !incremental) {
                for slot in 0..g.len() as u32 {
                    if !i.tombstones.contains(&slot) {
                        ids.push(g.id(slot));
                        w.write_all(bytemuck::cast_slice(&g.vector(slot)))?;
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
            // Metadata for the new index's columns comes from the log-held map
            // (delta records, and everything if the old index has no columns)
            // and otherwise from the old index's columns, which are immutable,
            // so both reads happen outside the lock.
            let metas: HashMap<u64, Arc<Metadata>> =
                i.meta.iter().map(|(&id, m)| (id, m.clone())).collect();
            let old_graph = i.graph.clone();
            let sparse_key = i.sparse_key().map(str::to_string);
            (ids, i.write_seq, base, metas, old_graph, sparse_key)
        };
        let meta_of = |id: u64| {
            metas.get(&id).cloned().or_else(|| {
                let g = old_graph.as_ref()?;
                g.metadata(g.slot(id)?).map(Arc::new)
            })
        };

        // 2. Build outside the lock. This is the slow part.
        let built = (|| -> Result<()> {
            // An empty snapshot is not mapped: Windows refuses to map a
            // zero-length file.
            let mmap = if ids.is_empty() {
                None
            } else {
                let file = io_ctx(std::fs::File::open(&vec_tmp), "open snapshot", &vec_tmp)?;
                // Safety: the snapshot file is complete and never written again.
                Some(io_ctx(
                    unsafe { memmap2::Mmap::map(&file) },
                    "map snapshot",
                    &vec_tmp,
                )?)
            };
            let vectors: &[f32] = mmap.as_ref().map_or(&[], |m| bytemuck::cast_slice(m));
            match &base {
                Some((graph, dead)) => graph::update_and_write(
                    &idx_new,
                    graph,
                    dead,
                    &ids,
                    vectors,
                    self.opts.graph,
                    &meta_of,
                    self.opts.text_field.as_deref(),
                    sparse_key.as_deref(),
                ),
                None => graph::build_and_write(
                    &idx_new,
                    dim,
                    self.header.metric,
                    &ids,
                    vectors,
                    self.opts.graph,
                    &meta_of,
                    self.opts.text_field.as_deref(),
                    sparse_key.as_deref(),
                ),
            }
        })();
        let incremental = base.is_some();
        drop(base);
        drop(old_graph);
        let _ = std::fs::remove_file(&vec_tmp);
        if let Err(e) = built {
            let _ = std::fs::remove_file(&idx_new);
            self.inner.write().deletes_since_snapshot = None;
            return Err(e);
        }

        // 3. Swap the log and reconcile with writes that happened meanwhile.
        let mut i = self.write()?;
        let graph = match GraphIndex::open(&idx_new, self.header.dim, self.header.metric) {
            Ok(Some(g)) => g,
            Ok(None) => return Err(Error::Corrupt("index vanished after build")),
            Err(e) => {
                let _ = std::fs::remove_file(&idx_new);
                return Err(e);
            }
        };
        let old_index = index_path_for(&self.path, i.snapshot);

        let deleted = i.deletes_since_snapshot.take().unwrap_or_default();
        // Records written during the build survive in the delta; they are
        // rewritten into the new log first, in this order, so their new
        // offsets can be matched up afterwards.
        let survivors: Vec<(u64, u64)> = i
            .delta
            .iter()
            .filter(|(id, _)| i.delta_seq.get(id).is_some_and(|&s| s > snap_seq))
            .collect();
        let mut tombstones = HashSet::new();
        for (id, _) in &survivors {
            if let Some(slot) = graph.slot(*id) {
                tombstones.insert(slot);
            }
        }
        // Deleted during the build: hide its graph slot unless it was
        // re-inserted (then the survivor loop already tombstoned it). These
        // ids also get a Delete frame in the new log, so a reopen replays the
        // tombstone instead of serving the stale graph node.
        let dead: Vec<u64> = deleted
            .into_iter()
            .filter(|id| i.delta.offset(*id).is_none())
            .filter(|id| graph.slot(*id).is_some())
            .collect();
        for id in &dead {
            tombstones.insert(graph.slot(*id).unwrap());
        }

        let log_tmp = sidecar(&self.path, ".tmp");
        let survivor_ids: HashSet<u64> = survivors.iter().map(|(id, _)| *id).collect();
        let inserts = survivors.iter().map(|&(id, off)| Entry::Insert {
            id,
            vector: i.wal.vector_at(off).to_vec(),
            meta: (*i.meta[&id]).clone(),
        });
        // The compacted log names its index first; graph records' metadata
        // lives in that index's columns, so the rest is the survivors and
        // the deletes.
        let names = std::iter::once(Entry::Snapshot { id: snap_id });
        let deletes = dead.iter().map(|&id| Entry::Delete { id });
        let new_offs =
            wal::write_snapshot(&log_tmp, self.header, names.chain(inserts).chain(deletes))?;
        // Same Windows rule for the log: close the old one before replacing
        // it. If the rename fails, reopen the old log (its contents are
        // unchanged, so the delta's offsets stay valid) and report the error.
        i.wal.release();
        if let Err(e) = io_ctx(
            fault::log_rename(&self.path).and_then(|_| std::fs::rename(&log_tmp, &self.path)),
            "replace log",
            &self.path,
        ) {
            let _ = std::fs::remove_file(&log_tmp);
            let _ = std::fs::remove_file(&idx_new);
            i.wal = Wal::open(&self.path, self.header)?.0;
            return Err(e);
        }
        // Durability of the renames themselves. Directory fsync is a Unix
        // notion; Windows cannot open a directory as a file.
        #[cfg(unix)]
        if let Some(dir) = self.path.parent().filter(|d| !d.as_os_str().is_empty()) {
            io_ctx(
                std::fs::File::open(dir).and_then(|d| d.sync_all()),
                "sync directory",
                dir,
            )?;
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
        i.snapshot = Some(snap_id);
        i.tombstones = tombstones;
        i.delta = delta;
        i.delta_seq = delta_seq;
        i.meta.retain(|id, _| survivor_ids.contains(id));
        i.log_entries = survivors.len() + dead.len() + 1;
        drop(i);
        // The previous index is unreferenced now. A reader elsewhere may
        // still map it; then the delete fails here (Windows) and the next
        // writer open collects it.
        if old_index != idx_new {
            let _ = std::fs::remove_file(&old_index);
        }
        Ok(incremental)
    }

    fn maybe_auto_checkpoint(
        &self,
        (garbage, delta, indexed): (usize, usize, usize),
    ) -> Result<()> {
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
    /// Apply one replayed log frame.
    fn replay(&mut self, e: Replayed) {
        match e {
            Replayed::Begin { .. } => self.pending = Some(Vec::new()),
            Replayed::Commit => {
                for e in self.pending.take().unwrap_or_default() {
                    self.replay_one(e);
                }
            }
            e if self.pending.is_some() => self.pending.as_mut().unwrap().push(e),
            e => self.replay_one(e),
        }
    }

    /// Apply a frame outside any batch.
    fn replay_one(&mut self, e: Replayed) {
        match e {
            Replayed::Begin { .. } | Replayed::Commit => {}
            Replayed::Insert {
                id,
                vector_off,
                meta,
            } => self.apply_insert(id, vector_off, meta),
            Replayed::Delete { id } => self.apply_delete(id),
            Replayed::Meta { id, meta } => {
                self.note_types(&meta);
                self.meta.insert(id, Arc::new(meta));
            }
            Replayed::Snapshot { id } => self.snapshot = Some(id),
        }
    }

    /// Live records: graph slots not superseded, plus the delta.
    fn len(&self) -> usize {
        self.graph
            .as_ref()
            .map_or(0, |g| g.len() - self.tombstones.len())
            + self.delta.len()
    }

    fn is_live(&self, id: u64) -> bool {
        self.delta.offset(id).is_some()
            || self
                .graph
                .as_ref()
                .and_then(|g| g.slot(id))
                .is_some_and(|s| !self.tombstones.contains(&s))
    }

    /// A live record's metadata: from the log-held map, else decoded from
    /// the index columns. Allocates; the `Filter` path avoids this.
    fn metadata_of(&self, id: u64) -> Option<Metadata> {
        if let Some(m) = self.meta.get(&id) {
            return Some((**m).clone());
        }
        let g = self.graph.as_ref()?;
        let slot = g.slot(id).filter(|s| !self.tombstones.contains(s))?;
        g.metadata(slot)
    }

    /// `Error::TypeMismatch` if any value's type differs from the one its
    /// key already has in this store.
    fn check_types(&self, meta: &Metadata) -> Result<()> {
        for (k, v) in meta {
            if let Some(&t) = self.types.get(k)
                && t != v.kind()
            {
                return Err(Error::TypeMismatch {
                    key: k.clone(),
                    expected: t,
                    got: v.kind(),
                });
            }
            if v.kind() == Kind::Sparse
                && let Some(have) = self.sparse_key()
                && have != k
            {
                return Err(Error::SecondSparseKey {
                    have: have.to_string(),
                    got: k.clone(),
                });
            }
        }
        Ok(())
    }

    /// The one metadata key holding sparse vectors, if any has been written.
    fn sparse_key(&self) -> Option<&str> {
        self.types
            .iter()
            .find(|(_, k)| **k == Kind::Sparse)
            .map(|(k, _)| k.as_str())
    }

    /// Record each key's type; first writer wins (used on replay, where the
    /// log is trusted).
    fn note_types(&mut self, meta: &Metadata) {
        for (k, v) in meta {
            self.types.entry(k.clone()).or_insert(v.kind());
        }
    }

    fn apply_insert(&mut self, id: u64, vector_off: u64, meta: Metadata) {
        self.note_types(&meta);
        self.write_seq += 1;
        if let Some(slot) = self.graph.as_ref().and_then(|g| g.slot(id)) {
            self.tombstones.insert(slot);
        }
        self.delta.insert(id, vector_off);
        self.delta_seq.insert(id, self.write_seq);
        self.meta.insert(id, Arc::new(meta));
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
    /// Append several frames with one flush (or fsync) at the end. Returns
    /// each frame's vector offset (meaningful for inserts).
    fn append_many(&mut self, entries: &[Entry], sync: bool) -> Result<Vec<u64>> {
        let mut offs = Vec::with_capacity(entries.len());
        for e in entries {
            offs.push(self.wal.append(e)?);
        }
        if sync {
            self.wal.sync()?;
        } else {
            self.wal.flush()?;
        }
        self.log_entries += entries.len();
        Ok(offs)
    }

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
        let indexed = self
            .graph
            .as_ref()
            .map_or(0, |g| g.len() - self.tombstones.len());
        (
            self.log_entries.saturating_sub(self.delta.len()),
            self.delta.len(),
            indexed,
        )
    }
}

/// `<path>.idx.<16 hex digits>` for a numbered snapshot, `<path>.idx` otherwise.
/// Size limits of the log's metadata encoding, and sparse vectors sorted,
/// merged and checked for finite weights.
fn check_meta(mut meta: Metadata) -> Result<Metadata> {
    if meta.len() > u16::MAX as usize
        || meta.iter().any(|(k, v)| {
            k.len() > u16::MAX as usize || v.as_str().is_some_and(|s| s.len() > u16::MAX as usize)
        })
    {
        return Err(Error::MetaTooLarge);
    }
    for v in meta.values_mut() {
        if let Value::Sparse(p) = v
            && !sparse::normalise(p)
        {
            return Err(Error::InvalidVector);
        }
    }
    Ok(meta)
}

/// Reciprocal rank fusion: a record scores `sum(1 / (60 + rank))` over the
/// lists it appears in; the result is the top `k` by that, score negated.
fn fuse(lists: &[Vec<Hit>], k: usize) -> Vec<Hit> {
    let mut fused: HashMap<u64, f32> = HashMap::new();
    for list in lists {
        for (rank, h) in list.iter().enumerate() {
            *fused.entry(h.id).or_insert(0.0) += 1.0 / (60.0 + rank as f32 + 1.0);
        }
    }
    let mut hits: Vec<Hit> = fused
        .into_iter()
        .map(|(id, s)| Hit { id, score: -s })
        .collect();
    hits.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.id.cmp(&b.id)));
    hits.truncate(k);
    hits
}

fn index_path_for(path: &Path, snapshot: Option<u64>) -> PathBuf {
    match snapshot {
        Some(id) => sidecar(path, &format!(".idx.{id:016x}")),
        None => sidecar(path, ".idx"),
    }
}

/// A fresh, non-zero snapshot id: time, pid and a counter, so two
/// checkpoints never reuse a file name even within one nanosecond.
fn new_snapshot_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let c = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mixed = (t ^ (u64::from(std::process::id()) << 40)).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ c;
    mixed.max(1)
}

fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}
