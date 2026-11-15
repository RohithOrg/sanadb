//! sanadb: a disk-first embedded vector + metadata store.
//!
//! Milestone 1: single-file, write-ahead-logged store with an exact (flat)
//! vector index and metadata filtering. Later milestones replace the flat
//! index with a disk-resident graph index so the dataset can exceed RAM.

mod index;
mod wal;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
}

impl Options {
    pub fn new(dim: u32) -> Self {
        Options { dim, metric: Metric::L2, sync_every_write: false }
    }
    pub fn metric(mut self, m: Metric) -> Self {
        self.metric = m;
        self
    }
    pub fn sync_every_write(mut self, on: bool) -> Self {
        self.sync_every_write = on;
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
    header: Header,
    sync_every_write: bool,
    wal: Wal,
    index: Flat,
    meta: HashMap<u64, Metadata>,
    /// Log entries since the last checkpoint. Drives automatic compaction.
    log_entries: usize,
}

impl Db {
    pub fn open(path: impl AsRef<Path>, opts: Options) -> Result<Db> {
        let path = path.as_ref().to_path_buf();
        let header = Header { dim: opts.dim, metric: opts.metric };
        let (wal, entries) = Wal::open(&path, header)?;
        let mut index = Flat::new(opts.dim as usize, opts.metric);
        let mut meta = HashMap::new();
        let log_entries = entries.len();
        for e in entries {
            match e {
                Entry::Insert { id, vector, meta: m } => {
                    index.insert(id, &vector);
                    meta.insert(id, m);
                }
                Entry::Delete { id } => {
                    index.remove(id);
                    meta.remove(&id);
                }
            }
        }
        Ok(Db { path, header, sync_every_write: opts.sync_every_write, wal, index, meta, log_entries })
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

    /// Insert or replace the record with this id.
    pub fn insert(&mut self, id: u64, vector: &[f32], meta: Metadata) -> Result<()> {
        if vector.len() != self.header.dim as usize {
            return Err(Error::DimMismatch { expected: self.header.dim, got: vector.len() });
        }
        if meta.len() > u16::MAX as usize || meta.iter().any(|(k, v)| k.len() > u16::MAX as usize || v.len() > u16::MAX as usize) {
            return Err(Error::MetaTooLarge);
        }
        self.append(&Entry::Insert { id, vector: vector.to_vec(), meta: meta.clone() })?;
        self.index.insert(id, vector);
        self.meta.insert(id, meta);
        Ok(())
    }

    /// Remove a record. Returns whether it existed.
    pub fn delete(&mut self, id: u64) -> Result<bool> {
        if !self.meta.contains_key(&id) {
            return Ok(false);
        }
        self.append(&Entry::Delete { id })?;
        self.index.remove(id);
        self.meta.remove(&id);
        Ok(true)
    }

    pub fn get(&self, id: u64) -> Option<(&[f32], &Metadata)> {
        Some((self.index.vector(id)?, self.meta.get(&id)?))
    }

    /// Exact k-nearest-neighbour search over every stored vector.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>> {
        self.search_filtered(query, k, |_| true)
    }

    /// k-NN restricted to records whose metadata passes `filter`.
    pub fn search_filtered(&self, query: &[f32], k: usize, filter: impl Fn(&Metadata) -> bool) -> Result<Vec<Hit>> {
        if query.len() != self.header.dim as usize {
            return Err(Error::DimMismatch { expected: self.header.dim, got: query.len() });
        }
        Ok(self.index.search(query, k, |id| self.meta.get(&id).is_some_and(&filter)))
    }

    /// fsync the log so acknowledged writes survive power loss.
    pub fn sync(&mut self) -> Result<()> {
        self.wal.sync()
    }

    /// Rewrite the file to hold only live records, dropping deleted and
    /// overwritten history. Atomic: a crash mid-checkpoint leaves the old file.
    pub fn checkpoint(&mut self) -> Result<()> {
        self.wal.sync()?;
        let tmp = self.path.with_extension("tmp");
        let entries = self.index.iter().map(|(id, vector)| Entry::Insert { id, vector: vector.to_vec(), meta: self.meta[&id].clone() });
        wal::write_snapshot(&tmp, self.header, entries)?;
        std::fs::rename(&tmp, &self.path)?;
        if let Some(dir) = self.path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::File::open(dir)?.sync_all()?;
        }
        let (wal, _) = Wal::open(&self.path, self.header)?;
        self.wal = wal;
        self.log_entries = self.meta.len();
        Ok(())
    }

    fn append(&mut self, e: &Entry) -> Result<()> {
        self.wal.append(e)?;
        if self.sync_every_write {
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
