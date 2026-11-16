//! Fault injection for tests: make a chosen I/O step fail once, so the
//! recovery paths (checkpoint aborts, torn log tails, reopen after a failed
//! swap) are exercised deterministically instead of by hoping a disk fills
//! at the right moment. One relaxed atomic load per checked step when no
//! fault is set. Hidden from the documented API; process-global.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    /// The index writer fails after this many bytes (disk full mid-build).
    IndexWriteAfter(u64),
    /// The index file's final fsync fails.
    IndexSync,
    /// The next log append fails before any byte is written.
    LogAppend,
    /// The rename that replaces the log at the end of a checkpoint fails.
    LogRename,
}

/// The armed fault and the store it applies to: every file of a store
/// (`<path>`, `<path>.tmp`, `<path>.idx.<id>`) starts with the store path,
/// so other stores in the same process, such as parallel tests, are
/// unaffected.
static FAULT: Mutex<Option<(PathBuf, Fault)>> = Mutex::new(None);
static ARMED: AtomicBool = AtomicBool::new(false);

/// Arm a fault for the store at `store`; it fires once and clears itself.
/// `None` clears.
#[doc(hidden)]
pub fn set_fault(store: impl AsRef<Path>, f: Option<Fault>) {
    *FAULT.lock().unwrap() = f.map(|f| (store.as_ref().to_path_buf(), f));
    ARMED.store(f.is_some(), Ordering::Release);
}

fn armed() -> bool {
    ARMED.load(Ordering::Acquire)
}

fn applies(store: &Path, file: &Path) -> bool {
    file.as_os_str()
        .as_encoded_bytes()
        .starts_with(store.as_os_str().as_encoded_bytes())
}

fn take_if(file: &Path, pred: impl Fn(Fault) -> bool) -> Option<Fault> {
    if !armed() {
        return None;
    }
    let mut g = FAULT.lock().unwrap();
    match &*g {
        Some((store, f)) if pred(*f) && applies(store, file) => {
            let f = *f;
            *g = None;
            ARMED.store(false, Ordering::Release);
            Some(f)
        }
        _ => None,
    }
}

fn injected(what: &str) -> std::io::Error {
    std::io::Error::other(format!("injected fault: {what}"))
}

/// Fail if the armed fault is `LogAppend`.
pub(crate) fn log_append(log: &Path) -> std::io::Result<()> {
    match take_if(log, |f| f == Fault::LogAppend) {
        Some(_) => Err(injected("log append")),
        None => Ok(()),
    }
}

/// Fail if the armed fault is `LogRename`.
pub(crate) fn log_rename(log: &Path) -> std::io::Result<()> {
    match take_if(log, |f| f == Fault::LogRename) {
        Some(_) => Err(injected("log rename")),
        None => Ok(()),
    }
}

/// Fail if the armed fault is `IndexSync`.
pub(crate) fn index_sync(index: &Path) -> std::io::Result<()> {
    match take_if(index, |f| f == Fault::IndexSync) {
        Some(_) => Err(injected("index sync")),
        None => Ok(()),
    }
}

/// A writer that fails once `IndexWriteAfter(n)` bytes have passed through.
pub(crate) struct Budgeted<W: std::io::Write> {
    inner: W,
    path: PathBuf,
    written: u64,
    budget: Option<u64>,
}

impl<W: std::io::Write> Budgeted<W> {
    pub(crate) fn new(path: &Path, inner: W) -> Self {
        let budget = if armed() {
            match &*FAULT.lock().unwrap() {
                Some((store, Fault::IndexWriteAfter(n))) if applies(store, path) => Some(*n),
                _ => None,
            }
        } else {
            None
        };
        Budgeted {
            inner,
            path: path.to_path_buf(),
            written: 0,
            budget,
        }
    }
    pub(crate) fn get_ref(&self) -> &W {
        &self.inner
    }
}

impl<W: std::io::Write> std::io::Write for Budgeted<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Some(b) = self.budget
            && self.written + buf.len() as u64 > b
        {
            take_if(&self.path, |f| matches!(f, Fault::IndexWriteAfter(_)));
            return Err(injected("index write past budget"));
        }
        let n = self.inner.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}
