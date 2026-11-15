//! Append-only log that is also the on-disk file format, and the backing
//! store for every vector written since the last checkpoint.
//!
//! Layout: `Header` followed by zero or more frames.
//! Frame: `len u32 LE | crc32 u32 LE (over body) | body`, where body is
//! `kind u8 | pad[3] | payload` and `len` is a multiple of 4. Because the
//! header is 16 bytes and every frame is 8 + len bytes, an insert frame's
//! vector (which follows `kind`, pad and an 8-byte id) always sits at a
//! 4-byte-aligned offset, so the memory-mapped log can be read as `&[f32]`
//! in place. The delta index keeps those offsets instead of copying vectors.
//!
//! A torn tail (short read or CRC mismatch) is truncated on open; everything
//! before it is trusted. The file is grown in large steps so the mapping is
//! rebuilt rarely; the region past the logical end is zero and parses as a
//! torn tail, so a crash with slack present is harmless.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use memmap2::Mmap;

use crate::{Error, Metadata, Metric, Result};

const MAGIC: &[u8; 8] = b"SANADB02";
/// Header byte 13. A file sets it to the lowest reader level that can open
/// it; this build reads levels up to `READER_LEVEL`. Additive changes (new
/// frame kinds, which old readers skip) leave it alone; only a change that
/// an old reader would misread raises it.
pub(crate) const READER_LEVEL: u8 = 1;
/// Format of 0.1.1: same frames without the 4-byte padding or the pad bytes
/// after `kind`. Converted in place on open.
const MAGIC_V1: &[u8; 8] = b"SANADB01";
pub(crate) const HEADER_LEN: u64 = 8 + 4 + 1 + 3; // magic, dim, metric, pad
/// The file is extended in steps of this size; each step costs one remap.
const GROW_STEP: u64 = 64 << 20;

const KIND_INSERT: u8 = 1;
const KIND_DELETE: u8 = 2;
/// Metadata only; the vector lives in the graph index sidecar.
const KIND_META: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Header {
    pub dim: u32,
    pub metric: Metric,
}

/// What gets written.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Entry {
    Insert { id: u64, vector: Vec<f32>, meta: Metadata },
    Delete { id: u64 },
    Meta { id: u64, meta: Metadata },
}

/// What replay yields. Insert carries the vector's byte offset in the file
/// rather than the vector.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Replayed {
    Insert { id: u64, vector_off: u64, meta: Metadata },
    Delete { id: u64 },
    Meta { id: u64, meta: Metadata },
}

pub(crate) struct Wal {
    /// `None` after `release`: the file is closed so it can be replaced.
    writer: Option<BufWriter<File>>,
    /// Logical end of the log: where the next frame goes.
    len: u64,
    /// Read-only mapping of the file, at least `len` bytes long. `None` only
    /// transiently: Windows refuses to resize or replace a mapped file, so
    /// the mapping is dropped around `set_len` and before a rename.
    map: Option<Mmap>,
    dim: usize,
}

impl Wal {
    /// Open or create the log. Returns the header and every intact entry.
    pub fn open(path: &Path, wanted: Header) -> Result<(Self, Vec<Replayed>)> {
        let exists = path.exists() && std::fs::metadata(path)?.len() > 0;
        if exists {
            migrate_v1(path, wanted)?;
        }
        let mut file = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path)?;
        let (entries, len) = if exists {
            let header = read_header(&mut file)?;
            if header != wanted {
                return Err(Error::HeaderMismatch { on_disk: (header.dim, header.metric), requested: (wanted.dim, wanted.metric) });
            }
            read_entries(&mut file, header.dim)?
        } else {
            write_header(&mut file, wanted)?;
            file.sync_all()?;
            (Vec::new(), HEADER_LEN)
        };
        // Drop any torn tail or growth slack; it is re-added on the next append.
        if len != file.metadata()?.len() {
            file.set_len(len)?;
        }
        // Safety: this process holds the store's exclusive lock, and the only
        // writer appends past `len`; bytes below `len` never change.
        let map = unsafe { Mmap::map(&file)? };
        file.seek(SeekFrom::Start(len))?;
        Ok((Wal { writer: Some(BufWriter::new(file)), len, map: Some(map), dim: wanted.dim as usize }, entries))
    }

    fn writer(&mut self) -> Result<&mut BufWriter<File>> {
        self.writer.as_mut().ok_or(Error::Corrupt("log used after release"))
    }

    /// Append a frame. For an insert, returns the byte offset of its vector.
    pub fn append(&mut self, entry: &Entry) -> Result<u64> {
        let mut body = Vec::new();
        encode(entry, &mut body);
        let needed = self.len + 8 + body.len() as u64;
        if needed > self.map.as_ref().map_or(0, |m| m.len() as u64) {
            self.grow(needed)?;
        }
        let w = self.writer()?;
        w.write_all(&(body.len() as u32).to_le_bytes())?;
        w.write_all(&crc32fast::hash(&body).to_le_bytes())?;
        w.write_all(&body)?;
        let vector_off = self.len + 8 + 4 + 8; // frame header, kind+pad, id
        self.len = needed;
        Ok(vector_off)
    }

    /// Extend the file in `GROW_STEP` multiples and remap. Called under the
    /// store's write lock, so no reader holds the old mapping's slices.
    fn grow(&mut self, needed: u64) -> Result<()> {
        let new_len = needed.div_ceil(GROW_STEP) * GROW_STEP;
        self.map = None; // Windows: no resizing while mapped.
        let w = self.writer()?;
        w.flush()?;
        let file = w.get_ref();
        file.set_len(new_len)?;
        // Safety: as in `open`.
        self.map = Some(unsafe { Mmap::map(file)? });
        Ok(())
    }

    /// Flush, drop the mapping, trim growth slack and close the file, so a
    /// rename over it succeeds on every platform. Called before the log is
    /// replaced at checkpoint and on drop. The `Wal` is unusable afterwards
    /// and must be replaced (or reopened with `Wal::open`).
    pub fn release(&mut self) {
        self.map = None;
        if let Some(mut w) = self.writer.take() {
            let _ = w.flush();
            let _ = w.get_ref().set_len(self.len);
        }
    }

    /// The vector stored at an offset returned by `append` or replay.
    pub fn vector_at(&self, off: u64) -> &[f32] {
        let o = off as usize;
        bytemuck::cast_slice(&self.map.as_ref().expect("log mapping released")[o..o + self.dim * 4])
    }

    /// Push buffered frames to the OS. Data survives a process crash, not a power loss.
    pub fn flush(&mut self) -> Result<()> {
        self.writer()?.flush()?;
        Ok(())
    }

    /// Flush and fsync. Data survives power loss.
    pub fn sync(&mut self) -> Result<()> {
        let w = self.writer()?;
        w.flush()?;
        w.get_ref().sync_data()?;
        Ok(())
    }
}

impl Drop for Wal {
    /// Give back the growth slack so a closed store is exactly as large as
    /// its contents. Errors are ignored: the slack is harmless on reopen.
    fn drop(&mut self) {
        self.release();
    }
}

/// Write a fresh log holding `entries` to `path` and fsync it. Returns, for
/// each insert entry in order, the byte offset of its vector.
pub(crate) fn write_snapshot(path: &Path, header: Header, entries: impl Iterator<Item = Entry>) -> Result<Vec<u64>> {
    let mut file = File::create(path)?;
    write_header(&mut file, header)?;
    let mut w = BufWriter::new(file);
    let mut len = HEADER_LEN;
    let mut offs = Vec::new();
    let mut body = Vec::new();
    for e in entries {
        body.clear();
        encode(&e, &mut body);
        w.write_all(&(body.len() as u32).to_le_bytes())?;
        w.write_all(&crc32fast::hash(&body).to_le_bytes())?;
        w.write_all(&body)?;
        if matches!(e, Entry::Insert { .. }) {
            offs.push(len + 8 + 4 + 8);
        }
        len += 8 + body.len() as u64;
    }
    w.flush()?;
    w.get_ref().sync_all()?;
    Ok(offs)
}

/// If `path` is a 0.1.1 (`SANADB01`) log, rewrite it in the current format
/// via a temp file and rename. Frames there were `len | crc | kind | payload`
/// with no padding; payloads are otherwise identical.
fn migrate_v1(path: &Path, wanted: Header) -> Result<()> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 8];
    if file.read_exact(&mut magic).is_err() || &magic != MAGIC_V1 {
        return Ok(());
    }
    let mut rest = [0u8; 8];
    file.read_exact(&mut rest).map_err(|_| Error::Corrupt("v1 file shorter than header"))?;
    let dim = u32::from_le_bytes(rest[..4].try_into().unwrap());
    let metric = Metric::from_u8(rest[4]).ok_or(Error::Corrupt("v1 unknown metric"))?;
    let header = Header { dim, metric };
    if header != wanted {
        return Err(Error::HeaderMismatch { on_disk: (dim, metric), requested: (wanted.dim, wanted.metric) });
    }
    let bytes = std::fs::read(path)?;
    let mut pos = HEADER_LEN as usize;
    let mut entries = Vec::new();
    while pos + 8 <= bytes.len() {
        let len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap());
        let Some(body) = bytes.get(pos + 8..pos + 8 + len) else { break };
        if len == 0 || crc32fast::hash(body) != crc {
            break;
        }
        // Re-frame the v1 body as a v2 body: insert the three pad bytes.
        let mut v2 = Vec::with_capacity(body.len() + 3);
        v2.push(body[0]);
        v2.extend_from_slice(&[0, 0, 0]);
        v2.extend_from_slice(&body[1..]);
        let e = match decode(&v2, dim, 0) {
            Decoded::Entry(e) => e,
            Decoded::Unknown => {
                pos += 8 + len;
                continue;
            }
            Decoded::Bad => break,
        };
        let entry = match e {
            Replayed::Insert { id, meta, .. } => {
                let vb = &body[1 + 8..1 + 8 + dim as usize * 4];
                let vector = vb.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
                Entry::Insert { id, vector, meta }
            }
            Replayed::Delete { id } => Entry::Delete { id },
            Replayed::Meta { id, meta } => Entry::Meta { id, meta },
        };
        entries.push(entry);
        pos += 8 + len;
    }
    let tmp = path.with_extension("migrate.tmp");
    write_snapshot(&tmp, header, entries.into_iter())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn write_header(file: &mut File, h: Header) -> io::Result<()> {
    let mut buf = Vec::with_capacity(HEADER_LEN as usize);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&h.dim.to_le_bytes());
    buf.push(h.metric as u8);
    buf.extend_from_slice(&[READER_LEVEL, 0, 0]);
    file.write_all(&buf)
}

fn read_header(file: &mut File) -> Result<Header> {
    let mut buf = [0u8; HEADER_LEN as usize];
    file.read_exact(&mut buf).map_err(|_| Error::Corrupt("file shorter than header"))?;
    if &buf[..8] != MAGIC {
        return Err(Error::Corrupt("bad magic"));
    }
    if buf[13] > READER_LEVEL {
        return Err(Error::NeedsNewerVersion { file_level: buf[13], supported: READER_LEVEL });
    }
    let dim = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    let metric = Metric::from_u8(buf[12]).ok_or(Error::Corrupt("unknown metric"))?;
    Ok(Header { dim, metric })
}

/// Parse frames until EOF or the first damaged frame. Returns entries and the
/// byte offset just past the last good frame.
fn read_entries(file: &mut File, dim: u32) -> Result<(Vec<Replayed>, u64)> {
    let mut reader = io::BufReader::with_capacity(1 << 20, file);
    reader.seek(SeekFrom::Start(HEADER_LEN))?;
    let mut offset = HEADER_LEN;
    let mut out = Vec::new();
    let mut body = Vec::new();
    loop {
        let mut prefix = [0u8; 8];
        if read_full(&mut reader, &mut prefix)? != 8 {
            break;
        }
        let len = u32::from_le_bytes(prefix[..4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(prefix[4..].try_into().unwrap());
        if len == 0 || !len.is_multiple_of(4) {
            break;
        }
        body.clear();
        body.resize(len, 0);
        if read_full(&mut reader, &mut body)? != len || crc32fast::hash(&body) != crc {
            break;
        }
        match decode(&body, dim, offset + 8) {
            Decoded::Entry(e) => out.push(e),
            // A well-formed frame of a kind this build does not know: written
            // by a newer version. Skip it rather than treat it as damage.
            Decoded::Unknown => {}
            Decoded::Bad => break,
        }
        offset += 8 + len as u64;
    }
    Ok((out, offset))
}

enum Decoded {
    Entry(Replayed),
    Unknown,
    Bad,
}

fn read_full<R: Read>(r: &mut R, buf: &mut [u8]) -> io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..])? {
            0 => break,
            k => n += k,
        }
    }
    Ok(n)
}

fn encode(entry: &Entry, out: &mut Vec<u8>) {
    match entry {
        Entry::Insert { id, vector, meta } => {
            out.extend_from_slice(&[KIND_INSERT, 0, 0, 0]);
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(bytemuck::cast_slice(vector));
            put_meta(out, meta);
        }
        Entry::Delete { id } => {
            out.extend_from_slice(&[KIND_DELETE, 0, 0, 0]);
            out.extend_from_slice(&id.to_le_bytes());
        }
        Entry::Meta { id, meta } => {
            out.extend_from_slice(&[KIND_META, 0, 0, 0]);
            out.extend_from_slice(&id.to_le_bytes());
            put_meta(out, meta);
        }
    }
    while !out.len().is_multiple_of(4) {
        out.push(0);
    }
}

fn put_meta(out: &mut Vec<u8>, meta: &Metadata) {
    out.extend_from_slice(&(meta.len() as u16).to_le_bytes());
    for (k, v) in meta {
        put_str(out, k);
        put_str(out, v);
    }
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u16).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// `body_off` is the file offset of `body`, used to report where a vector lies.
fn decode(body: &[u8], dim: u32, body_off: u64) -> Decoded {
    match decode_known(body, dim, body_off) {
        Some(Some(e)) => Decoded::Entry(e),
        Some(None) => Decoded::Unknown,
        None => Decoded::Bad,
    }
}

/// Outer `None`: malformed. Inner `None`: unknown kind.
fn decode_known(body: &[u8], dim: u32, body_off: u64) -> Option<Option<Replayed>> {
    let mut c = Cursor { buf: body, pos: 0 };
    let kind = c.u8()?;
    c.take(3)?;
    let e = match kind {
        KIND_INSERT => {
            let id = c.u64()?;
            let vector_off = body_off + c.pos as u64;
            c.take(dim as usize * 4)?;
            let meta = c.meta()?;
            Replayed::Insert { id, vector_off, meta }
        }
        KIND_DELETE => Replayed::Delete { id: c.u64()? },
        KIND_META => {
            let id = c.u64()?;
            Replayed::Meta { id, meta: c.meta()? }
        }
        _ => return Some(None),
    };
    // Only zero padding may follow.
    (body[c.pos..].len() < 4 && body[c.pos..].iter().all(|&b| b == 0)).then_some(Some(e))
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.buf.get(self.pos..self.pos + n)?;
        self.pos += n;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn str(&mut self) -> Option<String> {
        let n = self.u16()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).ok()
    }
    fn meta(&mut self) -> Option<Metadata> {
        let n = self.u16()? as usize;
        let mut meta = HashMap::with_capacity(n);
        for _ in 0..n {
            let k = self.str()?;
            let v = self.str()?;
            meta.insert(k, v);
        }
        Some(meta)
    }
}
