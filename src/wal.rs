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
    writer: BufWriter<File>,
    /// Logical end of the log: where the next frame goes.
    len: u64,
    /// Read-only mapping of the file, at least `len` bytes long.
    map: Mmap,
    dim: usize,
}

impl Wal {
    /// Open or create the log. Returns the header and every intact entry.
    pub fn open(path: &Path, wanted: Header) -> Result<(Self, Vec<Replayed>)> {
        let exists = path.exists() && std::fs::metadata(path)?.len() > 0;
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
        Ok((Wal { writer: BufWriter::new(file), len, map, dim: wanted.dim as usize }, entries))
    }

    /// Append a frame. For an insert, returns the byte offset of its vector.
    pub fn append(&mut self, entry: &Entry) -> Result<u64> {
        let mut body = Vec::new();
        encode(entry, &mut body);
        let needed = self.len + 8 + body.len() as u64;
        if needed > self.map.len() as u64 {
            self.grow(needed)?;
        }
        self.writer.write_all(&(body.len() as u32).to_le_bytes())?;
        self.writer.write_all(&crc32fast::hash(&body).to_le_bytes())?;
        self.writer.write_all(&body)?;
        let vector_off = self.len + 8 + 4 + 8; // frame header, kind+pad, id
        self.len = needed;
        Ok(vector_off)
    }

    /// Extend the file in `GROW_STEP` multiples and remap. Called under the
    /// store's write lock, so no reader holds the old mapping's slices.
    fn grow(&mut self, needed: u64) -> Result<()> {
        self.writer.flush()?;
        let new_len = needed.div_ceil(GROW_STEP) * GROW_STEP;
        let file = self.writer.get_ref();
        file.set_len(new_len)?;
        // Safety: as in `open`.
        self.map = unsafe { Mmap::map(file)? };
        Ok(())
    }

    /// The vector stored at an offset returned by `append` or replay.
    pub fn vector_at(&self, off: u64) -> &[f32] {
        let o = off as usize;
        bytemuck::cast_slice(&self.map[o..o + self.dim * 4])
    }

    /// Push buffered frames to the OS. Data survives a process crash, not a power loss.
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }

    /// Flush and fsync. Data survives power loss.
    pub fn sync(&mut self) -> Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        Ok(())
    }
}

impl Drop for Wal {
    /// Give back the growth slack so a closed store is exactly as large as
    /// its contents. Errors are ignored: the slack is harmless on reopen.
    fn drop(&mut self) {
        let _ = self.writer.flush();
        let _ = self.writer.get_ref().set_len(self.len);
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

fn write_header(file: &mut File, h: Header) -> io::Result<()> {
    let mut buf = Vec::with_capacity(HEADER_LEN as usize);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&h.dim.to_le_bytes());
    buf.push(h.metric as u8);
    buf.extend_from_slice(&[0, 0, 0]);
    file.write_all(&buf)
}

fn read_header(file: &mut File) -> Result<Header> {
    let mut buf = [0u8; HEADER_LEN as usize];
    file.read_exact(&mut buf).map_err(|_| Error::Corrupt("file shorter than header"))?;
    if &buf[..8] != MAGIC {
        return Err(Error::Corrupt("bad magic"));
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
            Some(e) => out.push(e),
            None => break,
        }
        offset += 8 + len as u64;
    }
    Ok((out, offset))
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
fn decode(body: &[u8], dim: u32, body_off: u64) -> Option<Replayed> {
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
        _ => return None,
    };
    // Only zero padding may follow.
    (body[c.pos..].len() < 4 && body[c.pos..].iter().all(|&b| b == 0)).then_some(e)
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
