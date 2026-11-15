//! Append-only log that is also the on-disk file format.
//!
//! Layout: `Header` followed by zero or more `Entry` frames.
//! Frame: `len u32 LE | crc32 u32 LE (over kind+payload) | kind u8 | payload`.
//! A torn tail (short read or CRC mismatch) is truncated on open; everything
//! before it is trusted.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::{Error, Metadata, Metric, Result};

const MAGIC: &[u8; 8] = b"SANADB01";
pub(crate) const HEADER_LEN: u64 = 8 + 4 + 1 + 3; // magic, dim, metric, pad

const KIND_INSERT: u8 = 1;
const KIND_DELETE: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Header {
    pub dim: u32,
    pub metric: Metric,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Entry {
    Insert { id: u64, vector: Vec<f32>, meta: Metadata },
    Delete { id: u64 },
}

pub(crate) struct Wal {
    writer: BufWriter<File>,
}

impl Wal {
    /// Open or create the log. Returns the header and every intact entry.
    pub fn open(path: &Path, wanted: Header) -> Result<(Self, Vec<Entry>)> {
        let exists = path.exists() && std::fs::metadata(path)?.len() > 0;
        let mut file = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path)?;
        let entries = if exists {
            let header = read_header(&mut file)?;
            if header != wanted {
                return Err(Error::HeaderMismatch { on_disk: (header.dim, header.metric), requested: (wanted.dim, wanted.metric) });
            }
            let (parsed, valid_len) = read_entries(&mut file, header.dim)?;
            if valid_len < file.metadata()?.len() {
                file.set_len(valid_len)?;
            }
            parsed
        } else {
            write_header(&mut file, wanted)?;
            file.sync_all()?;
            Vec::new()
        };
        file.seek(SeekFrom::End(0))?;
        Ok((Wal { writer: BufWriter::new(file) }, entries))
    }

    pub fn append(&mut self, entry: &Entry) -> Result<()> {
        let mut body = Vec::new();
        encode(entry, &mut body);
        let crc = crc32fast::hash(&body);
        self.writer.write_all(&(body.len() as u32).to_le_bytes())?;
        self.writer.write_all(&crc.to_le_bytes())?;
        self.writer.write_all(&body)?;
        Ok(())
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

/// Write a fresh log holding `entries` to `path` and fsync it.
pub(crate) fn write_snapshot(path: &Path, header: Header, entries: impl Iterator<Item = Entry>) -> Result<()> {
    let mut file = File::create(path)?;
    write_header(&mut file, header)?;
    let mut wal = Wal { writer: BufWriter::new(file) };
    for e in entries {
        wal.append(&e)?;
    }
    wal.writer.flush()?;
    wal.writer.get_ref().sync_all()?;
    Ok(())
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
fn read_entries(file: &mut File, dim: u32) -> Result<(Vec<Entry>, u64)> {
    let mut reader = BufReader::new(file);
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
        body.clear();
        body.resize(len, 0);
        if read_full(&mut reader, &mut body)? != len || crc32fast::hash(&body) != crc {
            break;
        }
        match decode(&body, dim) {
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
            out.push(KIND_INSERT);
            out.extend_from_slice(&id.to_le_bytes());
            for x in vector {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out.extend_from_slice(&(meta.len() as u16).to_le_bytes());
            for (k, v) in meta {
                put_str(out, k);
                put_str(out, v);
            }
        }
        Entry::Delete { id } => {
            out.push(KIND_DELETE);
            out.extend_from_slice(&id.to_le_bytes());
        }
    }
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u16).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn decode(body: &[u8], dim: u32) -> Option<Entry> {
    let mut c = Cursor { buf: body, pos: 0 };
    match c.u8()? {
        KIND_INSERT => {
            let id = c.u64()?;
            let mut vector = Vec::with_capacity(dim as usize);
            for _ in 0..dim {
                vector.push(f32::from_le_bytes(c.take(4)?.try_into().ok()?));
            }
            let n = c.u16()? as usize;
            let mut meta = HashMap::with_capacity(n);
            for _ in 0..n {
                let k = c.str()?;
                let v = c.str()?;
                meta.insert(k, v);
            }
            (c.pos == body.len()).then_some(Entry::Insert { id, vector, meta })
        }
        KIND_DELETE => {
            let id = c.u64()?;
            (c.pos == body.len()).then_some(Entry::Delete { id })
        }
        _ => None,
    }
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
}
