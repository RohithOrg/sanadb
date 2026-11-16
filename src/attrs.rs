//! Attribute columns: metadata laid out by graph slot inside the index file,
//! so a filter can be evaluated for a visited node with one array read, and
//! selective filters can enumerate their candidates from posting lists
//! instead of walking the graph.
//!
//! Section layout (appended after the PQ section, 8-byte aligned): one
//! region per attribute in sequence, then the attribute table. The index
//! file's footer records where the table starts (0.1.4 wrote a 16-byte
//! trailer `table_off u64 LE | "ATTRTBL1"` right after it instead; `graph.rs`
//! reads both). A file with neither has no columns and filters fall back to
//! the metadata map.
//!
//! Per attribute:
//! - presence: `n` bits, one per slot, in `u64` words.
//! - column: `Str` a `u32` dictionary code per slot; `Int` an `i64`; `Float`
//!   an `f64`; `Bool` a `u8`.
//! - dictionary (`Str` always, `Int` when at most `MAX_DICT` distinct values):
//!   sorted distinct values. Strings as `count u32 | pad u32 | offsets
//!   (count + 1) u64 | bytes`, ints as `count u32 | pad u32 | i64...`.
//! - postings (when the attribute has a dictionary with at most `MAX_DICT`
//!   entries, or is `Bool`): for every distinct value, the sorted slots that
//!   hold it: `offsets (count + 1) u64 | slots u32...`. Total size is one
//!   `u32` per present slot whatever the cardinality, which is why posting
//!   lists were chosen over per-value bitmaps.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use crate::text::TextIndex;
use crate::{Error, Filter, Kind, Metadata, Result, Value};

/// Trailer written by 0.1.4 right after the table; read for compatibility.
pub(crate) const LEGACY_TRAILER_MAGIC: &[u8; 8] = b"ATTRTBL1";
pub(crate) const LEGACY_TRAILER_LEN: usize = 16;
/// Dictionaries for `Int` (and so postings) are built for at most this many
/// distinct values; above it an `Int` attribute keeps only its raw column,
/// and a `Str` attribute keeps a dictionary for codes but no postings.
const MAX_DICT: usize = 1 << 16;
/// Range predicates enumerate candidates only over this many dictionary
/// codes; wider ranges go through the walk.
const MAX_RANGE_CODES: usize = 256;

#[derive(Debug, Clone)]
struct Attr {
    name: String,
    kind: Kind,
    n_distinct: u32,
    presence: usize,
    column: usize,
    /// Offset and length of the dictionary region.
    dict: Option<(usize, usize)>,
    postings: Option<usize>,
}

/// The attribute table of one index file. Column bytes are read from the
/// index's mapping on demand; nothing but the table lives in RAM.
#[derive(Debug, Default)]
pub(crate) struct Attrs {
    n: usize,
    attrs: Vec<Attr>,
    by_name: HashMap<String, usize>,
}

// ---------------------------------------------------------------- writing

/// Write the attribute section for `n` slots. `start` is the absolute file
/// offset at which the first byte lands (it must be 8-aligned); `meta_of`
/// yields a slot's metadata. Returns `(bytes written, absolute offset of
/// the attribute table)`; the caller records the table offset in the footer.
pub(crate) fn write(
    w: &mut impl Write,
    start: u64,
    n: usize,
    meta_of: &(dyn Fn(u32) -> Option<Arc<Metadata>> + Sync),
) -> std::io::Result<(u64, u64)> {
    debug_assert!(start.is_multiple_of(8));
    // Slot metadata is fetched once; the per-attribute passes borrow from it.
    let metas: Vec<Option<Arc<Metadata>>> = (0..n as u32).map(meta_of).collect();
    // Keys in a stable order, each with the kind of its first value.
    let mut kinds: Vec<(String, Kind)> = Vec::new();
    let mut seen: HashMap<&str, usize> = HashMap::new();
    for m in metas.iter().flatten() {
        for (k, v) in m.iter() {
            // Sparse vectors live in their own section, not in a column.
            if v.kind() != Kind::Sparse && !seen.contains_key(k.as_str()) {
                seen.insert(k.as_str(), kinds.len());
                kinds.push((k.clone(), v.kind()));
            }
        }
    }
    kinds.sort_by(|a, b| a.0.cmp(&b.0));

    let mut pos = start;
    let mut table: Vec<Attr> = Vec::with_capacity(kinds.len());
    let words = n.div_ceil(64);
    for (name, kind) in &kinds {
        let mut presence = vec![0u64; words];
        let value_at = |s: usize| {
            metas[s]
                .as_ref()
                .and_then(|m| m.get(name.as_str()))
                .filter(|v| v.kind() == *kind)
        };
        for s in 0..n {
            if value_at(s).is_some() {
                presence[s / 64] |= 1 << (s % 64);
            }
        }
        let presence_off = pos;
        w.write_all(bytemuck::cast_slice(&presence))?;
        pos += (words * 8) as u64;

        let mut attr = Attr {
            name: name.clone(),
            kind: *kind,
            n_distinct: 0,
            presence: presence_off as usize,
            column: 0,
            dict: None,
            postings: None,
        };
        let mut postings_by_code: Option<(usize, Vec<u32>)> = None; // (n_distinct, code per slot)
        match kind {
            Kind::Str => {
                let mut distinct: Vec<&str> = (0..n)
                    .filter_map(|s| value_at(s).and_then(Value::as_str))
                    .collect();
                distinct.sort_unstable();
                distinct.dedup();
                let code_of: HashMap<&str, u32> = distinct
                    .iter()
                    .enumerate()
                    .map(|(c, s)| (*s, c as u32))
                    .collect();
                let codes: Vec<u32> = (0..n)
                    .map(|s| {
                        value_at(s)
                            .and_then(Value::as_str)
                            .map_or(0, |v| code_of[v])
                    })
                    .collect();
                attr.column = pos as usize;
                w.write_all(bytemuck::cast_slice(&codes))?;
                pos += (n * 4) as u64;
                attr.n_distinct = distinct.len() as u32;
                let mut dict = Vec::new();
                dict.extend_from_slice(&(distinct.len() as u32).to_le_bytes());
                dict.extend_from_slice(&[0u8; 4]);
                let mut off = 0u64;
                for s in &distinct {
                    dict.extend_from_slice(&off.to_le_bytes());
                    off += s.len() as u64;
                }
                dict.extend_from_slice(&off.to_le_bytes());
                for s in &distinct {
                    dict.extend_from_slice(s.as_bytes());
                }
                while !dict.len().is_multiple_of(8) {
                    dict.push(0);
                }
                attr.dict = Some((pos as usize, dict.len()));
                w.write_all(&dict)?;
                pos += dict.len() as u64;
                if distinct.len() <= MAX_DICT {
                    postings_by_code = Some((distinct.len(), codes));
                }
            }
            Kind::Int => {
                let col: Vec<i64> = (0..n)
                    .map(|s| value_at(s).and_then(Value::as_int).unwrap_or(0))
                    .collect();
                attr.column = pos as usize;
                w.write_all(bytemuck::cast_slice(&col))?;
                pos += (n * 8) as u64;
                let mut distinct: Vec<i64> = (0..n)
                    .filter_map(|s| value_at(s).and_then(Value::as_int))
                    .collect();
                distinct.sort_unstable();
                distinct.dedup();
                attr.n_distinct = distinct.len().min(u32::MAX as usize) as u32;
                if distinct.len() <= MAX_DICT {
                    let mut dict = Vec::new();
                    dict.extend_from_slice(&(distinct.len() as u32).to_le_bytes());
                    dict.extend_from_slice(&[0u8; 4]);
                    dict.extend_from_slice(bytemuck::cast_slice(&distinct));
                    attr.dict = Some((pos as usize, dict.len()));
                    w.write_all(&dict)?;
                    pos += dict.len() as u64;
                    let codes: Vec<u32> = col
                        .iter()
                        .map(|v| distinct.binary_search(v).unwrap_or(0) as u32)
                        .collect();
                    postings_by_code = Some((distinct.len(), codes));
                }
            }
            Kind::Float => {
                let col: Vec<f64> = (0..n)
                    .map(|s| value_at(s).and_then(Value::as_float).unwrap_or(0.0))
                    .collect();
                attr.column = pos as usize;
                w.write_all(bytemuck::cast_slice(&col))?;
                pos += (n * 8) as u64;
            }
            Kind::Bool => {
                let col: Vec<u8> = (0..n)
                    .map(|s| value_at(s).and_then(Value::as_bool).unwrap_or(false) as u8)
                    .collect();
                attr.column = pos as usize;
                let padded = n.next_multiple_of(8);
                w.write_all(&col)?;
                w.write_all(&vec![0u8; padded - n])?;
                pos += padded as u64;
                attr.n_distinct = 2;
                postings_by_code = Some((2, col.iter().map(|&b| b as u32).collect()));
            }
            Kind::Sparse => unreachable!("sparse values are not columns"),
        }
        if let Some((n_distinct, codes)) = postings_by_code {
            // Counting sort of present slots by code.
            let mut counts = vec![0u64; n_distinct + 1];
            for s in 0..n {
                if presence[s / 64] >> (s % 64) & 1 == 1 {
                    counts[codes[s] as usize + 1] += 1;
                }
            }
            for i in 1..=n_distinct {
                counts[i] += counts[i - 1];
            }
            let total = counts[n_distinct] as usize;
            let mut slots = vec![0u32; total];
            let mut fill = counts.clone();
            for s in 0..n {
                if presence[s / 64] >> (s % 64) & 1 == 1 {
                    let c = codes[s] as usize;
                    slots[fill[c] as usize] = s as u32;
                    fill[c] += 1;
                }
            }
            attr.postings = Some(pos as usize);
            w.write_all(bytemuck::cast_slice(&counts))?;
            w.write_all(bytemuck::cast_slice(&slots))?;
            let pad = total.next_multiple_of(2) - total;
            w.write_all(&vec![0u8; pad * 4])?;
            pos += ((n_distinct + 1) * 8 + (total + pad) * 4) as u64;
        }
        table.push(attr);
    }

    // The table, then the trailer that points at it.
    let table_off = pos;
    let mut t = Vec::new();
    t.extend_from_slice(&(n as u64).to_le_bytes());
    t.extend_from_slice(&(table.len() as u32).to_le_bytes());
    t.extend_from_slice(&[0u8; 4]);
    for a in &table {
        t.extend_from_slice(&(a.name.len() as u16).to_le_bytes());
        t.extend_from_slice(a.name.as_bytes());
        while !t.len().is_multiple_of(8) {
            t.push(0);
        }
        t.push(a.kind as u8);
        t.push((a.dict.is_some() as u8) | ((a.postings.is_some() as u8) << 1));
        t.extend_from_slice(&[0u8; 2]);
        t.extend_from_slice(&a.n_distinct.to_le_bytes());
        for off in [
            a.presence as u64,
            a.column as u64,
            a.dict.map_or(0, |d| d.0 as u64),
            a.dict.map_or(0, |d| d.1 as u64),
            a.postings.map_or(0, |p| p as u64),
        ] {
            t.extend_from_slice(&off.to_le_bytes());
        }
    }
    w.write_all(&t)?;
    pos += t.len() as u64;
    Ok((pos - start, table_off))
}

// ---------------------------------------------------------------- reading

fn u64_at(b: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(o..o + 8)?.try_into().ok()?))
}
fn u32_at(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}

impl Attrs {
    /// Parse the attribute table that starts at `table_off` in a mapped index
    /// file whose fixed sections end at `min_off`.
    pub(crate) fn parse(mmap: &[u8], min_off: usize, table_off: usize, n: usize) -> Result<Attrs> {
        let bad = || Error::Corrupt("attribute table inconsistent");
        if table_off < min_off || table_off + 16 > mmap.len() {
            return Err(bad());
        }
        let t = &mmap[table_off..];
        if u64_at(t, 0).ok_or_else(bad)? as usize != n {
            return Err(bad());
        }
        let count = u32_at(t, 8).ok_or_else(bad)? as usize;
        let mut pos = 16;
        let mut attrs = Vec::with_capacity(count);
        let mut by_name = HashMap::with_capacity(count);
        for _ in 0..count {
            let len = u16::from_le_bytes(t.get(pos..pos + 2).ok_or_else(bad)?.try_into().unwrap())
                as usize;
            let name = std::str::from_utf8(t.get(pos + 2..pos + 2 + len).ok_or_else(bad)?)
                .map_err(|_| bad())?
                .to_string();
            pos = (pos + 2 + len).next_multiple_of(8);
            let kind = Kind::from_u8(*t.get(pos).ok_or_else(bad)?).ok_or_else(bad)?;
            let flags = *t.get(pos + 1).ok_or_else(bad)?;
            let n_distinct = u32_at(t, pos + 4).ok_or_else(bad)?;
            let f = |i: usize| {
                u64_at(t, pos + 8 + i * 8)
                    .ok_or_else(bad)
                    .map(|v| v as usize)
            };
            let (presence, column, dict_off, dict_len, postings) =
                (f(0)?, f(1)?, f(2)?, f(3)?, f(4)?);
            pos += 8 + 5 * 8;
            let width = match kind {
                Kind::Str => 4,
                Kind::Int | Kind::Float => 8,
                Kind::Bool => 1,
                Kind::Sparse => return Err(bad()),
            };
            if presence + n.div_ceil(64) * 8 > table_off || column + n * width > table_off {
                return Err(bad());
            }
            let dict = (flags & 1 != 0).then_some((dict_off, dict_len));
            if dict.is_some_and(|(o, l)| o + l > table_off || l < 8) {
                return Err(bad());
            }
            let postings = (flags & 2 != 0).then_some(postings);
            if postings.is_some_and(|p| p + (n_distinct as usize + 1) * 8 > table_off) {
                return Err(bad());
            }
            by_name.insert(name.clone(), attrs.len());
            attrs.push(Attr {
                name,
                kind,
                n_distinct,
                presence,
                column,
                dict,
                postings,
            });
        }
        Ok(Attrs { n, attrs, by_name })
    }

    /// Every key with its type.
    pub(crate) fn kinds(&self) -> impl Iterator<Item = (&str, Kind)> {
        self.attrs.iter().map(|a| (a.name.as_str(), a.kind))
    }

    /// The metadata of one slot, decoded from the columns.
    pub(crate) fn record(&self, b: &[u8], slot: u32) -> Metadata {
        let mut m = Metadata::new();
        for (a, attr) in self.attrs.iter().enumerate() {
            if !self.present(b, a, slot) {
                continue;
            }
            let v = match attr.kind {
                Kind::Str => Value::Str(self.dict_str(b, a, self.code_at(b, a, slot)).to_string()),
                Kind::Int => Value::Int(self.int_at(b, a, slot)),
                Kind::Float => Value::Float(self.float_at(b, a, slot)),
                Kind::Bool => Value::Bool(self.bool_at(b, a, slot)),
                Kind::Sparse => continue,
            };
            m.insert(attr.name.clone(), v);
        }
        m
    }

    fn present(&self, b: &[u8], a: usize, slot: u32) -> bool {
        let s = slot as usize;
        let w = u64::from_le_bytes(
            b[self.attrs[a].presence + s / 64 * 8..][..8]
                .try_into()
                .unwrap(),
        );
        w >> (s % 64) & 1 == 1
    }
    fn code_at(&self, b: &[u8], a: usize, slot: u32) -> u32 {
        u32::from_le_bytes(
            b[self.attrs[a].column + slot as usize * 4..][..4]
                .try_into()
                .unwrap(),
        )
    }
    fn int_at(&self, b: &[u8], a: usize, slot: u32) -> i64 {
        i64::from_le_bytes(
            b[self.attrs[a].column + slot as usize * 8..][..8]
                .try_into()
                .unwrap(),
        )
    }
    fn float_at(&self, b: &[u8], a: usize, slot: u32) -> f64 {
        f64::from_le_bytes(
            b[self.attrs[a].column + slot as usize * 8..][..8]
                .try_into()
                .unwrap(),
        )
    }
    fn bool_at(&self, b: &[u8], a: usize, slot: u32) -> bool {
        b[self.attrs[a].column + slot as usize] != 0
    }

    /// The `Str` dictionary entry for a code.
    /// Dictionary entry for a code. Every read is bounds-checked against the
    /// dictionary region: a damaged file yields "" rather than a panic.
    fn dict_str<'b>(&self, b: &'b [u8], a: usize, code: u32) -> &'b str {
        let Some((d, len)) = self.attrs[a].dict else {
            return "";
        };
        let region = &b[d..(d + len).min(b.len())];
        let Some(count) = u32_at(region, 0) else {
            return "";
        };
        let offs = 8;
        let heap = offs + (count as usize + 1) * 8;
        let (Some(lo), Some(hi)) = (
            u64_at(region, offs + code as usize * 8),
            u64_at(region, offs + (code as usize + 1) * 8),
        ) else {
            return "";
        };
        let (lo, hi) = (lo as usize, hi as usize);
        if lo > hi {
            return "";
        }
        region
            .get(heap + lo..heap + hi)
            .and_then(|s| std::str::from_utf8(s).ok())
            .unwrap_or("")
    }
    fn dict_int(&self, b: &[u8], a: usize, i: u32) -> i64 {
        let Some((d, len)) = self.attrs[a].dict else {
            return 0;
        };
        u64_at(&b[d..(d + len).min(b.len())], 8 + i as usize * 8).map_or(0, |v| v as i64)
    }
    /// Binary search a dictionary: `Ok(code)` or `Err(insertion point)`.
    fn dict_search(&self, b: &[u8], a: usize, v: &Value) -> std::result::Result<u32, u32> {
        let n = self.attrs[a].n_distinct;
        let (mut lo, mut hi) = (0u32, n);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let o = match self.attrs[a].kind {
                Kind::Str => self.dict_str(b, a, mid).cmp(v.as_str().unwrap_or("")),
                _ => self.dict_int(b, a, mid).cmp(&v.as_int().unwrap_or(0)),
            };
            match o {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return Ok(mid),
            }
        }
        Err(lo)
    }
    /// Slots holding dictionary code `c`, ascending.
    fn postings<'b>(&self, b: &'b [u8], a: usize, c: u32) -> &'b [u32] {
        let Some(p) = self.attrs[a].postings else {
            return &[];
        };
        let nd = self.attrs[a].n_distinct as usize;
        if c as usize >= nd {
            return &[];
        }
        let (Some(lo), Some(hi)) = (
            u64_at(b, p + c as usize * 8),
            u64_at(b, p + (c as usize + 1) * 8),
        ) else {
            return &[];
        };
        let (lo, hi) = (lo as usize, hi as usize);
        let slots = p + (nd + 1) * 8;
        // Offsets must be ordered, inside the file and 4-aligned; the slots
        // they name are checked by the caller against `n`.
        match (lo <= hi)
            .then(|| b.get(slots + lo * 4..slots + hi * 4))
            .flatten()
        {
            Some(bytes)
                if bytes.len().is_multiple_of(4) && (bytes.as_ptr() as usize).is_multiple_of(4) =>
            {
                bytemuck::cast_slice(bytes)
            }
            _ => &[],
        }
    }

    // ------------------------------------------------------------ predicates

    /// Compile a filter against this table. Keys the table lacks never match
    /// (`Eq`, ranges) or always fail `Exists`; values of the wrong type never
    /// match, as `Filter::eval` does.
    pub(crate) fn prepare(&self, b: &[u8], f: &Filter, text: Option<(&TextIndex, &str)>) -> Pred {
        let attr = |k: &str| self.by_name.get(k).copied();
        match f {
            Filter::Eq(k, v) => attr(k).map_or(Pred::False, |a| self.eq_pred(b, a, v)),
            Filter::Ne(k, v) => attr(k).map_or(Pred::False, |a| {
                let numeric = |k: Kind| matches!(k, Kind::Int | Kind::Float);
                if self.attrs[a].kind != v.kind()
                    && !(numeric(self.attrs[a].kind) && numeric(v.kind()))
                {
                    return Pred::False;
                }
                Pred::And(vec![
                    Pred::Exists(a),
                    Pred::Not(Box::new(self.eq_pred(b, a, v))),
                ])
            }),
            Filter::In(k, vs) => attr(k).map_or(Pred::False, |a| {
                Pred::Or(vs.iter().map(|v| self.eq_pred(b, a, v)).collect())
            }),
            Filter::Lt(k, v) => attr(k).map_or(Pred::False, |a| {
                self.range_pred(b, a, None, Some((v, false)))
            }),
            Filter::Le(k, v) => attr(k).map_or(Pred::False, |a| {
                self.range_pred(b, a, None, Some((v, true)))
            }),
            Filter::Gt(k, v) => attr(k).map_or(Pred::False, |a| {
                self.range_pred(b, a, Some((v, false)), None)
            }),
            Filter::Ge(k, v) => attr(k).map_or(Pred::False, |a| {
                self.range_pred(b, a, Some((v, true)), None)
            }),
            Filter::Prefix(k, p) => attr(k).map_or(Pred::False, |a| {
                if self.attrs[a].kind != Kind::Str {
                    return Pred::False;
                }
                // Codes are sorted by string, so a prefix is a code range:
                // [first string >= p, first string that does not start with p).
                let lo = self
                    .dict_search(b, a, &Value::Str(p.clone()))
                    .unwrap_or_else(|i| i);
                let mut hi = lo;
                while hi < self.attrs[a].n_distinct
                    && self.dict_str(b, a, hi).starts_with(p.as_str())
                {
                    hi += 1;
                }
                Pred::CodeRange(a, lo, hi)
            }),
            Filter::Contains(k, tok) => {
                let toks = crate::text::tokenize(tok);
                let Some(t) = toks.first() else {
                    return Pred::False;
                };
                match text {
                    Some((_, field)) if field == k => Pred::TextHas(t.clone()),
                    _ => attr(k)
                        .filter(|&a| self.attrs[a].kind == Kind::Str)
                        .map_or(Pred::False, |a| Pred::StrTok(a, t.clone())),
                }
            }
            Filter::Exists(k) => attr(k).map_or(Pred::False, Pred::Exists),
            Filter::And(fs) => Pred::And(fs.iter().map(|f| self.prepare(b, f, text)).collect()),
            Filter::Or(fs) => Pred::Or(fs.iter().map(|f| self.prepare(b, f, text)).collect()),
            Filter::Not(f) => Pred::Not(Box::new(self.prepare(b, f, text))),
        }
    }

    fn eq_pred(&self, b: &[u8], a: usize, v: &Value) -> Pred {
        match (self.attrs[a].kind, v) {
            (Kind::Str, Value::Str(_)) => self
                .dict_search(b, a, v)
                .map_or(Pred::False, |c| Pred::Code(a, c)),
            (Kind::Int, Value::Int(i)) => self.int_eq(b, a, *i),
            (Kind::Int, Value::Float(x)) if x.fract() == 0.0 && x.abs() < 9.0e18 => {
                self.int_eq(b, a, *x as i64)
            }
            (Kind::Float, Value::Float(x)) => Pred::FloatRange(a, *x, *x, true, true),
            (Kind::Float, Value::Int(i)) => Pred::FloatRange(a, *i as f64, *i as f64, true, true),
            (Kind::Bool, Value::Bool(x)) => Pred::Code(a, *x as u32),
            _ => Pred::False,
        }
    }

    fn int_eq(&self, b: &[u8], a: usize, i: i64) -> Pred {
        if self.attrs[a].dict.is_some() {
            self.dict_search(b, a, &Value::Int(i))
                .map_or(Pred::False, |c| Pred::Code(a, c))
        } else {
            Pred::IntRange(a, i, i)
        }
    }

    /// Bounds are `(value, inclusive)`.
    fn range_pred(
        &self,
        b: &[u8],
        a: usize,
        lo: Option<(&Value, bool)>,
        hi: Option<(&Value, bool)>,
    ) -> Pred {
        let kind = self.attrs[a].kind;
        let typed = |v: &Value| {
            matches!(
                (kind, v),
                (Kind::Str, Value::Str(_))
                    | (Kind::Bool, Value::Bool(_))
                    | (Kind::Int | Kind::Float, Value::Int(_) | Value::Float(_))
            )
        };
        if lo.is_some_and(|(v, _)| !typed(v)) || hi.is_some_and(|(v, _)| !typed(v)) {
            return Pred::False;
        }
        match kind {
            Kind::Float => {
                let l = lo.map_or((f64::NEG_INFINITY, true), |(v, inc)| {
                    (v.as_float().unwrap(), inc)
                });
                let h = hi.map_or((f64::INFINITY, true), |(v, inc)| {
                    (v.as_float().unwrap(), inc)
                });
                Pred::FloatRange(a, l.0, h.0, l.1, h.1)
            }
            Kind::Int if self.attrs[a].dict.is_none() => {
                let l = lo.map_or(i64::MIN, |(v, inc)| int_lower(v, inc));
                let h = hi.map_or(i64::MAX, |(v, inc)| int_upper(v, inc));
                Pred::IntRange(a, l, h)
            }
            Kind::Bool => {
                // Two codes: false = 0, true = 1.
                let l = lo.map_or(0, |(v, inc)| v.as_bool().unwrap() as u32 + (!inc) as u32);
                let h = hi.map_or(2, |(v, inc)| v.as_bool().unwrap() as u32 + inc as u32);
                Pred::CodeRange(a, l, h.max(l))
            }
            _ => {
                // Dictionary-coded Str or Int: a code range [lo, hi).
                let l = lo.map_or(0, |(v, inc)| {
                    let v = if kind == Kind::Int {
                        Value::Int(int_lower(v, inc))
                    } else {
                        v.clone()
                    };
                    match self.dict_search(b, a, &v) {
                        Ok(c) => c + (kind == Kind::Str && !inc) as u32,
                        Err(i) => i,
                    }
                });
                let h = hi.map_or(self.attrs[a].n_distinct, |(v, inc)| {
                    let v = if kind == Kind::Int {
                        Value::Int(int_upper(v, inc))
                    } else {
                        v.clone()
                    };
                    match self.dict_search(b, a, &v) {
                        Ok(c) => c + (kind == Kind::Int || inc) as u32,
                        Err(i) => i,
                    }
                });
                Pred::CodeRange(a, l, h.max(l))
            }
        }
    }

    /// Does `slot` pass? One or two array reads per leaf.
    pub(crate) fn pass(&self, b: &[u8], p: &Pred, slot: u32, tx: Option<&TextIndex>) -> bool {
        match p {
            Pred::TextHas(t) => tx.is_some_and(|x| x.has(b, t, slot)),
            Pred::StrTok(a, t) => {
                self.present(b, *a, slot)
                    && crate::text::tokenize(self.dict_str(b, *a, self.code_at(b, *a, slot)))
                        .iter()
                        .any(|w| w == t)
            }
            Pred::False => false,
            Pred::Exists(a) => self.present(b, *a, slot),
            Pred::Code(a, c) => self.present(b, *a, slot) && self.code_of(b, *a, slot) == *c,
            Pred::CodeRange(a, lo, hi) => {
                self.present(b, *a, slot) && {
                    let c = self.code_of(b, *a, slot);
                    *lo <= c && c < *hi
                }
            }
            Pred::IntRange(a, lo, hi) => {
                self.present(b, *a, slot) && {
                    let v = self.int_at(b, *a, slot);
                    *lo <= v && v <= *hi
                }
            }
            Pred::FloatRange(a, lo, hi, li, hi_inc) => {
                self.present(b, *a, slot) && {
                    let v = self.float_at(b, *a, slot);
                    (if *li { *lo <= v } else { *lo < v })
                        && (if *hi_inc { v <= *hi } else { v < *hi })
                }
            }
            Pred::And(ps) => ps.iter().all(|p| self.pass(b, p, slot, tx)),
            Pred::Or(ps) => ps.iter().any(|p| self.pass(b, p, slot, tx)),
            Pred::Not(p) => !self.pass(b, p, slot, tx),
        }
    }

    /// The dictionary code of a slot for `Str`, dictionary-coded `Int`
    /// (found by search, since the column holds raw values) and `Bool`.
    fn code_of(&self, b: &[u8], a: usize, slot: u32) -> u32 {
        match self.attrs[a].kind {
            Kind::Str => self.code_at(b, a, slot),
            Kind::Bool => self.bool_at(b, a, slot) as u32,
            Kind::Int => self
                .dict_search(b, a, &Value::Int(self.int_at(b, a, slot)))
                .unwrap_or(u32::MAX),
            Kind::Float | Kind::Sparse => u32::MAX,
        }
    }

    /// Estimated number of slots that pass, for choosing a strategy. Exact
    /// for posting-backed leaves, sampled for the rest; `And` takes the
    /// smallest child, `Or` the sum.
    pub(crate) fn estimate(&self, b: &[u8], p: &Pred, tx: Option<&TextIndex>) -> usize {
        let n = self.n;
        match p {
            Pred::TextHas(t) => tx.map_or(0, |x| x.df(b, t)),
            Pred::False => 0,
            Pred::Code(a, c) if self.attrs[*a].postings.is_some() => self.postings(b, *a, *c).len(),
            Pred::CodeRange(a, lo, hi)
                if self.attrs[*a].postings.is_some() && (hi - lo) as usize <= MAX_RANGE_CODES =>
            {
                (*lo..*hi).map(|c| self.postings(b, *a, c).len()).sum()
            }
            Pred::And(ps) => ps
                .iter()
                .map(|p| self.estimate(b, p, tx))
                .min()
                .unwrap_or(n),
            Pred::Or(ps) => ps
                .iter()
                .map(|p| self.estimate(b, p, tx))
                .sum::<usize>()
                .min(n),
            Pred::Not(p) => n - self.estimate(b, p, tx),
            _ => self.sample(b, p, tx),
        }
    }

    /// Fraction passing over up to 1,024 evenly spaced slots, scaled to `n`.
    fn sample(&self, b: &[u8], p: &Pred, tx: Option<&TextIndex>) -> usize {
        if self.n == 0 {
            return 0;
        }
        let step = (self.n / 1024).max(1);
        let slots = (0..self.n).step_by(step);
        let total = slots.clone().count();
        let hits = slots.filter(|&s| self.pass(b, p, s as u32, tx)).count();
        (hits as f64 / total as f64 * self.n as f64) as usize
    }

    /// Every slot that passes, found through posting lists, or `None` when
    /// no posting-backed leaf constrains the predicate. The returned list is
    /// exact (each candidate is re-checked against the whole predicate) and
    /// ascending.
    pub(crate) fn candidates(
        &self,
        b: &[u8],
        p: &Pred,
        tx: Option<&TextIndex>,
    ) -> Option<Vec<u32>> {
        let seed = self.seed(b, p, tx)?;
        let n = self.n as u32;
        Some(
            seed.into_iter()
                .filter(|&s| s < n && self.pass(b, p, s, tx))
                .collect(),
        )
    }

    /// Smallest posting-backed superset of the slots passing `p`.
    fn seed(&self, b: &[u8], p: &Pred, tx: Option<&TextIndex>) -> Option<Vec<u32>> {
        match p {
            Pred::TextHas(t) => Some(tx?.postings(b, t).map(|(s, _)| s).collect()),
            Pred::Code(a, c) if self.attrs[*a].postings.is_some() => {
                Some(self.postings(b, *a, *c).to_vec())
            }
            Pred::CodeRange(a, lo, hi)
                if self.attrs[*a].postings.is_some() && (hi - lo) as usize <= MAX_RANGE_CODES =>
            {
                let mut v: Vec<u32> = (*lo..*hi)
                    .flat_map(|c| self.postings(b, *a, c).iter().copied())
                    .collect();
                v.sort_unstable();
                Some(v)
            }
            Pred::And(ps) => ps
                .iter()
                .filter_map(|p| self.seed(b, p, tx))
                .min_by_key(Vec::len),
            Pred::Or(ps) => {
                let mut all = Vec::new();
                for p in ps {
                    all.extend(self.seed(b, p, tx)?);
                }
                all.sort_unstable();
                all.dedup();
                Some(all)
            }
            _ => None,
        }
    }
}

/// Smallest integer satisfying `>= v` (inclusive) or `> v`.
fn int_lower(v: &Value, inclusive: bool) -> i64 {
    match v {
        Value::Int(i) => i.saturating_add(!inclusive as i64),
        _ => {
            let x = v.as_float().unwrap_or(0.0);
            if inclusive {
                x.ceil() as i64
            } else {
                x.floor() as i64 + 1
            }
        }
    }
}
/// Largest integer satisfying `<= v` (inclusive) or `< v`.
fn int_upper(v: &Value, inclusive: bool) -> i64 {
    match v {
        Value::Int(i) => i.saturating_sub(!inclusive as i64),
        _ => {
            let x = v.as_float().unwrap_or(0.0);
            if inclusive {
                x.floor() as i64
            } else {
                x.ceil() as i64 - 1
            }
        }
    }
}

/// A filter compiled against one index's attribute table.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Pred {
    False,
    Exists(usize),
    /// Dictionary code equality (`Str`, dictionary `Int`, `Bool`).
    Code(usize, u32),
    /// Dictionary codes in `[lo, hi)`.
    CodeRange(usize, u32, u32),
    /// Raw `Int` column in `[lo, hi]`.
    IntRange(usize, i64, i64),
    /// `Float` column between the bounds; the booleans say inclusive.
    FloatRange(usize, f64, f64, bool, bool),
    /// The text index holds this token for the slot.
    TextHas(String),
    /// A `Str` column whose value, tokenised, contains the token (no index).
    StrTok(usize, String),
    And(Vec<Pred>),
    Or(Vec<Pred>),
    Not(Box<Pred>),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(records: &[Metadata]) -> (Vec<u8>, Attrs) {
        let metas: Vec<Arc<Metadata>> = records.iter().cloned().map(Arc::new).collect();
        let mut buf = vec![0u8; 64]; // stands in for the fixed sections
        let n = records.len();
        let (_, table_off) = write(&mut buf, 64, n, &|s| metas.get(s as usize).cloned()).unwrap();
        let attrs = Attrs::parse(&buf, 64, table_off as usize, n).unwrap();
        (buf, attrs)
    }

    fn recs() -> Vec<Metadata> {
        (0..200i64)
            .map(|i| {
                let mut m = Metadata::new();
                m.insert(
                    "lang".into(),
                    Value::from(["en", "de", "fr", "it"][i as usize % 4]),
                );
                m.insert("year".into(), Value::from(2000 + i % 25));
                m.insert("score".into(), Value::from(i as f64 / 200.0));
                if i % 3 != 0 {
                    m.insert("public".into(), Value::from(i % 2 == 0));
                }
                if i >= 100 {
                    m.insert("big".into(), Value::from(i * 1_000_003));
                }
                m
            })
            .collect()
    }

    /// Every prepared predicate must agree with `Filter::eval` on every record.
    fn agrees(f: &Filter) {
        let recs = recs();
        let (b, attrs) = build(&recs);
        let p = attrs.prepare(&b, f, None);
        for (s, m) in recs.iter().enumerate() {
            assert_eq!(
                attrs.pass(&b, &p, s as u32, None),
                f.eval(m),
                "{f:?} on slot {s}: {m:?}"
            );
        }
        if let Some(c) = attrs.candidates(&b, &p, None) {
            let want: Vec<u32> = (0..recs.len() as u32)
                .filter(|&s| f.eval(&recs[s as usize]))
                .collect();
            assert_eq!(c, want, "candidates for {f:?}");
            let e = attrs.estimate(&b, &p, None);
            match f {
                // And takes its most selective child, Or sums: both upper bounds.
                Filter::And(_) | Filter::Or(_) => assert!(
                    e >= want.len(),
                    "estimate {e} below {} for {f:?}",
                    want.len()
                ),
                _ => assert_eq!(e, want.len(), "estimate for {f:?}"),
            }
        }
    }

    #[test]
    fn columns_agree_with_in_memory_evaluation() {
        for f in [
            Filter::eq("lang", "de"),
            Filter::eq("lang", "zz"),
            Filter::eq("lang", 3i64),
            Filter::ne("lang", "de"),
            Filter::is_in("lang", ["en", "it", "xx"]),
            Filter::lt("lang", "fr"),
            Filter::ge("lang", "fr"),
            Filter::gt("lang", "de"),
            Filter::le("lang", "de"),
            Filter::prefix("lang", "d"),
            Filter::prefix("lang", ""),
            Filter::eq("year", 2003i64),
            Filter::eq("year", 2003.0f64),
            Filter::eq("year", 2003.5f64),
            Filter::gt("year", 2020i64),
            Filter::le("year", 2001.5f64),
            Filter::ge("year", 2024i64),
            Filter::lt("year", 2000i64),
            Filter::gt("year", 2024i64),
            Filter::is_in("year", [2000i64, 2024, 1999]),
            Filter::eq("score", 0.5f64),
            Filter::lt("score", 0.1f64),
            Filter::ge("score", 0.995f64),
            Filter::gt("score", 1i64),
            Filter::eq("public", true),
            Filter::eq("public", false),
            Filter::ne("public", true),
            Filter::ge("public", true),
            Filter::exists("public"),
            Filter::exists("nope"),
            Filter::eq("nope", 1i64),
            Filter::eq("big", 150i64 * 1_000_003),
            Filter::gt("big", 190i64 * 1_000_003),
            Filter::and([
                Filter::eq("lang", "en"),
                Filter::eq("public", true),
                Filter::gt("year", 2010i64),
            ]),
            Filter::or([Filter::eq("lang", "en"), Filter::eq("year", 2024i64)]),
            Filter::not(Filter::eq("lang", "en")),
            Filter::and([
                Filter::not(Filter::exists("public")),
                Filter::lt("score", 0.2f64),
            ]),
        ] {
            agrees(&f);
        }
    }

    #[test]
    fn estimates_without_postings_are_sampled() {
        let recs = recs();
        let (b, attrs) = build(&recs);
        let p = attrs.prepare(&b, &Filter::lt("score", 0.25f64), None);
        let e = attrs.estimate(&b, &p, None);
        assert!(
            (40..=60).contains(&e),
            "estimate {e} for a 25% float range over 200 records"
        );
        assert!(
            attrs.candidates(&b, &p, None).is_none(),
            "float ranges have no postings"
        );
    }

    #[test]
    fn damaged_table_is_corrupt_not_a_panic() {
        let (b, _) = build(&recs());
        assert!(
            Attrs::parse(&b, 64, b.len() + 8, 200).is_err(),
            "table offset past the end"
        );
        assert!(
            Attrs::parse(&b, 64, 8, 200).is_err(),
            "table offset inside the fixed sections"
        );
        assert!(
            Attrs::parse(&b, 64, 72, 200).is_err(),
            "garbage where the table should be"
        );
    }
}
