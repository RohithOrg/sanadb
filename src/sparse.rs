//! Learned sparse vectors (SPLADE, BGE-M3 sparse and the like): a few
//! hundred `(term id, weight)` pairs per record under one metadata key,
//! scored by dot product against a query of the same shape.
//!
//! Section layout in the index file (8-aligned start, found through the
//! footer's fourth slot):
//! - field name: `len u64 | bytes | pad to 8`
//! - forward offsets: `n + 1` u64, in pairs, slot `s` owns `[o[s], o[s+1])`
//! - pairs: `(index u32, weight f32)` per record, ascending index
//! - term dictionary: `count u64 | count × u32 ascending | pad to 8`
//! - posting offsets: `count + 1` u64, in postings
//! - postings: `(slot u32, weight f32)`, ascending slot within a term
//! - table, ten u64: `n`, name offset, name length, forward offsets offset,
//!   pairs offset, pairs count, dictionary offset, term count, posting
//!   offsets offset, postings offset
//!
//! Records written since the last checkpoint keep their pairs in the
//! metadata map and are scored by brute force, like the dense delta.

use std::collections::HashMap;
use std::io::Write;

use crate::{Error, Result};

pub(crate) type Pairs = Vec<(u32, f32)>;

/// Sort by index, merge duplicates by summing, drop zero weights. Returns
/// false when a weight is not finite.
pub(crate) fn normalise(pairs: &mut Pairs) -> bool {
    if pairs.iter().any(|(_, w)| !w.is_finite()) {
        return false;
    }
    pairs.sort_unstable_by_key(|p| p.0);
    let mut out: Pairs = Vec::with_capacity(pairs.len());
    for &(i, w) in pairs.iter() {
        match out.last_mut() {
            Some(last) if last.0 == i => last.1 += w,
            _ => out.push((i, w)),
        }
    }
    out.retain(|p| p.1 != 0.0);
    *pairs = out;
    true
}

/// Dot product of two normalised pair lists (merge join).
pub(crate) fn dot(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    let (mut i, mut j, mut s) = (0, 0, 0.0f32);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                s += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
        }
    }
    s
}

fn pad8(w: &mut impl Write, pos: u64) -> std::io::Result<u64> {
    let pad = pos.next_multiple_of(8) - pos;
    w.write_all(&vec![0u8; pad as usize])?;
    Ok(pad)
}

/// Write the section for `n` slots; `of(slot)` is the slot's pairs. Returns
/// `(bytes written, table offset)`.
pub(crate) fn write(
    w: &mut impl Write,
    start: u64,
    n: usize,
    name: &str,
    of: &(dyn Fn(u32) -> Option<Pairs> + Sync),
) -> std::io::Result<(u64, u64)> {
    debug_assert!(start.is_multiple_of(8));
    let mut pos = start;
    let name_off = pos + 8;
    w.write_all(&(name.len() as u64).to_le_bytes())?;
    w.write_all(name.as_bytes())?;
    pos += 8 + name.len() as u64;
    pos += pad8(w, pos)?;

    // Forward index, collecting the inverted lists on the way.
    let mut by_term: HashMap<u32, Vec<(u32, f32)>> = HashMap::new();
    let mut offs = Vec::with_capacity(n + 1);
    let mut pairs: Pairs = Vec::new();
    for s in 0..n as u32 {
        offs.push(pairs.len() as u64);
        if let Some(p) = of(s) {
            for &(i, wt) in &p {
                by_term.entry(i).or_default().push((s, wt));
            }
            pairs.extend(p);
        }
    }
    offs.push(pairs.len() as u64);
    let fwd_off = pos;
    w.write_all(bytemuck::cast_slice(&offs))?;
    pos += (offs.len() * 8) as u64;
    let pairs_off = pos;
    let mut buf = Vec::with_capacity(pairs.len() * 8);
    for (i, wt) in &pairs {
        buf.extend_from_slice(&i.to_le_bytes());
        buf.extend_from_slice(&wt.to_le_bytes());
    }
    w.write_all(&buf)?;
    pos += buf.len() as u64;

    let mut terms: Vec<u32> = by_term.keys().copied().collect();
    terms.sort_unstable();
    let dict_off = pos;
    w.write_all(&(terms.len() as u64).to_le_bytes())?;
    w.write_all(bytemuck::cast_slice(&terms))?;
    pos += 8 + (terms.len() * 4) as u64;
    pos += pad8(w, pos)?;
    let post_offs_off = pos;
    let mut total = 0u64;
    let mut poffs = Vec::with_capacity(terms.len() + 1);
    for t in &terms {
        poffs.push(total);
        total += by_term[t].len() as u64;
    }
    poffs.push(total);
    w.write_all(bytemuck::cast_slice(&poffs))?;
    pos += (poffs.len() * 8) as u64;
    let postings_off = pos;
    buf.clear();
    for t in &terms {
        for (s, wt) in &by_term[t] {
            buf.extend_from_slice(&s.to_le_bytes());
            buf.extend_from_slice(&wt.to_le_bytes());
        }
    }
    w.write_all(&buf)?;
    pos += buf.len() as u64;

    let table_off = pos;
    for v in [
        n as u64,
        name_off,
        name.len() as u64,
        fwd_off,
        pairs_off,
        pairs.len() as u64,
        dict_off,
        terms.len() as u64,
        post_offs_off,
        postings_off,
    ] {
        w.write_all(&v.to_le_bytes())?;
        pos += 8;
    }
    Ok((pos - start, table_off))
}

fn u64_at(b: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(o..o + 8)?.try_into().ok()?))
}

/// Parsed table of a sparse section; all data stays in the mapping.
#[derive(Debug, Clone)]
pub(crate) struct SparseIndex {
    n: usize,
    name_off: usize,
    name_len: usize,
    fwd: usize,
    pairs: usize,
    n_pairs: usize,
    dict: usize,
    count: usize,
    post_offs: usize,
    postings: usize,
}

impl SparseIndex {
    pub(crate) fn parse(
        mmap: &[u8],
        min_off: usize,
        table_off: usize,
        n: usize,
    ) -> Result<SparseIndex> {
        let bad = || Error::Corrupt("sparse index table inconsistent");
        if table_off < min_off || table_off + 80 > mmap.len() {
            return Err(bad());
        }
        let f = |i: usize| {
            u64_at(mmap, table_off + i * 8)
                .ok_or_else(bad)
                .map(|v| v as usize)
        };
        let ix = SparseIndex {
            n: f(0)?,
            name_off: f(1)?,
            name_len: f(2)?,
            fwd: f(3)?,
            pairs: f(4)?,
            n_pairs: f(5)?,
            dict: f(6)?,
            count: f(7)?,
            post_offs: f(8)?,
            postings: f(9)?,
        };
        if ix.n != n
            || ix.name_off < min_off
            || ix.name_off + ix.name_len > table_off
            || ix.fwd + (n + 1) * 8 > table_off
            || ix.pairs + ix.n_pairs * 8 > table_off
            || ix.dict + 8 + ix.count * 4 > table_off
            || ix.post_offs + (ix.count + 1) * 8 > table_off
            || ix.postings + ix.n_pairs * 8 > table_off
            || u64_at(mmap, ix.fwd + n * 8).is_none_or(|last| last as usize != ix.n_pairs)
            || u64_at(mmap, ix.post_offs + ix.count * 8)
                .is_none_or(|last| last as usize != ix.n_pairs)
            || std::str::from_utf8(&mmap[ix.name_off..ix.name_off + ix.name_len]).is_err()
        {
            return Err(bad());
        }
        Ok(ix)
    }

    /// The metadata key the section indexes.
    pub(crate) fn name<'b>(&self, b: &'b [u8]) -> &'b str {
        std::str::from_utf8(&b[self.name_off..self.name_off + self.name_len]).unwrap_or("")
    }

    /// The pairs of one slot; `None` when it has none.
    pub(crate) fn record(&self, b: &[u8], slot: u32) -> Option<Pairs> {
        if slot as usize >= self.n {
            return None;
        }
        let lo = u64_at(b, self.fwd + slot as usize * 8)? as usize;
        let hi = u64_at(b, self.fwd + (slot as usize + 1) * 8)? as usize;
        if lo >= hi || hi > self.n_pairs {
            return None;
        }
        let region = b.get(self.pairs + lo * 8..self.pairs + hi * 8)?;
        Some(
            region
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| {
                    (
                        u32::from_le_bytes(c[..4].try_into().unwrap()),
                        f32::from_le_bytes(c[4..].try_into().unwrap()),
                    )
                })
                .collect(),
        )
    }

    fn find(&self, b: &[u8], term: u32) -> Option<usize> {
        let (mut lo, mut hi) = (0usize, self.count);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let t = u32::from_le_bytes(b.get(self.dict + 8 + mid * 4..)?[..4].try_into().ok()?);
            match t.cmp(&term) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// `(slot, weight)` pairs holding `term`, ascending by slot.
    pub(crate) fn postings<'b>(
        &self,
        b: &'b [u8],
        term: u32,
    ) -> impl Iterator<Item = (u32, f32)> + 'b {
        let range = self.find(b, term).and_then(|i| {
            let lo = u64_at(b, self.post_offs + i * 8)? as usize;
            let hi = u64_at(b, self.post_offs + (i + 1) * 8)? as usize;
            (lo <= hi && hi <= self.n_pairs).then_some((lo, hi))
        });
        let (lo, hi) = range.unwrap_or((0, 0));
        let base = self.postings;
        let n = self.n as u32;
        (lo..hi).filter_map(move |i| {
            let o = base + i * 8;
            let slot = u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?);
            let w = f32::from_le_bytes(b.get(o + 4..o + 8)?.try_into().ok()?);
            (slot < n).then_some((slot, w))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalise_sorts_merges_and_drops_zero() {
        let mut p = vec![(5, 1.0), (2, 0.5), (5, 2.0), (9, 0.0)];
        assert!(normalise(&mut p));
        assert_eq!(p, vec![(2, 0.5), (5, 3.0)]);
        let mut bad = vec![(1, f32::NAN)];
        assert!(!normalise(&mut bad));
    }

    #[test]
    fn dot_is_a_merge_join() {
        assert_eq!(dot(&[(1, 2.0), (3, 1.0)], &[(3, 4.0), (7, 1.0)]), 4.0);
        assert_eq!(dot(&[], &[(1, 1.0)]), 0.0);
    }

    #[test]
    fn write_then_read_back() {
        let docs: Vec<Option<Pairs>> = vec![
            Some(vec![(1, 0.5), (4, 1.5)]),
            None,
            Some(vec![(4, 2.0), (8, 0.25)]),
        ];
        let mut buf = vec![0u8; 16];
        let (len, table) = write(&mut buf, 16, 3, "sp", &|s| docs[s as usize].clone()).unwrap();
        assert_eq!(buf.len() as u64, 16 + len);
        let ix = SparseIndex::parse(&buf, 16, table as usize, 3).unwrap();
        assert_eq!(ix.name(&buf), "sp");
        assert_eq!(ix.record(&buf, 0), docs[0]);
        assert_eq!(ix.record(&buf, 1), None);
        assert_eq!(ix.record(&buf, 2), docs[2]);
        assert_eq!(ix.record(&buf, 3), None);
        assert_eq!(
            ix.postings(&buf, 4).collect::<Vec<_>>(),
            vec![(0, 1.5), (2, 2.0)]
        );
        assert_eq!(ix.postings(&buf, 9).count(), 0);
        assert!(SparseIndex::parse(&buf, 16, table as usize, 4).is_err());
    }
}
