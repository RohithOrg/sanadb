//! Keyword search: a tokeniser, an inverted index section in the index file,
//! and BM25 scoring. See `docs/design/hybrid-search.md`.
//!
//! Section layout (8-aligned, found through the footer's third slot):
//! - lengths: `n` `u32` token counts, one per slot (0 = no text).
//! - dictionary: `count u32 | pad u32 | offsets (count + 1) u64 | bytes`,
//!   terms sorted, as the attribute dictionaries.
//! - posting offsets: `(count + 1) u64`, indexes into the postings array.
//! - postings: `slot u32 | tf u16 | pad u16` per entry, slots ascending.
//! - table: `n | n_docs | total_len | lengths_off | dict_off | dict_len |
//!   post_offs_off | postings_off | count`, all `u64`.

use std::collections::HashMap;
use std::io::Write;

use crate::{Error, Result};

/// BM25 constants: the Robertson defaults every reference implementation
/// uses, so numbers compare against them.
const K1: f32 = 1.2;
const B: f32 = 0.75;
/// Longer tokens are almost always noise (base64, hashes); drop them.
const MAX_TOKEN: usize = 64;

/// Lowercase, split on anything that is not alphanumeric. No stemming, no
/// stop words: predictable, language-neutral, and what BM25 baselines use.
pub(crate) fn tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty() && t.len() <= MAX_TOKEN)
        .map(|t| t.to_lowercase())
        .collect()
}

/// Term frequencies of one document.
pub(crate) fn term_freqs(tokens: &[String]) -> HashMap<&str, u16> {
    let mut m = HashMap::new();
    for t in tokens {
        let e = m.entry(t.as_str()).or_insert(0u16);
        *e = e.saturating_add(1);
    }
    m
}

/// One term's BM25 contribution.
pub(crate) fn bm25(tf: f32, df: f32, n_docs: f32, len: f32, avg_len: f32) -> f32 {
    let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln();
    let norm = K1 * (1.0 - B + B * len / avg_len.max(1e-6));
    idf * tf * (K1 + 1.0) / (tf + norm)
}

// ---------------------------------------------------------------- writing

/// Write the text section for `n` slots; `text_of` yields a slot's text.
/// Returns `(bytes written, table offset)`.
pub(crate) fn write(
    w: &mut impl Write,
    start: u64,
    n: usize,
    text_of: &(dyn Fn(u32) -> Option<String> + Sync),
) -> std::io::Result<(u64, u64)> {
    debug_assert!(start.is_multiple_of(8));
    // Tokenise every slot; collect postings per term.
    let mut lengths = vec![0u32; n];
    let mut by_term: HashMap<String, Vec<(u32, u16)>> = HashMap::new();
    let mut n_docs = 0u64;
    let mut total_len = 0u64;
    for s in 0..n {
        let Some(text) = text_of(s as u32) else {
            continue;
        };
        let tokens = tokenize(&text);
        if tokens.is_empty() {
            continue;
        }
        n_docs += 1;
        total_len += tokens.len() as u64;
        lengths[s] = tokens.len().min(u32::MAX as usize) as u32;
        for (term, tf) in term_freqs(&tokens) {
            by_term
                .entry(term.to_string())
                .or_default()
                .push((s as u32, tf));
        }
    }
    let mut terms: Vec<&String> = by_term.keys().collect();
    terms.sort_unstable();

    let mut pos = start;
    let lengths_off = pos;
    w.write_all(bytemuck::cast_slice(&lengths))?;
    pos += (n * 4) as u64;
    pos += pad_to_8(w, pos)?;

    let dict_off = pos;
    let mut dict = Vec::new();
    dict.extend_from_slice(&(terms.len() as u32).to_le_bytes());
    dict.extend_from_slice(&[0u8; 4]);
    let mut off = 0u64;
    for t in &terms {
        dict.extend_from_slice(&off.to_le_bytes());
        off += t.len() as u64;
    }
    dict.extend_from_slice(&off.to_le_bytes());
    for t in &terms {
        dict.extend_from_slice(t.as_bytes());
    }
    while !dict.len().is_multiple_of(8) {
        dict.push(0);
    }
    w.write_all(&dict)?;
    pos += dict.len() as u64;

    let post_offs_off = pos;
    let mut offs = Vec::with_capacity(terms.len() + 1);
    let mut total = 0u64;
    for t in &terms {
        offs.push(total);
        total += by_term[*t].len() as u64;
    }
    offs.push(total);
    w.write_all(bytemuck::cast_slice(&offs))?;
    pos += (offs.len() * 8) as u64;

    let postings_off = pos;
    let mut buf = Vec::with_capacity(8 * 1024);
    for t in &terms {
        let list = &by_term[*t];
        buf.clear();
        for &(slot, tf) in list {
            buf.extend_from_slice(&slot.to_le_bytes());
            buf.extend_from_slice(&tf.to_le_bytes());
            buf.extend_from_slice(&[0u8; 2]);
        }
        w.write_all(&buf)?;
        pos += buf.len() as u64;
    }

    let table_off = pos;
    for v in [
        n as u64,
        n_docs,
        total_len,
        lengths_off,
        dict_off,
        dict.len() as u64,
        post_offs_off,
        postings_off,
        terms.len() as u64,
    ] {
        w.write_all(&v.to_le_bytes())?;
        pos += 8;
    }
    Ok((pos - start, table_off))
}

fn pad_to_8(w: &mut impl Write, pos: u64) -> std::io::Result<u64> {
    let pad = pos.next_multiple_of(8) - pos;
    w.write_all(&vec![0u8; pad as usize])?;
    Ok(pad)
}

// ---------------------------------------------------------------- reading

/// Parsed table of a text section; all data stays in the mapping.
#[derive(Debug, Clone)]
pub(crate) struct TextIndex {
    n: usize,
    pub n_docs: u64,
    pub total_len: u64,
    lengths: usize,
    dict: usize,
    dict_len: usize,
    post_offs: usize,
    postings: usize,
    count: usize,
}

fn u64_at(b: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(o..o + 8)?.try_into().ok()?))
}

impl TextIndex {
    pub(crate) fn parse(
        mmap: &[u8],
        min_off: usize,
        table_off: usize,
        n: usize,
    ) -> Result<TextIndex> {
        let bad = || Error::Corrupt("text index table inconsistent");
        if table_off < min_off || table_off + 72 > mmap.len() {
            return Err(bad());
        }
        let f = |i: usize| {
            u64_at(mmap, table_off + i * 8)
                .ok_or_else(bad)
                .map(|v| v as usize)
        };
        let (tn, n_docs, total_len, lengths, dict, dict_len, post_offs, postings, count) = (
            f(0)?,
            f(1)?,
            f(2)?,
            f(3)?,
            f(4)?,
            f(5)?,
            f(6)?,
            f(7)?,
            f(8)?,
        );
        if tn != n
            || lengths + n * 4 > table_off
            || dict + dict_len > table_off
            || dict_len < 8
            || post_offs + (count + 1) * 8 > table_off
            || postings > table_off
            || n_docs > n
        {
            return Err(bad());
        }
        let last = u64_at(mmap, post_offs + count * 8).ok_or_else(bad)? as usize;
        if postings + last * 8 > table_off {
            return Err(bad());
        }
        Ok(TextIndex {
            n,
            n_docs: n_docs as u64,
            total_len: total_len as u64,
            lengths,
            dict,
            dict_len,
            post_offs,
            postings,
            count,
        })
    }

    pub(crate) fn len_of(&self, b: &[u8], slot: u32) -> u32 {
        if slot as usize >= self.n {
            return 0;
        }
        u32::from_le_bytes(
            b[self.lengths + slot as usize * 4..][..4]
                .try_into()
                .unwrap(),
        )
    }

    fn term<'b>(&self, b: &'b [u8], i: usize) -> &'b str {
        let region = &b[self.dict..self.dict + self.dict_len];
        let heap = 8 + (self.count + 1) * 8;
        let (Some(lo), Some(hi)) = (u64_at(region, 8 + i * 8), u64_at(region, 8 + (i + 1) * 8))
        else {
            return "";
        };
        region
            .get(heap + lo as usize..heap + hi as usize)
            .and_then(|s| std::str::from_utf8(s).ok())
            .unwrap_or("")
    }

    fn find(&self, b: &[u8], term: &str) -> Option<usize> {
        let (mut lo, mut hi) = (0usize, self.count);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.term(b, mid).cmp(term) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// `(slot, tf)` pairs for a term, ascending by slot; empty if absent.
    pub(crate) fn postings<'b>(
        &self,
        b: &'b [u8],
        term: &str,
    ) -> impl Iterator<Item = (u32, u16)> + 'b {
        let range = self.find(b, term).and_then(|i| {
            let lo = u64_at(b, self.post_offs + i * 8)? as usize;
            let hi = u64_at(b, self.post_offs + (i + 1) * 8)? as usize;
            (lo <= hi).then_some((lo, hi))
        });
        let (lo, hi) = range.unwrap_or((0, 0));
        let base = self.postings;
        let n = self.n as u32;
        (lo..hi).filter_map(move |i| {
            let o = base + i * 8;
            let slot = u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?);
            let tf = u16::from_le_bytes(b.get(o + 4..o + 6)?.try_into().ok()?);
            (slot < n).then_some((slot, tf))
        })
    }

    /// Document frequency of a term.
    pub(crate) fn df(&self, b: &[u8], term: &str) -> usize {
        self.find(b, term)
            .and_then(|i| {
                Some(
                    (u64_at(b, self.post_offs + (i + 1) * 8)? - u64_at(b, self.post_offs + i * 8)?)
                        as usize,
                )
            })
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokeniser_is_lowercase_alphanumeric_split() {
        assert_eq!(
            tokenize("Hello, World! tokio::select v2.1"),
            ["hello", "world", "tokio", "select", "v2", "1"]
        );
        assert_eq!(tokenize("Ünïcödé façade"), ["ünïcödé", "façade"]);
        assert!(tokenize("   ").is_empty());
    }

    #[test]
    fn section_round_trips_and_scores_like_the_formula() {
        let docs = ["the quick brown fox", "the lazy dog", "", "quick quick fox"];
        let mut buf = vec![0u8; 64];
        let (_, table) = write(&mut buf, 64, docs.len(), &|s| {
            Some(docs[s as usize].to_string())
        })
        .unwrap();
        let ix = TextIndex::parse(&buf, 64, table as usize, docs.len()).unwrap();
        assert_eq!(ix.n_docs, 3);
        assert_eq!(ix.total_len, 4 + 3 + 3);
        assert_eq!(ix.len_of(&buf, 0), 4);
        assert_eq!(ix.len_of(&buf, 2), 0);
        assert_eq!(
            ix.postings(&buf, "quick").collect::<Vec<_>>(),
            [(0, 1), (3, 2)]
        );
        assert_eq!(
            ix.postings(&buf, "the").collect::<Vec<_>>(),
            [(0, 1), (1, 1)]
        );
        assert_eq!(ix.df(&buf, "fox"), 2);
        assert_eq!(ix.df(&buf, "cat"), 0);
        assert!(ix.postings(&buf, "cat").next().is_none());
        // A rarer term scores higher than a common one at equal tf and length.
        let avg = ix.total_len as f32 / ix.n_docs as f32;
        assert!(bm25(1.0, 1.0, 3.0, 3.0, avg) > bm25(1.0, 2.0, 3.0, 3.0, avg));
        // Repetition helps with diminishing returns.
        let (a, b, c) = (
            bm25(1.0, 1.0, 3.0, 3.0, avg),
            bm25(2.0, 1.0, 3.0, 3.0, avg),
            bm25(3.0, 1.0, 3.0, 3.0, avg),
        );
        assert!(b > a && c > b && (c - b) < (b - a));
        // Damage is reported, not a panic.
        let mut bad = buf.clone();
        let l = bad.len();
        bad[l - 8..].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(TextIndex::parse(&bad, 64, table as usize, docs.len()).is_err());
    }
}
