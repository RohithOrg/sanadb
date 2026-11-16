# Hybrid search: BM25 and sparse vectors beside the graph

Status: steps 1 to 3 of "Order of work" are implemented (tokeniser without
the segmentation dependency, inverted index section, `search_text` with
term-at-a-time BM25, `search_hybrid` with reciprocal rank fusion, Python
bindings). Not yet: MaxScore skipping, block-max metadata, weighted fusion,
sparse vectors. SciFact nDCG@10 0.662 against rank_bm25's 0.654 and
Anserini's published 0.665.

## What users want

Two things, usually conflated:

1. **Keyword search over a text field**, scored with BM25, fused with the
   vector result. "Find passages about `tokio::select` near this embedding"
   where the identifier never embeds well. This is the common case.
2. **Learned sparse vectors** (SPLADE, BGE-M3 sparse): a few hundred
   `(term id, weight)` pairs per record, scored by dot product against the
   query's sparse vector. The retrieval machinery is the same inverted index
   with weights instead of term frequencies.

Both reduce to an inverted index over records, scored by a sum over the
query's terms, and a fusion step with the dense k-NN. One design covers
both; BM25 ships first.

## Storage

A new index section, written at checkpoint next to the attribute columns
and found through the footer:

- **Term dictionary**: sorted terms (strings for text, `u32` ids for sparse)
  with offsets into the postings, same shape as the attribute dictionaries.
- **Postings** per term: ascending `(slot u32, value)` pairs, `value` a `u16`
  term frequency for text or an `f32` weight for sparse. Delta-encoded,
  byte-aligned, no compression in the first version; SIFT-sized corpora
  with short passages are tens of MB. Block-max metadata (the largest value
  in each 128-posting block) is stored alongside so that MaxScore can skip.
- **Record lengths**: a `u32` per slot (token count) for BM25's length
  normalisation; average length and document count in the section header.
- **Tail**: records since the last checkpoint keep their tokens or sparse
  pairs in the in-RAM map, scored by brute force like the dense delta. A
  reader process gets them through the same refresh path.

Text comes from a metadata key the user names when opening the store
(`Options::text_field("body")`). The record's `Str` value is tokenised and
indexed; the raw text stays in the column as today. Sparse vectors are a new
`Value::Sparse(Vec<(u32, f32)>)`, so they travel through the log and the
batch API unchanged. Reader level of the log goes up by one for the new
value tag; the index section is additive.

## Tokeniser

Unicode word segmentation (`unicode-segmentation`, the one new dependency),
lowercase, no stemming, no stop words in the first version. Code
identifiers split on `_`, `::`, `.` and camelCase boundaries as well, since
code is the case where keywords beat embeddings most often. The tokeniser
is a trait, with the default one exported, so users with a language model's
tokeniser can bring it. Whatever tokeniser indexed a store is recorded in
the section header by name, and a query with a different one is an error.

## Query

```rust
db.search_hybrid(&query_vec, "tokio select cancellation", k, Fusion::Rrf { k: 60 })
db.search_sparse(&sparse_query, k)
```

- BM25 with k1 = 1.2, b = 0.75 over the postings, MaxScore to stop scoring
  records that cannot reach the current k-th score. Candidates are slots;
  tombstones and the filter (a `Filter` applies here too, evaluated on the
  columns) are checked as slots are scored.
- Dense k-NN as today, over the same filter.
- **Fusion**: reciprocal rank fusion by default (`1 / (k + rank)` summed
  over both lists), because it needs no score calibration between a BM25
  score and a distance. Weighted linear fusion of min-max-normalised scores
  is offered for users who tune it. Both lists are fetched with `k' = 3k`
  so the fused top-k has material to work with; that multiplier is an
  option.
- A keyword-only query (`search_text`) is the inverted index alone, and is
  what makes the store usable as a small full-text engine for metadata.

## Cost model

A text query touches the postings of its terms: a handful of sequential
reads from the mapping, then MaxScore over a few thousand candidates with
one `u32` length lookup each. Expected 1 to 5 ms warm for a 1M-passage
corpus with 5-term queries, dominated by the dense side it fuses with. Cold
it is a few sequential reads, far cheaper than the dense walk's random ones.

Index size: about 1.5 bytes per posting with delta encoding, so a 1M-record
corpus of 200-token passages is roughly 300 MB of postings, on disk, mapped,
not resident. The build is a counting sort by term over the tokenised
corpus, parallel by shard, minutes at 1M.

## What this does not do

- No phrase or proximity queries, no wildcards, no faceting. A `Filter`
  on columns covers the structured part; this covers ranked keyword match.
- No incremental index: like the graph, the inverted index is rebuilt or
  updated at checkpoint; the tail is brute-forced until then.
- No learned fusion. RRF and a weighted sum are what every published
  comparison lands on for small corpora.

## Measuring it

BEIR's small public sets with published qrels (SciFact, NFCorpus, FiQA)
give BM25, dense and hybrid nDCG@10 numbers to compare against the
reference implementations (Pyserini BM25 for keyword, the embedding model's
own dense score). The target is matching Pyserini's BM25 within noise and
showing the hybrid above both parts, which is what the literature reports.
Those three datasets are small enough (a few thousand documents) to run in
CI.

## Order of work

1. Tokeniser trait and default; `Options::text_field`; tokens in the tail.
2. Inverted index section at checkpoint; `search_text` with BM25 and
   MaxScore; BEIR SciFact against Pyserini.
3. `search_hybrid` with RRF, then weighted fusion; filters on both sides.
4. `Value::Sparse` and `search_sparse`, reusing the postings with weights.
5. Python: `search(query, text=..., fusion=...)`, `search_text`.
