# Typed metadata and filter-aware search

Status: steps 1 to 4 of "Order of work" shipped in 0.1.4 (typed values with
log reader level 2 and migration, attribute columns and posting lists in the
index, the `Filter` type with the Python operator dict, the exact-scan regime
and a filtered walk whose result slots admit only passing nodes). Not done:
shrinking `Inner::meta` to the delta (the columns exist, the RAM map is still
authoritative for `get` and for delta records), two-hop expansion through
rejected nodes, and the YFCC measurements. The regime thresholds below are
the shipped defaults; `Options::exact_filter_scan` is the one knob.

## Where we are

Metadata today is `HashMap<String, String>` per record, held entirely in RAM
in `Inner::meta`, written to the log as `(key, value)` string pairs inside
Insert and Meta frames. A filter is a Rust closure `Fn(&Metadata) -> bool`
(Python: a dict of equalities, or a callable). Search applies the filter to
the nodes the walk visited; if fewer than `k` pass, the walk runs again with
the beam doubled, up to `max_search_beam`.

What that costs, measured on SIFT1M warm (the 0.1.4 profile):

- The per-visited-node `HashMap<u64, Metadata>` lookup was 6% of an
  unfiltered query before 0.1.4 skipped it. A filtered query still pays it,
  plus the closure, per visited node.
- A 1% filter at `k = 10` needs about 1,000 visited nodes to pass 10, so the
  walk retries at beam 128, 256, 512, 1,024: five walks, roughly 15x the cost
  of the unfiltered query. At 0.1% selectivity it reaches `max_search_beam`
  (4,096) and may still come back short.
- No range, prefix, set-membership or numeric comparison without a callable,
  and a Python callable holds the GIL per visited node.
- All metadata lives in RAM. At 5M records with two short string fields that
  is about 700 MB of `HashMap<String, String>` overhead, more than the PQ
  codes (160 MB) that are the point of the disk-first design.

## Goals

1. Typed values: string, integer, float, boolean, and absent. Enough for
   `doc_id`, `timestamp`, `score`, `is_public`, `lang`.
2. Filters as data, not closures: an expression the engine can inspect, so
   it can estimate selectivity, use an index, and run without the GIL.
3. Filtered search whose cost follows the filter, so that a 0.1% filter is
   maybe 3 to 5x an unfiltered query, not 100x, and never silently short.
4. Metadata out of the RAM hot path: a slot-ordered column per attribute in
   the index file, read through the same mmap as the vectors.
5. Old stores keep opening. Old readers refuse new stores with
   `NeedsNewerVersion` rather than misreading them.

Non-goals: SQL, joins, secondary indexes on strings beyond equality, nested
documents, per-field schemas declared up front. Keys stay dynamic; the first
value written for a key fixes its type for that store.

## Values and encoding

```rust
pub enum Value { Str(String), Int(i64), Float(f64), Bool(bool) }
pub type Metadata = HashMap<String, Value>;
```

Log frames gain a tag byte per value: `tag u8 | payload`, tags 0 = Str
(u16 len + bytes, as today), 1 = Int (i64 LE), 2 = Float (f64 LE), 3 = Bool
(u8). Insert and Meta frames keep their kinds; only the metadata section
changes, so the vector offset arithmetic is untouched. Because an old reader
would parse the tag byte as the start of a string length, this is not an
additive change: the log header's reader level goes to 2 and `migrate_v1`
gains a sibling that rewrites a level-1 log, tagging every value as Str.
Python keeps accepting plain `str` values and maps `int`, `float`, `bool`
to the new tags; `from_str` conversions never happen implicitly.

A type conflict (an `Int` written for a key that already holds `Str`) is an
`Error::TypeMismatch` on insert, checked against the attribute table below.

## Attribute columns in the index file

At checkpoint, every key seen across live records gets a column laid out by
slot, written after the PQ section of the `.idx` file:

- `Int`, `Float`, `Bool`: fixed width, `n` entries plus an `n`-bit presence
  bitmap. 8 bytes per record per numeric attribute at 5M is 40 MB per
  attribute, on disk, mapped, not resident unless touched.
- `Str`: dictionary-encoded. Distinct values sorted, stored once; the column
  is a `u32` code per slot. Equality and `IN` become integer compares on the
  code; prefix and range on the sorted dictionary become a code range.
  Attributes with more than 2^24 distinct values fall back to an offset
  column into a string heap (equality only).
- Low-cardinality attributes (at most 256 distinct values, whatever the
  type) also get one roaring-style bitmap per value: the set of slots that
  hold it. This is what makes selective filters cheap (see below).

The attribute table (key, type, cardinality, column offset, bitmap offset)
sits in the index header. The index file's reader level also goes to 2.
Records in the delta keep their metadata in RAM as now; the delta is small
by construction (bounded by `auto_checkpoint_delta`).

The in-RAM `Inner::meta` map shrinks to delta records and tombstone
bookkeeping once columns exist; `get()` reads the column for graph records.
That removes the per-record `HashMap<String, String>` for the bulk of the
store, which is the memory goal above.

## Filter expressions

```rust
pub enum Filter {
    Eq(Key, Value), Ne(Key, Value),
    In(Key, Vec<Value>),
    Lt(Key, Value), Le(Key, Value), Gt(Key, Value), Ge(Key, Value),
    Prefix(Key, String),
    Exists(Key),
    And(Vec<Filter>), Or(Vec<Filter>), Not(Box<Filter>),
}
```

Python: a dict is still accepted and means `And` of `Eq`s, as today. A small
operator form covers the rest without a query language:
`{"year": {"$gte": 2020}, "lang": {"$in": ["en", "de"]}}`. The MongoDB-style
spelling is chosen because every vector-store user has already seen it
(Qdrant, Chroma, LanceDB all accept something close); we do not need to
invent one. Callables stay supported as the escape hatch and keep today's
post-filter path and today's cost.

Evaluation against a graph slot reads the columns directly: for `Eq` on a
dictionary string that is one `u32` load and compare, no hashing, no
allocation, no GIL. Against a delta record it reads the RAM map as now.

## Filter-aware traversal

Three regimes, chosen per query from the filter's estimated selectivity `s`
(fraction of live records that pass). The estimate comes from the bitmaps
(exact cardinality for low-cardinality predicates), the dictionary (code
range width for string ranges), or a 1,024-slot sample of the column for
numeric ranges; `And` multiplies, `Or` adds and clamps, which is wrong under
correlation but only has to pick a regime, not be exact.

**Permissive filters, `s` above about 10%.** Today's walk with the filter
applied to visited nodes, one retry at most. Nothing to gain from more
machinery; the walk visits a few hundred nodes and tens pass.

**Selective filters, `s` between about 0.05% and 10%.** The walk stays on
the full graph, since the graph's connectivity is what gives recall, but two
changes keep it from wandering:

- Predicate-aware expansion, as in ACORN (Patel et al., SIGMOD 2024): when a
  visited node fails the filter, its neighbours are still scored and the
  passing ones enter the candidate list, and if too few of its neighbours
  pass, their neighbours are scored too (a two-hop expansion, capped at
  `R * gamma` candidates per expansion with `gamma` chosen from `s`). The
  intuition is that a graph built without filters loses connectivity among
  the passing subset; expanding through failing nodes restores it without
  building per-filter graphs. The visited bitmap and heaps from 0.1.4 make
  the extra scoring cheap.
- The bound for entering the result heap is computed over passing nodes
  only; failing nodes are scored for navigation but never occupy result
  slots, so a beam of 64 means 64 passing candidates. Today a 1% filter
  fills the beam with 63 failing nodes and that is why it retries.

Expected cost on a 1% filter: 3 to 5x the unfiltered query at equal recall,
down from about 15x. ACORN's paper reports 2 to 10x over post-filtering on
graphs with similar degree; we will measure before claiming anything.

**Very selective filters, `s` below about 0.05%, or at most ~2,500 passing
records at 5M.** Skip the graph. Enumerate the passing slots from the
bitmaps (intersections and unions are word-wise AND/OR over mapped bitmaps),
read each vector once, exact distances, top-k. That is at most a few
thousand 512-byte reads, faster than any walk, and recall is 1.0 by
construction. This is the regime where graph methods fail silently and where
users complain; the switch makes it the easy case instead. Predicates with
no bitmap (numeric ranges on high-cardinality columns) fall back to the
selective regime.

The thresholds are `Options` with the defaults above. `search_filtered`
returns the regime used in a debug accessor so benchmarks can report it.

## What this does not fix

- Filters on delta records still go through the RAM map and a linear scan
  of the delta (already exact and small).
- Correlated predicates can make the selectivity estimate wrong by an order
  of magnitude; the consequence is a slower query, not a wrong one, because
  every regime applies the filter exactly.
- Filtered-DiskANN style label-aware graph construction (one entry point and
  one edge budget per label) is the other published approach; it gives
  better recall at very low selectivity but needs the labels at build time
  and fixes the label set. ACORN-style expansion keeps the build
  filter-agnostic, which matches an embedded store where the schema is not
  known when the graph is built. Revisit if measurements say the middle
  regime is not good enough.

## Format and compatibility

- Log: reader level 2, value tags. Migration on open rewrites a level-1 log
  through a temp file and rename (same shape as `migrate_v1`), tagging
  every value Str. Index: reader level 2, attribute table and columns
  appended; a level-1 index opens fine (no columns, filters use the RAM
  map) and the next checkpoint writes the new layout. So upgrading is: open
  the store once (log converted), checkpoint once (columns built). No
  downgrade, as before; CHANGELOG says so.
- `Metadata` changing from `HashMap<String, String>` to
  `HashMap<String, Value>` is a breaking Rust API change: 0.2.0. Python stays
  source compatible since plain strings still work.

## Measuring it

- Correctness: every regime against a brute-force filtered scan on random
  data with planted selectivities (10%, 1%, 0.1%, 0.01%), with `And`/`Or`
  combinations and conflicting types; property test that regime choice never
  changes the result set beyond the recall the walk already has.
- Performance: big-ann-benchmarks' filter track (YFCC-10M with per-record
  tag sets and query predicates; 100K-record subsets for the laptop) is the
  public dataset with real filter selectivities and published ground truth.
  Add it to `benchmarks/` next to the ann-benchmarks adapter and report
  recall and QPS per selectivity bucket against the post-filter baseline
  that 0.1.4 ships.
- Memory: `Inner::meta` size at 5M records before and after columns.

## Order of work

1. `Value` enum, tagged log encoding, reader level 2 on the log, migration,
   tests, Python type mapping. Ships as 0.2.0 with the API break and no
   behaviour change in search.
2. Attribute columns and bitmaps at checkpoint; `get()` and filters read
   them; `Inner::meta` shrinks to the delta. Index reader level 2.
3. `Filter` expression type, Python operator dict, selectivity estimate.
4. The very-selective regime (bitmap scan), since it is simple and removes
   the silent-short-result failure.
5. The selective regime (predicate-aware expansion), with the YFCC
   measurements that decide its defaults.
