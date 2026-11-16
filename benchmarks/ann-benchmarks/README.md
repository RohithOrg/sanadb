# ann-benchmarks adapters and results

Adapters for the [ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
harness: `sanadb/` (copy into `ann_benchmarks/algorithms/sanadb/`),
`usearch/` (same, the upstream repo has no usearch adapter), and
`faiss_hnsw.config.yml` (the upstream faiss config trimmed to the M=16 and
M=24 groups). Run in local mode, one algorithm at a time:

    python run.py --local --runs 2 --dataset glove-100-angular --algorithm sanadb
    python data_export.py --output results.csv

## Keeping measurements

A full four-library session on one dataset takes about 80 minutes, so
results are kept, not regenerated:

- `results/*.csv` in this directory is the export of the session the
  README and `docs/benchmarks.md` tables and charts come from, one file per
  dataset. `docs/benchmarks.md` names the date and machine.
  `python benchmarks/ann-benchmarks/tables.py results.csv` prints the
  tables; `python benchmarks/plots.py results.csv` draws the charts into
  `docs/img/`.
- The harness's own `results/<dataset>/10/<algorithm>/*.hdf5` files (one
  per configuration and query setting, about 170 MB for the two datasets)
  are the raw measurements. They live outside git in the harness checkout;
  keep a dated copy of that directory beside it after a session you intend
  to cite, and never pass `--force` for a library you are not deliberately
  re-measuring: `--force` overwrites that library's files.
- `SANADB_ANNB_KEEP=<dir>` makes the sanadb adapter keep each
  configuration's store on disk and reuse it when the index already holds
  every vector, so query-only reruns skip the build.

All numbers in the tables were taken in one session per dataset, on one
machine, single query thread, best of two runs, warm. Mixing sessions is
what the caveats in `docs/benchmarks.md` exist to prevent.
