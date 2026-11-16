# sanadb studio

A local web page for one [sanadb](https://crates.io/crates/sanadb) store:
a map of the vectors, every search mode, the graph walk of a search drawn
on the map, record inspection, metadata editing. One binary, nothing
leaves the machine.

```bash
pip install "sanadb[studio]"      # or: cargo install sanadb-studio
sanadb-studio data/vectors.sana
# sanadb studio 0.2.3: data/vectors.sana (1000000 records, dim 128, L2) at http://127.0.0.1:7070/
```

`sanadb-studio <store.sana> [--port 7070] [--text-field KEY] [--read-only]`.
Documentation and screenshots: https://rohithorg.github.io/sanadb/studio/
