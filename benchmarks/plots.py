"""Charts for docs/benchmarks.md, regenerated from measured data so they
cannot drift from the tables.

    python benchmarks/plots.py benchmarks/ann-benchmarks/results/*.csv

Writes SVGs into docs/img/. The ann-benchmarks curves come from the CSV
that `data_export.py` writes; the filtered-search and cold-cache charts
hold the numbers printed by examples/filtered.rs, examples/yfcc.rs and
examples/bigann.rs, copied here next to the tables they illustrate.
"""
import csv
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/img"
COLOURS = {
    "hnswlib": "#7a7a7a",
    "faiss HNSW": "#b08a3e",
    "usearch": "#5b8fb9",
    "sanadb": "#c0392b",
}


def style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 100,
        "svg.fonttype": "none",
    })


def label(row):
    a, p = row["algorithm"], row["parameters"]
    m = re.search(r"'M': (\d+)", p)
    if a in ("hnswlib", "usearch"):
        return a, f"{a} M={m.group(1)}"
    if a == "hnsw(faiss)":
        return "faiss HNSW", f"faiss HNSW M={m.group(1)}"
    if a == "sanadb":
        r = re.search(r"'R': (\d+)", p).group(1)
        pq = "PQ" if "'pq': True" in p else "no PQ"
        return "sanadb", f"sanadb R={r} {pq}"
    return None, None


def ann_curves(paths):
    rows = []
    for path in paths:
        # A harness export has a `dataset` column; the per-dataset CSVs in
        # benchmarks/ann-benchmarks/results/ are named after their dataset.
        name = re.sub(r"\.csv$", "", path.rsplit("/", 1)[-1])
        for r in csv.DictReader(open(path)):
            r.setdefault("dataset", name)
            rows.append(r)
    for ds, title in [("sift-128-euclidean", "SIFT1M, L2"), ("glove-100-angular", "GloVe, 1.18M, cosine")]:
        series = defaultdict(list)
        for r in rows:
            if r["dataset"] != ds:
                continue
            lib, name = label(r)
            if name:
                series[(lib, name)].append((float(r["k-nn"]), float(r["qps"])))
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for (lib, name), pts in sorted(series.items()):
            pts.sort()
            dashed = "M=24" in name or "no PQ" in name or "R=48" in name
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3,
                    lw=1.6 if lib == "sanadb" else 1.1, color=COLOURS[lib],
                    ls="--" if dashed else "-", label=name)
        ax.set_yscale("log")
        ax.set_xlabel("recall@10")
        ax.set_ylabel("queries per second (log)")
        ax.set_title(f"{title}: ann-benchmarks, single thread, warm")
        ax.legend(fontsize=8, ncol=2, frameon=False)
        lo = 0.7 if ds.startswith("glove") else 0.9
        ax.set_xlim(lo, 1.0)
        fig.tight_layout()
        fig.savefig(f"{OUT}/annb-{ds}.svg")
        plt.close(fig)


def filtered():
    # examples/filtered.rs, SIFT1M with labels, beam 64, warm (docs/benchmarks.md)
    sel = ["10%", "1%", "0.1%", "0.01%"]
    col = [0.995, 1.000, 1.000, 1.000]
    clo = [0.792, 0.900, 0.399, 0.044]
    col_ms = [1.04, 1.34, 0.110, 0.009]
    clo_ms = [0.42, 3.96, 10.4, 10.5]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    x = range(len(sel))
    w = 0.38
    a1.bar([i - w / 2 for i in x], col, w, color=COLOURS["sanadb"], label="Filter (columns)")
    a1.bar([i + w / 2 for i in x], clo, w, color=COLOURS["hnswlib"], label="closure (post-filter)")
    a1.set_xticks(list(x), sel)
    a1.set_ylim(0, 1.05)
    a1.set_ylabel("recall@10")
    a1.set_xlabel("records matching the filter")
    a1.legend(fontsize=8, frameon=False, loc="lower left")
    a2.bar([i - w / 2 for i in x], col_ms, w, color=COLOURS["sanadb"])
    a2.bar([i + w / 2 for i in x], clo_ms, w, color=COLOURS["hnswlib"])
    a2.set_yscale("log")
    a2.set_xticks(list(x), sel)
    a2.set_ylabel("mean latency, ms (log)")
    a2.set_xlabel("records matching the filter")
    fig.suptitle("Filtered search, SIFT1M with labels, beam 64, warm", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/filtered-sift1m.svg")
    plt.close(fig)

    # examples/yfcc.rs, 1M records of the big-ann filter track
    buckets = ["<100", "100-1K", "1K-10K", ">10K"]
    rec = [1.000, 1.000, 1.000, 0.994]
    post = [0.173, 0.422, 0.829, 0.818]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar([i - w / 2 for i in x], rec, w, color=COLOURS["sanadb"], label="Filter::contains on the text index")
    ax.bar([i + w / 2 for i in x], post, w, color=COLOURS["hnswlib"], label="post-filter")
    ax.set_xticks(list(x), buckets)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall@10")
    ax.set_xlabel("records carrying the query's labels")
    ax.set_title("YFCC filter track, first 1M records, beam 64", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/filtered-yfcc.svg")
    plt.close(fig)


def cold():
    # examples/bigann.rs, 5M vectors, after sudo purge (docs/benchmarks.md)
    cfg = ["PQ on\nwidth 1", "PQ off\nwidth 1", "PQ on\nwidth 4", "PQ on\nwidth 8"]
    mean = [8.6, 21.9, 2.75, 2.26]
    p99 = [39, 187, 4.2, 3.4]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = range(len(cfg))
    w = 0.38
    ax.bar([i - w / 2 for i in x], mean, w, color=COLOURS["sanadb"], label="cold mean")
    ax.bar([i + w / 2 for i in x], p99, w, color=COLOURS["usearch"], label="cold p99")
    ax.set_yscale("log")
    ax.set_xticks(list(x), cfg)
    ax.set_ylabel("latency, ms (log)")
    ax.set_title("5M vectors (SIFT1B subset), 3.5 GB index evicted from cache, beam 64", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/cold-5m.svg")
    plt.close(fig)


if __name__ == "__main__":
    style()
    if len(sys.argv) > 1:
        ann_curves(sys.argv[1:])
    filtered()
    cold()
    print("wrote", OUT)
