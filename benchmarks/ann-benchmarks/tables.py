"""Markdown tables for docs/benchmarks.md from an ann-benchmarks export.

    python benchmarks/ann-benchmarks/tables.py results.csv
"""
import csv, re, sys
rows=[]
for path in sys.argv[1:]:
    # A harness export has a `dataset` column; the per-dataset CSVs kept in
    # this directory are named after their dataset instead.
    name=re.sub(r'\.csv$','',path.rsplit('/',1)[-1])
    for r in csv.DictReader(open(path)):
        r.setdefault('dataset', name)
        rows.append(r)
def col(r):
    a=r['algorithm']; p=r['parameters']
    m=re.search(r"'M': (\d+)",p)
    if a in('hnswlib','usearch','hnsw(faiss)'):
        return {'hnswlib':'hnswlib','usearch':'usearch','hnsw(faiss)':'faiss HNSW'}[a]+' M='+m.group(1)
    if a=='sanadb':
        R=re.search(r"'R': (\d+)",p).group(1); pq='PQ' if "'pq': True" in p else 'no PQ'
        return f'sanadb R={R} {pq}'
COLS=['hnswlib M=16','hnswlib M=24','faiss HNSW M=16','faiss HNSW M=24','usearch M=16','usearch M=24','sanadb R=32 PQ','sanadb R=32 no PQ','sanadb R=48 PQ']
ANCH={'sift-128-euclidean':[0.93,0.96,0.98,0.99,0.996,0.999],'glove-100-angular':[0.73,0.81,0.86,0.89,0.93,0.95]}
for ds,anchors in ANCH.items():
    by={}
    for r in rows:
        if r['dataset']!=ds: continue
        c=col(r)
        if c: by.setdefault(c,[]).append((float(r['k-nn']),float(r['qps']),float(r['build'])))
    cols=[c for c in COLS if c in by]
    print(f'\n### {ds}\n')
    print('| recall@10 | '+' | '.join(cols)+' |'); print('|---'*(len(cols)+1)+'|')
    for a in anchors:
        cells=[]
        for c in cols:
            best=min(by[c],key=lambda t:abs(t[0]-a))
            tol=0.004 if a>=0.99 else 0.01
            cells.append(f'{best[1]:,.0f} ({best[0]:.3f})' if abs(best[0]-a)<=tol else '')
        print(f'| ~{a} | '+' | '.join(cells)+' |')
    print('| build | '+' | '.join(f'{by[c][0][2]:.0f} s' for c in cols)+' |')
