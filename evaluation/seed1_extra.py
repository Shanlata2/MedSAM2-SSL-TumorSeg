"""seed1_extra.py -- first-run tests of every baseline against supervised training (Holm correction per
setting) and per-seed Dice of MCG-PL ("OC"), supervised training and the design variants. Reads metrics_all.py output."""
import json, glob, os, math
import numpy as np
from scipy.stats import wilcoxon
D = '/data/shanta_islam/mcgpl/metrics_v2/percase'
SET = [('kits', 5), ('kits', 10), ('lits', 5), ('lits', 10)]
B = ['MeanTeacher', 'UAMT', 'DTC', 'URPC', 'MCNet', 'ADMT', 'DyCON', 'SAMburden']
def load(data, ratio, meth, seed, ev='organ'):
    f = f'{D}/{data}_{ratio}_{meth}_s{seed}.json'
    if not os.path.exists(f): return None
    out = {}
    for r in json.load(open(f))['rows']:
        if r.get('gt_vox', 1) == 0: continue
        if ev not in r: return None
        out[r['case']] = r[ev]
    return out
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = [0] * m; run = 0
    for i, j in enumerate(o):
        run = max(run, min(1, (m - i) * ps[j])); adj[j] = run
    return adj
def pair(a, b):
    ks = sorted(set(a) & set(b)); x = np.array([a[k]['dice'] for k in ks]); y = np.array([b[k]['dice'] for k in ks])
    return 100 * np.mean(x - y), wilcoxon(x, y).pvalue, len(ks)
def avg(data, ratio, meth, ev='organ'):
    cs = [load(data, ratio, meth, s, ev) for s in (1, 2, 3)]; cs = [c for c in cs if c]
    ks = set.intersection(*[set(c) for c in cs])
    return {k: {'dice': np.mean([c[k]['dice'] for c in cs])} for k in ks}, len(cs)
def mean(c): return 100 * np.mean([c[k]['dice'] for k in c])
print('=== H. each baseline s1 vs Supervised s1 (Holm over 8 per setting) ===')
for data, ratio in SET:
    sup = load(data, ratio, 'Supervised', 1); res = [pair(load(data, ratio, m, 1), sup) for m in B]; adj = holm([r[1] for r in res])
    for m, r, a in zip(B, res, adj): print(f'{data}{ratio:<3d}{m:12s} vs Sup: dDice {r[0]:+6.2f} p {r[1]:.4g} holm {a:.4g}')
print('=== I. per-seed organ / raw Dice (OC, Supervised, R2, Union) ===')
for data, ratio in SET:
    for m in ('OC', 'Supervised', 'R2', 'Union'):
        o = [round(mean(load(data, ratio, m, s)), 2) for s in (1, 2, 3) if load(data, ratio, m, s)]
        r = [round(mean(load(data, ratio, m, s, 'raw')), 2) for s in (1, 2, 3) if load(data, ratio, m, s, 'raw')]
        print(f'{data}{ratio:<3d}{m:11s} organ {o}  raw {r}')
print('=== J. within-method, per-case Dice averaged over 3 seeds ===')
for data, ratio in SET:
    for a_, b_ in (('OC', 'Union'), ('R2', 'Union'), ('OC', 'R2')):
        a, ka = avg(data, ratio, a_); b, kb = avg(data, ratio, b_); d, p, n = pair(a, b)
        print(f'{data}{ratio:<3d}{a_} vs {b_} ({ka}/{kb} seeds): dDice {d:+.2f} p {p:.4g}')
    a, _ = avg(data, ratio, 'OC', 'raw'); b, _ = avg(data, ratio, 'R2', 'raw'); d, p, n = pair(a, b)
    print(f'{data}{ratio:<3d}OC vs R2 RAW (3/3 seeds): dDice {d:+.2f} p {p:.4g}')
print('=== K. first-run tests within method ===')
for data, ratio in SET:
    d, p, n = pair(load(data, ratio, 'Union', 1), load(data, ratio, 'Ours', 1))
    print(f'{data}{ratio:<3d}Union s1 vs Ours(R1 prop) s1: dDice {d:+.2f} p {p:.4g}')
a, _ = avg('kits', 5, 'OC'); b, _ = avg('kits', 10, 'OC'); d, p, n = pair(b, a)
print(f'KiTS OC 10% vs 5% (seed-avg): dDice {d:+.2f} p {p:.4g} n {n}')
