"""seed1_tables.py -- tables of the paper from the per-case files of metrics_all.py ("OC" = MCG-PL):
first-run (seed 1) comparison of all methods (mean +/- SD over test cases; organ rule and raw), paired Wilcoxon
signed-rank tests of MCG-PL against every method with Holm correction, three-seed MCG-PL vs. supervised tests,
and the design variants."""
import json, glob, os, math
import numpy as np
from scipy.stats import wilcoxon
D = '/data/shanta_islam/mcgpl/metrics_v2/percase'
M = ['Supervised', 'MeanTeacher', 'UAMT', 'DTC', 'URPC', 'MCNet', 'ADMT', 'DyCON', 'SAMburden', 'OC']
SET = [('kits', 5), ('kits', 10), ('lits', 5), ('lits', 10)]
def load(data, ratio, meth, seed, ev='organ'):
    f = f'{D}/{data}_{ratio}_{meth}_s{seed}.json'
    if not os.path.exists(f): return None
    out = {}
    for r in json.load(open(f))['rows']:
        if r.get('gt_vox', 1) == 0: continue
        if ev not in r: return None
        out[r['case']] = r[ev]
    return out
def seeds(data, ratio, meth):
    s = sorted(int(f.rsplit('_s', 1)[1][:-5]) for f in glob.glob(f'{D}/{data}_{ratio}_{meth}_s*.json'))
    return [x for x in s if x <= 3]
fin = lambda v: isinstance(v, (int, float)) and math.isfinite(v)
SC = {'dice': 100, 'ja': 100, 'hd95': 1, 'asd': 1, 'assd': 1}
def vals(c, k): return [c[x][k] * SC[k] for x in c if fin(c[x].get(k))]
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = [0] * m; run = 0
    for i, j in enumerate(o):
        run = max(run, min(1, (m - i) * ps[j])); adj[j] = run
    return adj
def pair(a, b):
    ks = sorted(set(a) & set(b)); x = np.array([a[k]['dice'] for k in ks]); y = np.array([b[k]['dice'] for k in ks])
    return 100 * np.mean(x - y), wilcoxon(x, y).pvalue, len(ks)
def avg_seeds(data, ratio, meth, ev='organ'):
    cs = [load(data, ratio, meth, s, ev) for s in seeds(data, ratio, meth)]
    cs = [c for c in cs if c]; ks = set.intersection(*[set(c) for c in cs])
    return {k: {'dice': np.mean([c[k]['dice'] for c in cs])} for k in ks}, len(cs)
print('=== A. seed means, organ rule ===')
for data, ratio in SET:
    for m in M:
        per = []
        for s in seeds(data, ratio, m):
            c = load(data, ratio, m, s)
            if c: per.append([np.mean(vals(c, k)) for k in ('dice', 'ja', 'hd95', 'asd', 'assd')])
        if not per: print(f'{data}{ratio} {m:12s} MISSING'); continue
        p = np.array(per); sd0 = p[:, 0].std(ddof=0) if len(p) > 1 else 0; sd1 = p[:, 0].std(ddof=1) if len(p) > 1 else 0
        print(f'{data}{ratio:<3d}{m:12s} k={len(p)} Dice {p[:,0].mean():6.2f} sd0 {sd0:5.2f} sd1 {sd1:5.2f} | JA {p[:,1].mean():6.2f} HD95 {p[:,2].mean():6.2f} ASD {p[:,3].mean():6.2f} ASSD {p[:,4].mean():6.2f}')
print('=== B. raw seed means of MCG-PL ===')
for data, ratio in SET:
    per = [np.mean(vals(load(data, ratio, 'OC', s, 'raw'), 'dice')) for s in seeds(data, ratio, 'OC')]
    print(f'{data}{ratio} OC raw Dice per seed {[round(x,2) for x in per]} mean {np.mean(per):.2f}')
print('=== C. Table I: first run (s1), mean +- SD over test cases, organ rule ===')
for data, ratio in SET:
    for m in M:
        c = load(data, ratio, m, 1)
        if not c: print(f'{data}{ratio} {m} MISSING s1'); continue
        out = []
        for k in ('dice', 'ja', 'hd95', 'asd', 'assd'):
            v = vals(c, k); out.append(f'{k} {np.mean(v):6.2f}+-{np.std(v, ddof=1):5.2f}(n{len(v)})')
        print(f'{data}{ratio:<3d}{m:12s} ' + ' | '.join(out))
print('=== D. raw-metric table: first run, Dice mean +- SD over test cases ===')
for data, ratio in SET:
    print(f'{data}{ratio}: ' + '  '.join(f'{m} {np.mean(vals(load(data,ratio,m,1,"raw"),"dice")):.2f}+-{np.std(vals(load(data,ratio,m,1,"raw"),"dice"),ddof=1):.2f}' for m in M))
print('=== E. significance: OC s1 vs each method s1 (paired Wilcoxon, Holm over 9) ===')
for data, ratio in SET:
    oc = load(data, ratio, 'OC', 1); res = [pair(oc, load(data, ratio, m, 1)) for m in M[:-1]]
    adj = holm([r[1] for r in res])
    for m, r, a in zip(M[:-1], res, adj):
        print(f'{data}{ratio:<3d}{m:12s} dDice {r[0]:+6.2f}  p {r[1]:.4g}  holm {a:.4g}  n {r[2]}')
print('=== F. OC vs Supervised, per-case Dice averaged over seeds (primary hypotheses) ===')
for data, ratio in SET:
    a, ka = avg_seeds(data, ratio, 'OC'); b, kb = avg_seeds(data, ratio, 'Supervised')
    d, p, n = pair(a, b); print(f'{data}{ratio:<3d} OC({ka} seeds) vs Sup({kb} seeds): dDice {d:+.2f} p {p:.4g} n {n}')
print('=== G. design variants: first run and seed SDs ===')
for data, ratio in SET:
    row = []
    for m in ('Ours', 'Union', 'R2', 'OC'):
        c = load(data, ratio, m, 1); row.append(f'{m} {np.mean(vals(c, "dice")):.2f}' if c else f'{m} NA')
    d, p, n = pair(load(data, ratio, 'OC', 1), load(data, ratio, 'Ours', 1))
    sds = []
    for m in ('Ours', 'Union', 'R2', 'OC'):
        per = [np.mean(vals(load(data, ratio, m, s), 'dice')) for s in seeds(data, ratio, m) if load(data, ratio, m, s)]
        sds.append(f'{m} k{len(per)} {np.mean(per):.2f}+-{np.std(per, ddof=1) if len(per) > 1 else 0:.2f}')
    print(f'{data}{ratio:<3d} s1: ' + ' | '.join(row) + f' | OC-Ours {d:+.2f} p {p:.4g}')
    print(f'{data}{ratio:<3d} seeds: ' + ' | '.join(sds))
