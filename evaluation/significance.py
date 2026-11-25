"""
significance.py -- paired Wilcoxon signed-rank tests on per-case tumour Dice, versus Supervised,
with per-case Dice AVERAGED OVER SEEDS ON BOTH SIDES.

Reads the per-case files written by metrics_all.py.
Primary comparisons (pre-specified): Ours vs Supervised at 5% on KiTS21 and LiTS -> Holm-Bonferroni.
Everything else is exploratory (uncorrected), as in the paper.

Usage:  python3 significance.py /root/autodl-tmp/metrics_v2  [--metric dice|ja]
Prints, per (dataset, ratio): mean of each method (seed-averaged), number of seeds, diff vs Supervised,
p-value; plus each individual seed of Ours vs the seed-averaged Supervised baseline.
"""
import argparse
import collections
import glob
import json

import numpy as np
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("metrics_dir")
ap.add_argument("--metric", default="dice")
ap.add_argument("--variant", default="raw")
a = ap.parse_args()

# per[(data, ratio, method)][seed] = {case: value}
per = collections.defaultdict(dict)
for f in glob.glob(f"{a.metrics_dir}/percase/*.json"):
    d = json.load(open(f))
    c = d["cell"]
    if c.get("group") in ("upper", "reference"):
        continue
    per[(c["data"], c["ratio"], c["method"])][c["seed"]] = {r["case"]: r[a.variant][a.metric] for r in d["rows"]}


def seed_avg(seeds):
    common = set.intersection(*[set(s) for s in seeds.values()])
    return {k: float(np.mean([s[k] for s in seeds.values()])) for k in common}


def test(x_map, y_map):
    ks = sorted(set(x_map) & set(y_map))
    x = np.array([x_map[k] for k in ks])
    y = np.array([y_map[k] for k in ks])
    p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
    return len(ks), 100 * (x.mean() - y.mean()), p


rows, primary = [], []
for data, ratio in sorted({(k[0], k[1]) for k in per}):
    if (data, ratio, "Supervised") not in per:
        print(f"\n{data} {ratio}%: no Supervised baseline -- skipped")
        continue
    sup_seeds = per[(data, ratio, "Supervised")]
    sup = seed_avg(sup_seeds)
    print(f"\n==== {data.upper()} {ratio}%  | Supervised: {len(sup_seeds)} seed(s), mean {100*np.mean(list(sup.values())):.2f}")
    print(f"  {'method':12s} {'seeds':>5s} {'mean':>7s} {'diff':>7s} {'p':>8s}   n")
    for (dd, rr, m), seeds in sorted(per.items()):
        if dd != data or rr != ratio or m == "Supervised":
            continue
        x = seed_avg(seeds)
        n, diff, p = test(x, sup)
        sig = "*" if p < 0.05 else ""
        print(f"  {m:12s} {len(seeds):5d} {100*np.mean(list(x.values())):7.2f} {diff:+7.2f} {p:8.4f}{sig:2s} {n}")
        rows.append(dict(data=data, ratio=ratio, method=m, seeds=len(seeds), diff=diff, p=p, n=n))
        if m == "Ours" and ratio == 5:
            primary.append((f"{data} 5%", p))
        if m == "Ours" and len(seeds) > 1:
            for s, one in sorted(seeds.items()):
                n1, d1, p1 = test(one, sup)
                print(f"    Ours seed {s}: {100*np.mean(list(one.values())):6.2f}  diff {d1:+6.2f}  p={p1:.4f}")

# Holm-Bonferroni on the pre-specified primary comparisons
if primary:
    print("\n==== Holm-Bonferroni, primary comparisons (Ours vs Supervised at 5%), FWER 0.05")
    m = len(primary)
    still = True
    for i, (name, p) in enumerate(sorted(primary, key=lambda t: t[1])):
        thr = 0.05 / (m - i)
        rej = still and p < thr
        still = rej
        print(f"  {name:10s} p={p:.4f}  threshold {thr:.4f}  -> {'REJECT H0 (significant)' if rej else 'not significant'}")

json.dump(rows, open(f"{a.metrics_dir}/significance_{a.metric}.json", "w"), indent=1)
