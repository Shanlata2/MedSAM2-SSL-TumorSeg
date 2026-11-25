"""
metrics_all.py -- the single metric pipeline used for all reported results.
Every method, seed and setting is evaluated the same way:

  * predictions are read directly from each validation folder (never via summary.json paths)
  * tumour = label 2; KiTS21: all 63 validation cases; LiTS: tumour-bearing cases only (27)
  * Dice, Jaccard: empty prediction -> 0
  * 95HD and ASD in MILLIMETRES (voxel spacing from the reference header), medpy definitions:
      hd95 = 95th percentile of the pooled two-way surface distances, asd = mean pred->ref distance;
    the symmetric ASSD is stored too. Empty prediction -> undefined, excluded from the mean,
    and the number of such cases is reported (column n_empty).
  * optional --organ: also the organ-constrained variant used in Table V
    (keep tumour components touching the predicted organ dilated by 3 voxels)
  * volumes are cropped to the union bounding box (+5 voxel margin) before distance transforms:
    this is exact -- all surface voxels lie inside the crop -- and ~10x faster.

Usage:
  python3 metrics_all.py --out /root/autodl-tmp/metrics_v2              # all cells in cells.py
  python3 metrics_all.py --out /root/autodl-tmp/metrics_v2 --only kits:10 --organ   # Table V
  python3 metrics_all.py --out ... --workers 16
Outputs:  <out>/percase/<data>_<ratio>_<method>_s<seed>[_organ].json   and  <out>/summary.csv
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cells as C  # noqa: E402

TUMOR, ORGAN = 2, 1


# ---------- surface distances (identical to medpy.metric.binary.__surface_distances) ----------
def _surface_distances(result, reference, spacing):
    result = result.astype(bool)
    reference = reference.astype(bool)
    fp = ndimage.generate_binary_structure(result.ndim, 1)
    r_border = result ^ ndimage.binary_erosion(result, structure=fp, iterations=1)
    g_border = reference ^ ndimage.binary_erosion(reference, structure=fp, iterations=1)
    dt = ndimage.distance_transform_edt(~g_border, sampling=spacing)
    return dt[r_border]


def _crop(mask, arrays, margin=5):
    if not mask.any():
        return arrays
    sl = ndimage.find_objects(mask.astype(np.uint8))[0]
    sl = tuple(slice(max(0, s.start - margin), min(n, s.stop + margin)) for s, n in zip(sl, mask.shape))
    return [a[sl] for a in arrays]


def four(p, g, spacing):
    p = p.astype(bool)
    g = g.astype(bool)
    if p.sum() == 0:
        return dict(dice=0.0, ja=0.0, hd95=None, asd=None, assd=None, empty=True)
    inter = np.logical_and(p, g).sum()
    dice = 2.0 * inter / (p.sum() + g.sum())
    ja = inter / np.logical_or(p, g).sum()
    pc, gc = _crop(p | g, [p, g])
    d1 = _surface_distances(pc, gc, spacing)
    d2 = _surface_distances(gc, pc, spacing)
    hd95 = float(np.percentile(np.hstack((d1, d2)), 95))
    asd = float(d1.mean())
    assd = float(np.hstack((d1, d2)).mean())
    return dict(dice=float(dice), ja=float(ja), hd95=hd95, asd=asd, assd=assd, empty=False)


def organ_constrained(tum, org, dil=3):
    if tum.sum() == 0 or org.sum() == 0:
        return tum
    org_d = ndimage.binary_dilation(org.astype(bool), iterations=dil)
    lab, n = ndimage.label(tum)
    if n <= 1:
        return tum
    keep = np.unique(lab[org_d & (lab > 0)])
    keep = keep[keep > 0]
    if keep.size == 0:
        return tum
    return np.isin(lab, keep)


def one_case(args):
    pf, gf, data, do_organ = args
    gimg = nib.load(gf)
    gt = np.asanyarray(gimg.dataobj).astype(np.uint8)
    g = gt == TUMOR
    if data == "lits" and g.sum() == 0:
        return None
    pr = np.asanyarray(nib.load(pf).dataobj).astype(np.uint8)
    if pr.shape != gt.shape:
        return dict(case=os.path.basename(pf)[:-7], error=f"shape {pr.shape} vs {gt.shape}")
    spacing = tuple(float(z) for z in gimg.header.get_zooms()[:3])
    p = pr == TUMOR
    o = pr == ORGAN
    p, o, g = _crop(p | o | g, [p, o, g])          # exact: everything relevant is inside
    row = dict(case=os.path.basename(pf)[:-7], spacing=spacing, gt_vox=int(g.sum()))
    row["raw"] = four(p, g, spacing)
    if do_organ:
        row["organ"] = four(organ_constrained(p, o), g, spacing)
    return row


def agg(rows, key):
    d = [r[key] for r in rows]
    hd = [x["hd95"] for x in d if x["hd95"] is not None]
    asd = [x["asd"] for x in d if x["asd"] is not None]
    assd = [x["assd"] for x in d if x["assd"] is not None]
    return dict(n=len(d), n_empty=sum(x["empty"] for x in d),
                dice=100 * np.mean([x["dice"] for x in d]), ja=100 * np.mean([x["ja"] for x in d]),
                hd95=float(np.mean(hd)) if hd else float("nan"),
                asd=float(np.mean(asd)) if asd else float("nan"),
                assd=float(np.mean(assd)) if assd else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="e.g. kits:10 or lits or kits:5:Ours")
    ap.add_argument("--organ", action="store_true", help="also compute organ-constrained variant")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/percase", exist_ok=True)

    sel = [c for c in C.CELLS]
    if a.only:
        parts = a.only.split(":")
        sel = [c for c in sel if c["data"] == parts[0]
               and (len(parts) < 2 or str(c["ratio"]) == parts[1])
               and (len(parts) < 3 or c["method"] == parts[2])]
    summary = []
    with Pool(a.workers) as pool:
        for c in sel:
            tag = f"{c['data']}_{c['ratio']}_{c['method']}_s{c['seed']}"
            if not os.path.exists(c["path"] + "/summary.json"):
                print(f"SKIP {tag:32s} (missing {c['path']})", flush=True)
                continue
            preds = sorted(glob.glob(c["path"] + "/*.nii.gz"))
            jobs = [(pf, f"{C.GT[c['data']]}/{os.path.basename(pf)}", c["data"], a.organ)
                    for pf in preds if os.path.exists(f"{C.GT[c['data']]}/{os.path.basename(pf)}")]
            t0 = time.time()
            rows = [r for r in pool.map(one_case, jobs) if r is not None]
            errs = [r for r in rows if "error" in r]
            rows = [r for r in rows if "error" not in r]
            json.dump(dict(cell=c, rows=rows, errors=errs), open(f"{a.out}/percase/{tag}.json", "w"), indent=1)
            for key in (["raw", "organ"] if a.organ else ["raw"]):
                s = agg(rows, key)
                s.update(data=c["data"], ratio=c["ratio"], method=c["method"], seed=c["seed"],
                         group=c["group"], variant=key)
                summary.append(s)
                warn = "" if s["n"] == C.N_EXPECTED[c["data"]] else f"  <-- expected n={C.N_EXPECTED[c['data']]}"
                print(f"{tag:32s} {key:5s} n={s['n']:3d} empty={s['n_empty']:2d} | Dice {s['dice']:6.2f} "
                      f"JA {s['ja']:6.2f} | 95HD {s['hd95']:7.2f} mm  ASD {s['asd']:6.2f} mm "
                      f"(ASSD {s['assd']:6.2f}) | {time.time() - t0:5.0f}s{warn}"
                      + (f"  ERRORS {len(errs)}" if errs else ""), flush=True)

    fn = f"{a.out}/summary{'_' + a.only.replace(':', '-') if a.only else ''}.csv"
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["data", "ratio", "method", "seed", "group", "variant", "n", "n_empty",
                                          "dice", "ja", "hd95", "asd", "assd"])
        w.writeheader()
        for s in summary:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()})
    print(f"\nsaved {fn}")


if __name__ == "__main__":
    main()
