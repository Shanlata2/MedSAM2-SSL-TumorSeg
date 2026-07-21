import json, glob
from pathlib import Path
import numpy as np
import nibabel as nib

PROBS = Path("/root/autodl-tmp/conf_probs_out")
V3_DIR = Path("/root/autodl-tmp/predictions/v3_3d_on_unlabeled")
V2_DIR = Path("/root/autodl-tmp/sam2_v5_3d/v2_pseudo_labels")
QJSON = "/root/autodl-tmp/sam2_v5_3d/v2_pseudo_label_quality.json"
SAMPLE = "/root/autodl-tmp/sam2_v5_3d/sample_eval_results.json"
TUMOR = 2

def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    if a.sum() == 0 and b.sum() == 0:
        return 1.0
    return 2 * (a & b).sum() / (a.sum() + b.sum() + 1e-8)

q = {r["case"]: r for r in json.load(open(QJSON))["rows"]}
v1 = {r["case"]: r["sam2_dice"] for r in json.load(open(SAMPLE))["rows"]}

rows = []
for npz in sorted(glob.glob(str(PROBS / "*.npz"))):
    cid = Path(npz).stem
    if cid not in q:
        continue
    prob = np.load(npz)["probabilities"].astype(np.float32)
    prob = prob / np.clip(prob.sum(0, keepdims=True), 1e-6, None)
    pred = prob.argmax(0)
    tp = prob[TUMOR]
    tmask = pred == TUMOR
    fg = pred > 0
    if tmask.sum() > 0:
        conf_mean = float(tp[tmask].mean())
        conf_p90 = float((tp[tmask] > 0.9).mean())
        others = np.maximum(prob[0], prob[1])
        margin = float((tp[tmask] - others[tmask]).mean())
    else:
        conf_mean = conf_p90 = margin = 0.0
    ent = -(prob * np.log(prob + 1e-8)).sum(0)
    ent_fg = float(ent[fg].mean()) if fg.sum() > 0 else 5.0
    agree = np.nan
    try:
        v3 = nib.load(str(V3_DIR / f"{cid}.nii.gz")).get_fdata().astype(np.uint8)
        v2 = nib.load(str(V2_DIR / f"{cid}.nii.gz")).get_fdata().astype(np.uint8)
        agree = dice(v3 == TUMOR, v2 == TUMOR)
    except Exception:
        pass
    rows.append({"case": cid, "v3_dice": q[cid]["v3_dice"],
                 "conf_mean": round(conf_mean, 4), "conf_p90": round(conf_p90, 4),
                 "margin": round(margin, 4), "neg_entropy": round(-ent_fg, 4),
                 "agree_v3v2": round(float(agree), 4) if agree == agree else None})

print(f"Validated on {len(rows)} cases\n")
v3d = np.array([r["v3_dice"] for r in rows])

def report(name, vals):
    v = np.array(vals, float); ok = ~np.isnan(v)
    if ok.sum() < 5:
        print(f"  {name:<14}: insufficient"); return None
    c = float(np.corrcoef(v[ok], v3d[ok])[0, 1])
    print(f"  {name:<14}: corr with V3 quality = {c:+.3f}")
    return c

print("=" * 58)
print("SIGNAL CORRELATION with true V3 quality")
print("=" * 58)
signals = {
    "conf_mean":   [r["conf_mean"]   for r in rows],
    "conf_p90":    [r["conf_p90"]    for r in rows],
    "margin":      [r["margin"]      for r in rows],
    "neg_entropy": [r["neg_entropy"] for r in rows],
    "agree_v3v2":  [r["agree_v3v2"] if r["agree_v3v2"] is not None else np.nan for r in rows],
}
corrs = {k: report(k, v) for k, v in signals.items()}

best = max((k for k in corrs if corrs[k] is not None), key=lambda k: abs(corrs[k]))
print(f"\nBest signal: {best} (corr {corrs[best]:+.3f})")
print("=" * 58)
print("ROUTING SIMULATION (refine weak with v1, keep strong V3)")
print("=" * 58)
sig = np.array(signals[best], float)
low_is_weak = corrs[best] > 0
print(f"  V3 alone:          {v3d.mean():.4f}")
print(f"  ORACLE max(V3,v1): {np.array([max(r['v3_dice'], v1.get(r['case'], r['v3_dice'])) for r in rows]).mean():.4f}")
for pct in [20, 30, 40, 50]:
    thr = np.nanpercentile(sig, pct if low_is_weak else 100 - pct)
    refine = (sig <= thr) if low_is_weak else (sig >= thr)
    routed = np.array([v1.get(rows[i]["case"], v3d[i]) if refine[i] else v3d[i] for i in range(len(rows))])
    helped = sum(1 for i in range(len(rows)) if refine[i] and v1.get(rows[i]["case"], 0) > v3d[i])
    print(f"  refine bottom {pct}%: mean={routed.mean():.4f}  refined={int(refine.sum())}  helped={helped}")

json.dump(rows, open("confidence_validation.json", "w"), indent=2)
print("\nSaved confidence_validation.json")
