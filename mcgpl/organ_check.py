"""organ_check.py -- CPU only, read-only. Checks the round-2 training sets for wrong pseudo-labels (tumour outside the organ).
Part 1: list the wrong pseudo-labels (tumour outside the organ) in the round-2 training sets, and save a picture of the worst one.
Part 2: pre-declared training-data check of the fix "remove pseudo-label tumour components that do not touch the organ"
        (the same rule as the organ-constrained evaluation, now applied to the training pseudo-labels only).
        Decision rule, fixed BEFORE running: adopt the fix only if the mean admitted pseudo-label Dice (vs. reference labels of the
        training pool) improves on average over the 12 pools AND does not decrease in any of the 4 settings (seed-averaged)."""
import glob, json, os, sys
from multiprocessing import Pool
import nibabel as nib, numpy as np
from scipy import ndimage
R = "/root/autodl-tmp"; NR, NP = f"{R}/nnUNet_data/nnUNet_raw", f"{R}/nnUNet_data/nnUNet_preprocessed"
R2 = {"lits10": 554, "lits5": 557, "kits10": 560, "kits5": 563}
def filt(lab):
    """remove tumour (2) components that do not touch the organ (1) dilated by 3 voxels; same conventions as the organ rule:
    no organ -> unchanged; no component touches -> unchanged. Returns (new label, removed voxels)."""
    t, o = lab == 2, lab == 1
    if not t.any() or not o.any(): return lab, 0
    b = np.argwhere(t | o); bx = tuple(slice(max(lo - 4, 0), hi + 5) for lo, hi in zip(b.min(0), b.max(0)))
    tc, oc = t[bx], o[bx]
    cl, n = ndimage.label(tc); keep = np.unique(cl[ndimage.binary_dilation(oc, iterations=3) & (cl > 0)]); keep = keep[keep > 0]
    if keep.size == 0: return lab, 0
    rem = (cl > 0) & ~np.isin(cl, keep)
    if not rem.any(): return lab, 0
    new = lab.copy(); sub = new[bx]; sub[rem] = 0; new[bx] = sub   # removed tumour becomes background
    return new, int(rem.sum())
def dice(g, t): return 1.0 if g.sum() + t.sum() == 0 else 2 * (g & t).sum() / (g.sum() + t.sum())
def one(a):
    lp, gp = a
    lab = np.asanyarray(nib.load(lp).dataobj).astype(np.uint8); g = np.asanyarray(nib.load(gp).dataobj) == 2
    new, rem = filt(lab)
    return os.path.basename(lp)[:-7], dice(g, lab == 2), dice(g, new == 2), rem, int((lab == 2).sum())
def pools():
    for data in ("kits5", "kits10", "lits5", "lits10"):
        for s in (1, 2, 3):
            D = os.path.basename(glob.glob(f"{NR}/Dataset{R2[data] + s - 1}_*")[0])
            if data.startswith("kits"):
                gt = set(json.load(open(f"{NP}/Dataset501_V5_3D/splits_final.json"))[7 if data == "kits5" else 6]["train"]); SRC = "Dataset500_KiTS21"
            else:
                gt = set(json.load(open(f"{R}/lits_ssl_split.json"))["labeled_" + ("5pct" if data == "lits5" else "10pct")]); SRC = "Dataset507_Liver"
            tr = json.load(open(f"{NR}/{D}/our_split.json"))[0]["train"]
            yield data, s, D, SRC, sorted(c for c in tr if c not in gt), len(gt & set(tr))
if __name__ == "__main__":
    res, worst = {}, (0, None)
    with Pool(4) as P:
        for data, s, D, SRC, ps, ng in pools():
            r = P.map(one, [(f"{NR}/{D}/labelsTr/{c}.nii.gz", f"{NR}/{SRC}/labelsTr/{c}.nii.gz") for c in ps])
            b, a = np.mean([x[1] for x in r]), np.mean([x[2] for x in r]); ch = [x for x in r if x[3] > 0]
            tv, rv = sum(x[4] for x in r), sum(x[3] for x in r); res[(data, s)] = (b, a)
            print(f"{data:6s} seed {s} | reference {ng:2d} + pseudo {len(ps):3d} | wrong (out-of-organ) tumour in {len(ch):2d} pseudo-labels = {100*rv/max(tv,1):4.1f}% of pseudo tumour voxels"
                  f" | pseudo-label Dice {100*b:.2f} -> {100*a:.2f} after fix ({100*(a-b):+.2f})", flush=True)
            for c, db, da, rm, t in sorted(ch, key=lambda x: -x[3])[:3]:
                print(f"     {c}: {rm} wrong tumour voxels ({100*rm/t:.0f}% of its tumour) | Dice vs reference {100*db:.1f} -> {100*da:.1f}")
            if data.startswith("kits") and ch and max(x[3] for x in ch) > worst[0]:
                c = max(ch, key=lambda x: x[3]); worst = (c[3], (data, s, D, SRC, c[0]))
    print("\nSETTING MEANS (3 seeds): pseudo-label Dice before -> after fix")
    ok = True
    for data in ("kits5", "kits10", "lits5", "lits10"):
        b = np.mean([res[(data, s)][0] for s in (1, 2, 3)]); a = np.mean([res[(data, s)][1] for s in (1, 2, 3)]); ok &= a >= b
        print(f"  {data:6s} {100*b:.2f} -> {100*a:.2f} ({100*(a-b):+.2f})")
    B = np.mean([v[0] for v in res.values()]); A = np.mean([v[1] for v in res.values()]); ok &= A > B
    print(f"  ALL 12 {100*B:.2f} -> {100*A:.2f} ({100*(A-B):+.2f})\nDECISION (pre-declared rule): {'ADOPT the fix' if ok else 'DO NOT adopt the fix'}")
    if worst[1]:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            data, s, D, SRC, c = worst[1]
            im = nib.load(f"{NR}/{SRC}/imagesTr/{c}_0000.nii.gz"); ax = [i for i, k in enumerate(nib.aff2axcodes(im.affine)) if k in "SI"][0]
            ct = np.asanyarray(im.dataobj); lab = np.asanyarray(nib.load(f"{NR}/{D}/labelsTr/{c}.nii.gz").dataobj).astype(np.uint8)
            g = np.asanyarray(nib.load(f"{NR}/{SRC}/labelsTr/{c}.nii.gz").dataobj).astype(np.uint8); new, _ = filt(lab)
            wrong = (lab == 2) & (new != 2); k = int(np.argmax(wrong.sum(axis=tuple(i for i in range(3) if i != ax))))
            sl = lambda v: np.rot90(np.take(v, k, axis=ax))
            fig, axs = plt.subplots(1, 2, figsize=(10, 5.2))
            for a_, L, ttl in ((axs[0], lab, f"Pseudo-label used for training ({data}, seed {s})"), (axs[1], g, "Expert reference label")):
                a_.imshow(sl(np.clip(ct, -200, 300)), cmap="gray"); a_.axis("off"); a_.set_title(ttl, fontsize=10)
                ov = np.zeros(sl(L).shape + (4,)); ov[sl(L) == 1] = (0, .8, 0, .35); ov[sl(L) == 2] = (1, .85, 0, .55)
                if L is lab: ov[sl(wrong)] = (1, 0, 0, .75)
                a_.imshow(ov)
            fig.suptitle(f"{c}: green = kidney, yellow = tumour, RED = wrong tumour outside the kidney", fontsize=11)
            fig.tight_layout(); fig.savefig(f"{R}/wrong_pseudolabel_{c}.png", dpi=120); print(f"picture saved: {R}/wrong_pseudolabel_{c}.png")
        except Exception as e: print("picture failed:", e)
