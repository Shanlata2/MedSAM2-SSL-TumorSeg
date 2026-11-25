"""
SAM-burden baseline [arXiv:2411.12602]: per-slice 2D SAM pseudo-label refinement.
Forked from gen_v2_pseudo_labels_medsam2_5pct.py. IDENTICAL foundation model
(MedSAM2 CT-Lesion), inputs, [-200,300] windowing, connected-component filter,
Eq.(7) tumor/organ merge, QC logging, and output format.

KEY DIFFERENCES vs our BTC (what the comparison isolates):
  * Each axial slice refined INDEPENDENTLY as a 1-frame video (uses the same
    video-predictor code path as production, but with NO propagation across
    slices -> cannot recover slices the seed missed; the 2D limitation we expose).
  * Automatic prompts per component per slice (following SAM-burden [29]):
        tumor bounding box + center-of-mass positive point
        + up to N_NEG negative points from the organ (class 1).
  * NO confidence gate (all refined labels kept; reported ungated, analogous
    to Dataset505 refine-only).
Set env LIMIT=k to process only the first k cases (smoke test).
"""
from __future__ import annotations
import os, json, time, tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import scipy.ndimage as ndi
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t512.yaml"            # MedSAM2
SAM2_CKPT   = "/root/autodl-tmp/MedSAM2/checkpoints/MedSAM2_CTLesion.pt"
IMG_DIR     = Path("/root/autodl-tmp/kits21_repo/kits21/data")
V3_DIR      = Path("/root/autodl-tmp/predictions/v3_3d_on_unlabeled")
GT_DIR      = Path("/root/autodl-tmp/clean_labels_uint8")
SELECTION   = "/root/autodl-tmp/kits10_unlabeled121.json"
OUT_LABELS  = Path("/root/autodl-tmp/sam2_v5_3d/samburden_pseudo_labels_10pct")
OUT_JSON    = "/root/autodl-tmp/sam2_v5_3d/samburden_pseudo_label_quality_10pct.json"
CT_MIN, CT_MAX = -200.0, 300.0
TUMOR = 2
MIN_ABS, FRAC = 200, 0.25
N_NEG = 2
LIMIT = int(os.environ.get("LIMIT", "0"))


def dice(p, g):
    p, g = p.astype(bool), g.astype(bool)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    return 2.0 * (p & g).sum() / (p.sum() + g.sum() + 1e-8)


def ct_norm(v):
    v = np.clip(v, CT_MIN, CT_MAX)
    return ((v - CT_MIN) / (CT_MAX - CT_MIN)).astype(np.float32)


def to_rgb(s):
    s = (np.clip(s, 0, 1) * 255).astype(np.uint8)
    return np.stack([s, s, s], -1)


def comps_of(mask):
    """Same CC filter as production: keep comps >= max(200, 0.25*largest)."""
    lab, n = ndi.label(mask)
    cs = []
    for i in range(1, n + 1):
        c = (lab == i)
        ps = c.reshape(c.shape[0], -1).sum(1)
        zs = np.where(ps > 0)[0]
        cs.append({"vox": int(c.sum()), "z0": int(zs[0]), "z1": int(zs[-1]), "lab": i})
    cs.sort(key=lambda x: -x["vox"])
    if not cs:
        return [], lab
    thr = max(MIN_ABS, FRAC * cs[0]["vox"])
    return [c for c in cs if c["vox"] >= thr], lab


def refine_slice(predictor, fdir, rgb, seed_sl, organ_sl):
    """2D refine of one component on one slice via a fresh 1-frame video (no propagation)."""
    ys, xs = np.where(seed_sl)
    if len(xs) == 0:
        return np.zeros_like(seed_sl, bool)
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
    pts = [[float(xs.mean()), float(ys.mean())]]
    lbs = [1]
    oy, ox = np.where(organ_sl & ~seed_sl)
    if len(ox) > 0:
        for k in np.linspace(0, len(ox) - 1, min(N_NEG, len(ox))).astype(int):
            pts.append([float(ox[k]), float(oy[k])]); lbs.append(0)
    Image.fromarray(rgb).save(fdir / "00000.jpg", quality=92)
    with torch.inference_mode():
        state = predictor.init_state(video_path=str(fdir))
        _, _, ml = predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=1,
            box=box,
            points=np.array(pts, dtype=np.float32),
            labels=np.array(lbs, dtype=np.int32),
        )
    mask = (ml[0, 0] > 0.0).cpu().numpy().astype(bool)
    del state
    torch.cuda.empty_cache()
    return mask


def main():
    OUT_LABELS.mkdir(parents=True, exist_ok=True)
    cases = json.load(open(SELECTION))
    if LIMIT > 0:
        cases = cases[:LIMIT]
    print(f"SAM-burden per-slice refinement for {len(cases)} cases", flush=True)
    predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CKPT, device="cuda")
    tmp = tempfile.TemporaryDirectory()
    fdir = Path(tmp.name) / "f"
    fdir.mkdir(parents=True, exist_ok=True)

    rows = []
    t_all = time.time()
    for i, cid in enumerate(cases):
        t0 = time.time()
        v3_nii = nib.load(str(V3_DIR / f"{cid}.nii.gz"))
        v3 = v3_nii.get_fdata().astype(np.uint8)
        img = nib.load(str(IMG_DIR / cid / "imaging.nii.gz")).get_fdata().astype(np.float32)
        gt = nib.load(str(GT_DIR / f"{cid}.nii.gz")).get_fdata().astype(np.uint8)
        v3_t, v3_k, gt_t = (v3 == TUMOR), (v3 == 1), (gt == TUMOR)
        img_norm = ct_norm(img)

        sam2 = np.zeros_like(v3_t, bool)
        kept, lab = comps_of(v3_t)
        for c in kept:
            comp = (lab == c["lab"])
            for z in range(c["z0"], c["z1"] + 1):
                seed_sl = comp[z]
                if not seed_sl.any():
                    continue
                seed_sl = ndi.binary_fill_holes(seed_sl)
                sam2[z] |= refine_slice(predictor, fdir, to_rgb(img_norm[z]), seed_sl, v3_k[z])

        combined = np.zeros_like(v3, dtype=np.uint8)
        combined[v3_k] = 1
        combined[sam2] = 2
        out = nib.Nifti1Image(combined, v3_nii.affine, v3_nii.header)
        out.set_data_dtype(np.uint8)
        nib.save(out, str(OUT_LABELS / f"{cid}.nii.gz"))

        v3d, v2d = dice(v3_t, gt_t), dice(sam2, gt_t)
        r = {"case": cid, "v3_dice": round(v3d, 4), "v2_dice": round(v2d, 4),
             "delta": round(v2d - v3d, 4), "v3_vox": int(v3_t.sum()),
             "v2_vox": int(sam2.sum()), "n_comp_kept": len(kept), "sec": round(time.time() - t0, 1)}
        rows.append(r)
        flag = "OK" if r["delta"] > 0.02 else ("~" if r["delta"] > -0.02 else "!!")
        print(f"[{i+1:>3}/{len(cases)}] {cid}: seed={v3d:.3f} sb={v2d:.3f} ({r['delta']:+.3f}) {flag} "
              f"| {len(kept)}comp | {r['sec']}s", flush=True)

    v3m = float(np.mean([r["v3_dice"] for r in rows]))
    v2m = float(np.mean([r["v2_dice"] for r in rows]))
    disasters = [r["case"] for r in rows if r["delta"] < -0.15]
    json.dump({"mean_seed": round(v3m, 4), "mean_samburden": round(v2m, 4),
               "disasters": disasters, "rows": rows}, open(OUT_JSON, "w"), indent=2)
    tmp.cleanup()
    print("\n" + "=" * 60, flush=True)
    print(f"DONE in {(time.time()-t_all)/60:.1f} min", flush=True)
    print(f"  Mean seed Dice:        {v3m:.4f}", flush=True)
    print(f"  Mean SAM-burden Dice:  {v2m:.4f}", flush=True)
    print(f"  disasters (< -0.15): {len(disasters)}", flush=True)
    print(f"  Labels -> {OUT_LABELS}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
