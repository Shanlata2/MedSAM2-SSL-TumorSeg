"""
Generate V5.1-3D pseudo-labels for all 121 cases using v2 (multi-anchor + MASK prompt).
Label = V3 kidney (class 1) + SAM2-v2 tumor (class 2). Logs Dice vs GT for QC.
SAFE method: mask prompts + multi-anchor prevent the catastrophic collapses of box prompts.
"""
from __future__ import annotations
import os, sys, json, time, tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import scipy.ndimage as ndi
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t512.yaml"  # MedSAM2
SAM2_CKPT = "/root/autodl-tmp/MedSAM2/checkpoints/MedSAM2_CTLesion.pt"  # MedSAM2 CT-lesion
IMG_DIR = Path("/root/autodl-tmp/kits21_repo/kits21/data")
V3_DIR = Path("/root/autodl-tmp/predictions/sup5_on_unlabeled133")  # 5% model preds
GT_DIR = Path("/root/autodl-tmp/clean_labels_uint8")
SELECTION = "/root/autodl-tmp/ours5_unlabeled133.json"
OUT_LABELS = Path("/root/autodl-tmp/sam2_v5_3d/medsam2_pseudo_labels_5pct")
OUT_JSON = "/root/autodl-tmp/sam2_v5_3d/medsam2_pseudo_label_quality_5pct.json"
CT_MIN, CT_MAX = -200.0, 300.0
TUMOR = 2
MIN_ABS, FRAC, BUF = 200, 0.25, 5


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


def dump(vol, d):
    d.mkdir(parents=True, exist_ok=True)
    for z in range(vol.shape[0]):
        Image.fromarray(to_rgb(vol[z])).save(d / f"{z:05d}.jpg", quality=92)


def comps_of(mask):
    lab, n = ndi.label(mask)
    cs = []
    for i in range(1, n + 1):
        c = (lab == i)
        ps = c.reshape(c.shape[0], -1).sum(1)
        zs = np.where(ps > 0)[0]
        cs.append({"vox": int(c.sum()), "z0": int(zs[0]), "z1": int(zs[-1])})
    cs.sort(key=lambda x: -x["vox"])
    if not cs:
        return []
    thr = max(MIN_ABS, FRAC * cs[0]["vox"])
    return [c for c in cs if c["vox"] >= thr]


def comp_anchors(v3_t, z0, z1, max_anchors=6, min_gap=5):
    ps = v3_t.reshape(v3_t.shape[0], -1).sum(1)
    cand = [z for z in range(z0, z1 + 1) if ps[z] > 0]
    if not cand:
        return []
    densest = max(cand, key=lambda z: ps[z])
    target = min(max_anchors, max(1, (z1 - z0) // min_gap))
    picks = {densest}
    for t in np.linspace(z0, z1, target):
        picks.add(min(cand, key=lambda c: abs(c - t)))
    return sorted(picks)


def propagate_v2(predictor, img_norm, v3_t, z0, z1, anchors):
    Z, H, W = img_norm.shape
    full = np.zeros((Z, H, W), bool)
    tmp = tempfile.TemporaryDirectory()
    fdir = Path(tmp.name) / "f"
    dump(img_norm[z0:z1 + 1], fdir)
    state = predictor.init_state(video_path=str(fdir))
    for ag in anchors:
        m = v3_t[ag]
        if m.sum() == 0:
            continue
        predictor.add_new_mask(inference_state=state, frame_idx=ag - z0,
                               obj_id=1, mask=torch.as_tensor(m, dtype=torch.bool))
    res = {}
    for fi, _, ml in predictor.propagate_in_video(state):
        res[fi] = (ml[0, 0] > 0.0).cpu().numpy().astype(bool)
    for fi, _, ml in predictor.propagate_in_video(state, reverse=True):
        if fi not in res:
            res[fi] = (ml[0, 0] > 0.0).cpu().numpy().astype(bool)
    del state; torch.cuda.empty_cache()
    for lz, mm in res.items():
        gz = lz + z0
        if 0 <= gz < Z:
            full[gz] = mm
    tmp.cleanup()
    return full


def main():
    OUT_LABELS.mkdir(parents=True, exist_ok=True)
    cases = json.load(open(SELECTION))  # plain list of 133 case ids
    print(f"Generating v2 pseudo-labels for {len(cases)} cases", flush=True)
    predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CKPT, device="cuda")

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
        kept = comps_of(v3_t)
        for c in kept:
            z0 = max(0, c["z0"] - BUF)
            z1 = min(img.shape[0] - 1, c["z1"] + BUF)
            anchors = comp_anchors(v3_t, c["z0"], c["z1"])
            sam2 |= propagate_v2(predictor, img_norm, v3_t, z0, z1, anchors)

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
        flag = "✅" if r["delta"] > 0.02 else ("🟡" if r["delta"] > -0.02 else "🚨")
        print(f"[{i+1:>3}/{len(cases)}] {cid}: V3={v3d:.3f} v2={v2d:.3f} ({r['delta']:+.3f}) {flag} "
              f"| {len(kept)}comp | {r['sec']}s", flush=True)

    v3m = np.mean([r["v3_dice"] for r in rows])
    v2m = np.mean([r["v2_dice"] for r in rows])
    disasters = [r["case"] for r in rows if r["delta"] < -0.15]
    json.dump({"mean_v3": round(v3m, 4), "mean_v2": round(v2m, 4),
               "disasters": disasters, "rows": rows}, open(OUT_JSON, "w"), indent=2)
    print("\n" + "=" * 60, flush=True)
    print(f"DONE in {(time.time()-t_all)/60:.1f} min", flush=True)
    print(f"  Mean V3 pseudo-label Dice: {v3m:.4f}", flush=True)
    print(f"  Mean v2 pseudo-label Dice: {v2m:.4f}", flush=True)
    print(f"  v2 disasters (< -0.15): {len(disasters)} -> {disasters}", flush=True)
    print(f"  Labels saved to: {OUT_LABELS}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
