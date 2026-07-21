"""
SAM2 V5.1-3D — Component-aware propagation diagnostic.

Hypothesis: whole-volume multi-anchor propagation drifts across slices and
over-segments. Fix: connected-component analysis on V3 tumor, drop scattered
false positives, propagate each real component within its own z-window, union.
"""
from __future__ import annotations
import os, sys, time, tempfile, argparse
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import nibabel as nib
import torch
import scipy.ndimage as ndi
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor

SAM2_CONFIG = "configs/sam2/sam2_hiera_l.yaml"
SAM2_CHECKPOINT = "/root/autodl-tmp/sam2_transfer/sam2_hiera_large.pt"
IMAGING_DIR = Path("/root/autodl-tmp/kits21_repo/kits21/data")
V3_3D_PRED_DIR = Path("/root/autodl-tmp/predictions/v3_3d_on_unlabeled")
GT_DIR = Path("/root/autodl-tmp/clean_labels_uint8")
OUT_DIR = Path("/root/autodl-tmp/sam2_v5_3d/component_test_outputs")
CT_WINDOW_MIN, CT_WINDOW_MAX = -200.0, 300.0
TUMOR_CLASS = 2
JPG_QUALITY = 92


def dice_score(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    inter = (pred & gt).sum()
    return 2.0 * inter / (pred.sum() + gt.sum() + 1e-8)


def ct_normalize(volume):
    v = np.clip(volume, CT_WINDOW_MIN, CT_WINDOW_MAX)
    return ((v - CT_WINDOW_MIN) / (CT_WINDOW_MAX - CT_WINDOW_MIN)).astype(np.float32)


def slice_to_rgb(s):
    s = np.clip(s, 0.0, 1.0)
    s = (s * 255.0).astype(np.uint8)
    return np.stack([s, s, s], axis=-1)


def bbox_from_mask(mask_2d, pad=3):
    if mask_2d.sum() == 0:
        return None
    ys, xs = np.where(mask_2d)
    H, W = mask_2d.shape
    return np.array([max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
                     min(W, int(xs.max()) + pad + 1), min(H, int(ys.max()) + pad + 1)],
                    dtype=np.float32)


def dump_window_frames(vol_window, frames_dir):
    frames_dir.mkdir(parents=True, exist_ok=True)
    for z in range(vol_window.shape[0]):
        Image.fromarray(slice_to_rgb(vol_window[z])).save(
            frames_dir / f"{z:05d}.jpg", quality=JPG_QUALITY)


def analyze_components(mask_3d, min_voxels):
    """3D connected components, sorted by size descending."""
    labeled, n = ndi.label(mask_3d)
    comps = []
    for i in range(1, n + 1):
        comp = (labeled == i)
        vox = int(comp.sum())
        per_slice = comp.reshape(comp.shape[0], -1).sum(axis=1)
        zs = np.where(per_slice > 0)[0]
        densest = int(per_slice.argmax())
        coords = np.array(np.where(comp))
        centroid = coords.mean(axis=1)  # (z, y, x)
        comps.append({
            "id": i, "voxels": vox,
            "z_range": (int(zs[0]), int(zs[-1])),
            "n_slices": int(len(zs)),
            "densest_slice": densest,
            "densest_voxels": int(per_slice[densest]),
            "centroid_zyx": [round(float(c), 1) for c in centroid],
            "kept": vox >= min_voxels,
        })
    comps.sort(key=lambda c: -c["voxels"])
    return comps


def propagate_in_window(predictor, img_norm, v3_tumor, z_lo, z_hi, anchor_global):
    """Propagate SAM2 within [z_lo, z_hi], seeded by V3 bbox at anchor. Returns full-volume bool mask."""
    Z, H, W = img_norm.shape
    full = np.zeros((Z, H, W), dtype=bool)
    bb = bbox_from_mask(v3_tumor[anchor_global])
    if bb is None:
        return full

    n_win = z_hi - z_lo + 1
    a_local = anchor_global - z_lo

    tmp = tempfile.TemporaryDirectory()
    fdir = Path(tmp.name) / "frames"
    dump_window_frames(img_norm[z_lo:z_hi + 1], fdir)

    results = {}
    # forward
    st = predictor.init_state(video_path=str(fdir))
    predictor.add_new_points_or_box(inference_state=st, frame_idx=a_local, obj_id=1, box=bb)
    for fi, _, ml in predictor.propagate_in_video(st):
        results[fi] = (ml[0, 0] > 0.0).cpu().numpy().astype(bool)
    del st; torch.cuda.empty_cache()
    # backward
    st = predictor.init_state(video_path=str(fdir))
    predictor.add_new_points_or_box(inference_state=st, frame_idx=a_local, obj_id=1, box=bb)
    for fi, _, ml in predictor.propagate_in_video(st, reverse=True):
        if fi < a_local:
            results[fi] = (ml[0, 0] > 0.0).cpu().numpy().astype(bool)
    del st; torch.cuda.empty_cache()

    for local_z, m in results.items():
        gz = local_z + z_lo
        if 0 <= gz < Z:
            full[gz] = m
    tmp.cleanup()
    return full


def run(case_id, min_voxels, buffer):
    print("=" * 72)
    print(f"SAM2 V5.1-3D Component Test — {case_id}")
    print(f"min_voxels={min_voxels}, window buffer={buffer}")
    print("=" * 72)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(IMAGING_DIR / case_id / "imaging.nii.gz")).get_fdata().astype(np.float32)
    v3_nii = nib.load(str(V3_3D_PRED_DIR / f"{case_id}.nii.gz"))
    v3 = v3_nii.get_fdata().astype(np.uint8)
    gt = nib.load(str(GT_DIR / f"{case_id}.nii.gz")).get_fdata().astype(np.uint8)

    v3_tumor = (v3 == TUMOR_CLASS)
    gt_tumor = (gt == TUMOR_CLASS)
    v3_kidney = (v3 == 1)
    v3_dice = dice_score(v3_tumor, gt_tumor)

    print(f"\n[Data] shape {img.shape}")
    print(f"  V3 tumor voxels: {int(v3_tumor.sum()):,}   GT tumor voxels: {int(gt_tumor.sum()):,}")
    print(f"  V3-3D Tumor Dice (baseline): {v3_dice:.4f}")

    # --- Component analysis ---
    print(f"\n[V3 tumor connected components]")
    v3_comps = analyze_components(v3_tumor, min_voxels)
    for c in v3_comps:
        flag = "KEEP" if c["kept"] else "drop(scatter)"
        print(f"  comp {c['id']:>2}: {c['voxels']:>6,} vox | z {c['z_range'][0]:>3}-{c['z_range'][1]:<3} "
              f"({c['n_slices']:>3} sl) | densest z={c['densest_slice']} ({c['densest_voxels']} vox) "
              f"| centroid(z,y,x)={c['centroid_zyx']} | {flag}")

    print(f"\n[GT tumor connected components (for reference)]")
    gt_comps = analyze_components(gt_tumor, min_voxels=50)
    for c in gt_comps:
        print(f"  GT comp {c['id']:>2}: {c['voxels']:>6,} vox | z {c['z_range'][0]:>3}-{c['z_range'][1]:<3} "
              f"| centroid(z,y,x)={c['centroid_zyx']}")

    kept = [c for c in v3_comps if c["kept"]]
    print(f"\n[Strategy] propagating {len(kept)} kept component(s), window = comp z-range ± {buffer}")

    # --- Build predictor ---
    print("\n[Building SAM2 predictor...]")
    predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
    img_norm = ct_normalize(img)

    # --- Per-component windowed propagation, union ---
    sam2_union = np.zeros_like(v3_tumor, dtype=bool)
    largest_only = np.zeros_like(v3_tumor, dtype=bool)
    for idx, c in enumerate(kept):
        z_lo = max(0, c["z_range"][0] - buffer)
        z_hi = min(img.shape[0] - 1, c["z_range"][1] + buffer)
        anchor = c["densest_slice"]
        t0 = time.time()
        mask = propagate_in_window(predictor, img_norm, v3_tumor, z_lo, z_hi, anchor)
        sam2_union |= mask
        if idx == 0:
            largest_only = mask.copy()
        print(f"  comp {c['id']}: window z=[{z_lo},{z_hi}] ({z_hi-z_lo+1} sl), anchor {anchor}, "
              f"{time.time()-t0:.1f}s, propagated {int(mask.sum()):,} vox, "
              f"comp-Dice {dice_score(mask, gt_tumor):.4f}")

    # --- Results ---
    sam2_dice = dice_score(sam2_union, gt_tumor)
    largest_dice = dice_score(largest_only, gt_tumor)
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"  V3-3D Tumor Dice:              {v3_dice:.4f}  (baseline)")
    print(f"  SAM2 largest-component only:   {largest_dice:.4f}  ({largest_dice-v3_dice:+.4f})")
    print(f"  SAM2 all-kept-components union: {sam2_dice:.4f}  ({sam2_dice-v3_dice:+.4f})")
    print(f"  SAM2 union voxels: {int(sam2_union.sum()):,}  (GT: {int(gt_tumor.sum()):,})")
    best = max(sam2_dice, largest_dice)
    if best - v3_dice > 0.02:
        print(f"\n  ✅ SAM2 IMPROVES (+{best-v3_dice:.4f}) — component approach works!")
    elif best - v3_dice > -0.02:
        print(f"\n  🟡 SAM2 ≈ V3 — needs tuning.")
    else:
        print(f"\n  🚨 SAM2 worse — investigate.")

    # save the better variant
    out_tumor = sam2_union if sam2_dice >= largest_dice else largest_only
    combined = np.zeros_like(v3, dtype=np.uint8)
    combined[v3_kidney] = 1
    combined[out_tumor] = 2
    out_path = OUT_DIR / f"{case_id}_sam2_comp.nii.gz"
    nib.save(nib.Nifti1Image(combined, v3_nii.affine, v3_nii.header), str(out_path))
    print(f"\n  Saved: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="case_00000")
    ap.add_argument("--min_voxels", type=int, default=200,
                    help="Drop V3 tumor components smaller than this (false positives)")
    ap.add_argument("--buffer", type=int, default=5,
                    help="Slice buffer around each component's z-range")
    args = ap.parse_args()
    run(args.case, args.min_voxels, args.buffer)
