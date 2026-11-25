"""build_oc.py <pool | check>     pool = kits5_s1..3 | kits10_s1..3 | lits5_s1..3 | lits10_s1..3
MCG-PL round 2 + ORGAN-CONSISTENCY FILTER on the pseudo-labels (adopted by the pre-declared training-data rule, organ_check.py):
copies the round-2 training set (Dataset554-565, same images, same split, same reference/val labels) and removes from every
PSEUDO-label the tumour components that do not touch the organ dilated by 3 voxels (same rule/conventions as organ_check.py).
Reference labels and val labels are copied unchanged. Then fingerprint, plans of 501/507, preprocessing. Resumable.
  check : verifies all 12 source datasets and that the new IDs 581-592 are unused; writes nothing."""
import glob, json, os, shutil, subprocess, sys
import numpy as np
from scipy import ndimage
R = "/root/autodl-tmp"; NR, NP = f"{R}/nnUNet_data/nnUNet_raw", f"{R}/nnUNet_data/nnUNet_preprocessed"
SRC_ID = {"lits10": 554, "lits5": 557, "kits10": 560, "kits5": 563}     # round-2 datasets (seed 1; +1, +2)
NEW_ID = {"kits5": 581, "kits10": 584, "lits5": 587, "lits10": 590}     # new organ-filtered datasets (seed 1; +1, +2)
POOLS = [f"{d}_s{s}" for d in ("kits5", "kits10", "lits5", "lits10") for s in (1, 2, 3)]
def filt(lab):
    t, o = lab == 2, lab == 1
    if not t.any() or not o.any(): return lab, 0
    b = np.argwhere(t | o); bx = tuple(slice(max(lo - 4, 0), hi + 5) for lo, hi in zip(b.min(0), b.max(0)))
    tc, oc = t[bx], o[bx]
    cl, n = ndimage.label(tc); keep = np.unique(cl[ndimage.binary_dilation(oc, iterations=3) & (cl > 0)]); keep = keep[keep > 0]
    if keep.size == 0: return lab, 0
    rem = (cl > 0) & ~np.isin(cl, keep)
    if not rem.any(): return lab, 0
    new = lab.copy(); sub = new[bx]; sub[rem] = 0; new[bx] = sub
    return new, int(rem.sum())
def info(pool):
    data, s = pool.split("_s")[0], int(pool.split("_s")[1])
    src = glob.glob(f"{NR}/Dataset{SRC_ID[data] + s - 1}_*"); assert len(src) == 1, f"source dataset missing for {pool}: {src}"
    if data.startswith("kits"):
        gt = set(json.load(open(f"{NP}/Dataset501_V5_3D/splits_final.json"))[7 if data == "kits5" else 6]["train"]); PL = "501"
    else:
        gt = set(json.load(open(f"{R}/lits_ssl_split.json"))["labeled_" + ("5pct" if data == "lits5" else "10pct")]); PL = "507"
    sp = json.load(open(f"{src[0]}/our_split.json"))[0]
    return src[0], NEW_ID[data] + s - 1, f"Dataset{NEW_ID[data] + s - 1}_{pool}_r2oc", gt, sp, PL
if sys.argv[1] == "check":
    for p in POOLS:
        src, did, name, gt, sp, PL = info(p)
        other = [d for d in glob.glob(f"{NR}/Dataset{did}_*") + glob.glob(f"{NP}/Dataset{did}_*") + glob.glob(f"{R}/nnUNet_data/nnUNet_results/Dataset{did}_*") if not d.endswith(name)]
        assert not other, f"ID {did} already used: {other}"
        nl = len(glob.glob(f"{src}/labelsTr/*.nii.gz")); ps = [c for c in sp["train"] if c not in gt]
        assert nl == len(sp["train"]) + len(sp["val"]), (p, nl)
        print(f"OK {p:9s} {os.path.basename(src)} -> {name} | reference {len(sp['train']) - len(ps)} + pseudo {len(ps)} | val {len(sp['val'])}")
    print("CHECK PASSED"); sys.exit(0)
import nibabel as nib
pool = sys.argv[1]; src, did, name, gt, sp, PL = info(pool)
OUT, PP = f"{NR}/{name}", f"{NP}/{name}"
env = dict(os.environ, nnUNet_raw=NR, nnUNet_preprocessed=NP, nnUNet_compile="f")
def log(*a): print(f"[oc {pool}]", *a, flush=True)
if not os.path.exists(f"{OUT}/our_split.json"):
    shutil.rmtree(OUT, True); os.makedirs(f"{OUT}/imagesTr"); os.makedirs(f"{OUT}/labelsTr")
    for s_ in glob.glob(f"{src}/imagesTr/*.nii.gz"): os.symlink(os.path.realpath(s_), f"{OUT}/imagesTr/{os.path.basename(s_)}")
    nch, nrem, ntum = 0, 0, 0
    for c in sp["train"] + sp["val"]:
        f = f"{src}/labelsTr/{c}.nii.gz"
        if c in gt or c in sp["val"]: shutil.copy(f, f"{OUT}/labelsTr/{c}.nii.gz"); continue
        img = nib.load(f); lab = np.asanyarray(img.dataobj).astype(np.uint8); new, rem = filt(lab); ntum += int((lab == 2).sum())
        if rem: nch += 1; nrem += rem
        nib.save(nib.Nifti1Image(new, img.affine, img.header), f"{OUT}/labelsTr/{c}.nii.gz")
    shutil.copy(f"{src}/dataset.json", f"{OUT}/dataset.json"); shutil.copy(f"{src}/our_split.json", f"{OUT}/our_split.json")
    log(f"filter changed {nch} pseudo-labels, removed {nrem} wrong tumour voxels ({100*nrem/max(ntum,1):.1f}% of pseudo tumour)")
n_img = len(glob.glob(f"{OUT}/imagesTr/*.nii.gz")); n_lab = len(glob.glob(f"{OUT}/labelsTr/*.nii.gz"))
assert n_lab == len(sp["train"]) + len(sp["val"]) and n_img >= n_lab, (n_img, n_lab)
if not os.path.exists(f"{PP}/splits_final.json"):
    for cmd in (["nnUNetv2_extract_fingerprint", "-d", str(did), "--verify_dataset_integrity"],
                ["nnUNetv2_move_plans_between_datasets", "-s", PL, "-t", str(did), "-sp", "nnUNetPlans", "-tp", "nnUNetPlans"]):
        subprocess.run(cmd, check=True, env=env)
    shutil.copy(f"{OUT}/dataset.json", f"{PP}/dataset.json")
    subprocess.run(["nnUNetv2_preprocess", "-d", str(did), "-plans_name", "nnUNetPlans", "-c", "3d_fullres", "-np", "2"], check=True, env=env)
    n = len(glob.glob(f"{PP}/nnUNetPlans_3d_fullres/*.pkl")); assert n == n_lab, f"preprocessing incomplete: {n}"
    shutil.copy(f"{OUT}/our_split.json", f"{PP}/splits_final.json")
log(f"READY -> nnUNetv2_train {did} 3d_fullres 0")
