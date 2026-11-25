"""build_union.py <pool> <dsid>    pool = kits5_s1..3 | kits10_s1..3 | lits5_s1..3 | lits10_s1..3
MCG-PL with UNION pseudo-labels: tumour = MedSAM2-refined tumour OR seed tumour (refinement may add, never delete).
Everything else unchanged: same seed predictions, same confidence c, same gate c >= 0.80, nnU-Net dataset with the plans of
501 (KiTS21) / 507 (LiTS), preprocessing -np 2, case-count check, split.  Resumable.  CPU only."""
import glob, json, os, shutil, subprocess, sys
import nibabel as nib, numpy as np
R = "/root/autodl-tmp"
NR, NP, RES = f"{R}/nnUNet_data/nnUNet_raw", f"{R}/nnUNet_data/nnUNet_preprocessed", f"{R}/nnUNet_data/nnUNet_results"
pool, dsid = sys.argv[1], int(sys.argv[2])
data, seed = pool.split("_s")[0], int(pool.split("_s")[1])
W = f"{R}/e2e/{pool}"
if data.startswith("kits"):
    k = 7 if data == "kits5" else 6
    sp = json.load(open(f"{NP}/Dataset501_V5_3D/splits_final.json"))[k]
    gt, val = sorted(sp["train"]), sorted(sp["val"])
    unl = sorted(json.load(open(f"{R}/ours5_unlabeled133.json")) if data == "kits5"
                 else json.load(open(f"{R}/v5_3d_pseudo_label_selection.json"))["pseudo_label_case_ids"])
    SRC, PLANS, expect = f"{NR}/Dataset500_KiTS21", "501", (12, 133, 63) if data == "kits5" else (24, 121, 63)
    SD, RD, CJ = f"{W}/seed_preds", f"{W}/medsam2_labels", f"{W}/conf.json"
else:
    r = "5pct" if data == "lits5" else "10pct"
    S = json.load(open(f"{R}/lits_ssl_split.json"))
    gt, val, unl = sorted(S[f"labeled_{r}"]), sorted(S["val"]), sorted(S[f"unlabeled_{r}"])
    SRC, PLANS, expect = f"{NR}/Dataset507_Liver", "507", (5, 95, 31) if data == "lits5" else (10, 90, 31)
    if seed == 1:
        SD = f"{R}/predictions/liver_sup5_on_unlabeled95" if data == "lits5" else f"{R}/predictions/liver_sup10_on_unlabeled90"
        RD, CJ = f"{R}/lits_build/liver_medsam2_pseudo_labels_{r}", f"{R}/lits_conf{r[:-3]}.json"
    else:
        SD, RD, CJ = f"{W}/seed_preds", f"{W}/medsam2_labels", f"{W}/conf.json"
assert (len(gt), len(unl), len(val)) == expect, (len(gt), len(unl), len(val))
assert not (set(gt) & set(unl)) and not (set(gt) & set(val)) and not (set(unl) & set(val))
conf = json.load(open(CJ))
if isinstance(conf, list): conf = {x["case"]: x.get("conf_p90", x.get("conf")) for x in conf}
assert all(c in conf for c in unl), "confidence missing for some unlabeled cases"
name, UL = f"Dataset{dsid}_{pool}_union", f"{R}/union/{pool}"
env = dict(os.environ, nnUNet_raw=NR, nnUNet_preprocessed=NP, nnUNet_results=RES, nnUNet_compile="f")
kept = sorted(c for c in unl if conf[c] >= 0.80)
# 1. union labels for the admitted cases (organ label from the refined file; tumour = refined OR seed)
os.makedirs(UL, exist_ok=True); add = tot = 0
for c in kept:
    out = f"{UL}/{c}.nii.gz"
    if os.path.exists(out): continue
    img = nib.load(f"{RD}/{c}.nii.gz"); lab = np.asanyarray(img.dataobj).astype(np.uint8)
    s = np.asanyarray(nib.load(f"{SD}/{c}.nii.gz").dataobj) == 2
    assert s.shape == lab.shape, (c, s.shape, lab.shape)
    new = s & (lab != 2); add += int(new.sum()); tot += int((lab == 2).sum()); lab[new] = 2
    nib.save(nib.Nifti1Image(lab, img.affine, img.header), out)
print(f"[{name}] gate keeps {len(kept)}/{len(unl)} | union added {100 * add / max(tot, 1):.1f}% tumour voxels (new files) | train {len(gt) + len(kept)} val {len(val)}", flush=True)
assert all(os.path.exists(f"{UL}/{c}.nii.gz") for c in kept)
# 2. nnU-Net dataset
OUT, PP = f"{NR}/{name}", f"{NP}/{name}"
if not os.path.exists(f"{OUT}/our_split.json"):
    other = [d for d in glob.glob(f"{NR}/Dataset{dsid}_*") if d != OUT]
    assert not other, f"ID {dsid} already used: {other}"
    shutil.rmtree(OUT, True); os.makedirs(f"{OUT}/imagesTr"); os.makedirs(f"{OUT}/labelsTr")
    for c in gt + kept + val:
        im = glob.glob(f"{SRC}/imagesTr/{c}_*.nii.gz"); assert im, f"no image for {c}"
        for s_ in im: os.symlink(os.path.realpath(s_), f"{OUT}/imagesTr/{os.path.basename(s_)}")
    for c in gt + val: shutil.copy(f"{SRC}/labelsTr/{c}.nii.gz", f"{OUT}/labelsTr/{c}.nii.gz")
    for c in kept: shutil.copy(f"{UL}/{c}.nii.gz", f"{OUT}/labelsTr/{c}.nii.gz")
    dj = json.load(open(f"{SRC}/dataset.json")); dj["numTraining"] = len(gt + kept + val)
    json.dump(dj, open(f"{OUT}/dataset.json", "w"), indent=2)
    json.dump([{"train": sorted(gt + kept), "val": val}], open(f"{OUT}/our_split.json", "w"), indent=2)
# 3. preprocess with the plans of 501 / 507 (identical to every other model of that dataset)
if not os.path.exists(f"{PP}/splits_final.json"):
    for cmd in (["nnUNetv2_extract_fingerprint", "-d", str(dsid), "--verify_dataset_integrity"],
                ["nnUNetv2_move_plans_between_datasets", "-s", PLANS, "-t", str(dsid), "-sp", "nnUNetPlans", "-tp", "nnUNetPlans"]):
        subprocess.run(cmd, check=True, env=env)
    shutil.copy(f"{OUT}/dataset.json", f"{PP}/dataset.json")
    subprocess.run(["nnUNetv2_preprocess", "-d", str(dsid), "-plans_name", "nnUNetPlans", "-c", "3d_fullres", "-np", "2"], check=True, env=env)
    n = len(glob.glob(f"{PP}/nnUNetPlans_3d_fullres/*.pkl"))
    assert n == len(gt + kept + val), f"preprocessing incomplete: {n}/{len(gt + kept + val)}"
    shutil.copy(f"{OUT}/our_split.json", f"{PP}/splits_final.json")
print(f"[{name}] READY -> nnUNetv2_train {dsid} 3d_fullres 0", flush=True)
