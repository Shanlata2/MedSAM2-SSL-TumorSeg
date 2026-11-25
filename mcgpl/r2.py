"""r2.py <gpu|build> <pool>      pool = lits10_s1..3 | lits5_s1..3 | kits10_s1..3 | kits5_s1..3
ROUND 2 of MCG-PL (iterative self-training): the round-1 student (union labels, same seed) becomes the seed model.
Everything else is exactly round 1: confidence c = fraction of predicted-tumour voxels with p > 0.9, the ORIGINAL MedSAM2
generator (only paths changed), additive merge (refined OR seed tumour), gate c >= 0.80, plans of 501/507, same split.
  gpu   : step 1 (student predicts the unlabeled pool + softmax -> c) and step 2 (MedSAM2)  -> writes GPU_DONE
  build : step 3 (union labels, gate, nnU-Net dataset) and step 4 (preprocessing)          -> READY (CPU only)
Training-pool pseudo-label Dice (admitted cases) is logged for the pre-declared adoption check. Resumable."""
import glob, json, os, re, shutil, subprocess, sys
import numpy as np
R = "/root/autodl-tmp"
NR, NP, RES = f"{R}/nnUNet_data/nnUNet_raw", f"{R}/nnUNet_data/nnUNet_preprocessed", f"{R}/nnUNet_data/nnUNet_results"
TR = {1: "nnUNetTrainer_250epochs", 2: "nnUNetTrainer_250epochs_seed2", 3: "nnUNetTrainer_250epochs_seed3", 4: "nnUNetTrainer_250epochs_seed4", 5: "nnUNetTrainer_250epochs_seed5"}
R1 = {"lits10": 542, "lits5": 545, "kits5": 548, "kits10": 551}          # round-1 union student IDs (seed 1; +1, +2)
R2 = {"lits10": 554, "lits5": 557, "kits10": 560, "kits5": 563}          # round-2 dataset IDs (seed 1; +1, +2)
mode, pool = sys.argv[1], sys.argv[2]
data, seed = pool.split("_s")[0], int(pool.split("_s")[1])
r1id, dsid, tr = R1[data] + seed - 1, R2[data] + seed - 1, TR[seed]
EXTRA = {("kits5", 4): (570, 574), ("kits5", 5): (571, 575), ("kits10", 4): (572, 576), ("kits10", 5): (573, 577)}   # extra KiTS21 seeds
if (data, seed) in EXTRA: r1id, dsid = EXTRA[(data, seed)]
W, name = f"{R}/e2e_r2/{pool}", f"Dataset{dsid}_{pool}_r2"
os.makedirs(W, exist_ok=True)
if data.startswith("kits"):
    sp = json.load(open(f"{NP}/Dataset501_V5_3D/splits_final.json"))[7 if data == "kits5" else 6]
    gt, val = sorted(sp["train"]), sorted(sp["val"])
    unl = sorted(json.load(open(f"{R}/ours5_unlabeled133.json")) if data == "kits5"
                 else json.load(open(f"{R}/v5_3d_pseudo_label_selection.json"))["pseudo_label_case_ids"])
    SRC, PLANS, expect = f"{NR}/Dataset500_KiTS21", "501", (12, 133, 63) if data == "kits5" else (24, 121, 63)
    GEN = f"{R}/sam2_v5_3d/gen_v2_pseudo_labels_medsam2_5pct.py"
else:
    r = "5pct" if data == "lits5" else "10pct"
    S = json.load(open(f"{R}/lits_ssl_split.json"))
    gt, val, unl = sorted(S[f"labeled_{r}"]), sorted(S["val"]), sorted(S[f"unlabeled_{r}"])
    SRC, PLANS, expect = f"{NR}/Dataset507_Liver", "507", (5, 95, 31) if data == "lits5" else (10, 90, 31)
    GEN = f"{R}/lits_build/gen_liver_medsam2_{r}.py"
assert (len(gt), len(unl), len(val)) == expect, (len(gt), len(unl), len(val))
assert not (set(gt) & set(unl)) and not (set(gt) & set(val)) and not (set(unl) & set(val))
r1name = os.path.basename(glob.glob(f"{NR}/Dataset{r1id}_*")[0])
CKPT = f"{RES}/{r1name}/{tr}__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
assert os.path.exists(CKPT), f"round-1 student not found: {CKPT}"
SEG, CONF, MED, QJ, UL = f"{W}/seed_preds", f"{W}/conf.json", f"{W}/medsam2_labels", f"{W}/medsam2_quality.json", f"{W}/union_labels"
env = dict(os.environ, nnUNet_raw=NR, nnUNet_preprocessed=NP, nnUNet_results=RES, nnUNet_compile="f")
def log(*a): print(f"[r2 {pool}]", *a, flush=True)

if mode == "gpu":
    # 1. round-1 student predicts the unlabeled pool (+ softmax) -> confidence c (identical formula to round 1)
    os.makedirs(SEG, exist_ok=True)
    conf = json.load(open(CONF)) if os.path.exists(CONF) else {}
    todo = [c for c in unl if not (c in conf and os.path.exists(f"{SEG}/{c}.nii.gz"))]
    log(f"step1 seed model = {r1name} / {tr}; {len(unl) - len(todo)}/{len(unl)} already predicted")
    for b in range(0, len(todo), 10):
        batch, inp, out = todo[b:b + 10], f"{W}/in", f"{W}/out"
        shutil.rmtree(inp, True); shutil.rmtree(out, True); os.makedirs(inp); os.makedirs(out)
        for c in batch:
            for s in glob.glob(f"{SRC}/imagesTr/{c}_*.nii.gz"): os.symlink(os.path.realpath(s), f"{inp}/{os.path.basename(s)}")
        subprocess.run(["nnUNetv2_predict", "-i", inp, "-o", out, "-d", str(r1id), "-c", "3d_fullres", "-f", "0",
                        "-tr", tr, "--save_probabilities", "-npp", "2", "-nps", "2"], check=True, env=env, stdout=subprocess.DEVNULL)
        for c in batch:
            d = np.load(f"{out}/{c}.npz"); p = d["probabilities"].astype(np.float32)
            p = p / np.clip(p.sum(0, keepdims=True), 1e-6, None); tm = p.argmax(0) == 2
            conf[c] = float((p[2][tm] > 0.9).mean()) if tm.any() else 0.0
            d.close(); shutil.move(f"{out}/{c}.nii.gz", f"{SEG}/{c}.nii.gz")
        json.dump(conf, open(CONF, "w"), indent=1); shutil.rmtree(out, True); shutil.rmtree(inp, True)
        log(f"step1 {len(conf)}/{len(unl)}")
    assert all(c in conf for c in unl)
    # 2. MedSAM2 refinement with the ORIGINAL generator, only paths changed (as in round 1)
    if len(glob.glob(f"{MED}/*.nii.gz")) < len(unl) or not os.path.exists(QJ):
        src = open(GEN).read()
        reps = [("V3_DIR", f'Path("{SEG}")'), ("OUT_LABELS", f'Path("{MED}")'), ("OUT_JSON", f'"{QJ}"')]
        if data == "kits10":
            json.dump(unl, open(f"{W}/selection.json", "w")); reps.append(("SELECTION", f'"{W}/selection.json"'))
        for var, v in reps:
            src, n = re.subn(rf"^{var} = .*$", f"{var} = {v}", src, flags=re.M); assert n == 1, f"could not set {var}"
        gen_copy = f"{os.path.dirname(GEN)}/_r2_{pool}_{os.path.basename(GEN)}"
        open(gen_copy, "w").write(src); log("step2 MedSAM2 ->", gen_copy)
        subprocess.run(["python3", gen_copy], check=True, cwd=os.path.dirname(GEN), env=env)
    assert len(glob.glob(f"{MED}/*.nii.gz")) >= len(unl), "MedSAM2 output incomplete"
    open(f"{W}/GPU_DONE", "w").write("ok"); log("GPU steps done")
    sys.exit(0)

# ---------------- build (CPU) ----------------
import nibabel as nib
assert os.path.exists(f"{W}/GPU_DONE"), "GPU steps not done"
conf = json.load(open(CONF)); kept = sorted(c for c in unl if conf[c] >= 0.80)
os.makedirs(UL, exist_ok=True); dices = []
for c in kept:
    out = f"{UL}/{c}.nii.gz"
    img = nib.load(f"{MED}/{c}.nii.gz"); lab = np.asanyarray(img.dataobj).astype(np.uint8)
    s = np.asanyarray(nib.load(f"{SEG}/{c}.nii.gz").dataobj) == 2
    assert s.shape == lab.shape, (c, s.shape, lab.shape)
    lab[s] = 2
    if not os.path.exists(out): nib.save(nib.Nifti1Image(lab, img.affine, img.header), out)
    g = np.asanyarray(nib.load(f"{SRC}/labelsTr/{c}.nii.gz").dataobj) == 2; t = lab == 2   # training pool only
    dices.append(1.0 if g.sum() + t.sum() == 0 else 2 * (g & t).sum() / (g.sum() + t.sum()))
log(f"gate keeps {len(kept)}/{len(unl)} | admitted union pseudo-label Dice (training pool) {np.mean(dices):.4f} | train {len(gt) + len(kept)} val {len(val)}")
json.dump(dict(kept=len(kept), pl_dice_admitted=float(np.mean(dices))), open(f"{W}/pl_quality_r2.json", "w"))
OUT, PP = f"{NR}/{name}", f"{NP}/{name}"
if not os.path.exists(f"{OUT}/our_split.json"):
    other = [d for d in glob.glob(f"{NR}/Dataset{dsid}_*") if d != OUT]; assert not other, f"ID {dsid} already used: {other}"
    shutil.rmtree(OUT, True); os.makedirs(f"{OUT}/imagesTr"); os.makedirs(f"{OUT}/labelsTr")
    for c in gt + kept + val:
        im = glob.glob(f"{SRC}/imagesTr/{c}_*.nii.gz"); assert im, f"no image for {c}"
        for s_ in im: os.symlink(os.path.realpath(s_), f"{OUT}/imagesTr/{os.path.basename(s_)}")
    for c in gt + val: shutil.copy(f"{SRC}/labelsTr/{c}.nii.gz", f"{OUT}/labelsTr/{c}.nii.gz")
    for c in kept: shutil.copy(f"{UL}/{c}.nii.gz", f"{OUT}/labelsTr/{c}.nii.gz")
    dj = json.load(open(f"{SRC}/dataset.json")); dj["numTraining"] = len(gt + kept + val)
    json.dump(dj, open(f"{OUT}/dataset.json", "w"), indent=2)
    json.dump([{"train": sorted(gt + kept), "val": val}], open(f"{OUT}/our_split.json", "w"), indent=2)
if not os.path.exists(f"{PP}/splits_final.json"):
    for cmd in (["nnUNetv2_extract_fingerprint", "-d", str(dsid), "--verify_dataset_integrity"],
                ["nnUNetv2_move_plans_between_datasets", "-s", PLANS, "-t", str(dsid), "-sp", "nnUNetPlans", "-tp", "nnUNetPlans"]):
        subprocess.run(cmd, check=True, env=env)
    shutil.copy(f"{OUT}/dataset.json", f"{PP}/dataset.json")
    subprocess.run(["nnUNetv2_preprocess", "-d", str(dsid), "-plans_name", "nnUNetPlans", "-c", "3d_fullres", "-np", "2"], check=True, env=env)
    n = len(glob.glob(f"{PP}/nnUNetPlans_3d_fullres/*.pkl")); assert n == len(gt + kept + val), f"preprocessing incomplete: {n}"
    shutil.copy(f"{OUT}/our_split.json", f"{PP}/splits_final.json")
log(f"READY -> nnUNetv2_train {dsid} 3d_fullres 0 -tr {tr}")
