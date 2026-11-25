"""
e2e_pseudo.py <kits|lits> <seed>
End-to-end seed s of MCG-PL, starting from SUPERVISED SEED s (the same model used as the
supervised baseline), with the paper's exact settings:
  1. seed model predicts the unlabeled 5% pool (+ softmax), confidence c = fraction of predicted-tumour
     voxels with p_tumour > 0.9            (identical to ours5_step2.sh / predict_conf_sup10.py)
  2. MedSAM2 refinement = your original generator script, only input/output paths changed
  3. gate c >= 0.80, build nnU-Net dataset  (identical to build_dataset504 / build_d511)
  4. fingerprint + plans copied from 501/507 + preprocess + split
The student is trained afterwards with the same seed trainer.  Re-running resumes where it stopped.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

R = "/root/autodl-tmp"
NR, NP, RES = f"{R}/nnUNet_data/nnUNet_raw", f"{R}/nnUNet_data/nnUNet_preprocessed", f"{R}/nnUNet_data/nnUNet_results"
TR = {1: "nnUNetTrainer_250epochs", 2: "nnUNetTrainer_250epochs_seed2", 3: "nnUNetTrainer_250epochs_seed3", 4: "nnUNetTrainer_250epochs_seed4", 5: "nnUNetTrainer_250epochs_seed5"}
IDS = {("kits", 1): 526, ("kits", 2): 527, ("kits", 3): 528, ("lits", 2): 529, ("lits", 3): 530, ("kits10", 1): 534, ("kits10", 2): 538, ("kits10", 3): 539, ("lits10", 2): 540, ("lits10", 3): 541, ("kits", 4): 566, ("kits", 5): 567, ("kits10", 4): 568, ("kits10", 5): 569}
TAU = 0.80

data, seed = sys.argv[1], int(sys.argv[2])
dsid, tr = IDS[(data, seed)], TR[seed]
W = f"{R}/e2e/{data}_s{seed}" if data in ("kits10", "lits10") else f"{R}/e2e/{data}5_s{seed}"
os.makedirs(W, exist_ok=True)
if data == "kits":
    sp = json.load(open(f"{NP}/Dataset501_V5_3D/splits_final.json"))[7]
    gt, val = sorted(sp["train"]), sorted(sp["val"])
    unl = sorted(json.load(open(f"{R}/ours5_unlabeled133.json")))
    SRC, MDS, FOLD, PLANS = f"{NR}/Dataset500_KiTS21", "Dataset501_V5_3D", "7", "501"
    GEN = f"{R}/sam2_v5_3d/gen_v2_pseudo_labels_medsam2_5pct.py"
    name, expect = f"Dataset{dsid}_kits5_ours_s{seed}", (12, 133, 63)
elif data == "kits10":
    sp = json.load(open(f"{NP}/Dataset501_V5_3D/splits_final.json"))[6]
    gt, val = sorted(sp["train"]), sorted(sp["val"])
    allc = sorted({os.path.basename(f)[:-12] for f in glob.glob(f"{NR}/Dataset500_KiTS21/imagesTr/*_0000.nii.gz")})
    unl = sorted(json.load(open(f"{R}/v5_3d_pseudo_label_selection.json"))["pseudo_label_case_ids"])
    SRC, MDS, FOLD, PLANS = f"{NR}/Dataset500_KiTS21", "Dataset501_V5_3D", "6", "501"
    GEN = f"{R}/sam2_v5_3d/gen_v2_pseudo_labels_medsam2_5pct.py"
    name, expect = f"Dataset{dsid}_kits10_ours_s{seed}", (24, 121, 63)
elif data == "lits10":   # LiTS 10%: seed model = Dataset507 fold 0 (10 labeled / 31 test), original 10% generator
    S = json.load(open(f"{R}/lits_ssl_split.json"))
    gt, val, unl = sorted(S["labeled_10pct"]), sorted(S["val"]), sorted(S["unlabeled_10pct"])
    SRC, MDS, FOLD, PLANS = f"{NR}/Dataset507_Liver", "Dataset507_Liver", "0", "507"
    GEN = f"{R}/lits_build/gen_liver_medsam2_10pct.py"
    name, expect = f"Dataset{dsid}_lits10_ours_s{seed}", (10, 90, 31)
else:
    S = json.load(open(f"{R}/lits_ssl_split.json"))
    gt, val, unl = sorted(S["labeled_5pct"]), sorted(S["val"]), sorted(S["unlabeled_5pct"])
    SRC, MDS, FOLD, PLANS = f"{NR}/Dataset507_Liver", "Dataset507_Liver", "1", "507"
    GEN = f"{R}/lits_build/gen_liver_medsam2_5pct.py"
    name, expect = f"Dataset{dsid}_lits5_ours_s{seed}", (5, 95, 31)
assert (len(gt), len(unl), len(val)) == expect, (len(gt), len(unl), len(val))
assert not (set(gt) & set(unl)) and not (set(gt) & set(val)) and not (set(unl) & set(val))
CKPT = f"{RES}/{MDS}/{tr}__nnUNetPlans__3d_fullres/fold_{FOLD}/checkpoint_final.pth"
assert os.path.exists(CKPT), f"seed model not finished: {CKPT}"
SEG, CONF, MED, QJ = f"{W}/seed_preds", f"{W}/conf.json", f"{W}/medsam2_labels", f"{W}/medsam2_quality.json"
env = dict(os.environ, nnUNet_raw=NR, nnUNet_preprocessed=NP, nnUNet_results=RES, nnUNet_compile="f")


def log(*a):
    print(f"[{os.path.basename(W)}]", *a, flush=True)


# ---------- 1. seed pseudo-labels + confidence ----------
os.makedirs(SEG, exist_ok=True)
conf = json.load(open(CONF)) if os.path.exists(CONF) else {}
todo = [c for c in unl if not (c in conf and os.path.exists(f"{SEG}/{c}.nii.gz"))]
log(f"step1 seed predictions: {len(unl) - len(todo)}/{len(unl)} already done")
for b in range(0, len(todo), 10):
    batch = todo[b:b + 10]
    inp, out = f"{W}/in", f"{W}/out"
    shutil.rmtree(inp, True)
    shutil.rmtree(out, True)
    os.makedirs(inp)
    os.makedirs(out)
    for c in batch:
        for s in glob.glob(f"{SRC}/imagesTr/{c}_*.nii.gz"):
            os.symlink(os.path.realpath(s), f"{inp}/{os.path.basename(s)}")
    subprocess.run(["nnUNetv2_predict", "-i", inp, "-o", out, "-d", PLANS, "-c", "3d_fullres", "-f", FOLD,
                    "-tr", tr, "--save_probabilities", "-npp", "2", "-nps", "2"],
                   check=True, env=env, stdout=subprocess.DEVNULL)
    for c in batch:
        d = np.load(f"{out}/{c}.npz")
        p = d["probabilities"].astype(np.float32)
        p = p / np.clip(p.sum(0, keepdims=True), 1e-6, None)
        tm = p.argmax(0) == 2
        conf[c] = float((p[2][tm] > 0.9).mean()) if tm.any() else 0.0
        d.close()
        shutil.move(f"{out}/{c}.nii.gz", f"{SEG}/{c}.nii.gz")
    json.dump(conf, open(CONF, "w"), indent=1)
    shutil.rmtree(out, True)
    shutil.rmtree(inp, True)
    log(f"step1 {len(conf)}/{len(unl)}")
assert all(c in conf for c in unl)

# ---------- 2. MedSAM2 refinement with the ORIGINAL generator (only paths changed) ----------
if len(glob.glob(f"{MED}/*.nii.gz")) < len(unl):
    src = open(GEN).read()
    reps = [("V3_DIR", f'Path("{SEG}")'), ("OUT_LABELS", f'Path("{MED}")'), ("OUT_JSON", f'"{QJ}"')]
    if data == "kits10":   # same KiTS generator, only the case list differs (121 unlabeled at 10%)
        json.dump(unl, open(f"{W}/selection.json", "w"))
        reps.append(("SELECTION", f'"{W}/selection.json"'))
    for var, val_ in reps:
        src, n = re.subn(rf"^{var} = .*$", f"{var} = {val_}", src, flags=re.M)
        assert n == 1, f"could not set {var} in {GEN}"
    gen_copy = f"{os.path.dirname(GEN)}/_e2e_{os.path.basename(W)}_{os.path.basename(GEN)}"
    open(gen_copy, "w").write(src)
    log("step2 MedSAM2 refinement ->", gen_copy)
    subprocess.run(["python3", gen_copy], check=True, cwd=os.path.dirname(GEN), env=env)
assert len(glob.glob(f"{MED}/*.nii.gz")) >= len(unl), "MedSAM2 output incomplete"
q = json.load(open(QJ))
log(f"step2 pseudo-label Dice vs GT: seed {q['mean_v3']:.4f} -> MedSAM2 {q['mean_v2']:.4f}")

# ---------- 3. gate + build dataset ----------
kept = sorted(c for c in unl if conf[c] >= TAU)
OUT = f"{NR}/{name}"
if not os.path.exists(f"{OUT}/our_split.json"):
    assert not glob.glob(f"{NR}/Dataset{dsid}_*") or os.path.exists(OUT), f"ID {dsid} used by another dataset"
    shutil.rmtree(OUT, True)
    os.makedirs(f"{OUT}/imagesTr")
    os.makedirs(f"{OUT}/labelsTr")
    for c in gt + kept + val:
        for s in glob.glob(f"{SRC}/imagesTr/{c}_*.nii.gz"):
            os.symlink(os.path.realpath(s), f"{OUT}/imagesTr/{os.path.basename(s)}")
    for c in gt + val:
        shutil.copy(f"{SRC}/labelsTr/{c}.nii.gz", f"{OUT}/labelsTr/{c}.nii.gz")
    for c in kept:
        shutil.copy(f"{MED}/{c}.nii.gz", f"{OUT}/labelsTr/{c}.nii.gz")
    dj = json.load(open(f"{SRC}/dataset.json"))
    dj["numTraining"] = len(gt + kept + val)
    json.dump(dj, open(f"{OUT}/dataset.json", "w"), indent=2)
    json.dump([{"train": sorted(gt + kept), "val": val}], open(f"{OUT}/our_split.json", "w"), indent=2)
log(f"step3 gate keeps {len(kept)}/{len(unl)} | {name}: train {len(gt) + len(kept)} val {len(val)}")

if os.environ.get("E2E_LABELS_ONLY") == "1":
    log("E2E_LABELS_ONLY=1 -> labels and gate done, stopping before preprocessing"); sys.exit(0)

# ---------- 4. preprocess with identical plans ----------
PP = f"{NP}/{name}"
if not os.path.exists(f"{PP}/splits_final.json"):
    for cmd in (["nnUNetv2_extract_fingerprint", "-d", str(dsid), "--verify_dataset_integrity"],
                ["nnUNetv2_move_plans_between_datasets", "-s", PLANS, "-t", str(dsid), "-sp", "nnUNetPlans", "-tp", "nnUNetPlans"]):
        subprocess.run(cmd, check=True, env=env)
    shutil.copy(f"{OUT}/dataset.json", f"{PP}/dataset.json")
    subprocess.run(["nnUNetv2_preprocess", "-d", str(dsid), "-plans_name", "nnUNetPlans", "-c", "3d_fullres", "-np", "2"],
                   check=True, env=env)
    n_case = len(glob.glob(f"{PP}/nnUNetPlans_3d_fullres/*.pkl"))
    assert n_case == len(gt + kept + val), f"preprocessing incomplete: {n_case}/{len(gt + kept + val)} cases"
    shutil.copy(f"{OUT}/our_split.json", f"{PP}/splits_final.json")
n_pp = len(glob.glob(f"{PP}/nnUNetPlans_3d_fullres/*"))
log(f"step4 preprocessed files: {n_pp} | READY -> train: nnUNetv2_train {dsid} 3d_fullres 0 -tr {tr}")
json.dump(dict(dsid=dsid, name=name, trainer=tr, kept=len(kept), n_unl=len(unl),
               pl_dice_seed=q["mean_v3"], pl_dice_medsam2=q["mean_v2"]), open(f"{W}/READY.json", "w"), indent=1)
