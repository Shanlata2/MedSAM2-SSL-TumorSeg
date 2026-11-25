"""
build_ablation.py <id> <variant>     variant: plain | gate | refine | tau070 | tau090
KiTS21 5% ablation rebuilt from END-TO-END SEED 1 (the same seed model / pseudo-labels as the full method, Dataset526):
  plain  : 12 GT + all 133 seed pseudo-labels            (no refinement, no gate)
  gate   : 12 GT + seed pseudo-labels with c >= 0.80      (gate only)
  refine : 12 GT + all 133 MedSAM2-refined labels         (refinement only)
  (full  = Dataset526: 12 GT + MedSAM2 labels with c >= 0.80)
Then fingerprint + plans from 501 + preprocess (-np 2) + case-count check + split.  Resumable.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

R = "/root/autodl-tmp"
NR, NP, RES = f"{R}/nnUNet_data/nnUNet_raw", f"{R}/nnUNet_data/nnUNet_preprocessed", f"{R}/nnUNet_data/nnUNet_results"
W = f"{R}/e2e/kits5_s1"
dsid, variant = int(sys.argv[1]), sys.argv[2]
name = f"Dataset{dsid}_kits5_abl_{variant}_s1"
env = dict(os.environ, nnUNet_raw=NR, nnUNet_preprocessed=NP, nnUNet_results=RES, nnUNet_compile="f")

sp = json.load(open(f"{NP}/Dataset501_V5_3D/splits_final.json"))[7]
gt, val = sorted(sp["train"]), sorted(sp["val"])
unl = sorted(json.load(open(f"{R}/ours5_unlabeled133.json")))
conf = json.load(open(f"{W}/conf.json"))
assert (len(gt), len(unl), len(val)) == (12, 133, 63) and all(c in conf for c in unl)
assert not (set(gt) & set(unl)) and not (set(gt) & set(val)) and not (set(unl) & set(val))
TAUV = {"tau070": 0.70, "tau090": 0.90}   # tau sweep: refined labels gated at another tau (full method uses 0.80)
kept = sorted(c for c in unl if conf[c] >= TAUV.get(variant, 0.80))
src_dir = {"plain": f"{W}/seed_preds", "gate": f"{W}/seed_preds", "refine": f"{W}/medsam2_labels",
           "tau070": f"{W}/medsam2_labels", "tau090": f"{W}/medsam2_labels"}[variant]
pseudo = kept if variant in ("gate", "tau070", "tau090") else unl
SRC = f"{NR}/Dataset500_KiTS21"
OUT, PP = f"{NR}/{name}", f"{NP}/{name}"
print(f"[{name}] pseudo={len(pseudo)} (gate keeps {len(kept)}) from {src_dir}", flush=True)

if not os.path.exists(f"{OUT}/our_split.json"):
    other = [d for d in glob.glob(f"{NR}/Dataset{dsid}_*") if d != OUT]
    assert not other, f"ID {dsid} already used: {other}"
    shutil.rmtree(OUT, True)
    os.makedirs(f"{OUT}/imagesTr")
    os.makedirs(f"{OUT}/labelsTr")
    for c in gt + pseudo + val:
        for s in glob.glob(f"{SRC}/imagesTr/{c}_*.nii.gz"):
            os.symlink(os.path.realpath(s), f"{OUT}/imagesTr/{os.path.basename(s)}")
    for c in gt + val:
        shutil.copy(f"{SRC}/labelsTr/{c}.nii.gz", f"{OUT}/labelsTr/{c}.nii.gz")
    for c in pseudo:
        shutil.copy(f"{src_dir}/{c}.nii.gz", f"{OUT}/labelsTr/{c}.nii.gz")
    dj = json.load(open(f"{SRC}/dataset.json"))
    dj["numTraining"] = len(gt + pseudo + val)
    json.dump(dj, open(f"{OUT}/dataset.json", "w"), indent=2)
    json.dump([{"train": sorted(gt + pseudo), "val": val}], open(f"{OUT}/our_split.json", "w"), indent=2)

if not os.path.exists(f"{PP}/splits_final.json"):
    for cmd in (["nnUNetv2_extract_fingerprint", "-d", str(dsid), "--verify_dataset_integrity"],
                ["nnUNetv2_move_plans_between_datasets", "-s", "501", "-t", str(dsid), "-sp", "nnUNetPlans", "-tp", "nnUNetPlans"]):
        subprocess.run(cmd, check=True, env=env)
    shutil.copy(f"{OUT}/dataset.json", f"{PP}/dataset.json")
    subprocess.run(["nnUNetv2_preprocess", "-d", str(dsid), "-plans_name", "nnUNetPlans", "-c", "3d_fullres", "-np", "2"],
                   check=True, env=env)
    n_case = len(glob.glob(f"{PP}/nnUNetPlans_3d_fullres/*.pkl"))
    assert n_case == len(gt + pseudo + val), f"preprocessing incomplete: {n_case}/{len(gt + pseudo + val)}"
    shutil.copy(f"{OUT}/our_split.json", f"{PP}/splits_final.json")
print(f"[{name}] READY train {len(gt) + len(pseudo)} val {len(val)} -> nnUNetv2_train {dsid} 3d_fullres 0 -tr nnUNetTrainer_250epochs", flush=True)
