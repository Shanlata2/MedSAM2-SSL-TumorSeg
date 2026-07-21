import json, itertools
import numpy as np
from scipy.stats import wilcoxon

R = "/root/autodl-tmp/nnUNet_data/nnUNet_results"

# (label) -> (dataset_folder, trainer, fold, kits_or_lits)
CELLS = {
    "KiTS 5%":  dict(ds="Dataset501_V5_3D", fold=7, organ="kits",
                     ours=("Dataset504_ours5medsam2", 0)),
    "KiTS 10%": dict(ds="Dataset501_V5_3D", fold=6, organ="kits",
                     ours=("Dataset506_ours10medsam2", 0)),
    "LiTS 5%":  dict(ds="Dataset507_Liver", fold=1, organ="lits",
                     ours=("Dataset511_liver_ours5medsam2", 0)),
    "LiTS 10%": dict(ds="Dataset507_Liver", fold=0, organ="lits",
                     ours=("Dataset513_liver_ours10medsam2", 0)),
}
BASELINES = {
    "Supervised":  None,   # filled per-organ below
    "MeanTeacher": "nnUNetTrainerMeanTeacher",
    "UA-MT":       "nnUNetTrainerUAMT",
    "URPC":        "nnUNetTrainerURPC",
    "DTC":         "nnUNetTrainerDTC",
}
# supervised result folders (separate datasets)
SUP = {
    "KiTS 5%":  ("Dataset500_KiTS21", "nnUNetTrainer", 6),
    "KiTS 10%": ("Dataset500_KiTS21", "nnUNetTrainer", 0),
    "LiTS 5%":  ("Dataset510_liver_sup5",  "nnUNetTrainer", 0),
    "LiTS 10%": ("Dataset512_liver_sup10", "nnUNetTrainer", 0),
}

def per_case(summary_path, organ):
    m = json.load(open(summary_path))["metric_per_case"]
    out = {}
    for c in m:
        key = c.get("reference_file", "") or c.get("prediction_file", "")
        key = key.split("/")[-1]
        d2 = c["metrics"]["2"]
        if organ == "lits" and d2["n_ref"] <= 0:
            continue            # tumor-bearing only for LiTS
        out[key] = d2["Dice"]
    return out

def summ(ds, trainer, fold):
    return f"{R}/{ds}/{trainer}__nnUNetPlans__3d_fullres/fold_{fold}/validation/summary.json"

print(f"{'Cell':<9} {'vs Baseline':<12} {'mean Ours':>9} {'mean Base':>9} {'Δ':>7} {'p-value':>9}  sig")
print("-"*70)
for cell, cfg in CELLS.items():
    ours = per_case(summ(cfg["ours"][0], "nnUNetTrainer", cfg["ours"][1]), cfg["organ"])
    for bname, btr in BASELINES.items():
        if bname == "Supervised":
            ds, tr, fold = SUP[cell]
            base = per_case(summ(ds, tr, fold), cfg["organ"])
        else:
            base = per_case(summ(cfg["ds"], btr, cfg["fold"]), cfg["organ"])
        keys = sorted(set(ours) & set(base))
        if len(keys) < 5:
            print(f"{cell:<9} {bname:<12}  (only {len(keys)} shared cases - skipped)")
            continue
        o = np.array([ours[k] for k in keys]); b = np.array([base[k] for k in keys])
        try:
            p = wilcoxon(o, b).pvalue
        except ValueError:
            p = 1.0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{cell:<9} {bname:<12} {o.mean()*100:9.2f} {b.mean()*100:9.2f} {(o.mean()-b.mean())*100:+7.2f} {p:9.4f}  {sig}")
    print()