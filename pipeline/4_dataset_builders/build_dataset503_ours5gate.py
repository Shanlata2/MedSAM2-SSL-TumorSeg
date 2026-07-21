import json, os, glob, shutil

D500_RAW    = "/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset500_KiTS21"
D500_SPLITS = "/root/autodl-tmp/nnUNet_data/nnUNet_preprocessed/Dataset500_KiTS21/splits_final.json"
CONF        = "/root/autodl-tmp/sam2_v5_3d/conf_scores_5pct.json"
RAW_PREDS   = "/root/autodl-tmp/predictions/sup5_on_unlabeled133"   # raw 5% preds = pseudo-labels
D503_RAW    = "/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset503_ours5gate"
TAU = 0.80

splits = json.load(open(D500_SPLITS))
gt  = sorted(splits[6]["train"])                  # 12 labeled (5%)
val = sorted(splits[6]["val"])                    # 63 held-out
conf = json.load(open(CONF))
kept = sorted([r["case"] for r in conf if r["conf_p90"] >= TAU])   # 76 gated

# --- sanity BEFORE building ---
assert len(gt)==12, f"expected 12 GT, got {len(gt)}"
assert len(val)==63, f"expected 63 val, got {len(val)}"
print(f"GT={len(gt)}  gated pseudo={len(kept)}  val={len(val)}  total={len(gt)+len(kept)+len(val)}")
assert not (set(gt) & set(kept)), "gt/kept overlap!"
assert not (set(gt) & set(val)),  "gt/val overlap!"
assert not (set(kept) & set(val)),"kept/val overlap!"
# every source file must exist
for c in gt+val:
    assert os.path.exists(f"{D500_RAW}/labelsTr/{c}.nii.gz"), f"missing GT label {c}"
for c in kept:
    assert os.path.exists(f"{RAW_PREDS}/{c}.nii.gz"), f"missing pseudo {c}"
for c in gt+kept+val:
    assert glob.glob(f"{D500_RAW}/imagesTr/{c}_*.nii.gz"), f"missing image {c}"
print("all source files present ✓")

# --- clean rebuild ---
shutil.rmtree(D503_RAW, ignore_errors=True)
os.makedirs(f"{D503_RAW}/imagesTr"); os.makedirs(f"{D503_RAW}/labelsTr")
all_cases = gt + kept + val
for c in all_cases:                                # images for everyone (symlink)
    for s in glob.glob(f"{D500_RAW}/imagesTr/{c}_*.nii.gz"):
        os.symlink(s, f"{D503_RAW}/imagesTr/{os.path.basename(s)}")
for c in gt + val:                                 # real GT for labeled + val
    shutil.copy(f"{D500_RAW}/labelsTr/{c}.nii.gz", f"{D503_RAW}/labelsTr/{c}.nii.gz")
for c in kept:                                     # raw-5% preds as pseudo-labels
    shutil.copy(f"{RAW_PREDS}/{c}.nii.gz", f"{D503_RAW}/labelsTr/{c}.nii.gz")

dj = json.load(open(f"{D500_RAW}/dataset.json"))   # identical label config
dj["numTraining"] = len(all_cases)
json.dump(dj, open(f"{D503_RAW}/dataset.json", "w"), indent=2)
json.dump([{"train": sorted(gt + kept), "val": sorted(val)}],
          open(f"{D503_RAW}/our_split.json", "w"), indent=2)

n_img = len(glob.glob(f"{D503_RAW}/imagesTr/*.nii.gz"))
n_lab = len(glob.glob(f"{D503_RAW}/labelsTr/*.nii.gz"))
print(f"Built Dataset503: {n_img} images, {n_lab} labels")
print(f"  train = {len(gt)} GT + {len(kept)} gated-raw-pseudo = {len(gt)+len(kept)}")
print(f"  val   = {len(val)} (held-out 63)")
print(f"  labels config: {dj['labels']}")
