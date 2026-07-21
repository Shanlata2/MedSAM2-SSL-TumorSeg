import json, os, glob, shutil
D500_RAW="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset500_KiTS21"
D500_SPLITS="/root/autodl-tmp/nnUNet_data/nnUNet_preprocessed/Dataset500_KiTS21/splits_final.json"
CONF="/root/autodl-tmp/sam2_v5_3d/conf_scores_5pct.json"
MEDSAM2="/root/autodl-tmp/sam2_v5_3d/medsam2_pseudo_labels_5pct"
D504_RAW="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset504_ours5medsam2"
TAU=0.80
sp=json.load(open(D500_SPLITS)); gt=sorted(sp[6]["train"]); val=sorted(sp[6]["val"])
kept=sorted([r["case"] for r in json.load(open(CONF)) if r["conf_p90"]>=TAU])
assert len(gt)==12 and len(val)==63
print(f"GT={len(gt)} gated={len(kept)} val={len(val)} total={len(gt)+len(kept)+len(val)}")
assert not(set(gt)&set(kept)) and not(set(gt)&set(val)) and not(set(kept)&set(val))
for c in gt+val: assert os.path.exists(f"{D500_RAW}/labelsTr/{c}.nii.gz"), f"GT {c}"
for c in kept:   assert os.path.exists(f"{MEDSAM2}/{c}.nii.gz"), f"medsam2 {c}"
print("all sources present ✓")
shutil.rmtree(D504_RAW, ignore_errors=True)
os.makedirs(f"{D504_RAW}/imagesTr"); os.makedirs(f"{D504_RAW}/labelsTr")
for c in gt+kept+val:
    for s in glob.glob(f"{D500_RAW}/imagesTr/{c}_*.nii.gz"):
        os.symlink(s, f"{D504_RAW}/imagesTr/{os.path.basename(s)}")
for c in gt+val: shutil.copy(f"{D500_RAW}/labelsTr/{c}.nii.gz", f"{D504_RAW}/labelsTr/{c}.nii.gz")
for c in kept:   shutil.copy(f"{MEDSAM2}/{c}.nii.gz", f"{D504_RAW}/labelsTr/{c}.nii.gz")
dj=json.load(open(f"{D500_RAW}/dataset.json")); dj["numTraining"]=len(gt+kept+val)
json.dump(dj, open(f"{D504_RAW}/dataset.json","w"), indent=2)
json.dump([{"train":sorted(gt+kept),"val":sorted(val)}], open(f"{D504_RAW}/our_split.json","w"), indent=2)
print(f"Built D504: {len(glob.glob(D504_RAW+'/imagesTr/*.nii.gz'))} imgs / {len(glob.glob(D504_RAW+'/labelsTr/*.nii.gz'))} labels | train {len(gt)+len(kept)} val {len(val)}")
