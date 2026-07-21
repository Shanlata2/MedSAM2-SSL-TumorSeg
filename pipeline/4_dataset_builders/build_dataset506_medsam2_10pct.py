import json, os, glob, shutil
D500_RAW="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset500_KiTS21"
D501_SPLITS="/root/autodl-tmp/nnUNet_data/nnUNet_preprocessed/Dataset501_V5_3D/splits_final.json"
SEL="/root/autodl-tmp/v5_3d_pseudo_label_selection.json"
CONF="/root/autodl-tmp/sam2_v5_3d/all_conf_scores.json"
MEDSAM2="/root/autodl-tmp/sam2_v5_3d/medsam2_pseudo_labels_10pct"
D506_RAW="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset506_ours10medsam2"
TAU=0.80
sp=json.load(open(D501_SPLITS)); train0,val0=sp[0]["train"],sp[0]["val"]
pid=set(json.load(open(SEL))["pseudo_label_case_ids"])
gt=sorted([c for c in train0 if c not in pid]); val=sorted(val0)
conf={r['case']:r['conf_p90'] for r in json.load(open(CONF))}
kept=sorted([c for c in pid if conf.get(c,0)>=TAU])
print(f"GT={len(gt)} gated={len(kept)} val={len(val)} total={len(gt)+len(kept)+len(val)}")
assert len(gt)==24 and len(val)==63, f"got {len(gt)}/{len(val)}"
assert not(set(gt)&set(kept)) and not(set(gt)&set(val)) and not(set(kept)&set(val))
for c in gt+val: assert os.path.exists(f"{D500_RAW}/labelsTr/{c}.nii.gz"), f"GT {c}"
for c in kept:   assert os.path.exists(f"{MEDSAM2}/{c}.nii.gz"), f"medsam2 {c}"
print("all sources present ✓")
shutil.rmtree(D506_RAW, ignore_errors=True)
os.makedirs(f"{D506_RAW}/imagesTr"); os.makedirs(f"{D506_RAW}/labelsTr")
for c in gt+kept+val:
    for s in glob.glob(f"{D500_RAW}/imagesTr/{c}_*.nii.gz"):
        os.symlink(s, f"{D506_RAW}/imagesTr/{os.path.basename(s)}")
for c in gt+val: shutil.copy(f"{D500_RAW}/labelsTr/{c}.nii.gz", f"{D506_RAW}/labelsTr/{c}.nii.gz")
for c in kept:   shutil.copy(f"{MEDSAM2}/{c}.nii.gz", f"{D506_RAW}/labelsTr/{c}.nii.gz")
dj=json.load(open(f"{D500_RAW}/dataset.json")); dj["numTraining"]=len(gt+kept+val)
json.dump(dj, open(f"{D506_RAW}/dataset.json","w"), indent=2)
json.dump([{"train":sorted(gt+kept),"val":sorted(val)}], open(f"{D506_RAW}/our_split.json","w"), indent=2)
print(f"Built D506: {len(glob.glob(D506_RAW+'/imagesTr/*.nii.gz'))} imgs / {len(glob.glob(D506_RAW+'/labelsTr/*.nii.gz'))} labels | train {len(gt)+len(kept)} val {len(val)}")
