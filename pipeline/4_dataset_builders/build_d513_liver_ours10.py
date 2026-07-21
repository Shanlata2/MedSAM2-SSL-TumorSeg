import json, os, glob, shutil
D507="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset507_Liver"
S=json.load(open("/root/autodl-tmp/lits_ssl_split.json"))
CONF=json.load(open("/root/autodl-tmp/lits_conf10.json"))
MED="/root/autodl-tmp/lits_build/liver_medsam2_pseudo_labels_10pct"
OUT="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset513_liver_ours10medsam2"
TAU=0.80
gt=S["labeled_10pct"]; val=S["val"]
kept=sorted([c for c in S["unlabeled_10pct"] if CONF.get(c,0)>=TAU])
print(f"GT={len(gt)} gated={len(kept)} val={len(val)} total={len(gt)+len(kept)+len(val)}")
assert len(gt)==10 and len(val)==31
assert not(set(gt)&set(kept)) and not(set(gt)&set(val)) and not(set(kept)&set(val))
for c in gt+val: assert os.path.exists(f"{D507}/labelsTr/{c}.nii.gz")
for c in kept:   assert os.path.exists(f"{MED}/{c}.nii.gz")
print("sources OK")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(f"{OUT}/imagesTr"); os.makedirs(f"{OUT}/labelsTr")
for c in gt+kept+val:
    for s in glob.glob(f"{D507}/imagesTr/{c}_*.nii.gz"):
        os.symlink(s, f"{OUT}/imagesTr/{os.path.basename(s)}")
for c in gt+val: shutil.copy(f"{D507}/labelsTr/{c}.nii.gz", f"{OUT}/labelsTr/{c}.nii.gz")
for c in kept:   shutil.copy(f"{MED}/{c}.nii.gz", f"{OUT}/labelsTr/{c}.nii.gz")
dj=json.load(open(f"{D507}/dataset.json")); dj["numTraining"]=len(gt+kept+val)
json.dump(dj, open(f"{OUT}/dataset.json","w"), indent=2)
json.dump([{"train":sorted(gt+kept),"val":sorted(val)}], open(f"{OUT}/our_split.json","w"), indent=2)
print(f"Built D511: {len(glob.glob(OUT+'/imagesTr/*.nii.gz'))} imgs / {len(glob.glob(OUT+'/labelsTr/*.nii.gz'))} labels | train {len(gt)+len(kept)} val {len(val)}")
