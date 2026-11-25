import json, os, glob, shutil
D500_RAW="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset500_KiTS21"
D500_SPLITS="/root/autodl-tmp/nnUNet_data/nnUNet_preprocessed/Dataset500_KiTS21/splits_final.json"
UNLAB="/root/autodl-tmp/ours5_unlabeled133.json"
SAMB="/root/autodl-tmp/sam2_v5_3d/samburden_pseudo_labels_5pct"
DOUT="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset520_samburden5"
sp=json.load(open(D500_SPLITS)); gt=sorted(sp[6]["train"]); val=sorted(sp[6]["val"])
unlab=sorted(json.load(open(UNLAB)))
assert len(gt)==12 and len(val)==63, f"gt={len(gt)} val={len(val)}"
print(f"GT={len(gt)} unlab={len(unlab)} val={len(val)} total={len(gt)+len(unlab)+len(val)}")
assert not(set(gt)&set(unlab)) and not(set(gt)&set(val)) and not(set(unlab)&set(val))
for c in gt+val: assert os.path.exists(f"{D500_RAW}/labelsTr/{c}.nii.gz"), f"GT {c}"
for c in unlab: assert os.path.exists(f"{SAMB}/{c}.nii.gz"), f"samburden {c}"
print("all sources present ✓")
shutil.rmtree(DOUT, ignore_errors=True)
os.makedirs(f"{DOUT}/imagesTr"); os.makedirs(f"{DOUT}/labelsTr")
for c in gt+unlab+val:
    for s in glob.glob(f"{D500_RAW}/imagesTr/{c}_*.nii.gz"):
        os.symlink(s, f"{DOUT}/imagesTr/{os.path.basename(s)}")
for c in gt+val: shutil.copy(f"{D500_RAW}/labelsTr/{c}.nii.gz", f"{DOUT}/labelsTr/{c}.nii.gz")
for c in unlab: shutil.copy(f"{SAMB}/{c}.nii.gz", f"{DOUT}/labelsTr/{c}.nii.gz")
dj=json.load(open(f"{D500_RAW}/dataset.json")); dj["numTraining"]=len(gt+unlab+val)
json.dump(dj, open(f"{DOUT}/dataset.json","w"), indent=2)
json.dump([{"train":sorted(gt+unlab),"val":sorted(val)}], open(f"{DOUT}/our_split.json","w"), indent=2)
print(f"Built D520: {len(glob.glob(DOUT+'/imagesTr/*.nii.gz'))} imgs / {len(glob.glob(DOUT+'/labelsTr/*.nii.gz'))} labels | train {len(gt)+len(unlab)} val {len(val)}")
