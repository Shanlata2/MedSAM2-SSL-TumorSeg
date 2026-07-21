import json, os, glob, subprocess, numpy as np
os.environ["nnUNet_raw"]="/root/autodl-tmp/nnUNet_data/nnUNet_raw"
os.environ["nnUNet_preprocessed"]="/root/autodl-tmp/nnUNet_data/nnUNet_preprocessed"
os.environ["nnUNet_results"]="/root/autodl-tmp/nnUNet_data/nnUNet_results"
S=json.load(open("/root/autodl-tmp/lits_ssl_split.json"))
unlab=S["unlabeled_10pct"]
D507="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset507_Liver"
OUT="/root/autodl-tmp/predictions/liver_sup10_on_unlabeled90"
os.makedirs(OUT, exist_ok=True)
CONF="/root/autodl-tmp/lits_conf10.json"
conf=json.load(open(CONF)) if os.path.exists(CONF) else {}
B=15
batches=[unlab[i:i+B] for i in range(0,len(unlab),B)]
print(f"{len(unlab)} unlabeled -> {len(batches)} batches of {B}", flush=True)
for bi,batch in enumerate(batches):
    inp=f"/tmp/b{bi}"; os.makedirs(inp, exist_ok=True)
    for c in batch:
        for s in glob.glob(f"{D507}/imagesTr/{c}_*.nii.gz"):
            dst=f"{inp}/{os.path.basename(s)}"
            if not os.path.lexists(dst): os.symlink(s, dst)
    print(f"[batch {bi+1}/{len(batches)}] predicting {len(batch)}...", flush=True)
    subprocess.run(["nnUNetv2_predict","-i",inp,"-o",OUT,"-d","512","-c","3d_fullres",
                    "-tr","nnUNetTrainer_250epochs","-f","0","--save_probabilities"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for c in batch:
        f=f"{OUT}/{c}.npz"
        if not os.path.exists(f): print("  MISSING npz",c); continue
        try:
            d=np.load(f); k='probabilities' if 'probabilities' in d else d.files[0]
            p=d[k]; pred=p.argmax(0); tm=(pred==2); tp=p[2]
            conf[c]=float((tp[tm]>0.9).mean()) if tm.any() else 0.0
            d.close()
        except Exception as e: print("  BAD",c,e)
    json.dump(conf, open(CONF,"w"), indent=2)
    for f in glob.glob(f"{OUT}/*.npz"): os.remove(f)
    free=subprocess.run(["df","-h","/root/autodl-tmp"],capture_output=True,text=True).stdout.splitlines()[-1].split()[3]
    print(f"  conf total {len(conf)} | npz cleared | free {free}", flush=True)
gated=sum(1 for v in conf.values() if v>=0.80)
print(f"DONE: seeds {len(glob.glob(OUT+'/*.nii.gz'))}/90 | conf {len(conf)}/90 | gated(>=0.80) {gated}", flush=True)
