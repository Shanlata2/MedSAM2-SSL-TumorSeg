import os, sys, glob, json, math
import numpy as np
import nibabel as nib
from medpy import metric

# Edit this list to point at each method's 4 cells.
# Format: (label, validation_dir, gt_dir, kind)
METHOD = sys.argv[1] if len(sys.argv) > 1 else 'DyCON'

RESULTS = "/root/autodl-tmp/nnUNet_data/nnUNet_results"
GT_LITS = "/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset507_Liver/labelsTr"
GT_KITS = "/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset501_V5_3D/labelsTr"

# map method -> trainer folder name
TRAINERS = {
    'DyCON': 'nnUNetTrainerDyCON',
    'MCNet': 'nnUNetTrainerMCNet',
    'MeanTeacher': 'nnUNetTrainerMeanTeacher',
    'UAMT': 'nnUNetTrainerUAMT',
    'URPC': 'nnUNetTrainerURPC',
    'DTC': 'nnUNetTrainerDTC',
}
tr = TRAINERS.get(METHOD, METHOD)

cells = [
    (f'{METHOD} LiTS5',  f'{RESULTS}/Dataset507_Liver/{tr}__nnUNetPlans__3d_fullres/fold_1/validation', GT_LITS, 'lits'),
    (f'{METHOD} LiTS10', f'{RESULTS}/Dataset507_Liver/{tr}__nnUNetPlans__3d_fullres/fold_0/validation', GT_LITS, 'lits'),
    (f'{METHOD} KiTS5',  f'{RESULTS}/Dataset501_V5_3D/{tr}__nnUNetPlans__3d_fullres/fold_7/validation', GT_KITS, 'kits'),
    (f'{METHOD} KiTS10', f'{RESULTS}/Dataset501_V5_3D/{tr}__nnUNetPlans__3d_fullres/fold_6/validation', GT_KITS, 'kits'),
]

def metrics_case(pred, gt):
    p = (pred == 2).astype(np.uint8)
    g = (gt == 2).astype(np.uint8)
    if g.sum() == 0:
        return None
    if p.sum() == 0:
        return {'Dice': 0.0, 'JA': 0.0, 'HD95': None, 'ASD': None}
    dice = metric.binary.dc(p, g)
    ja   = metric.binary.jc(p, g)
    try:
        hd95 = metric.binary.hd95(p, g)
        asd  = metric.binary.asd(p, g)
    except Exception:
        hd95, asd = None, None
    return {'Dice': dice, 'JA': ja, 'HD95': hd95, 'ASD': asd}

def avg(rows, key):
    vals = [r[key] for r in rows if r[key] is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
    return sum(vals)/len(vals) if vals else float('nan')

out = {}
for name, val_dir, gt_dir, kind in cells:
    if not os.path.isdir(val_dir):
        print(f'{name}: SKIP (no validation dir yet)')
        continue
    preds = sorted(glob.glob(val_dir + '/*.nii.gz'))
    rows = []
    for i, pf in enumerate(preds):
        cid = os.path.basename(pf)[:-7]
        gf = os.path.join(gt_dir, cid + '.nii.gz')
        if not os.path.exists(gf):
            continue
        pred = nib.load(pf).get_fdata()
        gt   = nib.load(gf).get_fdata()
        r = metrics_case(pred, gt)
        if kind == 'kits' and r is None:
            r = {'Dice':0.0,'JA':0.0,'HD95':None,'ASD':None}  # KiTS: keep all
        if r is not None:
            rows.append(r)
        print(f'  {name}: {i+1}/{len(preds)}', end='\r')
    n=len(rows)
    res={'n':n,'Dice':avg(rows,'Dice')*100,'JA':avg(rows,'JA')*100,'HD95':avg(rows,'HD95'),'ASD':avg(rows,'ASD')}
    out[name]=res
    print(f'\n{name:16s} n={n:3d} | Dice {res["Dice"]:6.2f} | JA {res["JA"]:6.2f} | HD95 {res["HD95"]:7.2f} | ASD {res["ASD"]:6.2f}')

# save
with open(f'/root/autodl-tmp/metrics_{METHOD}.json','w') as f:
    json.dump(out, f, indent=2)
print(f'\nSaved to metrics_{METHOD}.json')
