import os, sys, glob, json
import numpy as np, nibabel as nib
from medpy import metric
from scipy import ndimage

val_dir = sys.argv[1]; gt_dir = sys.argv[2]; kind = sys.argv[3]
TUMOR, ORGAN = 2, 1

def four(p,g):
    if p.sum()==0: return dict(D=0.0,J=0.0,H=None,A=None)
    return dict(D=metric.binary.dc(p,g), J=metric.binary.jc(p,g),
                H=metric.binary.hd95(p,g), A=metric.binary.asd(p,g))

def organ_constrained(tum, org, dil=3):
    """keep tumor components that touch/lie within the predicted organ (dilated)"""
    if tum.sum()==0 or org.sum()==0: return tum
    org_d = ndimage.binary_dilation(org, iterations=dil)
    lab,n = ndimage.label(tum)
    keep = np.zeros_like(tum)
    for i in range(1,n+1):
        c = (lab==i)
        if (c & org_d).sum() > 0:          # overlaps organ -> anatomically plausible
            keep[c]=1
    return keep.astype(np.uint8) if keep.sum()>0 else tum

def dist_from_main(tum, max_ratio=1.0):
    """drop components whose distance from the LARGEST component exceeds
       max_ratio x (largest component's own diameter)"""
    lab,n = ndimage.label(tum)
    if n<=1: return tum
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1,n+1))
    main_i = int(np.argmax(sizes))+1
    main = (lab==main_i)
    # main's extent
    pts = np.array(np.nonzero(main)).T
    diam = np.linalg.norm(pts.max(0)-pts.min(0)) + 1e-6
    dmap = ndimage.distance_transform_edt(~main)
    keep = main.copy()
    for i in range(1,n+1):
        if i==main_i: continue
        c = (lab==i)
        d = dmap[c].min()
        if d <= max_ratio*diam:
            keep |= c
    return keep.astype(np.uint8)

rows=[]
for i,pf in enumerate(sorted(glob.glob(val_dir+'/*.nii.gz'))):
    cid=os.path.basename(pf)[:-7]; gf=os.path.join(gt_dir,cid+'.nii.gz')
    if not os.path.exists(gf): continue
    pr=nib.load(pf).get_fdata(); gt=nib.load(gf).get_fdata()
    p=(pr==TUMOR).astype(np.uint8); o=(pr==ORGAN).astype(np.uint8); g=(gt==TUMOR).astype(np.uint8)
    if kind=='lits' and g.sum()==0: continue
    if g.sum()==0: continue
    r=dict(case=cid)
    r['raw']   = four(p,g)
    r['organ'] = four(organ_constrained(p,o), g)
    r['dist']  = four(dist_from_main(p,1.0), g)
    r['both']  = four(dist_from_main(organ_constrained(p,o),1.0), g)
    rows.append(r); print(f'  {i+1} {cid}',end='\r')
print()
def A(k,s):
    v=[r[k][s] for r in rows if r[k][s] is not None]; return sum(v)/len(v) if v else float('nan')
def AD(k):
    return sum(r[k]['D'] for r in rows)/len(rows)*100
def AJ(k):
    return sum(r[k]['J'] for r in rows)/len(rows)*100
print('='*70)
print(f'{"variant":22s} {"Dice":>7s} {"JA":>7s} {"HD95":>8s} {"ASD":>7s}')
for k,name in [('raw','raw (reported)'),('organ','organ-constrained'),
               ('dist','distance-from-main'),('both','organ + distance')]:
    print(f'{name:22s} {AD(k):7.2f} {AJ(k):7.2f} {A(k,"H"):8.2f} {A(k,"A"):7.2f}')
print('='*70)
json.dump(rows, open('/root/autodl-tmp/hd95_fix_v2.json','w'), indent=2, default=float)
