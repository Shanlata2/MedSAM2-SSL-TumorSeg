import os, sys, glob, json
import numpy as np, nibabel as nib
from medpy import metric
from scipy import ndimage

val_dir = sys.argv[1]
gt_dir  = sys.argv[2]
kind    = sys.argv[3]           # 'lits' (tumor-bearing) or 'kits' (all)
TUMOR = 2

def comps(p):
    lab, n = ndimage.label(p)
    if n == 0: return lab, n, []
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n+1))
    return lab, n, sizes

def filt_lcc(p):
    lab, n, sizes = comps(p)
    if n <= 1: return p
    return (lab == (int(np.argmax(sizes)) + 1)).astype(np.uint8)

def filt_rel(p, frac=0.10):
    """drop components smaller than frac x largest (keeps genuine multi-focal tumors)"""
    lab, n, sizes = comps(p)
    if n <= 1: return p
    thr = max(sizes) * frac
    keep = np.zeros_like(p)
    for i, s in enumerate(sizes, start=1):
        if s >= thr: keep[lab == i] = 1
    return keep.astype(np.uint8)

def four(p, g):
    if p.sum() == 0: return dict(D=0.0, J=0.0, H=None, A=None)
    return dict(D=metric.binary.dc(p,g), J=metric.binary.jc(p,g),
                H=metric.binary.hd95(p,g), A=metric.binary.asd(p,g))

rows=[]
files = sorted(glob.glob(val_dir + '/*.nii.gz'))
for i, pf in enumerate(files):
    cid = os.path.basename(pf)[:-7]
    gf = os.path.join(gt_dir, cid + '.nii.gz')
    if not os.path.exists(gf): continue
    pred = nib.load(pf).get_fdata(); gt = nib.load(gf).get_fdata()
    p = (pred==TUMOR).astype(np.uint8); g = (gt==TUMOR).astype(np.uint8)
    if kind=='lits' and g.sum()==0: continue
    if g.sum()==0 and kind=='kits':
        rows.append(dict(case=cid, gt_vox=0)); continue

    lab, n, sizes = comps(p)
    largest = int(max(sizes)) if n>0 else 0
    r = dict(case=cid, gt_vox=int(g.sum()), pred_vox=int(p.sum()),
             ncomp=int(n), largest=largest,
             largest_frac=float(largest/max(1,p.sum())))
    r['raw'] = four(p, g)
    r['lcc'] = four(filt_lcc(p), g)
    r['rel10'] = four(filt_rel(p, 0.10), g)
    rows.append(r)
    print(f'  {i+1}/{len(files)}  {cid}', end='\r')

print()
def avg(key, sub):
    v=[r[key][sub] for r in rows if key in r and r[key][sub] is not None]
    return sum(v)/len(v) if v else float('nan')
def avgD(key):
    v=[r[key]['D'] for r in rows if key in r]
    return sum(v)/len(v)*100 if v else float('nan')
def avgJ(key):
    v=[r[key]['J'] for r in rows if key in r]
    return sum(v)/len(v)*100 if v else float('nan')

print('='*72)
print(f'{"variant":10s} {"Dice":>7s} {"JA":>7s} {"HD95":>8s} {"ASD":>7s}')
for k,name in [('raw','raw'),('lcc','largest-CC'),('rel10','drop<10%')]:
    print(f'{name:10s} {avgD(k):7.2f} {avgJ(k):7.2f} {avg(k,"H"):8.2f} {avg(k,"A"):7.2f}')
print('='*72)

# the offenders
have = [r for r in rows if 'raw' in r and r['raw']['H'] is not None]
have.sort(key=lambda r: -r['raw']['H'])
print('\nWORST HD95 CASES (raw):')
print(f'{"case":16s} {"HD95":>8s} {"Dice":>6s} {"#comp":>6s} {"largest%":>9s} | {"HD95_lcc":>9s} {"Dice_lcc":>9s}')
for r in have[:10]:
    hl = r['lcc']['H']; dl = r['lcc']['D']
    print(f'{r["case"]:16s} {r["raw"]["H"]:8.1f} {r["raw"]["D"]*100:6.1f} {r["ncomp"]:6d} {r["largest_frac"]*100:8.1f}% | '
          f'{(hl if hl else -1):9.1f} {dl*100:9.1f}')

nmulti = sum(1 for r in rows if r.get('ncomp',0) > 1)
print(f'\ncases with >1 predicted component: {nmulti}/{len(rows)}')
json.dump(rows, open('/root/autodl-tmp/hd95_diag.json','w'), indent=2, default=float)
print('saved hd95_diag.json')
