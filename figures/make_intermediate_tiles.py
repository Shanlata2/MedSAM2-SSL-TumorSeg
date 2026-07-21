import os, numpy as np, nibabel as nib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage

OUT="/root/autodl-tmp/fig_tiles"; os.makedirs(OUT, exist_ok=True)
D507="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset507_Liver"
SEED="/root/autodl-tmp/predictions/liver_sup5_on_unlabeled95"
MED="/root/autodl-tmp/lits_build/liver_medsam2_pseudo_labels_5pct"
CID="liver_76"; CT_MIN,CT_MAX=-200,300

img=nib.load(f"{D507}/imagesTr/{CID}_0000.nii.gz").get_fdata()
seed=nib.load(f"{SEED}/{CID}.nii.gz").get_fdata() if os.path.exists(f"{SEED}/{CID}.nii.gz") else None
med=nib.load(f"{MED}/{CID}.nii.gz").get_fdata() if os.path.exists(f"{MED}/{CID}.nii.gz") else None
gt=nib.load(f"{D507}/labelsTr/{CID}.nii.gz").get_fdata()

# pick slice with most tumor in the MedSAM2-refined (or GT)
ref = med if med is not None else gt
z=int(np.argmax((ref==2).sum(axis=(0,1)) if (ref==2).any() else (gt==2).sum(axis=(0,1))))
print(f"{CID} slice {z}")

def win(s): 
    s=np.clip(s,CT_MIN,CT_MAX); return ((s-CT_MIN)/(CT_MAX-CT_MIN)).T
ct=win(img[:,:,z])
# crop around liver
fg=(gt[:,:,z].T>0)|((seed[:,:,z].T>0) if seed is not None else False)
ys,xs=np.where(fg); m=25
y0,y1=max(0,ys.min()-m),min(ct.shape[0],ys.max()+m); x0,x1=max(0,xs.min()-m),min(ct.shape[1],xs.max()+m)
def cr(a): return a[y0:y1,x0:x1]
ctc=cr(ct)

def save_gray(arr,name):
    fig,ax=plt.subplots(figsize=(2.4,2.4),dpi=200); ax.imshow(arr,cmap='gray',vmin=0,vmax=1); ax.axis('off')
    plt.subplots_adjust(0,0,1,1); plt.savefig(f"{OUT}/{name}.png",bbox_inches='tight',pad_inches=0); plt.close()

def save_overlay(base,mask2,name,color='red',pts=None):
    fig,ax=plt.subplots(figsize=(2.4,2.4),dpi=200); ax.imshow(base,cmap='gray',vmin=0,vmax=1)
    if mask2 is not None and mask2.any():
        ax.contour(mask2.astype(float),[0.5],colors=color,linewidths=1.8)
    if pts is not None:
        ax.plot(pts[1],pts[0],'o',color='yellow',markersize=6,markeredgecolor='black')
    ax.axis('off'); plt.subplots_adjust(0,0,1,1); plt.savefig(f"{OUT}/{name}.png",bbox_inches='tight',pad_inches=0); plt.close()

# TILE 1: windowed CT
save_gray(ctc,"t1_windowed_ct")
# TILE 2: seed tumor mask (binary, tumor only)
if seed is not None:
    seedt=cr((seed[:,:,z].T==2).astype(float)); save_gray(seedt,"t2_seed_tumor_mask")
    save_overlay(ctc,cr(seed[:,:,z].T==2),"t2b_seed_overlay",'red')
# TILE 3: connected components of seed tumor (color-label kept components)
if seed is not None:
    lab,n=ndimage.label(seed[:,:,z].T==2)
    fig,ax=plt.subplots(figsize=(2.4,2.4),dpi=200); ax.imshow(ctc,cmap='gray',vmin=0,vmax=1)
    ax.imshow(np.ma.masked_where(cr(lab)==0,cr(lab)),cmap='tab10',alpha=0.7)
    ax.axis('off'); plt.subplots_adjust(0,0,1,1); plt.savefig(f"{OUT}/t3_components.png",bbox_inches='tight',pad_inches=0); plt.close()
# TILE 4: anchor slice with prompt point (centroid of largest component)
if seed is not None and (seed[:,:,z].T==2).any():
    cy,cx=ndimage.center_of_mass(cr(seed[:,:,z].T==2))
    save_overlay(ctc,cr(seed[:,:,z].T==2),"t4_anchor_prompt",'red',pts=(cy,cx))
# TILE 5,6: propagation frames (adjacent slices, MedSAM2 refined)
if med is not None:
    for off,tag in [(-4,"t5_prop_below"),(4,"t6_prop_above")]:
        zz=max(0,min(img.shape[2]-1,z+off)); base=win(img[:,:,zz]); basec=base[y0:y1,x0:x1]
        save_overlay(basec,cr(med[:,:,zz].T==2) if (med[:,:,zz].T==2).any() else None,tag,'#39a96b')
# TILE 7: merged organ+tumor label (MedSAM2 refined)
if med is not None:
    fig,ax=plt.subplots(figsize=(2.4,2.4),dpi=200); ax.imshow(ctc,cmap='gray',vmin=0,vmax=1)
    if (med[:,:,z].T>=1).any(): ax.contour(cr(med[:,:,z].T>=1).astype(float),[0.5],colors='#3D8BE0',linewidths=1.5)
    if (med[:,:,z].T==2).any(): ax.contour(cr(med[:,:,z].T==2).astype(float),[0.5],colors='#39a96b',linewidths=1.9)
    ax.axis('off'); plt.subplots_adjust(0,0,1,1); plt.savefig(f"{OUT}/t7_merged.png",bbox_inches='tight',pad_inches=0); plt.close()

print("Saved tiles:", sorted(os.listdir(OUT)))
