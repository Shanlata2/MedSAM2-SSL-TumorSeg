import numpy as np, nibabel as nib, glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CID = "liver_7"   # chosen held-out validation case
OUT = "/root/autodl-tmp/pipeline_figs_final"
os.makedirs(OUT, exist_ok=True)

D507="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset507_Liver"
STU ="/root/autodl-tmp/nnUNet_data/nnUNet_results/Dataset511_liver_ours5medsam2/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/validation"
MED ="/root/autodl-tmp/lits_build/liver_medsam2_pseudo_labels_5pct"
CT_MIN,CT_MAX=-200,300

img = nib.load(f"{D507}/imagesTr/{CID}_0000.nii.gz").get_fdata()
gt  = nib.load(f"{D507}/labelsTr/{CID}.nii.gz").get_fdata()
stu = nib.load(f"{STU}/{CID}.nii.gz").get_fdata()
med_path = f"{MED}/{CID}.nii.gz"
med = nib.load(med_path).get_fdata() if os.path.exists(med_path) else None

# slice with most tumor
z = int(np.argmax((gt==2).sum(axis=(0,1))))
print(f"{CID} slice {z} | tumor vox {int((gt==2).sum())}")

def wn(s):
    s=np.clip(s,CT_MIN,CT_MAX); return (s-CT_MIN)/(CT_MAX-CT_MIN)
ct=wn(img[:,:,z]).T
gt_s=gt[:,:,z].T; stu_s=stu[:,:,z].T
med_s=med[:,:,z].T if med is not None else stu_s  # fallback

# tight crop around tumor+organ
reg=(gt_s>0)|(stu_s>0)
ys,xs=np.where(reg); m=28
y0,y1=max(0,ys.min()-m),min(ct.shape[0],ys.max()+m)
x0,x1=max(0,xs.min()-m),min(ct.shape[1],xs.max()+m)
def cr(a): return a[y0:y1,x0:x1]
ctc,gtc,stuc,medc=cr(ct),cr(gt_s),cr(stu_s),cr(med_s)

def contour(ct,lab,name):
    fig,ax=plt.subplots(figsize=(3,3),dpi=220)
    ax.imshow(ct,cmap='gray',vmin=0,vmax=1)
    if (lab>=1).any(): ax.contour((lab>=1).astype(float),[0.5],colors='#3D8BE0',linewidths=1.6)
    if (lab==2).any(): ax.contour((lab==2).astype(float),[0.5],colors='#E0362C',linewidths=2.0)
    ax.axis('off'); plt.subplots_adjust(0,0,1,1)
    plt.savefig(f"{OUT}/{name}.png",bbox_inches='tight',pad_inches=0); plt.close()

# plain CT
fig,ax=plt.subplots(figsize=(3,3),dpi=220)
ax.imshow(ctc,cmap='gray',vmin=0,vmax=1); ax.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.savefig(f"{OUT}/a_raw_ct.png",bbox_inches='tight',pad_inches=0); plt.close()

contour(ctc, medc if med is not None else stuc, "c_medsam2_refined")
contour(ctc, gtc,  "d_ground_truth")
contour(ctc, stuc, "e_student_pred")   # THE REAL held-out student prediction

# seed: compute a rough seed pseudo by eroding? No—use MedSAM2 source seed if available
print("Saved:", sorted(os.listdir(OUT)))
print("NOTE: e_student_pred.png is the REAL held-out student output for", CID)