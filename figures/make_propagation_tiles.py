import os, numpy as np, nibabel as nib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage

OUT="/root/autodl-tmp/fig_masks"; os.makedirs(OUT, exist_ok=True)
D507="/root/autodl-tmp/nnUNet_data/nnUNet_raw/Dataset507_Liver"
seed=nib.load("/root/autodl-tmp/predictions/liver_sup5_on_unlabeled95/liver_76.nii.gz").get_fdata()
med =nib.load("/root/autodl-tmp/lits_build/liver_medsam2_pseudo_labels_5pct/liver_76.nii.gz").get_fdata()

SLICES=[124,133,140,146]   # 133 = anchor
ANCHOR=133

# ONE common square crop across ALL slices (union of everything shown)
union=np.zeros((seed.shape[1],seed.shape[0]),dtype=bool)
for z in SLICES:
    union |= (seed[:,:,z].T==2) | (med[:,:,z].T==2)
ys,xs=np.where(union); m=25
y0,y1=max(0,ys.min()-m),min(union.shape[0],ys.max()+m)
x0,x1=max(0,xs.min()-m),min(union.shape[1],xs.max()+m)
h,w=y1-y0,x1-x0; s=max(h,w)
y1=min(union.shape[0],y0+s); x1=min(union.shape[1],x0+s)
def cr(a): return a[y0:y1,x0:x1]

def save_bw(mask,name,dot=None):
    c=cr(mask).astype(float)
    fig,ax=plt.subplots(figsize=(1.8,1.8),dpi=200)
    ax.imshow(c,cmap='gray',vmin=0,vmax=1)
    if dot is not None:
        ax.plot(dot[1],dot[0],'o',color='#ff3b30',markersize=7,markeredgecolor='white',markeredgewidth=0.9)
    ax.axis('off'); plt.subplots_adjust(0,0,1,1)
    plt.savefig(f"{OUT}/{name}.png",bbox_inches='tight',pad_inches=0,facecolor='black'); plt.close()
    print(f"  {name}: {int(cr(mask).sum())} px")

print("=== SEED row ===")
for z in SLICES:
    dot=None
    if z==ANCHOR and (seed[:,:,z].T==2).any():
        cy,cx=ndimage.center_of_mass(cr(seed[:,:,z].T==2)); dot=(cy,cx)
    save_bw(seed[:,:,z].T==2, f"s_seed_z{z}", dot=dot)
print("=== MEDSAM2 row ===")
for z in SLICES:
    save_bw(med[:,:,z].T==2, f"s_med_z{z}")
print("\nAll tiles:", sorted([f for f in os.listdir(OUT) if f.startswith('s_')]))
