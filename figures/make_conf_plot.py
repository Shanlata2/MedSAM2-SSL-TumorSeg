import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

conf=json.load(open('/root/autodl-tmp/lits_conf5.json'))
vals=np.array(sorted(conf.values()))
tau=0.80
kept=vals[vals>=tau]; drop=vals[vals<tau]
print(f'n={len(vals)} | kept(>={tau}) = {len(kept)} | discarded = {len(drop)}')
print(f'kept  mean conf = {kept.mean():.3f}')
print(f'drop  mean conf = {drop.mean():.3f}')

fig,ax=plt.subplots(figsize=(4.2,1.5),dpi=220)
x=np.arange(len(vals))
ax.scatter(x[vals<tau], vals[vals<tau], s=14, c='#d6362c', label=f'discard (n={len(drop)})', zorder=3)
ax.scatter(x[vals>=tau], vals[vals>=tau], s=14, c='#3a9e68', label=f'keep (n={len(kept)})', zorder=3)
ax.axhline(tau, color='#e29620', lw=1.6, ls='--', zorder=2)
ax.text(len(vals)*0.02, tau+0.012, r'$\tau=0.80$', color='#b8760f', fontsize=8, weight='bold')
ax.set_ylabel(r'$c(\hat{y})$', fontsize=9)
ax.set_xlabel('unlabeled cases (sorted by confidence)', fontsize=8)
ax.tick_params(labelsize=7)
ax.spines[['top','right']].set_visible(False)
ax.legend(fontsize=7, frameon=False, loc='lower right')
plt.tight_layout()
plt.savefig('/root/autodl-tmp/fig_masks/d_confidence_dist.png', bbox_inches='tight', dpi=220)
print('Saved d_confidence_dist.png')
