"""Build reproducible diagnostics from committed summaries (no fabricated scores)."""
from pathlib import Path
import csv, json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                     'figure.facecolor':'white','axes.facecolor':'white','font.size':9})

# Dataset composition from the committed public-benchmark summary.
rows = list(csv.DictReader((ROOT/'paper/results/ade_conll04/dataset_statistics.csv').open()))
ds = sorted({r['dataset'] for r in rows}); splits=['train','validation','test']
fig, ax = plt.subplots(figsize=(5.8,3.0)); w=.34; import numpy as np
x=np.arange(len(splits))
for j, d in enumerate(ds):
    vals=[int(next(r['sentences'] for r in rows if r['dataset']==d and r['split']==s)) for s in splits]
    ax.bar(x+(j-(len(ds)-1)/2)*w, vals, width=w, label=d)
ax.set_xticks(x, splits); ax.set_ylabel('Sentences'); ax.set_title('Public benchmark split distribution')
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/'data_distribution_public.pdf',bbox_inches='tight'); fig.savefig(OUT/'data_distribution_public.png',dpi=400,bbox_inches='tight'); plt.close(fig)

# Loss diagnostic from the existing committed run log.
loss = list(csv.DictReader((ROOT/'paper/results/ade_conll04/training_loss_seed42.csv').open()))
fig, ax = plt.subplots(figsize=(5.8,3.0))
for system in sorted({r['system'] for r in loss}):
    x=[int(r['step']) for r in loss if r['system']==system]; y=[float(r['loss']) for r in loss if r['system']==system]
    ax.plot(x,y,marker='.',linewidth=1.2,label=system)
ax.set_xlabel('Optimization step'); ax.set_ylabel('Loss'); ax.set_title('Training-loss diagnostic (committed pilot run)')
ax.legend(frameon=False,ncol=2); fig.tight_layout(); fig.savefig(OUT/'training_loss_diagnostic.pdf',bbox_inches='tight'); fig.savefig(OUT/'training_loss_diagnostic.png',dpi=400,bbox_inches='tight'); plt.close(fig)

# Results figure is generated only when a new experiment CSV exists.
result_file = ROOT/'paper/cross-residual/results/results.csv'
if result_file.exists():
    rr=list(csv.DictReader(result_file.open())); methods=sorted({r['method'] for r in rr})
    fig, ax=plt.subplots(figsize=(6.2,3.2))
    for m in methods:
        q=[r for r in rr if r['method']==m]; ax.plot([r['dataset'] for r in q],[float(r['relation_f1']) for r in q],marker='o',label=m)
    ax.set_ylabel('Relation F1'); ax.set_title('Cross-residual model comparison'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/'results_comparison.pdf',bbox_inches='tight'); fig.savefig(OUT/'results_comparison.png',dpi=400,bbox_inches='tight')
