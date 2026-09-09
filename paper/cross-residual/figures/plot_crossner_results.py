#!/usr/bin/env python3
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]; RES=ROOT/'paper/cross-residual/results/crossner'; OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'axes.titlesize':12,'axes.labelsize':11,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'figure.facecolor':'white','axes.facecolor':'white','savefig.dpi':400})
runs=list(csv.DictReader((RES/'runs.csv').open())); hist=list(csv.DictReader((RES/'history.csv').open()))
domains=['ai','literature','music','politics','science']; methods=['bigru','cross_residual']; names={'bigru':'BiGRU','cross_residual':'Cross-Residual'}; colors={'bigru':'#0072B2','cross_residual':'#009E73'}

agg=[]
for d in domains:
    for m in methods:
        q=[float(x['f1']) for x in runs if x['domain']==d and x['method']==m]
        agg.append({'domain':d,'method':m,'mean':np.mean(q),'std':np.std(q,ddof=1)})
with (RES/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=agg[0].keys()); w.writeheader(); w.writerows(agg)

x=np.arange(len(domains)); width=.34; fig,ax=plt.subplots(figsize=(7.0,3.4))
for j,m in enumerate(methods):
    q=[next(a for a in agg if a['domain']==d and a['method']==m) for d in domains]
    ax.bar(x+(j-.5)*width,[a['mean']*100 for a in q],width,yerr=[a['std']*100 for a in q],capsize=3,label=names[m],color=colors[m],edgecolor='black',linewidth=.5)
ax.set_xticks(x,[d.title() for d in domains]); ax.set_ylabel('Entity-level micro-F1 (%)'); ax.set_title('CrossNER results (mean ± SD, three seeds)'); ax.grid(axis='y',color='#dddddd',linewidth=.5); ax.legend(frameon=False); fig.tight_layout()
fig.savefig(OUT/'crossner_f1.pdf',bbox_inches='tight'); fig.savefig(OUT/'crossner_f1.png',bbox_inches='tight'); plt.close(fig)

fig,ax=plt.subplots(figsize=(7.0,3.4))
for m,ls,mk in [('bigru','--','s'),('cross_residual','-','o')]:
    q=[h for h in hist if h['domain']=='science' and h['method']==m and h['seed']=='13']
    ax.plot([int(h['epoch']) for h in q],[float(h['loss']) for h in q],label=names[m],color=colors[m],linestyle=ls,marker=mk,markersize=4,linewidth=1.8)
ax.set_xlabel('Epoch'); ax.set_ylabel('Training loss'); ax.set_title('Optimization behavior on CrossNER-Science'); ax.grid(color='#dddddd',linewidth=.5); ax.legend(frameon=False); fig.tight_layout()
fig.savefig(OUT/'crossner_loss.pdf',bbox_inches='tight'); fig.savefig(OUT/'crossner_loss.png',bbox_inches='tight'); plt.close(fig)

counts=[]
for d in domains:
    base=ROOT/'tools/external-baselines/oneke/data/datasets/CrossNER'/d
    counts.append((len(json.loads((base/'train.json').read_text())),len(json.loads((base/'test.json').read_text()))))
fig,ax=plt.subplots(figsize=(7.0,3.4)); ax.bar(x-.17,[a for a,b in counts],.34,label='Train',color='#56B4E9'); ax.bar(x+.17,[b for a,b in counts],.34,label='Test',color='#E69F00'); ax.set_xticks(x,[d.title() for d in domains]); ax.set_ylabel('Sentences'); ax.set_title('CrossNER domain distribution'); ax.legend(frameon=False); ax.grid(axis='y',color='#dddddd',linewidth=.5); fig.tight_layout(); fig.savefig(OUT/'crossner_distribution.pdf',bbox_inches='tight'); fig.savefig(OUT/'crossner_distribution.png',bbox_inches='tight')
