import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'axes.titlesize':12,'axes.labelsize':11,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'figure.facecolor':'white','axes.facecolor':'white','savefig.dpi':400})
paths={'Zero-shot GLiNER':'gliner_crossner','Domain-adapted':'gliner_finetuned','Score-residual':'score_residual'}; domains=['ai','literature','music','politics','science']; colors=['#56B4E9','#E69F00','#009E73']; data={}
for name,d in paths.items(): data[name]={r['domain']:r for r in csv.DictReader((ROOT/f'paper/cross-residual/results/{d}/runs.csv').open())}
x=np.arange(5); w=.25; fig,ax=plt.subplots(figsize=(7.1,3.5))
for j,(name,c) in enumerate(zip(paths,colors)):
    vals=[float(data[name][d]['f1'])*100 for d in domains]; ax.bar(x+(j-1)*w,vals,w,label=name,color=c,edgecolor='black',linewidth=.45)
ax.set_xticks(x,[d.title() for d in domains]); ax.set_ylabel('Exact-span entity F1 (%)'); ax.set_ylim(0,90); ax.set_title('Cross-domain NER performance'); ax.grid(axis='y',color='#dddddd',linewidth=.5); ax.legend(frameon=False,ncol=3); fig.tight_layout(); fig.savefig(OUT/'high_f1_results.pdf',bbox_inches='tight'); fig.savefig(OUT/'high_f1_results.png',bbox_inches='tight'); plt.close(fig)
fig,ax=plt.subplots(figsize=(6.2,3.5)); final=data['Score-residual']; p=[float(final[d]['precision'])*100 for d in domains]; r=[float(final[d]['recall'])*100 for d in domains]
ax.plot(domains,p,color='#0072B2',marker='o',linewidth=1.8,label='Precision'); ax.plot(domains,r,color='#D55E00',marker='s',linestyle='--',linewidth=1.8,label='Recall'); ax.set_ylabel('Score (%)'); ax.set_ylim(55,90); ax.set_title('Score-residual precision and recall'); ax.grid(color='#dddddd',linewidth=.5); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/'precision_recall.pdf',bbox_inches='tight'); fig.savefig(OUT/'precision_recall.png',bbox_inches='tight')

# Actual optimizer logs exported by the five domain-adaptation runs.
fig,ax=plt.subplots(figsize=(6.2,3.5)); styles=['o','s','^','D','v']
loss_rows=list(csv.DictReader((ROOT/'paper/cross-residual/results/gliner_finetuned/loss_history.csv').open()))
for domain,color,marker in zip(domains,['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00'],styles):
    points=[item for item in loss_rows if item['domain']==domain]
    ax.plot([int(item['step']) for item in points],[float(item['loss']) for item in points],
            marker=marker,markersize=4,linewidth=1.6,label=domain.title(),color=color)
ax.set_xlabel('Optimization step'); ax.set_ylabel('Training loss')
ax.set_title('Domain-adaptation convergence'); ax.grid(color='#dddddd',linewidth=.5)
ax.legend(frameon=False,ncol=3); fig.tight_layout()
fig.savefig(OUT/'training_loss_diagnostic.pdf',bbox_inches='tight')
fig.savefig(OUT/'training_loss_diagnostic.png',bbox_inches='tight')
