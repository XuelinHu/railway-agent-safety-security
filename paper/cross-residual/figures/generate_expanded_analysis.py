#!/usr/bin/env python3
"""Generate additional figures from official CrossNER data and result summaries."""
import csv
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "tmp/crossner-official/ner_data"
RESULT = ROOT / "paper/cross-residual/results/official_crossner"
OUT = ROOT / "paper/cross-residual/figures"
DOMAINS = ["ai", "literature", "music", "politics", "science"]
plt.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman","DejaVu Serif"], "axes.labelsize":10, "xtick.labelsize":8, "ytick.labelsize":8, "legend.fontsize":8, "figure.facecolor":"white", "savefig.dpi":400})

def read_bio(path):
    rows, tokens, tags = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        if line.strip():
            tok, tag = line.rsplit(maxsplit=1); tokens.append(tok); tags.append(tag)
        elif tokens:
            ents=[]; start=None; typ=None
            for i, tag in enumerate(tags + ["O"]):
                p, t = tag.split("-",1) if "-" in tag else ("O",None)
                if start is not None and (p != "I" or t != typ): ents.append((start,i,typ)); start=None; typ=None
                if p == "B" or (p == "I" and start is None): start,typ=i,t
            rows.append((tokens[:],ents)); tokens=[]; tags=[]
    return rows

profile=[]
for domain in DOMAINS:
    train=read_bio(DATA/domain/"train.txt")
    all_rows=sum((read_bio(DATA/domain/(split+".txt")) for split in ["train","dev","test"]),[])
    ents=[e for _,es in all_rows for e in es]; counts=Counter(e[2] for _,es in train for e in es)
    lengths=[e[1]-e[0] for e in ents]; density=[len(es)/len(ts) for ts,es in all_rows if ts]
    profile.append((domain,len(counts),np.median(lengths),np.median(density),sorted(counts.values(),reverse=True)))

fig,axes=plt.subplots(1,2,figsize=(7.0,2.65)); x=np.arange(len(DOMAINS)); width=.34
axes[0].bar(x-width/2,[p[1] for p in profile],width,label="Entity types",color="#0072B2",edgecolor="black",linewidth=.35)
axes[0].bar(x+width/2,[p[2] for p in profile],width,label="Median span length",color="#E69F00",edgecolor="black",linewidth=.35)
axes[0].set_xticks(x,[d.title() for d in DOMAINS]); axes[0].set_ylabel("Count / tokens"); axes[0].set_title("Taxonomy and span profile"); axes[0].legend(frameon=False); axes[0].grid(axis="y",color="#ddd",linewidth=.4)
axes[1].bar(x,[p[3] for p in profile],color="#009E73",edgecolor="black",linewidth=.35); axes[1].set_xticks(x,[d.title() for d in DOMAINS]); axes[1].set_ylabel("Entities per token"); axes[1].set_title("Entity density"); axes[1].grid(axis="y",color="#ddd",linewidth=.4)
fig.tight_layout(); fig.savefig(OUT/"dataset_profile_analysis.pdf",bbox_inches="tight"); fig.savefig(OUT/"dataset_profile_analysis.png",bbox_inches="tight"); plt.close(fig)

fig,axes=plt.subplots(1,5,figsize=(7.1,2.15),sharey=True)
for ax,(domain,_,_,_,vals) in zip(axes,profile):
    ax.plot(np.arange(1,len(vals)+1),vals,"o-",color="#0072B2",markersize=2.5,linewidth=1); ax.set_title(domain.title(),fontsize=9); ax.set_yscale("log"); ax.grid(color="#ddd",linewidth=.35); ax.set_xlabel("Rank",fontsize=8)
axes[0].set_ylabel("Train mentions (log)"); fig.tight_layout(); fig.savefig(OUT/"label_frequency_longtail.pdf",bbox_inches="tight"); fig.savefig(OUT/"label_frequency_longtail.png",bbox_inches="tight"); plt.close(fig)

runs=list(csv.DictReader((RESULT/"repeated_macro.csv").open())); fig,ax=plt.subplots(figsize=(4.2,2.7)); adapt=[float(r["adapted_f1"]) for r in runs]; resid=[float(r["residual_f1"]) for r in runs]
for i,(a,f) in enumerate(zip(adapt,resid),1): ax.plot([0,1],[a,f],marker="o",linewidth=1.4,label=f"Run {i}")
ax.set_xticks([0,1],["Domain-adapted","Score-residual"]); ax.set_ylabel("Macro F1 (%)"); ax.set_title("Matched repeated-run comparison"); ax.grid(axis="y",color="#ddd",linewidth=.4); ax.legend(frameon=False,ncol=3,loc="lower right"); fig.tight_layout(); fig.savefig(OUT/"paired_run_gains.pdf",bbox_inches="tight"); fig.savefig(OUT/"paired_run_gains.png",bbox_inches="tight"); plt.close(fig)

raw=list(csv.DictReader((RESULT/"runs.csv").open())); gates={d:Counter(float(r["alpha"]) for r in raw if r["method"]=="score-residual" and r["domain"]==d) for d in DOMAINS}; fig,ax=plt.subplots(figsize=(5.0,2.8)); alphas=[.75,1.,1.25,1.5]; x=np.arange(len(DOMAINS)); width=.18
for j,alpha in enumerate(alphas): ax.bar(x+(j-1.5)*width,[gates[d][alpha] for d in DOMAINS],width,label=f"$\\alpha={alpha:g}$",edgecolor="black",linewidth=.3)
ax.set_xticks(x,[d.title() for d in DOMAINS]); ax.set_ylabel("Runs selected"); ax.set_ylim(0,3.5); ax.set_title("Development-selected gate frequencies"); ax.legend(frameon=False,ncol=4); ax.grid(axis="y",color="#ddd",linewidth=.4); fig.tight_layout(); fig.savefig(OUT/"gate_selection_frequency.pdf",bbox_inches="tight"); fig.savefig(OUT/"gate_selection_frequency.png",bbox_inches="tight"); plt.close(fig)
print("Generated expanded analysis figures in",OUT)
