import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "figure.facecolor": "white", "axes.facecolor": "white"})
fig, ax = plt.subplots(figsize=(7.0, 2.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis("off")
def box(x, y, w, h, txt, fc):
    p = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.08",
                       linewidth=1.0, edgecolor="#333333", facecolor=fc)
    ax.add_patch(p); ax.text(x+w/2,y+h/2,txt,ha="center",va="center",wrap=True)
box(.3,1.8,2.0,1.0,"Trainable token\nembeddings","#e6f2f8")
box(3.0,3.1,2.3,1.0,"BiGRU contextual\nbranch","#d9ead3")
box(3.0,.5,2.3,1.0,"CNN local-pattern\nbranch","#fce5cd")
box(6.2,1.8,2.2,1.0,"Cross-residual\nfusion","#cfe2f3")
box(9.2,1.8,2.0,1.0,"Token classifier\n+ softmax","#eadcf8")
box(12.0,1.8,1.6,1.0,"BIO entity\nlabels","#fff2cc")
def arr(a,b,style='-'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,linewidth=1.1,
                                 color="#333333",linestyle=style,connectionstyle="arc3,rad=0"))
arr((2.3,2.3),(3.0,3.55)); arr((2.3,2.3),(3.0,1.0)); arr((5.3,3.6),(6.2,2.6)); arr((5.3,1.0),(6.2,2.0)); arr((8.4,2.3),(9.2,2.3)); arr((11.2,2.3),(12.0,2.3))
arr((4.1,3.1),(7.2,2.8),'--'); ax.text(5.45,3.35,'identity path',ha='center',va='bottom',fontsize=8)
ax.text(7.2,.12,'The gate controls the CNN correction while the BiGRU identity path remains available.',ha='center',fontsize=8)
fig.tight_layout(pad=.2)
fig.savefig('cross_residual_architecture.pdf',bbox_inches='tight')
fig.savefig('cross_residual_architecture.png',dpi=400,bbox_inches='tight')
