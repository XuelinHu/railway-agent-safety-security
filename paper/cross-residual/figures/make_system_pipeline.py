import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':9,'figure.facecolor':'white','axes.facecolor':'white','savefig.dpi':400})
fig,ax=plt.subplots(figsize=(7.0,1.75)); ax.set_xlim(0,15); ax.set_ylim(0,3); ax.axis('off')
items=[(.2,'Official\nBIO files'),(3.1,'Train / dev / test\nprotocol'),(6.0,'Parallel\nspan scores'),(8.9,'Dev-only\ngate selection'),(11.8,'Repeated test\nevaluation')]
cols=['#e6f2f8','#fff2cc','#d9ead3','#fce5cd','#eadcf8']
for (x,t),c in zip(items,cols):
    ax.add_patch(FancyBboxPatch((x,.9),2.1,1.1,boxstyle='round,pad=.04,rounding_size=.08',facecolor=c,edgecolor='#333333',linewidth=1)); ax.text(x+1.05,1.45,t,ha='center',va='center',fontsize=8)
for i in range(4):
    ax.add_patch(FancyArrowPatch((items[i][0]+2.1,1.45),(items[i+1][0],1.45),arrowstyle='-|>',mutation_scale=12,color='#333333',linewidth=1.1))
ax.text(7.5,.28,'Outputs: strict precision, recall, F1, runtime, aggregate table, and figures',ha='center',fontsize=8)
fig.tight_layout(pad=.2); fig.savefig('system_pipeline.pdf',bbox_inches='tight'); fig.savefig('system_pipeline.png',bbox_inches='tight')
