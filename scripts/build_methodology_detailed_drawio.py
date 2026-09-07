#!/usr/bin/env python3
"""Build an editable Draw.io version of methodology_detailed_draft.svg."""
from pathlib import Path
import xml.etree.ElementTree as ET
from build_paper_diagrams import Diagram

ROOT = Path(__file__).resolve().parents[1]

BLUE, PURPLE, GREEN, RED, GOLD, INK = '#2f6f9f', '#7257a6', '#2f7d5b', '#b84c6a', '#c9892f', '#17324d'

def main():
    d = Diagram('methodology-detailed', 'Detailed Qwen3-4B methodology', 1900, 1100)
    # Render math using Draw.io/MathJax rather than rasterized text.
    def v(cid, value, x, y, w, h, fill, stroke, fs=14, bold=False, dashed=False, shape=None, extra=''):
        d.vertex(cid, value, x, y, w, h, fill=fill, stroke=stroke, font_color=INK,
                 bold=bold, dashed=dashed, shape=shape, extra=f'fontSize={fs};{extra}')
    def e(cid, a, b, value='', color=INK, dashed=False, extra=''):
        d.edge(cid, a, b, value, color=color, dashed=dashed, width=2, extra=extra)

    v('title','<b>Detailed Methodology: Source-Grounded Qwen3-4B + QLoRA Extraction</b>',40,20,1820,55,'#ffffff','#ffffff',22,True)
    v('subtitle','All examples are copied from released ADE/CoNLL04 benchmark records.',430,75,1040,30,'#ffffff','#ffffff',12)

    v('input','<b>REAL ADE TEST SENTENCE</b><br>We report a case of fulminant hepatic failure<br>associated with didanosine ...<br><br>Tokens: 0 We · 1 report · 2 a · 3 case · 4 of<br>5 fulminant · 6 hepatic · 7 failure · 8 associated · 9 with · 10 didanosine',40,130,380,175,'#fcecef',RED,12,True)
    v('anchor','<b>SOURCE ANCHORING</b><br>We report a case of<br>[fulminant hepatic failure] associated with [didanosine]<br><br>Adverse-Effect span = [20,45)<br>Drug span = [62,72)<br>Copied text; deterministic offset reconstruction',40,335,380,180,'#fff7e6',GOLD,12,True)
    e('input_anchor','input','anchor',color=INK)
    v('kg','<b>DATASET-SPECIFIC TRAINING KG (ADE)</b><br><font color="#526777">Nodes and directed edges are derived only from training annotations.</font>',40,550,380,300,'#fcecef',RED,12,True)
    # KG nodes and directed edges.
    v('n1','heart<br>block<br><font color="#b84c6a">Adverse-Effect</font>',80,630,105,70,'#ffffff',RED,10,True,shape='ellipse')
    v('n2','CHB<br><font color="#b84c6a">Adverse-Effect</font>',255,630,105,70,'#ffffff',RED,10,True,shape='ellipse')
    v('n3','disopyramide<br>phosphate<br><font color="#2f6f9f">Drug</font>',80,735,105,75,'#ffffff',BLUE,9,True,shape='ellipse')
    v('n4','Norpace<br><font color="#2f6f9f">Drug</font>',255,735,105,75,'#ffffff',BLUE,10,True,shape='ellipse')
    e('kg_e1','n1','n3','Adverse-Effect',color=RED); e('kg_e2','n1','n4','Adverse-Effect',color=RED)
    e('kg_e3','n2','n3','Adverse-Effect',color=RED); e('kg_e4','n2','n4','Adverse-Effect',color=RED)
    v('kg_ev','<font color="#526777">Evidence: “developed complete heart block (CHB) following administration of<br>disopyramide phosphate (Norpace)”</font>',58,815,345,28,'#fcecef',RED,9)
    e('anchor_kg','anchor','kg',color=GOLD,dashed=True)

    v('model','<b>FROZEN QWEN3-4B TRANSFORMER BACKBONE</b><br><br>Embedding: \\(X=E[\\mathrm{token IDs}],\\;X\\in\\mathbb{R}^{n\\times d}\\)<br>RMSNorm \\rightarrow Q/K/V projections<br>\\(Q=XW_q+XA_qB_q,\\;K=XW_k+XA_kB_k,\\;V=XW_v+XA_vB_v\\)<br><br>Self-attention: \\(S=\\mathrm{softmax}(QK^{\\top}/\\sqrt{d_k}),\\;H=SV\\)<br>Output projection: \\(Y=HW_o+HA_oB_o+\\mathrm{residual}\\)<br><br>Gated FFN (SwiGLU): \\(U=XW_u+XA_uB_u,\\;G=XW_g+XA_gB_g\\)<br>\\(Z=\\mathrm{SiLU}(G)\\odot U\\)<br>Down projection: \\(O=ZW_d+ZA_dB_d+\\mathrm{residual}\\)<br><br>Next Transformer block (repeated \\(L\\) times)',x=460,y=130,w=760,h=405,fill='#eaf2fb',stroke=BLUE,fs=12,bold=True,extra='math=1;')
    e('anchor_model','anchor','model',color=INK)
    e('kg_model','kg','model',color=RED,dashed=True)
    v('lora','<b>QLoRA update for every adapted matrix</b><br>\\(W\\prime=W+(\\alpha/r)BA\\)<br><font color="#526777">W is frozen; only low-rank A and B receive gradients</font><br>Attention: \\(W_q,W_k,W_v,W_o\\) &nbsp;&nbsp; FFN: \\(W_{gate},W_{up},W_{down}\\)',460,555,760,92,'#fff7e6',GOLD,12,True,extra='math=1;')
    e('model_lora','model','lora',color=GOLD,dashed=True)

    branches=[('soe','<b>SOE — Source-Only Extraction</b><br>source + ontology only<br><font face="Courier New">{entities:[...], relations:[...]}</font>'),
              ('eae','<b>EAE — Exact-Anchor Extraction</b><br>source + exact concept hints<br><font face="Courier New">{entity:{text:"didanosine", type:"Drug"}}</font>'),
              ('hrge','<b>HRGE — High-Recall Graph Extraction</b><br>source + anchors + edge priors<br><font face="Courier New">{relation:{source:"effect", target:"drug", evidence:"..."}}</font>')]
    for i,(cid,text) in enumerate(branches):
        x=470+i*250
        v(cid,text,x,680,225,100,'#f1ecfa',PURPLE,10,False,True)
        e(cid+'_down','model',cid,color=PURPLE,dashed=True)
    v('branch_note','three learned QLoRA adapters',730,790,250,28,'#ffffff','#ffffff',10)
    v('gates','<b>DETERMINISTIC EVIDENCE GATES (NOT NEURAL LAYERS)</b><br>Entity: EAE agreement OR valid HRGE anchor OR verified endpoint<br>Relation: valid references AND legal type signature AND explicit status AND common source evidence<br><font color="#526777">Rule-based verification; not a graph neural network</font>',460,835,760,125,'#eaf6ef',GREEN,11,True)
    for cid in ('soe','eae','hrge'): e(cid+'_gate',cid,'gates',color=PURPLE)

    v('out','<b>OUTPUT KNOWLEDGE GRAPH (GRAPH DATA, NOT A GNN)</b><br><br>ADE:<br>[fulminant hepatic failure]<font color="#b84c6a">Adverse-Effect</font>  ── <b>Adverse-Effect</b> ──▶  [didanosine]<font color="#2f6f9f">Drug</font><br><br>CoNLL04:<br>[John Wilkes Booth]<font color="#2f6f9f">Peop</font>  ── <b>Kill</b> ──▶  [President Abraham Lincoln]<font color="#2f6f9f">Peop</font><br><br><font color="#526777">Nodes: surface text, type, span · Edges: direction, evidence, provenance</font>',1260,130,600,300,'#eaf6ef',GREEN,12,True)
    e('gates_out','gates','out',color=GREEN)
    v('audit','<b>AUDIT RECORD (AUTOMATIC)</b><br>Evidence quote · character offsets · entity types<br>Relation direction · provenance · accept/reject signal · rejection reason<br><br><font color="#526777">Optional human spot-check; no manual review of every prediction required.</font>',1260,500,600,155,'#fff7e6',GOLD,12,True)
    e('out_audit','out','audit',color=GOLD,dashed=True)
    v('pge','<b>PGE OUTPUT</b><br>accepted spans + typed directed relations<br>with provenance linked back to source text',1260,740,600,115,'#eaf6ef',GREEN,13,True)
    e('gates_pge','gates','pge',color=GREEN)
    v('legend','Blue = frozen Transformer computation &nbsp;&nbsp; Purple = trainable LoRA adapters &nbsp;&nbsp; Green = verified graph output &nbsp;&nbsp; Gold = deterministic audit / offsets',500,1010,900,28,'#ffffff','#ffffff',10)
    d.write('ade_conll04/methodology_detailed_draft.drawio')
    path = ROOT/'paper/figures/ade_conll04/methodology_detailed_draft.drawio'
    text = path.read_text(encoding='utf-8').replace('math="0"','math="1"')
    path.write_text(text,encoding='utf-8')

if __name__ == '__main__': main()
