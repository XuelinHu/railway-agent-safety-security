#!/usr/bin/env python3
"""Publication-sized two-benchmark view of the existing PGE architecture."""
try:
    from .diagram_utils import Diagram
except ImportError:  # direct execution: python scripts/build_ade_conll04_diagram.py
    from diagram_utils import Diagram


def main():
    d=Diagram('pge-two-benchmark','ADE and CoNLL04 PGE architecture',1000,1120)
    palette={'source':('#fcecef','#b84c6a'),'model':('#f1ecfa','#7257a6'),
             'gate':('#eaf6ef','#2f7d5b'),'control':('#eaf2fb','#2f6f9f')}
    def box(key,text,x,y,w,h,kind='gate',dashed=False):
        fill,stroke=palette[kind]
        d.vertex(key,text,x,y,w,h,fill=fill,stroke=stroke,font_color='#17324d',
                 dashed=dashed,extra='fontSize=21;')
    def edge(key,a,b,**kwargs):d.edge(key,a,b,**kwargs)
    box('source','<b>ADE or CoNLL04 sentence</b><br>Published tokens to source character offsets<br>No annotation-guided rescue windows',20,20,460,110,'source')
    box('kg','<b>Dataset-specific training KG</b><br>Typed mentions, directed edges, provenance<br>Train-derived labels and endpoint constraints',520,20,460,110,'source')
    box('context','<b>Provenance-isolated context construction</b><br>Exact matching + frozen BGE-M3 relation-pattern retrieval<br>Clear graph hints for exact train/evaluation text overlaps',200,175,780,110,'source')
    edge('s_context','source','context');edge('kg_context','kg','context')
    for name,x,detail in [('SOE',20,'Source only; no KG'),('EAE',350,'Exact concept hints'),('HRGE',680,'Anchors / edges / patterns')]:
        box(name,f'<b>{name}: QLoRA adapter</b><br>{detail}<br>Frozen Qwen3-4B<br>Structured candidate JSON',x,335,300,135,'model',name=='SOE')
        box(name+'_spans','<b>Span reconstruction</b><br>Copied text to exact offsets<br>Resolve relation endpoints',x,510,300,100,'model')
        edge(name+'_expand',name,name+'_spans')
    edge('source_soe','source','SOE',points=[(90,145),(90,315)],extra='exitX=0.15;exitY=1;entryX=0.2;entryY=0;')
    edge('context_eae','context','EAE');edge('context_hrge','context','HRGE')
    box('baseline','<b>SOE comparison</b><br>Not used in PGE fusion',20,655,300,85,'control',True)
    edge('soe_eval','SOE_spans','baseline',dashed=True)
    box('entity','<b>Entity acceptance: ANY</b><br>EAE agreement OR anchor<br>OR verified endpoint<br>Deduplicate accepted entities',350,655,300,145)
    box('relation','<b>Relation gate: ALL</b><br>References; type signatures;<br>explicit status; evidence<br>Output: EVGE relations',680,655,300,145)
    edge('agreement','EAE_spans','entity');edge('relation_input','HRGE_spans','relation')
    edge('entity_input','HRGE_spans','entity',points=[(665,630),(610,630)],extra='exitX=0;exitY=1;entryX=0.87;entryY=0;')
    edge('verified_endpoint','relation','entity',extra='exitX=0;exitY=0.7;entryX=1;entryY=0.7;')
    box('cfe','<b>CFE validation ablation</b><br>Accepted entities + raw HRGE relations<br>Retain only surviving endpoints',20,850,460,105,'control',True)
    box('pge','<b>PGE evidence-linked assertion graph</b><br>Accepted entities + EVGE relations<br>Retain only surviving endpoints',520,850,460,105)
    edge('entity_cfe','entity','cfe',dashed=True,points=[(420,825),(240,825)],extra='exitX=0.25;exitY=1;entryX=0.5;entryY=0;')
    edge('entity_pge','entity','pge',points=[(560,825),(620,825)],extra='exitX=0.7;exitY=1;entryX=0.22;entryY=0;')
    edge('relation_pge','relation','pge',extra='exitX=0.5;exitY=1;entryX=0.7;entryY=0;')
    box('audit','<b>Audit records accompany predictions</b><br>Source spans, relation direction, provenance and acceptance/rejection signals<br>Rules establish structural admissibility, not semantic truth or clinical causality',20,1000,960,100)
    d.write('ade_conll04/03-pge-architecture.drawio')


if __name__=='__main__':main()
