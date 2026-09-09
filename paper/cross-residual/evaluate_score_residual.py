#!/usr/bin/env python3
"""Fuse frozen and domain-adapted GLiNER span scores through a residual path."""
import csv,json,random,time
from pathlib import Path
from gliner import GLiNER
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'tools/external-baselines/oneke/data/datasets/CrossNER'; BASE=ROOT/'paper/cross-residual/results/gliner_finetuned'; OUT=ROOT/'paper/cross-residual/results/score_residual'; OUT.mkdir(parents=True,exist_ok=True)
def gold(row):
    out=set(); s=row['sentence']
    for e in row.get('entity_list',[]):
        i=s.lower().find(e['name'].lower())
        if i>=0: out.add((i,i+len(e['name']),e['type']))
    return out
def cache(model,rows,labels):
    ans=[]
    for r in rows:
        ans.append({(e['start'],e['end'],e['label']):e['score'] for e in model.predict_entities(r['sentence'],labels,threshold=.03)})
    return ans
def score(rows,a,b,alpha,thr):
    tp=fp=fn=0
    for row,x,y in zip(rows,a,b):
        keys=set(x)|set(y); pred={k for k in keys if x.get(k,0)+alpha*(y.get(k,0)-x.get(k,0))>=thr}; g=gold(row)
        tp+=len(pred&g); fp+=len(pred-g); fn+=len(g-pred)
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; f=2*p*r/(p+r) if p+r else 0
    return p,r,f
base=GLiNER.from_pretrained('urchade/gliner_small-v2.1',local_files_only=True).to('cuda'); rows=[]; start=time.time()
for d in ['ai','literature','music','politics','science']:
    labels=json.loads((DATA/d/'class.json').read_text()); tr=json.loads((DATA/d/'train.json').read_text()); te=json.loads((DATA/d/'test.json').read_text()); random.Random(13).shuffle(tr); val=tr[int(.8*len(tr)):]
    adapted=GLiNER.from_pretrained(str(BASE/d/'checkpoint-300'),local_files_only=True).to('cuda')
    bv,av=cache(base,val,labels),cache(adapted,val,labels); bt,at=cache(base,te,labels),cache(adapted,te,labels)
    configs=[(a,t,score(val,bv,av,a,t)) for a in (.75,1.0,1.25,1.5) for t in (.25,.35,.45,.55,.65)]
    alpha,thr,_=max(configs,key=lambda z:z[2][2]); p,r,f=score(te,bt,at,alpha,thr); rec={'domain':d,'alpha':alpha,'threshold':thr,'precision':p,'recall':r,'f1':f}; rows.append(rec); print(json.dumps(rec),flush=True)
    del adapted
with (OUT/'runs.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
(OUT/'status.json').write_text(json.dumps({'status':'complete','seconds':time.time()-start},indent=2))
