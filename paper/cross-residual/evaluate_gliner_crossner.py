#!/usr/bin/env python3
import csv,json,random,time
from pathlib import Path
from gliner import GLiNER
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'tools/external-baselines/oneke/data/datasets/CrossNER'; OUT=ROOT/'paper/cross-residual/results/gliner_crossner'; OUT.mkdir(parents=True,exist_ok=True)
def gold_spans(row):
    s=row['sentence']; out=set()
    for e in row.get('entity_list',[]):
        start=s.lower().find(e['name'].lower())
        if start>=0: out.add((start,start+len(e['name']),e['type']))
    return out
def score(model,rows,labels,thr):
    tp=fp=fn=0
    for row in rows:
        g=gold_spans(row); p={(e['start'],e['end'],e['label']) for e in model.predict_entities(row['sentence'],labels,threshold=thr)}
        tp+=len(g&p); fp+=len(p-g); fn+=len(g-p)
    pr=tp/(tp+fp) if tp+fp else 0; re=tp/(tp+fn) if tp+fn else 0; f=2*pr*re/(pr+re) if pr+re else 0
    return pr,re,f
model=GLiNER.from_pretrained('urchade/gliner_small-v2.1',local_files_only=True).to('cuda'); result=[]; start=time.time()
for d in ['ai','literature','music','politics','science']:
    labels=json.loads((DATA/d/'class.json').read_text()); train=json.loads((DATA/d/'train.json').read_text()); test=json.loads((DATA/d/'test.json').read_text()); random.Random(13).shuffle(train); val=train[int(.8*len(train)):]
    trials=[(t,score(model,val,labels,t)) for t in (.2,.3,.4,.5,.6)]; threshold=max(trials,key=lambda x:x[1][2])[0]
    p,r,f=score(model,test,labels,threshold); rec={'domain':d,'threshold':threshold,'precision':p,'recall':r,'f1':f}; result.append(rec); print(json.dumps(rec),flush=True)
with (OUT/'runs.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=result[0]); w.writeheader(); w.writerows(result)
(OUT/'status.json').write_text(json.dumps({'status':'complete','seconds':time.time()-start},indent=2))
