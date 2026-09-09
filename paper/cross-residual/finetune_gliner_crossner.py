#!/usr/bin/env python3
import argparse,csv,json,random,time
from pathlib import Path
from gliner import GLiNER
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'tools/external-baselines/oneke/data/datasets/CrossNER'; OUT=ROOT/'paper/cross-residual/results/gliner_finetuned'
def sample(row):
    words=row['sentence'].split(); low=[w.lower() for w in words]; ner=[]
    for e in row.get('entity_list',[]):
        q=[w.lower() for w in e['name'].split()]
        for i in range(len(words)-len(q)+1):
            if low[i:i+len(q)]==q: ner.append((i,i+len(q)-1,e['type'])); break
    return {'tokenized_text':words,'ner':ner}
def gold(row):
    s=row['sentence']; out=set()
    for e in row.get('entity_list',[]):
        i=s.lower().find(e['name'].lower())
        if i>=0: out.add((i,i+len(e['name']),e['type']))
    return out
def score(model,rows,labels,thr=.4):
    tp=fp=fn=0
    for row in rows:
        g=gold(row); p={(e['start'],e['end'],e['label']) for e in model.predict_entities(row['sentence'],labels,threshold=thr)}
        tp+=len(g&p); fp+=len(p-g); fn+=len(g-p)
    pr=tp/(tp+fp) if tp+fp else 0; re=tp/(tp+fn) if tp+fn else 0; f=2*pr*re/(pr+re) if pr+re else 0
    return pr,re,f
ap=argparse.ArgumentParser(); ap.add_argument('--domains',nargs='+',default=['ai','literature','music','politics','science']); ap.add_argument('--steps',type=int,default=300); a=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True); results=[]; loss_rows=[]; start=time.time()
for domain in a.domains:
    random.seed(13); tr=json.loads((DATA/domain/'train.json').read_text()); te=json.loads((DATA/domain/'test.json').read_text()); random.shuffle(tr); cut=int(.8*len(tr)); train=[sample(x) for x in tr[:cut]]; val=[sample(x) for x in tr[cut:]]; labels=json.loads((DATA/domain/'class.json').read_text())
    model=GLiNER.from_pretrained('urchade/gliner_small-v2.1',local_files_only=True)
    args=model.create_training_args(output_dir=str(OUT/domain),learning_rate=1e-5,others_lr=5e-5,max_steps=a.steps,per_device_train_batch_size=8,per_device_eval_batch_size=8,logging_steps=50,save_steps=a.steps,save_total_limit=1,bf16=True,report_to='none')
    model.train_model(train,val,training_args=args); model.to('cuda')
    state=json.loads((OUT/domain/f'checkpoint-{a.steps}'/'trainer_state.json').read_text())
    loss_rows.extend({'domain':domain,'step':item['step'],'loss':item['loss']} for item in state['log_history'] if 'loss' in item)
    trials=[(t,score(model,tr[cut:],labels,t)) for t in (.25,.35,.45,.55)]; thr=max(trials,key=lambda z:z[1][2])[0]; p,r,f=score(model,te,labels,thr); rec={'domain':domain,'threshold':thr,'precision':p,'recall':r,'f1':f}; results.append(rec); print(json.dumps(rec),flush=True)
with (OUT/'runs.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=results[0]); w.writeheader(); w.writerows(results)
with (OUT/'loss_history.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['domain','step','loss']); w.writeheader(); w.writerows(loss_rows)
(OUT/'status.json').write_text(json.dumps({'status':'complete','seconds':time.time()-start},indent=2))
