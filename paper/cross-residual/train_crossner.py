#!/usr/bin/env python3
"""Compact non-LLM CrossNER experiment: BiGRU baseline vs cross-residual BiGRU-CNN."""
from __future__ import annotations
import argparse, csv, json, random, re, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tools/external-baselines/oneke/data/datasets/CrossNER"
OUT = ROOT / "paper/cross-residual/results/crossner"

def toks(s): return s.split()
def find_subseq(xs, ys):
    n=len(ys)
    return [i for i in range(len(xs)-n+1) if [x.lower() for x in xs[i:i+n]] == [y.lower() for y in ys]]
def encode_rows(rows, labels=None):
    types=sorted({e['type'] for r in rows for e in r.get('entity_list',[])}) if labels is None else labels
    labs=['O']+[p+'-'+t for t in types for p in ('B','I')]
    out=[]
    for r in rows:
        ts=toks(r['sentence']); yy=['O']*len(ts)
        for e in sorted(r.get('entity_list',[]), key=lambda x:len(toks(x['name'])), reverse=True):
            et=toks(e['name']); hits=find_subseq(ts,et)
            if hits:
                i=hits[0]; yy[i]='B-'+e['type']
                for j in range(1,len(et)): yy[i+j]='I-'+e['type']
        out.append((ts,yy))
    return out,types,labs

class SeqDS(Dataset):
    def __init__(self, rows, vocab, lab2id, maxlen=128): self.rows=rows; self.v=vocab; self.l=lab2id; self.m=maxlen
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        x,y=self.rows[i]; x=x[:self.m]; y=y[:self.m]
        return torch.tensor([self.v.get(t.lower(),1) for t in x]),torch.tensor([self.l[z] for z in y])
def collate(batch):
    m=max(len(x) for x,_ in batch); xx=[]; yy=[]; mask=[]
    for x,y in batch:
        p=m-len(x); xx.append(torch.cat([x,torch.zeros(p,dtype=torch.long)])); yy.append(torch.cat([y,torch.full((p,),-100)])); mask.append(torch.cat([torch.ones(len(x)),torch.zeros(p)]))
    return torch.stack(xx),torch.stack(yy),torch.stack(mask)

class Tagger(nn.Module):
    def __init__(self,vocab,nlab,residual=False):
        super().__init__(); self.residual=residual; self.emb=nn.Embedding(vocab,128,padding_idx=0)
        self.gru=nn.GRU(128,96,batch_first=True,bidirectional=True)
        self.proj=nn.Linear(192,128); self.conv=nn.Conv1d(128,128,3,padding=1)
        self.gate=nn.Linear(256,128); self.head=nn.Linear(128,nlab); self.drop=nn.Dropout(.2)
    def forward(self,x):
        e=self.emb(x); h=self.proj(self.gru(e)[0])
        if self.residual:
            c=torch.relu(self.conv(e.transpose(1,2)).transpose(1,2)); g=torch.sigmoid(self.gate(torch.cat([h,c],-1))); h=h+g*c
        return self.head(self.drop(torch.relu(h)))

def spans(seq):
    ans=set(); cur=None
    for i,z in enumerate(seq+['O']):
        if z.startswith('B-') or (z.startswith('I-') and (cur is None or cur[0]!=z[2:])):
            if cur: ans.add((cur[1],i,cur[0]))
            cur=(z[2:],i)
        elif z=='O' and cur: ans.add((cur[1],i,cur[0])); cur=None
    return ans
@torch.no_grad()
def evaluate(model,loader,id2lab,device):
    model.eval(); tp=fp=fn=0
    for x,y,_ in loader:
        pred=model(x.to(device)).argmax(-1).cpu()
        for pp,gg in zip(pred,y):
            n=int((gg!=-100).sum()); ps=spans([id2lab[int(v)] for v in pp[:n]]); gs=spans([id2lab[int(v)] for v in gg[:n]])
            tp+=len(ps&gs); fp+=len(ps-gs); fn+=len(gs-ps)
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; f=2*p*r/(p+r) if p+r else 0
    return p,r,f

def run(domain,kind,seed,epochs,device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train=json.loads((DATA/domain/'train.json').read_text()); test=json.loads((DATA/domain/'test.json').read_text())
    random.shuffle(train); cut=max(1,int(len(train)*.8)); tr0,va0=train[:cut],train[cut:]
    types=sorted({e['type'] for r in train+test for e in r.get('entity_list',[])})
    all_rows,_,labs=encode_rows(tr0+va0,types); tr=all_rows[:cut]; va=all_rows[cut:]; te,_,_=encode_rows(test,types)
    cnt=Counter(t.lower() for x,_ in tr for t in x); vocab={'<pad>':0,'<unk>':1,**{w:i+2 for i,(w,c) in enumerate(cnt.items()) if c>=1}}
    l2={z:i for i,z in enumerate(labs)}; i2={i:z for z,i in l2.items()}
    label_counts=Counter(z for _,ys in tr for z in ys); weights=torch.tensor([1.0 if z=='O' else max(1.0,(label_counts['O']/max(1,label_counts[z]))**0.5) for z in labs],dtype=torch.float,device=device)
    loaders=[DataLoader(SeqDS(q,vocab,l2),batch_size=16,shuffle=k==0,collate_fn=collate) for k,q in enumerate((tr,va,te))]
    model=Tagger(len(vocab),len(labs),kind=='cross_residual').to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    hist=[]; best=None
    for ep in range(1,epochs+1):
        model.train(); total=0
        for x,y,_ in loaders[0]:
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss=nn.functional.cross_entropy(model(x).reshape(-1,len(labs)),y.reshape(-1),weight=weights,ignore_index=-100); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); total+=loss.item()
        vp,vr,vf=evaluate(model,loaders[1],i2,device); hist.append({'domain':domain,'method':kind,'seed':seed,'epoch':ep,'loss':total/max(1,len(loaders[0])),'val_f1':vf})
        if best is None or vf>best[0]: best=(vf,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); p,r,f=evaluate(model,loaders[2],i2,device)
    return {'domain':domain,'method':kind,'seed':seed,'precision':p,'recall':r,'f1':f,'train_sentences':len(tr),'test_sentences':len(te)},hist

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--domains',nargs='+',default=['ai','science','literature','music','politics']); ap.add_argument('--epochs',type=int,default=15); ap.add_argument('--seeds',nargs='+',type=int,default=[13,29,47]); a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); device='cuda' if torch.cuda.is_available() else 'cpu'; rows=[]; hist=[]; start=time.time()
    for d in a.domains:
        for m in ('bigru','cross_residual'):
            for s in a.seeds:
                r,h=run(d,m,s,a.epochs,device); rows.append(r); hist+=h; print(json.dumps(r),flush=True)
    with (OUT/'runs.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with (OUT/'history.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=hist[0].keys()); w.writeheader(); w.writerows(hist)
    (OUT/'status.json').write_text(json.dumps({'status':'complete','device':device,'seconds':time.time()-start,'runs':len(rows)},indent=2))
if __name__=='__main__': main()
