"""Evaluate calibrated Ollama drafts with lightweight residual consistency filters."""
import csv, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
src=ROOT/'paper/cross-residual/results/ollama_draft.jsonl'; out=ROOT/'paper/cross-residual/results/results.csv'
rows=[]
for line in src.read_text(encoding='utf-8').splitlines():
    rec=json.loads(line); raw=rec.get('response','')
    m=re.search(r'\{.*\}',raw,re.S)
    try: obj=json.loads(m.group(0)) if m else {}
    except Exception: obj={}
    ents=obj.get('entities',[]) if isinstance(obj,dict) else []; rels=obj.get('relations',[]) if isinstance(obj,dict) else []
    names={str(e.get('text','')).strip().lower() for e in ents if isinstance(e,dict)}
    valid=[r for r in rels if isinstance(r,dict) and str(r.get('source','')).strip().lower() in names and str(r.get('target','')).strip().lower() in names]
    rows.append({'file':rec['file'],'entities':len(ents),'relations':len(rels),'valid_relations':len(valid),'validity':len(valid)/len(rels) if rels else 1.0})
with out.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['file','entities','relations','valid_relations','validity']); w.writeheader(); w.writerows(rows)
print(json.dumps({'documents':len(rows),'entities':sum(r['entities'] for r in rows),'relations':sum(r['relations'] for r in rows),'valid_relations':sum(r['valid_relations'] for r in rows),'mean_validity':sum(r['validity'] for r in rows)/len(rows)},ensure_ascii=False))
