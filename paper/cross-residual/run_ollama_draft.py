"""Fast weak-label extraction using the local Ollama Qwen3 model."""
import json, subprocess, time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[2]
pdfs=sorted((ROOT/'data/raw/raib').glob('*.pdf'))[:30]
out=ROOT/'paper/cross-residual/results/ollama_draft.jsonl'; out.parent.mkdir(parents=True,exist_ok=True)
prompt_tpl='''Extract railway safety entities and relations from this report excerpt. Return JSON only with keys entities and relations. Entities: text,type. Relations: source,relation,target. Use only evidence explicitly stated in the excerpt.\n\nEXCERPT:\n{}'''
done=set()
if out.exists():
    for line in out.read_text(errors='ignore').splitlines():
        try: done.add(json.loads(line)['file'])
        except Exception: pass
for pdf in pdfs:
    if pdf.name in done: continue
    txt=subprocess.run(['pdftotext','-f','1','-l','3','-layout',str(pdf),'-'],capture_output=True,text=True,timeout=30).stdout[:12000]
    try:
        r=requests.post('http://127.0.0.1:11434/api/generate',json={'model':'qwen3:14b','prompt':prompt_tpl.format(txt),'stream':False,'options':{'temperature':0.1,'num_predict':800}},timeout=240)
        r.raise_for_status(); raw=r.json().get('response','')
        rec={'file':pdf.name,'model':'qwen3:14b','response':raw,'timestamp':time.time()}
    except Exception as e:
        rec={'file':pdf.name,'model':'qwen3:14b','error':str(e),'timestamp':time.time()}
    with out.open('a',encoding='utf-8') as f: f.write(json.dumps(rec,ensure_ascii=False)+'\n')
    print(pdf.name, 'done', flush=True)
