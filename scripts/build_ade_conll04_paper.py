#!/usr/bin/env python3
"""Read completed ADE/CoNLL04 runs; build a frozen, reproducible paper snapshot.

No training, inference, threshold selection, or source-artifact mutation.
The already released test artifacts are used for reporting and overlap sensitivity.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np

import evaluate_public_validation_spans as evaluator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/results/ade_conll04"
DATASETS = ("ade", "conll04")
NAMES = {"ade": "ADE", "conll04": "CoNLL04"}
LABEL_NAMES = {
    "Adverse-Effect": "ADVERSE EFFECT",
    "Drug": "DRUG",
    "Loc": "LOCATION",
    "Org": "ORGANIZATION",
    "Other": "OTHER",
    "Peop": "PERSON",
    "Kill": "KILL",
    "Live_In": "LIVE IN",
    "Located_In": "LOCATED IN",
    "OrgBased_In": "ORGANIZATION BASED IN",
    "Work_For": "WORK FOR",
}
SPLITS = ("train", "validation", "test")
SYSTEMS = ("soe", "eae", "hrge", "evge", "cfe", "pge")
HASHES = {}


def read(path):
    path = ROOT / path
    data = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def rows(path):
    path = ROOT / path
    data = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def csv_write(name, values):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def score(ds, split, system, predictions):
    base = ROOT / "data/processed/public_benchmarks_full" / ds
    path = OUT / "metrics" / f"{ds}_{split}_{system}.json"
    args = argparse.Namespace(gold=base/f"{split}_gold.jsonl",
        gold_index=base/f"{split}_index.jsonl", predictions=ROOT/predictions,
        jobs=base/f"{split}_baseline_jobs.jsonl", output=path,
        allow_non_validation=split == "test")
    for p in (args.gold, args.gold_index, args.predictions, args.jobs):
        HASHES[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.run(args)
    d = json.loads(path.read_text())
    assert d["jobs_missing_predictions"] == 0
    assert d["resolution"]["unresolved_gold_entities"] == 0
    return d


def percent(value):
    return f"{100*value:.2f}"


def tex_escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def table(name, header, body, columns):
    content = [r"\begin{tabular}{"+columns+"}", r"\toprule", " & ".join(header)+r" \\", r"\midrule"]
    content += [" & ".join(map(tex_escape, line))+r" \\" for line in body]
    content += [r"\bottomrule", r"\end{tabular}"]
    (OUT / name).write_text("\n".join(content)+"\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats, labels, lengths, results, sensitivity, evidence = [], [], {}, {}, {}, []
    reconciliation, bootstrap = {}, {}
    for ds in DATASETS:
        base = Path("data/processed/public_benchmarks_full") / ds
        lengths[ds] = {}
        for split in SPLITS:
            gold = rows(base / f"{split}_gold.jsonl")
            jobs = rows(base / f"{split}_baseline_jobs.jsonl")
            assert len(gold) == len(jobs)
            ec = Counter(e["type"] for g in gold for e in g["entities"])
            rc = Counter(r["type"] for g in gold for r in g["relations"])
            ll = [len(" ".join(s["text"] for s in j["segments"]).split()) for j in jobs]
            lengths[ds][split] = ll
            stats.append(dict(dataset=ds, split=split, sentences=len(gold),
                entities=sum(ec.values()), relations=sum(rc.values()),
                entity_types=len(ec), relation_types=len(rc),
                mean_tokens=round(float(np.mean(ll)),2), median_tokens=float(np.median(ll)),
                p95_tokens=float(np.percentile(ll,95)), max_tokens=max(ll),
                zero_relation_sentences=sum(not g["relations"] for g in gold)))
            for kind, counts in (("entity",ec),("relation",rc)):
                for label, count in sorted(counts.items()):
                    labels.append(dict(dataset=ds, split=split, kind=kind, label=label,
                                       count=count, proportion=count/sum(counts.values())))

        test = Path("outputs/public_formal_matrix/internal/seed42") / ds / "test"
        assert read(test / "complete.json")["status"] == "complete"
        results[ds] = {"validation": {}, "test": {}}
        for split in ("validation", "test"):
            for system in SYSTEMS if split == "validation" else ("soe", "pge"):
                if split == "validation":
                    run = Path("outputs/public_pge_validation_seed42") / ds
                    if system == "soe":
                        prediction = Path(f"outputs/public_full_stage1/validation_analysis/completed_predictions/{ds}_baseline_expanded.jsonl")
                    else:
                        prediction = run / ({"eae":"eae_validation_expanded.jsonl", "hrge":"hrge_validation_expanded.jsonl"}.get(system, f"{system}_validation.jsonl"))
                else:
                    prediction = test / ("soe_expanded.jsonl" if system == "soe" else "pge.jsonl")
                results[ds][split][system] = score(ds,split,system,prediction)

        horizontal = Path("outputs/public_formal_matrix/horizontal")
        for system, prediction in {
            "spert": horizontal / "spert_fresh_seed42" / ds / "test_predictions.jsonl",
            "qwen_zero": horizontal / "qwen3_4b_zero_shot" / f"{ds}_test_verified.jsonl",
            "gliner_glirel": horizontal / "gliner_glirel_calibrated" / f"{ds}_test.jsonl",
        }.items():
            results[ds]["test"][system] = score(ds,"test",system,prediction)

        quarantine = read(Path("data/processed/public_benchmarks_hrge_test_v1")/ds/"audits/train_test_exact_text_quarantine.json")
        excluded = set(quarantine["excluded_training_documents_by_test_job"])
        sensitivity[ds] = {"excluded_test_sentences":len(excluded), "systems":{}}
        reconciliation[ds] = {}
        for system in ("soe", "pge"):
            d = results[ds]["test"][system]
            kept = [v for k,v in d["per_job"].items() if k not in excluded]
            sensitivity[ds]["systems"][system] = {"sentences":len(kept),
                **{field:evaluator.aggregate(kept,field) for field in ("entity_strict","relation_strict")}}
            # Record the effect of uniform set-based duplicate handling.
            original = read(test / "metrics" / f"{system}_span.json")
            reconciliation[ds][system] = {}
            for field, orig_field in (("entity_strict","entity"),("relation_strict","relation")):
                assert d[field]['gold'] == original['overall'][orig_field]['gold']
                reconciliation[ds][system][field] = {'old':original['overall'][orig_field], 'common':d[field]}
        for system in ("soe", "hrge", "pge"):
            ev = read(test/"metrics"/f"{system}_evidence.json")["overall"]
            evidence.append(dict(dataset=ds,system=system,**ev["counts"],
                                 invalid_relation_rate=ev["invalid_relation_rate"]))
        # Paired sentence bootstrap, fixed predictions, no tuning or new inference.
        left=results[ds]['test']['soe']['per_job'];right=results[ds]['test']['pge']['per_job']
        ids=sorted(left);assert set(ids)==set(right)
        rng=np.random.default_rng(20260830)
        indices=rng.integers(0,len(ids),size=(20000,len(ids)))
        bootstrap[ds]={'iterations':20000,'seed':20260830,'unit':'sentence','fields':{}}
        for field in ('entity_strict','relation_strict'):
            a=np.array([[left[k][field][c] for c in ('gold','predicted','correct')] for k in ids])
            b=np.array([[right[k][field][c] for c in ('gold','predicted','correct')] for k in ids])
            aa=a[indices].sum(axis=1);bb=b[indices].sum(axis=1)
            delta=2*bb[:,2]/np.maximum(bb[:,0]+bb[:,1],1)-2*aa[:,2]/np.maximum(aa[:,0]+aa[:,1],1)
            at=a.sum(axis=0);bt=b.sum(axis=0)
            bootstrap[ds]['fields'][field]={'difference':float(2*bt[2]/(bt[0]+bt[1])-2*at[2]/(at[0]+at[1])),
                'ci95':[float(v) for v in np.quantile(delta,[.025,.975])]}

    csv_write("dataset_statistics.csv", stats)
    csv_write("label_distribution.csv", labels)
    csv_write("test_evidence.csv", evidence)
    dump("overlap_sensitivity.json", sensitivity)
    dump("evaluator_reconciliation.json", reconciliation)
    dump("paired_test_bootstrap.json", bootstrap)
    summary = {ds:{split:{sys:{k:v for k,v in met.items() if k not in ("per_job","by_language")}
                   for sys,met in sm.items()} for split,sm in d.items()} for ds,d in results.items()}
    dump("results_snapshot.json",summary)
    dump("lengths.json",lengths)
    # Keep a small public-method manifest without unrelated dataset entries or local paths.
    import yaml
    cfg_path = ROOT/'configs/public_pge_seed42.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    HASHES[str(cfg_path.relative_to(ROOT))] = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    for script in ('build_ade_conll04_paper.py','evaluate_public_validation_spans.py',
                   'import_spert_benchmarks.py','prepare_public_hrge_cpu.py',
                   'fuse_kg_v1_v2_predictions.py','train_qlora.py'):
        path=ROOT/'scripts'/script
        HASHES[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    prep=read('data/processed/public_benchmarks_hrge_test_v1/ade/preparation_manifest.json')
    dump('protocol_snapshot.json',{'datasets':{d:cfg['datasets'][d] for d in DATASETS},
        'training_seed':42,'qwen_family':cfg['model']['family'],
        'qwen_revision':cfg['model']['revision'],
        'bge_m3_revision':prep['semantic_model']['revision'],
        'training':cfg['training'],'inference':cfg['inference'],
        'scope':'retrospective two-benchmark reporting; frozen model predictions',
        'primary_evaluator':'strict-source-character-span with set deduplication'})
    dump("source_hashes.json",HASHES)

    table("dataset_statistics.tex",["Dataset","Split","Sentences","Entities","Relations","Mean length"],
          [[NAMES[s['dataset']],s['split'],s['sentences'],s['entities'],s['relations'],s['mean_tokens']] for s in stats],"llrrrr")
    test_rows=[]
    for ds in DATASETS:
        for sys,name in (("qwen_zero","Qwen3-ZS + verifier"),("gliner_glirel","GLiNER + GLiREL"),("spert","SpERT"),("soe","SOE"),("pge","PGE")):
            d=results[ds]["test"][sys]
            test_rows.append([NAMES[ds],name]+[percent(d[f][m]) for f in ("entity_strict","relation_strict") for m in ("precision","recall","f1")])
    table("test_results.tex",["Dataset","System","Ent. P","Ent. R","Ent. F1","Rel. P","Rel. R","Rel. F1"],test_rows,"llrrrrrr")
    table("validation_ablation.tex",["Dataset","System","Ent. P","Ent. R","Ent. F1","Rel. P","Rel. R","Rel. F1"],
          [[NAMES[ds],sys.upper()]+[percent(results[ds]['validation'][sys][f][m]) for f in ('entity_strict','relation_strict') for m in ('precision','recall','f1')] for ds in DATASETS for sys in SYSTEMS],"llrrrrrr")
    label_rows=[]
    for ds in DATASETS:
        for kind in ("entity","relation"):
            for label in sorted({r['label'] for r in labels if r['dataset']==ds and r['kind']==kind}):
                label_rows.append([NAMES[ds], kind.title(), LABEL_NAMES[label]] +
                    [next(r['count'] for r in labels if r['dataset']==ds and
                          r['kind']==kind and r['label']==label and r['split']==split)
                     for split in SPLITS])
    table("label_counts.tex",["Dataset","Kind","Label","Train","Validation","Test"],label_rows,"lllrrr")
    print(json.dumps({'statistics':stats,'test':{d:{s:{f:results[d]['test'][s][f]['f1'] for f in ('entity_strict','relation_strict')} for s in results[d]['test']} for d in DATASETS},'sensitivity':sensitivity},indent=2))


if __name__ == '__main__':
    main()
