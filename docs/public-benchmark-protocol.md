# Public Benchmark Protocol

This branch evaluates the existing provenance-preserving, evidence-gated KG
augmentation method on small public entity-relation datasets. It is not a
replacement for the railway benchmark and it does not mix public data with
railway training, validation, or sealed test artifacts.

## Research Claim

The transferable contribution is a controlled use of graph knowledge:

1. A graph is built only from the selected training split.
2. The graph can nominate entity types and relation candidates, but cannot
   establish a relation in a validation or test text.
3. Every retained entity must be an exact source span.
4. Every retained relation must have an evidence quote containing both endpoints.
5. Relation endpoint signatures are inferred only from selected training labels.
6. The system abstains when source evidence or a legal signature is absent.

This yields four comparable variants: text-only extraction, text plus
training-KG candidates, candidates plus evidence gate, and candidates plus
evidence gate plus training-derived signature verifier. The primary result is
the precision/recall trade-off together with unsupported-relation and invalid-
signature rates, not an isolated post-filtered precision value.

## Public Datasets

- CoNLL04 is the initial small smoke test: 4 entity types and 5 relation types.
- SciERC is the primary transfer dataset: scientific entity-relation extraction
  with six entity types and seven relation types.
- ADE publishes train/test only; the adapter derives a deterministic 10%
  development slice from train and removes it from the training pool.

The `scripts/import_spert_benchmarks.py` adapter reads SpERT JSON files and
produces jobs, gold annotations, a split manifest, a training-only ontology,
and training-only graph edges. Its sentence text and offsets are reconstructed
deterministically from the released tokens, so the existing evidence and
relation verifier can be reused unchanged.

## First 3090 Run

Use a small fixed development protocol first, never the test split for tuning:

```bash
python3 scripts/import_spert_benchmarks.py \
  --dataset conll04 --train-limit 100 --validation-limit 20 --test-limit 30

python3 scripts/build_experiment_jobs.py \
  --jobs data/processed/public_benchmarks/conll04/jobs.jsonl \
  --manifest data/processed/public_benchmarks/conll04/split_manifest.jsonl \
  --mentions data/processed/public_benchmarks/conll04/knowledge_graph/mentions.jsonl \
  --ontology data/processed/public_benchmarks/conll04/ontology.yaml \
  --split validation --mode kg_constrained \
  --output data/processed/public_benchmarks/conll04/validation_kg_jobs.jsonl
```

Freeze limits, seed, prompt, model, and gate thresholds before producing the
30-example held-out test result. Then repeat with full SciERC splits.

The converter also writes `{train,validation,test}_{gold,index}.jsonl`, which
can be passed directly to the existing `scripts/train_qlora.py` and evaluation
scripts. Use the compact target first on 100 training examples to keep a single
3090 run short; increase to the full split only after the smoke test is valid.
