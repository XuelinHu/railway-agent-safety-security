# Fresh SpERT Public Baseline

This path trains SpERT from clean base encoders on the exact public
train/validation splits used by the current experiment. It never reads or
materializes a test split.

## Split Contract

| Dataset | Training input | Validation input | Base encoder |
|---|---|---|---|
| CoNLL04 | Published train, 922 rows | Published dev, 231 rows | `bert-base-cased` |
| SciERC | Published train, 1,861 rows | Published dev, 275 rows | SciBERT |
| ADE | Published train minus deterministic dev, 3,461 rows | 10% deterministic slice, 384 rows | `bert-base-cased` |

ADE uses seed `public-full-42`, matching `scripts/import_spert_benchmarks.py`.
The preparation command verifies the native SpERT entity spans and relation
indices against the existing train and validation gold annotations. Every
input and output receives a SHA-256 digest in `manifest.json`.

The downloaded SpERT task checkpoints are not valid validation checkpoints:
the CoNLL04 and SciERC releases were trained on train+dev, while the ADE
release was trained on a split containing the examples used for the current
derived validation set. They remain suitable only for separately labelled
upstream test reproduction after the experiment protocol permits test access.

## Compatibility

Upstream SpERT imports the removed `transformers.AdamW` class and saves the
legacy `pytorch_model.bin` format. `scripts/run_spert_compat.py` restores the
Transformers 4.1 AdamW update, including `correct_bias=False`, and forces the
legacy checkpoint format inside that process only. It does not modify the
external repository or the shared Python environment.

## Commands

Run the CPU-only preparation and model-load check first:

```bash
bash scripts/run_spert_fresh_baseline.sh preflight
bash scripts/run_spert_fresh_baseline.sh status
```

When the GPU queue permits training, run one dataset at a time:

```bash
SPERT_SEED=42 bash scripts/run_spert_fresh_baseline.sh train conll04
SPERT_SEED=42 bash scripts/run_spert_fresh_baseline.sh validation conll04

SPERT_SEED=42 bash scripts/run_spert_fresh_baseline.sh train scierc
SPERT_SEED=42 bash scripts/run_spert_fresh_baseline.sh validation scierc

SPERT_SEED=42 bash scripts/run_spert_fresh_baseline.sh train ade
SPERT_SEED=42 bash scripts/run_spert_fresh_baseline.sh validation ade
```

Training defaults to 20 epochs, batch size 2, final-only validation, and seed
42. Override only through recorded environment values such as
`SPERT_EPOCHS`, `SPERT_TRAIN_BATCH_SIZE`, and `SPERT_SAMPLING_PROCESSES`.
Training should remain a single GPU job. Once fresh checkpoints exist,
validation inference is small enough to run two datasets concurrently on a
24 GB RTX 3090, provided no large-model service owns the GPU.
