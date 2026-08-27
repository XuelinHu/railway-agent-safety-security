# Railway Agent for Safety and Security

Research workspace for an English paper on AI agents, railway safety, complex-network substructure analysis, rare-event risk discovery, cascading risk, and resilience assessment.

## Research direction

The initial paper will study how multiple LLM agents can extract latent risk substructures from railway accident and near-miss reports, verify them against railway safety regulations, and map them to a railway network for cascading-risk and resilience analysis.

Provisional title:

> Discovering Latent Risk Substructures from Railway Accident Reports Using Large Language Model Agents

## Repository contents

- [`docs/research-requirements.md`](docs/research-requirements.md): research requirements, corpus plan, domestic and international sources, book recommendations, and reproducibility rules.
- [`scripts/download_public_corpus.ps1`](scripts/download_public_corpus.ps1): rate-limited downloader for legally public RAIB reports.
- [`data/catalog/book_metadata.csv`](data/catalog/book_metadata.csv): book metadata and licensed purchase links; no copyrighted ebook full text is stored.

## Planned agent pipeline

1. Extraction Agent: identify events, assets, people, environmental conditions, controls, and failures.
2. Causal Analysis Agent: reconstruct timelines, precursor chains, causal relations, and cascading paths.
3. Regulation Verification Agent: compare discovered risks with official safety-management requirements.
4. Resilience Assessment Agent: map risk substructures to railway networks and estimate critical nodes, propagation paths, and recovery capability.

## Data policy

Official public reports and open datasets are preferred for experiments. Purchased books are reference material for terminology, ontology design, and methodology; copyrighted books must not be redistributed or bulk-ingested without permission.

## Status

- [x] Define the initial research problem
- [x] Identify domestic and international corpus sources
- [x] Create the local Git repository
- [x] Build the source inventory
- [x] Implement evidence-preserving PDF/DOCX/TXT preprocessing
- [x] Parse legacy `.doc` files with project-local `antiword`/`catdoc` tools
- [x] Define the safety-risk ontology and annotation schemas
- [x] Prepare a 28-document bilingual pilot set
- [x] Run and validate 69 serial teacher pre-annotation jobs through the local OpenAI-compatible proxy
- [x] Screen and bulk-accept the 28-document pilot annotations
- [x] Build the first provenance-bearing safety knowledge graph snapshot
- [x] Create document-level train/validation/test splits
- [x] Run initial Qwen3 KG-prompt and local QLoRA pilot experiments
- [x] Generate the initial strict evaluation report with compact-output failure analysis
- [ ] Implement and evaluate the multi-agent pipeline
- [ ] Draft the English manuscript

Current research and implementation tasks are tracked in [`docs/work-plan.md`](docs/work-plan.md). The annotation ontology and output contract are defined in [`configs/risk_ontology.yaml`](configs/risk_ontology.yaml), [`schemas/preannotation_candidate.schema.json`](schemas/preannotation_candidate.schema.json), and [`schemas/risk_annotation.schema.json`](schemas/risk_annotation.schema.json).

The reproducible experiment entry points are [`scripts/build_experiment_jobs.py`](scripts/build_experiment_jobs.py), [`scripts/evaluate_annotations.py`](scripts/evaluate_annotations.py), [`scripts/train_qlora.py`](scripts/train_qlora.py), and [`scripts/run_qlora_inference.py`](scripts/run_qlora_inference.py). Current pilot numbers and limitations are recorded in [`docs/experiment-report.md`](docs/experiment-report.md).
