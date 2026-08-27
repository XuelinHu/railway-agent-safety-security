# Initial Safety Extraction Experiments

> This is an initial pilot report. The frozen test split contains four documents and is too small for final manuscript claims.

## Data

- Gold schema version: `0.1.0`; ontology version: `1.0.0`; review policy: `pilot_bulk_accepted_plus_manual_v1_1_gate_review`.
- Documents: 29; reviewed chunks: 72; train/validation/test: {'train': 21, 'test': 4, 'validation': 4}.
- Test entities: 88; test relations: 26.

## Strict Results

| System | Entity precision | Entity recall | Entity F1 | Relation precision | Relation recall | Relation F1 | Evaluated jobs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-14B baseline | 18.42% | 7.95% | 11.11% | 0.00% | 0.00% | 0.00% | 4 |
| Qwen3-14B + KG prompt | 13.75% | 12.50% | 13.10% | 0.00% | 0.00% | 0.00% | 4 |
| Qwen3-4B + QLoRA + KG prompt | - | - | - | - | - | - | 0 |
| Qwen3-4B + compact QLoRA + KG prompt | 36.67% | 12.50% | 18.64% | 0.00% | 0.00% | 0.00% | 4 |
| Qwen3-4B + v1.2 compact QLoRA + KG prompt | 15.05% | 20.29% | 17.28% | 20.00% | 4.76% | 7.69% | 3 |
| Qwen3-4B + v1.2 windowed compact QLoRA + KG prompt | 15.73% | 20.29% | 17.72% | 20.00% | 4.76% | 7.69% | 3 |
| Qwen3.8-27B GGUF + windowed compact extraction | 24.62% | 36.36% | 29.36% | 6.12% | 11.54% | 8.00% | 4 |

## QLoRA Status

- Base model: local Qwen3-4B; LoRA rank 8; training examples 56; optimization steps 14.
- Final training loss: `0.5822190642356873`; mean loss: `0.6156739784138543`.
- Compact target training: `True`; final loss: `0.6014580726623535`; mean loss: `0.4517741607768195`.
- Both adapter training runs completed successfully.
- The full-schema adapter did not produce the required top-level `entities`/`relations` JSON object on the four test jobs; its F1 is not treated as a valid extraction result.
- The compact adapter produced parseable envelopes for all four jobs, but Chinese jobs still returned incomplete structures and relation candidates were removed by schema constraints; the compact F1 is an engineering pilot result, not a final claim.
- Auditable inverse-direction repair recovered five schema-valid compact relations after missing claim statuses were conservatively marked `uncertain`; none matched the 26 gold relations, so strict relation F1 remains 0%. Relation-signature repair is retained as a trustworthiness constraint, not claimed as a standalone accuracy innovation.

## Interpretation

- The KG prompt variant improved entity F1 in this pilot from 11.11% to 13.10%, but relation F1 remained 0%; this is a smoke signal, not a statistically supported claim.
- The compact target improved entity F1 in this pilot, but relation extraction and Chinese structured output remain unresolved; the next engineering task is constrained decoding with explicit relation-signature repair, followed by regeneration on validation and test splits.
- Final experiments should add a second reviewer, expand the gold set, and report confidence intervals or paired tests.
- The teacher-prompt evidence-copy gate reduced normalization findings on three fixed Chinese chunks from 75 to 47 and increased retained relations from 14 to 19. These are pipeline-quality measurements, not test-set extraction-accuracy results; the three accepted chunks are in train and are excluded from the frozen pilot test.

## Updated QLoRA Run

- The reviewed training set now contains 59 chunks from 21 documents. Full-schema and compact-target adapters used Qwen3-4B, NF4, rank 8, one epoch, and 15 optimization steps.
- Expanded full-schema mean loss: `0.5559417386849721`; compact-target mean loss: `0.57162946164608`.
- Full-schema generations did not produce usable annotation envelopes on the four pilot test jobs and are recorded as format failures rather than extraction F1.
- The EOS-preserving compact adapter produced valid annotations on 2/4 pilot test jobs: both English jobs were parseable, while both Chinese jobs were format failures. On the valid English jobs, pooled strict entity F1 was 25.32% and relation F1 was 0%. These are engineering diagnostics, not final cross-language claims.
- The v1.2 compact adapter replaced the conflicting full-schema prompt, preserved a 12K default input window, and trained with mean loss `0.41431158085664116`. It produced valid structured outputs on 3/4 jobs; the long English job over-generated entities until the 4K and 8K output budgets were exhausted.
- On the 3 valid v1.2 jobs, pooled strict entity F1 was 17.28% and relation F1 was 7.69%; Chinese entity F1 was 11.62% and Chinese relation F1 was 0.00%. Because one test job failed structurally, these remain engineering diagnostics and are not final manuscript claims.
- The v1.2 long-Chinese diagnostic required an explicit 32K input window but exceeded the 24GB GPU memory budget when a concurrent local model occupied the GPU; its 29-entity/28-relation output was nevertheless recovered from the saved log and evaluated after structural repair.
- Tokenizer-budgeted windowing split the long Chinese job into two overlapping windows and produced 3 valid window outputs from 5. After document-level merge, pooled strict entity F1 was 17.72%, relation F1 was 7.69%, and Chinese entity F1 was 12.01%; this is a resource/robustness diagnostic over 3 evaluated documents.
- Qwen3.8-27B-UD-Q4_K_M GGUF was evaluated through llama.cpp with 4K output budgets and tokenizer-budgeted windows. All 5 windows and 4 documents produced valid parsed outputs; pooled strict entity F1 was 29.36% and relation F1 was 8.00%. Chinese entity F1 was 27.26%, while Chinese relation F1 remained 0.00%.
- The Qwen3.8 GGUF grammar-constrained path failed during llama.cpp sampler initialization on the current build, including a minimal schema. The reported Qwen3.8 result therefore uses ordinary generation followed by JSON parsing, evidence recovery, ontology validation, and ID normalization; constrained decoding remains a separate pending experiment after a compatible runtime is available.
