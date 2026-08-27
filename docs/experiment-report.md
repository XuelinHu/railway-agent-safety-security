# Initial Safety Extraction Experiments

> This is an initial pilot report. The frozen test split contains four documents and is too small for final manuscript claims.

## Data

- Gold version: `0.1.0`; review policy: `bulk_accepted_after_human_screening`.
- Documents: 28; train/validation/test: {'test': 4, 'validation': 4, 'train': 20}.
- Test entities: 88; test relations: 26.

## Strict Results

| System | Entity precision | Entity recall | Entity F1 | Relation precision | Relation recall | Relation F1 | Evaluated jobs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-14B baseline | 18.42% | 7.95% | 11.11% | 0.00% | 0.00% | 0.00% | 4 |
| Qwen3-14B + KG prompt | 13.75% | 12.50% | 13.10% | 0.00% | 0.00% | 0.00% | 4 |
| Qwen3-4B + QLoRA + KG prompt | - | - | - | - | - | - | 0 |
| Qwen3-4B + compact QLoRA + KG prompt | 36.67% | 12.50% | 18.64% | 0.00% | 0.00% | 0.00% | 4 |

## QLoRA Status

- Base model: local Qwen3-4B; LoRA rank 8; training examples 56; optimization steps 14.
- Final training loss: `0.5822190642356873`; mean loss: `0.6156739784138543`.
- Compact target training: `True`; final loss: `0.6014580726623535`; mean loss: `0.4517741607768195`.
- Both adapter training runs completed successfully.
- The full-schema adapter did not produce the required top-level `entities`/`relations` JSON object on the four test jobs; its F1 is not treated as a valid extraction result.
- The compact adapter produced parseable envelopes for all four jobs, but Chinese jobs still returned incomplete structures and relation candidates were removed by schema constraints; the compact F1 is an engineering pilot result, not a final claim.

## Interpretation

- The KG prompt variant improved entity F1 in this pilot from 11.11% to 13.10%, but relation F1 remained 0%; this is a smoke signal, not a statistically supported claim.
- The compact target improved entity F1 in this pilot, but relation extraction and Chinese structured output remain unresolved; the next engineering task is constrained decoding with explicit relation-signature repair, followed by regeneration on validation and test splits.
- Final experiments should add a second reviewer, expand the gold set, and report confidence intervals or paired tests.
