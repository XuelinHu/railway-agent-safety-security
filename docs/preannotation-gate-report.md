# Teacher pre-annotation quality gate

## Scope

- Date: 27 August 2026
- Teacher model: `gpt-5.6-terra` through the local OpenAI-compatible proxy
- Ontology: `safety_risk_ontology` version `1.0.0`
- Annotation schema: `0.1.0`
- Prompt under test: `teacher-preannotation-v1.1.0`
- Gate data: three fixed chunks from one long Chinese emergency-plan document

The gate compares the previous prompt with a prompt requiring each entity's `evidence.text` to be byte-for-byte identical to `entity.text`. Both variants received the same entity types, relation types, legal relation signatures, output schema, source chunks, and deterministic normalizer.

## Results

| Job | v1 retained entities | v1 retained relations | v1 findings / repairs | v1.1 retained entities | v1.1 retained relations | v1.1 findings / repairs |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 26 | 4 | 38 / 16 | 26 | 6 | 17 / 3 |
| C2 | 19 | 0 | 35 / 15 | 17 | 3 | 18 / 0 |
| C3 | 17 | 10 | 2 / 0 | 23 | 10 | 12 / 0 |
| **Total** | **62** | **14** | **75 / 31** | **66** | **19** | **47 / 3** |

All three v1.1 normalized records passed the annotation schema, source-offset checks, entity-reference checks, and ontology relation signatures. The result supports freezing v1.1 for the next bounded annotation batch, but it is not an extraction-accuracy result because these new candidates have not been human-labeled.

## Resource observations

The v1.1 gate consumed 63,962 prompt tokens and 14,247 completion tokens, 78,209 tokens in total. Serial wall time was 286.3 seconds, or 95.4 seconds per job. A direct extrapolation to 301 jobs is about 7.8 million tokens and 8.0 serial hours; actual usage will vary by document length and output density.

Therefore, remaining jobs should be run in bounded resumable batches with normalization after each batch. A batch must be paused if schema failures, unsupported evidence, illegal signatures, latency, or token usage materially exceed the gate range.

## Compact QLoRA control

Deterministic inverse-direction repair recovered five schema-valid relations from the compact QLoRA pilot. None matched the 26 gold test relations, and strict relation F1 remained zero. Relation-signature repair is retained as a trustworthiness constraint, not claimed as a standalone accuracy innovation.
