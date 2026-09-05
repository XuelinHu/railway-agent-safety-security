# Low-resource result audit

This snapshot includes validation metrics only when the run directory contains a valid `pipeline_complete.json` gate. The sealed formal-test namespace was not read.
Strict span metrics are recomputed after merging and de-duplicating all windows of each document. Original experiment directories are read-only.

## Included validation groups

- D10: 1 completed seed(s): 20260830
- D25: 1 completed seed(s): 20260830
- D100: 3 completed seed(s): 20260830, 20260831, 20260901

D50 has no completed validation pipeline and is excluded from validation tables and scaling plots. D10 and D25 are single-seed observations; D100 is summarized across three seeds.

## Training evidence

- Completed training runs: 36/36
- Parsed loss logs: 36/36
- Total recorded loss observations: 1053
- Prompt truncations: 0
- Answer truncations: 0
- Skipped overlength examples: 0

## Completion gates read

- `/ds1/workspace/ai/railway-agent-safety-security/data/processed/experiments/formal/low_resource_v2/d010/seed20260830/derived_validation/pipeline_complete.json`
- `/ds1/workspace/ai/railway-agent-safety-security/data/processed/experiments/formal/low_resource_v2/d025/seed20260830/derived_validation/pipeline_complete.json`
- `/ds1/workspace/ai/railway-agent-safety-security/data/processed/experiments/formal/low_resource_v2/d100/seed20260830/derived_validation/pipeline_complete.json`
- `/ds1/workspace/ai/railway-agent-safety-security/data/processed/experiments/formal/low_resource_v2/d100/seed20260831/derived_validation/pipeline_complete.json`
- `/ds1/workspace/ai/railway-agent-safety-security/data/processed/experiments/formal/low_resource_v2/d100/seed20260901/derived_validation/pipeline_complete.json`
