# Protocol v2 d100 first-row execution gate

Status: **passed_all_d100**

Formal test remained sealed. No validation metric was read or calculated.

| Run | System | Seed | Status |
|---|---|---:|---|
| `lr_v2_d100_seed20260830_baseline` | `baseline` | 20260830 | complete |
| `lr_v2_d100_seed20260830_kg_v1` | `kg_v1` | 20260830 | complete |
| `lr_v2_d100_seed20260830_kg_v2` | `kg_v2` | 20260830 | complete |
| `lr_v2_d100_seed20260831_baseline` | `baseline` | 20260831 | complete |
| `lr_v2_d100_seed20260831_kg_v1` | `kg_v1` | 20260831 | complete |
| `lr_v2_d100_seed20260831_kg_v2` | `kg_v2` | 20260831 | complete |
| `lr_v2_d100_seed20260901_baseline` | `baseline` | 20260901 | complete |
| `lr_v2_d100_seed20260901_kg_v1` | `kg_v1` | 20260901 | complete |
| `lr_v2_d100_seed20260901_kg_v2` | `kg_v2` | 20260901 | complete |

## First-row result

The baseline trained all 1266 examples for 317 optimizer steps. No answer or prompt was truncated, and no overlength example was skipped.

Training took 5282.756 seconds. Peak PyTorch allocated/reserved memory was 12025.17/14378.0 MiB; peak whole-device use was 15155.0 MiB. Mean training loss was 0.17285245775659377.

## Decision

The protocol v2 memory gate passed. The remaining eight d100 rows may run sequentially under the same frozen protocol. Lower-budget rows remain blocked until all nine d100 trainable rows complete and pass their artifact/telemetry checks.
