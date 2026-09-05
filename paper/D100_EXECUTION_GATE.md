# 100-document execution gate audit

Status: **failed_terminal**

The formal test remained sealed. No validation metric was read or
calculated during this gate.

| Run | System | Seed | Status |
|---|---|---:|---|
| `lr_v1_d100_seed20260830_baseline` | `baseline` | 20260830 | failed_terminal |
| `lr_v1_d100_seed20260830_kg_v1` | `kg_v1` | 20260830 | not_started |
| `lr_v1_d100_seed20260830_kg_v2` | `kg_v2` | 20260830 | not_started |
| `lr_v1_d100_seed20260831_baseline` | `baseline` | 20260831 | not_started |
| `lr_v1_d100_seed20260831_kg_v1` | `kg_v1` | 20260831 | not_started |
| `lr_v1_d100_seed20260831_kg_v2` | `kg_v2` | 20260831 | not_started |
| `lr_v1_d100_seed20260901_baseline` | `baseline` | 20260901 | not_started |
| `lr_v1_d100_seed20260901_kg_v1` | `kg_v1` | 20260901 | not_started |
| `lr_v1_d100_seed20260901_kg_v2` | `kg_v2` | 20260901 | not_started |

## Finding

The first baseline attempt failed before model loading because the
new telemetry call was incompatible with the frozen PyTorch build.
The one allowed mechanical retry preserved all model and training
parameters, loaded 986 examples without truncation or dropping, and
then failed during the first long-sequence backward pass with CUDA
out-of-memory. No adapter checkpoint or training metrics were emitted.

The retry reached peak whole-device use of 23952.0 MiB on the
24 GiB RTX 3090. The retained maximum sequence lengths are 5,027
tokens for baseline, 5,086 for KG V1, and 5,007 for KG V2, so the
remaining eight rows were not launched after the execution gate failed.

## Decision

Protocol v1 is immutable and failed its hardware execution gate. Do not
lower the frozen sequence length, alter batch settings, discard the
failed row, or start lower-budget runs under v1. Continuing requires a
new versioned runtime-only protocol amendment, frozen before another
attempt and applied identically to every system and seed.
