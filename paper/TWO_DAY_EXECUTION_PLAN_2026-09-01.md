# Two-day experiment execution plan

Frozen at: 2026-09-01 10:30 Asia/Shanghai  
Compute deadline: 2026-09-03 approximately 08:00 Asia/Shanghai  
Formal-test status: sealed

## Objective

Use the remaining single-GPU window to produce the smallest experiment package
that supports a defensible paper claim. Multi-seed evidence for the full-budget
main comparison takes priority over completing every low-resource cell.

## Measured runtime basis

- The first 126 D10 baseline validation jobs averaged about 18 seconds/job.
- A baseline row projects to about 2.5 GPU-hours at that rate.
- Historical KG V2 validation required about 8.5 GPU-hours for a comparable
  row because its high-recall outputs were longer.
- One complete budget/seed group (baseline, KG V1, KG V2, plus derived V3 and
  metrics) is therefore planned at roughly 13--15 GPU-hours.

## Priority order

1. D100, seed 20260830: baseline, KG V1, KG V2, then derived systems.
2. D100, seed 20260831: baseline, KG V1, KG V2, then derived systems.
3. D100, seed 20260901: baseline, KG V1, KG V2, then derived systems.
4. Resume the preserved D10 seed 20260830 group if time remains.
5. Run D50 seed 20260830, then D25 seed 20260830 if further time remains.
6. Additional low-budget seeds are last priority and are not expected within
   the 46-hour compute window.

This order targets the pre-registered three-seed D100 comparison first. The
D10 partial inference is preserved at 126/491 terminal jobs and is resumable;
no completed output is discarded.

## Public-data scope

The 61-example, all-relation Geo-NRE diagnostic is already complete for the
Qwen3-4B base model, railway KG V1 adapter, railway KG V2 adapter, and three
V3-style selection proxies. It consumes no further GPU time in this plan.

Geo-NRE remains an external-domain compatibility/abstention diagnostic rather
than a primary railway benchmark because its schema lacks railway entity
types, relation signatures, claim status, character offsets, and evidence
spans. Downloading and adapting REBEL or Wiki-NRE is deferred because a fair
task-native run would displace the D100 multi-seed evidence.

## Stop and reporting rules

- The user service has a 46-hour runtime limit and may stop between jobs or in
  a resumable in-progress row.
- Failed or missing generations remain in the denominator as empty
  predictions; no favorable seed or row is silently removed.
- Every completed seed group immediately produces strict span-aware metrics,
  evidence/graph metrics, V2 verification, V3 raw/final outputs, and frozen
  paired bootstrap comparisons.
- The formal test remains unopened. A one-time formal-test run requires a
  separate explicit authorization after the validation-selected pipeline is
  frozen.
- The paper will report the completed matrix honestly. Unfinished low-resource
cells remain pending rather than being extrapolated or imputed.

## Two-lane inference amendment

A controlled 20-job concurrency probe was run without changing decoding
parameters. With the existing D100 seed-20260830 baseline continuing in the
first process and seed-20260831 baseline in the second process:

- two model processes used about 8,865 MiB of 24,576 MiB device memory;
- GPU utilization reached 99%, power was about 240 W, and temperature was
  about 70 C;
- no CUDA OOM or host-memory pressure occurred;
- during the 949.6-second probe, lane B completed 20 jobs while lane A
  completed 38 jobs, for about 65% higher aggregate throughput than the prior
  single-lane rate;
- probe outputs are ordinary frozen-parameter validation outputs and remain in
  their run directory for resume; they are not discarded or rerun.

The remaining execution therefore uses two disjoint controller lanes. Lane A
prioritizes D100 seeds 20260830 and 20260901. Lane B prioritizes D100 seed
20260831 and then resumes D10 seed 20260830. Each lane writes only to its own
budget/seed groups. Three-way concurrency is not used because two processes
already saturate the GPU and a third would increase memory and tail-latency
risk without a demonstrated throughput benefit. The original wall-clock
deadline is retained rather than restarted.
