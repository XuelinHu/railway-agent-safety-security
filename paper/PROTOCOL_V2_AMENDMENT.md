# Protocol v2 amendment: semantic token-bounded execution

Protocol ID: `low-resource-provenance-study-v2`  
Frozen: 2026-08-30 (Asia/Shanghai)  
Parent: `low-resource-provenance-study-v1`  
Machine-readable specification: `configs/low_resource_protocol_v2.yaml`

## Reason for amendment

Protocol v1 terminated at its first 100-document baseline row. Its permitted
mechanical retry reached a 5,027-token sequence and failed during the first
backward pass with CUDA out-of-memory on the 24 GiB RTX 3090. The failure and
all v1 outputs remain immutable.

Changing hardware is not available. Protocol v2 therefore changes only the
runtime/windowing boundary needed to execute the registered comparison on the
same GPU. Model revision, document subsets, seeds, optimizer, QLoRA settings,
KG retrieval settings, evaluation metrics, statistics, and formal-test gate are
unchanged.

## Frozen windowing amendment

- Training `max_length` is 4,096 tokens for every system and seed.
- Window construction targets at most 3,840 complete prompt-plus-gold tokens,
  leaving room for differences among the final subset-specific KG prompts.
- Long source segments are split preferentially at sentence, paragraph, or line
  boundaries, with an 800-character segment cap and 160-character overlap.
- Entity source spans and relation evidence quotes are protected from splitting.
- A relation whose endpoints do not co-occur in a sequential window receives a
  minimal evidence-focused rescue window containing both endpoints and the
  relation evidence. Noncontiguous source provenance remains explicit.
- Every original entity and relation must occur in at least one window. Any
  coverage loss, prompt/target truncation, or overlength skip is a hard failure.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is applied identically to
  all runs.

The frozen train windows contain 1,266 examples and cover all 4,335 entities and
1,510 relations from the 244 train records. Of these, 286 are relation-rescue
windows. The frozen validation windows contain 491 examples, including 57
rescue windows, and cover all 1,001 entities and 343 relations. Formal test was
not read or materialized.

For the d100 assets, exact full-sequence maxima are 3,165 tokens for baseline,
3,968 for KG V1, and 3,902 for KG V2. Each system retains all 1,266 examples;
none exceeds the 4,096-token hard limit.

## First-row execution gate

`lr_v2_d100_seed20260830_baseline` completed 317 optimizer steps over all 1,266
examples. No answer or prompt was truncated and no example was skipped.
Training took 5,282.756 seconds, with mean loss 0.172852. Peak CUDA allocated
and reserved memory was 12,025.17 and 14,378 MiB; peak whole-device use was
15,155 MiB. The adapter and exact metrics/telemetry hashes are preserved in the
run manifest.

The first-row memory gate therefore passes. The remaining eight d100 rows may
run sequentially. The 10/25/50-document runs remain blocked until all d100 rows
complete their artifact and telemetry checks. Formal test remains sealed.
