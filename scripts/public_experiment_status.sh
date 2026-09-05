#!/usr/bin/env bash
# Compact read-only status view suitable for this or another terminal session.
set -u

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

printf '%s\n' '--- branch ---'
git branch --show-current
printf '%s\n' '--- managed services ---'
for unit in \
  public-validation-analysis.service \
  public-hrge-cpu-preparation.service \
  public-cpu-smoke.service \
  public-gpu-validation-queue.service \
  public-pge-validation.service \
  public-spert-fresh-validation.service \
  public-gliner-glirel-t0-validation.service \
  public-glirel-calibration-conll04.service \
  public-glirel-calibration-scierc.service \
  public-glirel-calibration-ade.service \
  public-gliner-glirel-calibrated-validation.service \
  public-gliner-entity-only-validation.service \
  public-post-pge-bootstrap.service \
  public-validation-audit.service \
  public-external-formal-preflight.service \
  public-formal-release.service \
  public-formal-internal-matrix.service \
  public-formal-spert.service \
  public-formal-qwen-zeroshot.service \
  public-formal-gliner-glirel.service \
  public-formal-oneke-canary.service \
  public-formal-oneke-validation.service \
  public-gliner-validation.service \
  public-glirel-validation.service \
  public-qwen-zeroshot-probe.service \
  public-qwen-zeroshot-validation.service \
  public-model-download-hf.service \
  public-model-download-spert.service \
  public-model-download-repos.service; do
  printf '%-45s %s\n' "$unit" "$(systemctl --user is-active "$unit" 2>/dev/null || true)"
done
printf '%s\n' '--- CPU-only public smoke ---'
if [[ -f outputs/public_cpu_smoke/status.json ]]; then
  jq -r '
    if .schema_version == "public-cpu-smoke-v1" then
      "status=\(.status) stage=\(.active_stage) run_id=\(.run_id) threads=\(.environment.torch_threads) cuda=\(.environment.torch_cuda_available)",
      "pytest=\(.pytest.status) tests=\(.pytest.tests // 0) failures=\(.pytest.failures // 0) errors=\(.pytest.errors // 0)",
      "gliner_glirel=\(.gliner_glirel.status) gpu_runtime=\(.gpu_runtime.status)",
      (.datasets | to_entries[] | "\(.key)\t\(.value.status)\tpreprocessing=\(.value.preprocessing)\tevaluation=\(.value.evaluation.status)")
    else
      "status=\(.status) current=\(.current_check // "none") passed=\(.counts.passed) failed=\(.counts.failed) warnings=\(.counts.warnings) threads=\(.execution.thread_limit)",
      (.checks[] | "\(.check)\t\(.severity)\t\(.result)")
    end
  ' outputs/public_cpu_smoke/status.json
else
  printf '%s\n' 'status=not-started'
fi
printf '%s\n' '--- public HRGE CPU preparation ---'
for dataset in conll04 scierc ade; do
  manifest="data/processed/public_benchmarks_hrge_v1/$dataset/preparation_manifest.json"
  status="pending"
  if [[ -f "$manifest" ]]; then
    status="$(jq -r '.status // "unknown"' "$manifest" 2>/dev/null || printf invalid)"
  fi
  printf '%-8s %s\n' "$dataset" "$status"
done
printf '%s\n' '--- GPU validation queue ---'
if [[ -f outputs/public_gpu_validation_queue/status.json ]]; then
  jq -r '"status=\(.status) stage=\(.stage) qwen_retry=\(.qwen_retry_status // "-") qwen_failed=\(.qwen_remaining_failed_jobs // 0) poll_seconds=\(.poll_seconds) updated_at=\(.updated_at)"' \
    outputs/public_gpu_validation_queue/status.json
else
  printf '%s\n' 'status=not-started'
fi
printf '%s\n' '--- public PGE validation ---'
if [[ -f outputs/public_pge_validation_seed42/status.json ]]; then
  jq -r '"status=\(.status) stage=\(.stage) dataset=\(.active_dataset // "-") system=\(.active_system // "-") updated_at=\(.updated_at)"' \
    outputs/public_pge_validation_seed42/status.json
elif [[ -f outputs/public_pge_validation_seed42/launcher_status.json ]]; then
  jq -r '"status=\(.status) upstream=\(.upstream_status // "-") qwen_failed=\(.upstream_qwen_remaining_failed_jobs // 0) poll_seconds=\(.poll_seconds) updated_at=\(.updated_at)"' \
    outputs/public_pge_validation_seed42/launcher_status.json
else
  printf '%s\n' 'status=not-started'
fi
printf '%s\n' '--- fresh SpERT validation ---'
if [[ -f outputs/public_horizontal_validation/spert_fresh/status.json ]]; then
  jq -r '"status=\(.status) split=\(.split) seed=\(.seed) finished_at=\(.finished_at)"' \
    outputs/public_horizontal_validation/spert_fresh/status.json
elif [[ -f outputs/public_horizontal_validation/spert_fresh/launcher_status.json ]]; then
  jq -r '"status=\(.status) upstream=\(.upstream_status // "-") poll_seconds=\(.poll_seconds) updated_at=\(.updated_at)"' \
    outputs/public_horizontal_validation/spert_fresh/launcher_status.json
elif [[ -f outputs/public_horizontal_validation/spert_fresh/preflight.json ]]; then
  jq -r '"status=preflight-\(.status) mode=\(.mode) test_split_access=\(.test_split_access)"' \
    outputs/public_horizontal_validation/spert_fresh/preflight.json
else
  printf '%s\n' 'status=not-started'
fi
printf '%s\n' '--- GLiNER + GLiREL relation-threshold=0 sensitivity ---'
t0_root="outputs/public_horizontal_validation/gliner_glirel_t0"
if [[ -f "$t0_root/compatibility_canary.json" ]]; then
  jq -r '"canary=\(.status) runtime_compatible=\(.runtime_compatible) checked_at=\(.checked_at) source_revision=\(.provenance.glirel_source_revision // "-") transformers=\(.provenance.transformers // "-")"' \
    "$t0_root/compatibility_canary.json"
else
  printf '%s\n' 'canary=not-run runtime_compatible=false'
fi
if [[ -f "$t0_root/status.json" ]]; then
  jq -r '"status=\(.status) dataset=\(.active_dataset // "-") workers=\(.parallel_workers)/\(.requested_parallel_workers) entity_threshold=\(.parameters.entity_threshold) relation_threshold=\(.parameters.relation_threshold) dtype=\(.parameters.dtype) updated_at=\(.updated_at)"' \
    "$t0_root/status.json"
elif [[ -f "$t0_root/launcher_status.json" ]]; then
  jq -r '"status=\(.status) upstream=\(.upstream_status // "-") workers=\(.parameters.requested_workers) entity_threshold=\(.parameters.entity_threshold) relation_threshold=\(.parameters.relation_threshold) dtype=\(.parameters.dtype) poll_seconds=\(.poll_seconds) updated_at=\(.updated_at)"' \
    "$t0_root/launcher_status.json"
else
  printf '%s\n' 'status=not-started'
fi
for dataset in conll04 scierc ade; do
  total="$(awk 'NF {count++} END {print count + 0}' \
    "data/processed/public_benchmarks_full/$dataset/validation_baseline_jobs.jsonl")"
  merged=0
  parts=0
  prediction="$t0_root/${dataset}_validation.jsonl"
  [[ -f "$prediction" ]] && merged="$(awk 'NF {count++} END {print count + 0}' "$prediction")"
  for part in "$t0_root/${dataset}_validation.part"*.jsonl; do
    [[ -f "$part" ]] || continue
    count="$(awk 'NF {count++} END {print count + 0}' "$part")"
    parts=$((parts + count))
  done
  completed="$merged"
  (( parts > completed )) && completed="$parts"
  metrics="pending"
  [[ -f "$t0_root/${dataset}_validation.character_span_metrics.json" ]] && metrics="complete"
  printf '%-8s %4d/%-4d %6.2f%% metrics=%s\n' \
    "$dataset" "$completed" "$total" "$((completed * 10000 / total))e-2" "$metrics"
done
printf '%s\n' '--- GLiREL train-only calibration ---'
for dataset in conll04 scierc ade; do
  calibration="outputs/public_horizontal_validation/glirel_train_calibration/$dataset/status.json"
  if [[ -f "$calibration" ]]; then
    jq -r '"\(.dataset)\tstatus=\(.status)\tprogress=\(.completed_mode_jobs)/\(.expected_mode_jobs)\tmode=\(.active_label_mode // "-")\tselected=\(.selected_configuration.label_mode // "-")@\(.selected_configuration.threshold // "-")\tupdated_at=\(.updated_at)"' \
      "$calibration"
  else
    printf '%s\tstatus=not-started\n' "$dataset"
  fi
done
printf '%s\n' '--- train-calibrated GLiNER + GLiREL validation ---'
calibrated_root="outputs/public_horizontal_validation/gliner_glirel_calibrated"
if [[ -f "$calibrated_root/status.json" ]]; then
  jq -r '"status=\(.status) dataset=\(.active_dataset // "-") selection=\(.selection_split) validation_gold_selection=\(.validation_gold_used_for_selection) workers=\(.parallel_workers)/\(.requested_parallel_workers) updated_at=\(.updated_at)", (.datasets | to_entries[] | "\(.key)\t\(.value.prediction_rows)/\(.value.expected_jobs)\tmode=\(.value.configuration.label_mode)\tthreshold=\(.value.configuration.relation_threshold)\texecution=\(.value.configuration.execution)\tmetrics=\(.value.character_span_metrics_ready)")' \
    "$calibrated_root/status.json"
elif [[ -f "$calibrated_root/launcher_status.json" ]]; then
  jq -r '"status=\(.status) t0=\(.upstream.t0 // "-") calibration=\(.upstream.train_calibration | to_entries | map("\(.key):\(.value // "-")") | join(",")) poll_seconds=\(.poll_seconds) updated_at=\(.updated_at)"' \
    "$calibrated_root/launcher_status.json"
else
  printf '%s\n' 'status=not-started'
fi
printf '%s\n' '--- formal GLiNER entity-only validation ---'
entity_root="outputs/public_horizontal_validation/gliner_entity_only"
if [[ -f "$entity_root/status.json" ]]; then
  jq -r '"status=\(.status) split=\(.split) scope=\(.scope) gpu=\(.execution.gpu_used) test_access=\(.execution.test_split_access) updated_at=\(.updated_at)", (.datasets | to_entries[] | "\(.key)\tstatus=\(.value.status)\tjobs=\(.value.jobs)\tentity_f1=\(.value.entity_strict.f1)\trelations=\(.value.relations)")' \
    "$entity_root/status.json"
else
  printf '%s\n' 'status=not-started'
fi
printf '%s\n' '--- post-PGE validation bootstrap ---'
post_pge_root="outputs/public_post_pge_validation_seed42"
if [[ -f "$post_pge_root/status.json" ]]; then
  jq -r '"status=\(.status) stage=\(.stage) upstream=\(.upstream_status // "-") completed=\(.completed_datasets)/3 split=\(.selection_split) formal_test_read=\(.formal_test_read) updated_at=\(.updated_at)"' \
    "$post_pge_root/status.json"
else
  printf '%s\n' 'status=not-started'
fi
printf '%s\n' '--- unified public validation audit ---'
audit_root="outputs/public_validation_audit"
if [[ -f "$audit_root/status.json" ]]; then
  jq -r '"status=\(.status) rows=\(.counts.passed_system_dataset_rows // 0)/\((.counts.passed_system_dataset_rows // 0) + (.counts.failed_system_dataset_rows // 0)) comparisons=\(.counts.passed_comparisons // 0)/\((.counts.passed_comparisons // 0) + (.counts.failed_comparisons // 0)) errors=\(.counts.errors // 0) test_namespace=\(.test_namespace_status) finished_at=\(.finished_at)"' \
    "$audit_root/status.json"
elif [[ -f "$audit_root/launcher_status.json" ]]; then
  jq -r '"status=\(.status) ready=\(.ready) upstream_ready=\([.upstreams[].ready | select(.)] | length)/\(.upstreams | length) waiting=\([.upstreams | to_entries[] | select(.value.ready | not) | .key] | join(",")) poll_seconds=\(.poll_seconds) updated_at=\(.updated_at)"' \
    "$audit_root/launcher_status.json"
else
  printf '%s\n' 'status=not-started'
fi
printf '%s\n' '--- formal public-test release and matrix ---'
if [[ -f outputs/public_formal_matrix/gate_review_status.json ]]; then
  jq -r '"gate_review=\(.status) gpu_release_allowed=\(.formal_gpu_release_allowed // false) reason=\(.reason // "-")"' \
    outputs/public_formal_matrix/gate_review_status.json
else
  printf '%s\n' 'gate_review=missing gpu_release_allowed=false'
fi
if [[ -f outputs/public_formal_matrix/release_status.json ]]; then
  jq -r '"release=\(.status) stage=\(.stage) promotion=\(.promotion_status) updated_at=\(.updated_at)", (.datasets | to_entries[] | "\(.key)\tpreparation=\(.value.preparation_status)")' \
    outputs/public_formal_matrix/release_status.json
else
  printf '%s\n' 'release=not-started'
fi
if [[ -f outputs/public_formal_matrix/internal/status.json ]]; then
  jq -r '"internal=\(.status) stage=\(.stage) seed=\(.seed // "-") dataset=\(.active_dataset // "-") split=\(.active_split // "-") system=\(.active_system // "-") formal_test_read=\(.formal_test_read) updated_at=\(.updated_at)"' \
    outputs/public_formal_matrix/internal/status.json
elif [[ -f outputs/public_formal_matrix/internal_launcher_status.json ]]; then
  jq -r '"internal=\(.status) poll_seconds=\(.poll_seconds) updated_at=\(.updated_at)"' \
    outputs/public_formal_matrix/internal_launcher_status.json
else
  printf '%s\n' 'internal=not-started'
fi
printf '%s\n' '--- formal horizontal test queue ---'
for pair in \
  'spert:outputs/public_formal_matrix/horizontal/spert/status.json' \
  'qwen3_4b_zero_shot:outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot/status.json' \
  'gliner_glirel_calibrated:outputs/public_formal_matrix/horizontal/gliner_glirel_calibrated/status.json' \
  'oneke:outputs/public_external_formal/oneke/status.json'; do
  name="${pair%%:*}"
  marker="${pair#*:}"
  if [[ -f "$marker" ]]; then
    printf '%s\t' "$name"
    jq -r '"status=\(.status) stage=\(.stage // "-") dataset=\(.active_dataset // "-") formal_test_read=\(.formal_test_read // false) updated_at=\(.updated_at // "-")"' "$marker"
  else
    printf '%s\tstatus=not-started\n' "$name"
  fi
done
if [[ -f outputs/public_external_formal/preflight_status.json ]]; then
  jq -r '"external_preflight=\(.status) queue=\(.queue_counts // {}) gpu_started=\(.safety.gpu_process_started) test_gold_read=\(.scope.test_gold_read) generated_at=\(.generated_at)"' \
    outputs/public_external_formal/preflight_status.json
fi
if [[ -f outputs/public_external_formal/oneke/canary_launcher_status.json ]]; then
  jq -r '"oneke_canary=\(.status) stage=\(.stage) terminal=\(.terminal // false) updated_at=\(.updated_at)"' \
    outputs/public_external_formal/oneke/canary_launcher_status.json
fi
if [[ -f outputs/public_external_formal/oneke/gpu_canary.json ]]; then
  jq -r '"oneke_canary_result=\(.status) gpu=\(.actual_gpu_name // "-") peak_gib=\(((.peak_allocated_bytes // 0) / 1073741824 * 100 | round) / 100) test_gold_read=\(.test_gold_read) finished_at=\(.finished_at // "-")"' \
    outputs/public_external_formal/oneke/gpu_canary.json
fi
printf '%s\n' '--- downloads ---'
bash scripts/public_baseline_download_status.sh
printf '%s\n' '--- gpu ---'
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader
printf '%s\n' '--- qwen zero-shot validation progress ---'
for dataset in conll04 scierc ade; do
  total="$(awk 'NF {count++} END {print count + 0}' \
    "data/processed/public_benchmarks_full/$dataset/validation_baseline_jobs.jsonl")"
  logs=()
  for part in outputs/public_horizontal_validation/qwen3_4b_zero_shot/"${dataset}_validation.log.part"*; do
    [[ -f "$part" ]] || continue
    logs+=("$part")
  done
  terminal=0
  successful=0
  failed=0
  if (( ${#logs[@]} > 0 )); then
    summary="$(jq -s '
      reduce .[] as $row ({};
        if (($row.job_id | type) == "string") then .[$row.job_id] = ($row.status // "unknown")
        else . end
      ) |
      {
        terminal: length,
        successful: ([.[] | select(. == "success")] | length),
        failed: ([.[] | select(. != "success")] | length)
      }
    ' "${logs[@]}" 2>/dev/null || printf '{"terminal":0,"successful":0,"failed":0}')"
    terminal="$(jq -r '.terminal' <<< "$summary")"
    successful="$(jq -r '.successful' <<< "$summary")"
    failed="$(jq -r '.failed' <<< "$summary")"
  fi
  metrics="pending"
  [[ -f "outputs/public_horizontal_validation/qwen3_4b_zero_shot/${dataset}_validation_metrics.json" ]] && metrics="complete"
  printf '%-8s terminal=%4d/%-4d success=%4d failed=%3d %6.2f%% metrics=%s\n' \
    "$dataset" "$terminal" "$total" "$successful" "$failed" \
    "$((terminal * 10000 / total))e-2" "$metrics"
done
if [[ -f outputs/public_horizontal_validation/qwen3_4b_zero_shot/retry_status.json ]]; then
  jq -r '"retry_status=\(.status) recorded_remaining_failures=\(.remaining_failed_jobs) updated_at=\(.updated_at)"' \
    outputs/public_horizontal_validation/qwen3_4b_zero_shot/retry_status.json
fi
printf '%s\n' '--- GLiNER + GLiREL validation progress ---'
for dataset in conll04 scierc ade; do
  total="$(awk 'NF {count++} END {print count + 0}' \
    "data/processed/public_benchmarks_full/$dataset/validation_baseline_jobs.jsonl")"
  merged=0
  parts=0
  prediction="outputs/public_horizontal_validation/gliner_glirel/${dataset}_validation.jsonl"
  [[ -f "$prediction" ]] && merged="$(awk 'NF {count++} END {print count + 0}' "$prediction")"
  for part in outputs/public_horizontal_validation/gliner_glirel/"${dataset}_validation.part"*.jsonl; do
    [[ -f "$part" ]] || continue
    count="$(awk 'NF {count++} END {print count + 0}' "$part")"
    parts=$((parts + count))
  done
  completed="$merged"
  (( parts > completed )) && completed="$parts"
  metrics="pending"
  [[ -f "outputs/public_horizontal_validation/gliner_glirel/${dataset}_validation.character_span_metrics.json" ]] && metrics="complete"
  printf '%-8s %4d/%-4d %6.2f%% metrics=%s\n' \
    "$dataset" "$completed" "$total" "$((completed * 10000 / total))e-2" "$metrics"
done
printf '%s\n' '--- result markers ---'
find \
  outputs/public_full_stage1/validation_analysis \
  outputs/public_horizontal_validation \
  outputs/public_post_pge_validation_seed42 \
  outputs/public_validation_audit \
  outputs/public_formal_matrix \
  outputs/public_external_formal \
  -name status.json -print 2>/dev/null | sort
printf '%s\n' '--- watchdog ---'
tail -40 outputs/public_experiment_watchdog/latest.log 2>/dev/null || true
