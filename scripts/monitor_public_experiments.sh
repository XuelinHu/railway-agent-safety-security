#!/usr/bin/env bash
# One-shot watchdog for public-baseline downloads and managed validation jobs.
set -uo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
state_root="outputs/public_experiment_watchdog"
mkdir -p "$state_root"

exec 9>"$state_root/watchdog.lock"
flock -n 9 || exit 0

timestamp="$(date --iso-8601=seconds)"
log="$state_root/watchdog.log"
gpu_liveness_grace_seconds="${GPU_LIVENESS_GRACE_SECONDS:-600}"
gpu_restart_cooldown_seconds="${GPU_RESTART_COOLDOWN_SECONDS:-3600}"
gpu_compute_pids=""
gpu_query_status="not_queried"

if [[ ! "$gpu_liveness_grace_seconds" =~ ^[1-9][0-9]*$ ]]; then
  gpu_liveness_grace_seconds=600
fi
if [[ ! "$gpu_restart_cooldown_seconds" =~ ^[1-9][0-9]*$ ]]; then
  gpu_restart_cooldown_seconds=3600
fi

status_complete() {
  local name="$1"
  [[ -f "outputs/public_baseline_downloads/${name}.status.json" ]] &&
    [[ "$(jq -r '.status // ""' "outputs/public_baseline_downloads/${name}.status.json" 2>/dev/null)" == "complete" ]]
}

lane_complete() {
  local lane="$1"
  local component
  case "$lane" in
    hf) components=(instructuie glirel oneke scibert roberta_large) ;;
    spert) components=(spert_models) ;;
    repos) components=(mirror_code glirel_code pl_marker_code oneke_code mirror_models) ;;
    *) return 1 ;;
  esac
  for component in "${components[@]}"; do
    status_complete "$component" || return 1
  done
}

launch_download_lane() {
  local lane="$1"
  local unit="public-model-download-${lane}.service"
  systemd-run --user --unit="${unit%.service}" --collect --no-block \
    --property=Type=exec \
    --property=Restart=on-failure \
    --property=RestartSec=120 \
    /usr/bin/bash -lc \
    "cd '$root' && exec > 'outputs/public_baseline_downloads/${lane}.log' 2>&1; exec scripts/download_public_baseline_models.sh '$lane'"
}

marker_status() {
  local marker="$1"
  [[ -f "$marker" ]] || return 1
  jq -r '.status // empty' "$marker" 2>/dev/null
}

marker_complete() {
  local marker="$1" expected="$2" actual
  local candidate
  local accepted=()
  actual="$(marker_status "$marker" 2>/dev/null || true)"
  IFS='|' read -r -a accepted <<< "$expected"
  for candidate in "${accepted[@]}"; do
    [[ "$actual" == "$candidate" ]] && return 0
  done
  return 1
}

latest_progress_mtime() {
  local path candidate=0 mtime
  for path in "$@"; do
    if [[ -f "$path" ]]; then
      mtime="$(stat -c %Y "$path" 2>/dev/null || printf 0)"
      (( mtime > candidate )) && candidate="$mtime"
    elif [[ -d "$path" ]]; then
      mtime="$(find "$path" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -n 1)"
      mtime="${mtime%%.*}"
      [[ "$mtime" =~ ^[0-9]+$ ]] || mtime=0
      (( mtime > candidate )) && candidate="$mtime"
    fi
  done
  printf '%s\n' "$candidate"
}

unit_active_age_seconds() {
  local unit="$1" entered entered_epoch now
  entered="$(systemctl --user show "$unit" -p ActiveEnterTimestamp --value 2>/dev/null || true)"
  entered_epoch="$(date -d "$entered" +%s 2>/dev/null || printf 0)"
  now="$(date +%s)"
  if [[ "$entered_epoch" =~ ^[0-9]+$ ]] && (( entered_epoch > 0 && entered_epoch <= now )); then
    printf '%s\n' "$(( now - entered_epoch ))"
  else
    printf '0\n'
  fi
}

refresh_gpu_snapshot() {
  local output
  if output="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
    gpu_compute_pids="$(printf '%s\n' "$output" | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/, ""); print}')"
    gpu_query_status=ok
  else
    gpu_compute_pids=""
    gpu_query_status=failed
  fi
}

read_unit_runtime() {
  local unit="$1" control_group pid gpu_pid cmdline process_state
  runtime_cgroup_pids=""
  runtime_cuda_pids=""
  runtime_gpu_worker_pids=""
  runtime_cpu_phase_pids=""
  runtime_d_state_pids=""
  control_group="$(systemctl --user show "$unit" -p ControlGroup --value 2>/dev/null || true)"
  [[ -n "$control_group" && -r "/sys/fs/cgroup${control_group}/cgroup.procs" ]] || return 0
  runtime_cgroup_pids="$(<"/sys/fs/cgroup${control_group}/cgroup.procs")"
  for pid in $runtime_cgroup_pids; do
    for gpu_pid in $gpu_compute_pids; do
      if [[ "$pid" == "$gpu_pid" ]]; then
        runtime_cuda_pids+="${runtime_cuda_pids:+ }$pid"
        break
      fi
    done
    [[ -r "/proc/$pid/cmdline" ]] || continue
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" == *"train_qlora.py"* ||
          "$cmdline" == *"run_qlora_inference.py"* ||
          "$cmdline" == *"run_spert_compat.py"* ||
          "$cmdline" == *"run_public_oneke_formal.py"* ||
          ( "$cmdline" == *"run_gliner_glirel_validation.py"* &&
            "$cmdline" != *"--preflight-only"* ) ]]; then
      runtime_gpu_worker_pids+="${runtime_gpu_worker_pids:+ }$pid"
    fi
    if [[ "$cmdline" == *"--preflight-only"* ||
          "$cmdline" == *"preflight_"* ||
          "$cmdline" == *"prepare_"* ||
          "$cmdline" == *"evaluate_"* ||
          "$cmdline" == *"convert_"* ||
          "$cmdline" == *"merge_"* ||
          "$cmdline" == *"materialize_"* ||
          "$cmdline" == *"expand_"* ||
          "$cmdline" == *"verify_"* ||
          "$cmdline" == *"bootstrap_"* ||
          "$cmdline" == *"calibrate_"* ||
          "$cmdline" == *"audit_"* ]]; then
      runtime_cpu_phase_pids+="${runtime_cpu_phase_pids:+ }$pid"
    fi
    if [[ -r "/proc/$pid/status" ]]; then
      process_state="$(awk '/^State:/ {print $2; exit}' "/proc/$pid/status" 2>/dev/null || true)"
      [[ "$process_state" == D ]] && runtime_d_state_pids+="${runtime_d_state_pids:+ }$pid"
    fi
  done
}

gpu_phase_is_expected() {
  local job="$1" state stage dataset execution
  case "$job" in
    qwen)
      [[ "$(systemctl --user is-active public-qwen-zeroshot-validation.service 2>/dev/null || true)" == active ]] &&
        ! marker_complete outputs/public_horizontal_validation/qwen3_4b_zero_shot/status.json validation_complete
      ;;
    gpu_queue)
      [[ "$(marker_status outputs/public_gpu_validation_queue/status.json 2>/dev/null || true)" == running ]]
      ;;
    pge)
      state="$(marker_status outputs/public_pge_validation_seed42/status.json 2>/dev/null || true)"
      stage="$(jq -r '.stage // empty' outputs/public_pge_validation_seed42/status.json 2>/dev/null || true)"
      [[ "$state" == running && ( "$stage" == training || "$stage" == validation_inference ) ]]
      ;;
    spert_fresh)
      [[ "$(marker_status outputs/public_horizontal_validation/spert_fresh/launcher_status.json 2>/dev/null || true)" == running_spert_fresh ]]
      ;;
    glirel_t0)
      [[ "$(marker_status outputs/public_horizontal_validation/gliner_glirel_t0/status.json 2>/dev/null || true)" == running ]]
      ;;
    glirel_calibrated)
      state="$(marker_status outputs/public_horizontal_validation/gliner_glirel_calibrated/status.json 2>/dev/null || true)"
      dataset="$(jq -r '.active_dataset // empty' outputs/public_horizontal_validation/gliner_glirel_calibrated/status.json 2>/dev/null || true)"
      execution=""
      if [[ -n "$dataset" && -f outputs/public_horizontal_validation/gliner_glirel_calibrated/protocol.json ]]; then
        execution="$(jq -r --arg dataset "$dataset" '.datasets[$dataset].execution // empty' \
          outputs/public_horizontal_validation/gliner_glirel_calibrated/protocol.json 2>/dev/null || true)"
      fi
      [[ "$state" == running && "$execution" == fresh_inference ]]
      ;;
    formal_internal)
      state="$(marker_status outputs/public_formal_matrix/internal/status.json 2>/dev/null || true)"
      stage="$(jq -r '.stage // empty' outputs/public_formal_matrix/internal/status.json 2>/dev/null || true)"
      [[ "$state" == running && ( "$stage" == training || "$stage" == *_inference ) ]]
      ;;
    formal_qwen)
      state="$(marker_status outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot/status.json 2>/dev/null || true)"
      stage="$(jq -r '.stage // empty' outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot/status.json 2>/dev/null || true)"
      [[ "$state" == running && "$stage" == test_inference ]]
      ;;
    formal_glirel)
      state="$(marker_status outputs/public_formal_matrix/horizontal/gliner_glirel_calibrated/status.json 2>/dev/null || true)"
      stage="$(jq -r '.stage // empty' outputs/public_formal_matrix/horizontal/gliner_glirel_calibrated/status.json 2>/dev/null || true)"
      [[ "$state" == running && "$stage" == test_inference ]]
      ;;
    oneke_canary)
      state="$(marker_status outputs/public_external_formal/oneke/canary_launcher_status.json 2>/dev/null || true)"
      stage="$(jq -r '.stage // empty' outputs/public_external_formal/oneke/canary_launcher_status.json 2>/dev/null || true)"
      [[ "$state" == running && "$stage" == gpu_canary ]]
      ;;
    oneke_formal)
      state="$(marker_status outputs/public_external_formal/oneke/status.json 2>/dev/null || true)"
      stage="$(jq -r '.stage // empty' outputs/public_external_formal/oneke/status.json 2>/dev/null || true)"
      [[ "$state" == running && "$stage" == inference ]]
      ;;
    *)
      return 1
      ;;
  esac
}

clear_gpu_suspicion() {
  local record="$state_root/$1.gpu_suspect"
  [[ -e "$record" ]] && : > "$record"
}

check_gpu_liveness() {
  local job="$1" unit="$2" completion_marker="$3" completion_states="$4"
  local gpu_expected="$5" phase="$6"
  shift 6
  local progress_paths=("$@")
  local state substate latest_mtime progress_age active_age quiet_age now
  local restart_record last_restart=0 restart_age
  local suspect_record previous_phase="" previous_workers="" previous_mtime="0"
  local previous_strikes=0 current_strikes=0 rechecked_mtime

  if marker_complete "$completion_marker" "$completion_states"; then
    clear_gpu_suspicion "$job"
    printf '%s gpu_liveness job=%s expected=no reason=complete action=none\n' \
      "$timestamp" "$job" >> "$log"
    return 0
  fi
  if gpu_phase_is_expected "$job"; then
    gpu_expected=yes
  else
    clear_gpu_suspicion "$job"
    printf '%s gpu_liveness job=%s expected=no phase=%s reason=phase_changed action=none_after_recheck\n' \
      "$timestamp" "$job" "${phase:-unknown}" >> "$log"
    return 0
  fi
  state="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
  substate="$(systemctl --user show "$unit" -p SubState --value 2>/dev/null || true)"
  if [[ "$state" != active || "$substate" != running ]]; then
    clear_gpu_suspicion "$job"
    printf '%s gpu_liveness job=%s expected=%s phase=%s state=%s/%s action=lifecycle_repair\n' \
      "$timestamp" "$job" "$gpu_expected" "${phase:-unknown}" \
      "${state:-not-found}" "${substate:-unknown}" >> "$log"
    return 0
  fi
  if [[ "$gpu_expected" != yes ]]; then
    clear_gpu_suspicion "$job"
    printf '%s gpu_liveness job=%s expected=no phase=%s state=active/running action=none\n' \
      "$timestamp" "$job" "${phase:-unknown}" >> "$log"
    return 0
  fi
  if [[ "$gpu_query_status" != ok ]]; then
    clear_gpu_suspicion "$job"
    printf '%s gpu_liveness job=%s expected=yes phase=%s query=failed action=none\n' \
      "$timestamp" "$job" "${phase:-unknown}" >> "$log"
    return 0
  fi

  read_unit_runtime "$unit"
  if [[ -n "$runtime_cuda_pids" ]]; then
    clear_gpu_suspicion "$job"
    printf '%s gpu_liveness job=%s expected=yes phase=%s running=yes cuda_pids=%s action=none\n' \
      "$timestamp" "$job" "${phase:-unknown}" "$runtime_cuda_pids" >> "$log"
    return 0
  fi

  latest_mtime="$(latest_progress_mtime "${progress_paths[@]}")"
  now="$(date +%s)"
  active_age="$(unit_active_age_seconds "$unit")"
  if (( latest_mtime > 0 && latest_mtime <= now )); then
    progress_age="$(( now - latest_mtime ))"
  else
    progress_age="$active_age"
  fi
  quiet_age="$progress_age"
  (( active_age < quiet_age )) && quiet_age="$active_age"

  if [[ -n "$runtime_d_state_pids" ]]; then
    clear_gpu_suspicion "$job"
    printf '%s gpu_liveness job=%s expected=yes phase=%s running=no d_state_pids=%s quiet_seconds=%s action=none\n' \
      "$timestamp" "$job" "${phase:-unknown}" "$runtime_d_state_pids" "$quiet_age" >> "$log"
    return 0
  fi
  if [[ -n "$runtime_cpu_phase_pids" ]]; then
    clear_gpu_suspicion "$job"
    printf '%s gpu_liveness job=%s expected=yes phase=%s running=no cpu_phase_pids=%s quiet_seconds=%s action=none\n' \
      "$timestamp" "$job" "${phase:-unknown}" "$runtime_cpu_phase_pids" "$quiet_age" >> "$log"
    return 0
  fi
  if (( quiet_age <= gpu_liveness_grace_seconds )); then
    clear_gpu_suspicion "$job"
    printf '%s gpu_liveness job=%s expected=yes phase=%s running=no gpu_worker_pids=%s quiet_seconds=%s grace_seconds=%s action=grace\n' \
      "$timestamp" "$job" "${phase:-unknown}" "${runtime_gpu_worker_pids:-none}" \
      "$quiet_age" "$gpu_liveness_grace_seconds" >> "$log"
    return 0
  fi

  restart_record="$state_root/${job}.last_gpu_restart"
  if [[ -f "$restart_record" ]]; then
    last_restart="$(<"$restart_record")"
    [[ "$last_restart" =~ ^[0-9]+$ ]] || last_restart=0
  fi
  restart_age="$(( now - last_restart ))"
  if (( last_restart > 0 && restart_age < gpu_restart_cooldown_seconds )); then
    printf '%s gpu_liveness job=%s expected=yes phase=%s running=no quiet_seconds=%s restart_age_seconds=%s action=cooldown\n' \
      "$timestamp" "$job" "${phase:-unknown}" "$quiet_age" "$restart_age" >> "$log"
    return 0
  fi

  # Re-query immediately before recovery so a model that just reached CUDA is not interrupted.
  refresh_gpu_snapshot
  if [[ "$gpu_query_status" != ok ]]; then
    clear_gpu_suspicion "$job"
    printf '%s gpu_liveness job=%s expected=yes phase=%s running=unknown query=failed action=none_after_recheck\n' \
      "$timestamp" "$job" "${phase:-unknown}" >> "$log"
    return 0
  fi
  if marker_complete "$completion_marker" "$completion_states"; then
    clear_gpu_suspicion "$job"
    printf '%s gpu_liveness job=%s expected=no phase=%s reason=completed_during_recheck action=none\n' \
      "$timestamp" "$job" "${phase:-unknown}" >> "$log"
    return 0
  fi
  if ! gpu_phase_is_expected "$job"; then
    clear_gpu_suspicion "$job"
    printf '%s gpu_liveness job=%s expected=no phase=%s reason=phase_changed action=none_after_recheck\n' \
      "$timestamp" "$job" "${phase:-unknown}" >> "$log"
    return 0
  fi
  state="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
  substate="$(systemctl --user show "$unit" -p SubState --value 2>/dev/null || true)"
  if [[ "$state" != active || "$substate" != running ]]; then
    clear_gpu_suspicion "$job"
    printf '%s gpu_liveness job=%s expected=yes phase=%s state=%s/%s action=none_after_recheck\n' \
      "$timestamp" "$job" "${phase:-unknown}" "${state:-not-found}" \
      "${substate:-unknown}" >> "$log"
    return 0
  fi
  read_unit_runtime "$unit"
  if [[ -n "$runtime_cuda_pids" ]]; then
    clear_gpu_suspicion "$job"
    printf '%s gpu_liveness job=%s expected=yes phase=%s running=yes cuda_pids=%s action=none_after_recheck\n' \
      "$timestamp" "$job" "${phase:-unknown}" "$runtime_cuda_pids" >> "$log"
    return 0
  fi
  if [[ -n "$runtime_d_state_pids" || -n "$runtime_cpu_phase_pids" ]]; then
    clear_gpu_suspicion "$job"
    printf '%s gpu_liveness job=%s expected=yes phase=%s running=no cpu_phase_pids=%s d_state_pids=%s action=none_after_recheck\n' \
      "$timestamp" "$job" "${phase:-unknown}" "${runtime_cpu_phase_pids:-none}" \
      "${runtime_d_state_pids:-none}" >> "$log"
    return 0
  fi
  rechecked_mtime="$(latest_progress_mtime "${progress_paths[@]}")"
  if (( rechecked_mtime > latest_mtime )); then
    clear_gpu_suspicion "$job"
    printf '%s gpu_liveness job=%s expected=yes phase=%s running=no reason=progress_advanced action=none_after_recheck\n' \
      "$timestamp" "$job" "${phase:-unknown}" >> "$log"
    return 0
  fi

  # A GPU command may spend time loading/tokenizing on CPU before it creates a
  # CUDA context. Require the same worker and unchanged progress on two
  # consecutive 20-minute polls before recovering that case. If the expected
  # GPU worker vanished entirely, the stale wrapper can be recovered now.
  if [[ -n "$runtime_gpu_worker_pids" ]]; then
    suspect_record="$state_root/${job}.gpu_suspect"
    if [[ -f "$suspect_record" ]]; then
      IFS=$'\t' read -r previous_phase previous_workers previous_mtime previous_strikes \
        < "$suspect_record" || true
      [[ "$previous_mtime" =~ ^[0-9]+$ ]] || previous_mtime=0
      [[ "$previous_strikes" =~ ^[0-9]+$ ]] || previous_strikes=0
    fi
    if [[ "$previous_phase" == "$phase" &&
          "$previous_workers" == "$runtime_gpu_worker_pids" &&
          "$previous_mtime" == "$rechecked_mtime" ]]; then
      current_strikes=$(( previous_strikes + 1 ))
    else
      current_strikes=1
    fi
    printf '%s\t%s\t%s\t%s\n' "$phase" "$runtime_gpu_worker_pids" \
      "$rechecked_mtime" "$current_strikes" > "$suspect_record"
    if (( current_strikes < 2 )); then
      printf '%s gpu_liveness job=%s expected=yes phase=%s running=no gpu_worker_pids=%s quiet_seconds=%s strike=%s/2 action=observe\n' \
        "$timestamp" "$job" "${phase:-unknown}" "$runtime_gpu_worker_pids" \
        "$quiet_age" "$current_strikes" >> "$log"
      return 0
    fi
  fi
  if systemctl --user restart "$unit" >> "$log" 2>&1; then
    clear_gpu_suspicion "$job"
    printf '%s\n' "$now" > "$restart_record"
    printf '%s gpu_liveness job=%s expected=yes phase=%s running=no quiet_seconds=%s action=restarted\n' \
      "$timestamp" "$job" "${phase:-unknown}" "$quiet_age" >> "$log"
  else
    printf '%s gpu_liveness job=%s expected=yes phase=%s running=no quiet_seconds=%s action=restart_failed\n' \
      "$timestamp" "$job" "${phase:-unknown}" "$quiet_age" >> "$log"
  fi
}

glirel_t0_blocked() {
  local launcher="outputs/public_horizontal_validation/gliner_glirel_t0/launcher_status.json"
  local canary="outputs/public_horizontal_validation/gliner_glirel_t0/compatibility_canary.json"
  [[ "$(marker_status "$launcher" 2>/dev/null || true)" == "blocked_incompatible_runtime" ]] &&
    ! marker_complete "$canary" passed
}

launch_managed_job() {
  local job="$1"
  case "$job" in
    hrge)
      systemd-run --user --unit=public-hrge-cpu-preparation --collect --no-block \
        --property=Type=exec --property=Restart=on-failure --property=RestartSec=120 \
        --property=Nice=10 --property=CPUWeight=25 \
        --working-directory="$root" \
        /usr/bin/bash -lc \
        'mkdir -p outputs/public_hrge_preparation; exec >>outputs/public_hrge_preparation/runner.log 2>&1; exec /home/xuelin/miniconda3/bin/python scripts/prepare_public_hrge_cpu.py --threads 8 --batch-size 16'
      ;;
    qwen)
      systemd-run --user --unit=public-qwen-zeroshot-validation --collect --no-block \
        --property=Type=exec --property=Restart=on-failure --property=RestartSec=120 \
        --working-directory="$root" \
        /usr/bin/bash -lc \
        'mkdir -p outputs/public_horizontal_validation/qwen3_4b_zero_shot; exec >outputs/public_horizontal_validation/qwen3_4b_zero_shot/runner.log 2>&1; exec scripts/run_qwen_zeroshot_validation.sh'
      ;;
    gpu_queue)
      systemd-run --user --unit=public-gpu-validation-queue --collect --no-block \
        --property=Type=exec --property=Restart=on-failure --property=RestartSec=120 \
        --working-directory="$root" \
        /usr/bin/bash -lc \
        'mkdir -p outputs/public_gpu_validation_queue; exec >outputs/public_gpu_validation_queue/runner.log 2>&1; exec scripts/run_public_gpu_validation_queue.sh'
      ;;
    pge)
      systemd-run --user --unit=public-pge-validation --collect --no-block \
        --property=Type=exec --property=Restart=on-failure --property=RestartSec=120 \
        --working-directory="$root" \
        /usr/bin/bash -lc \
        'mkdir -p outputs/public_pge_validation_seed42; exec >outputs/public_pge_validation_seed42/runner.log 2>&1; exec scripts/run_public_pge_after_horizontal.sh'
      ;;
    spert_fresh)
      systemd-run --user --unit=public-spert-fresh-validation --collect --no-block \
        --property=Type=exec --property=Restart=on-failure --property=RestartSec=120 \
        --working-directory="$root" \
        /usr/bin/bash -lc \
        'mkdir -p outputs/public_horizontal_validation/spert_fresh; exec >outputs/public_horizontal_validation/spert_fresh/runner.log 2>&1; exec scripts/run_spert_after_pge.sh'
      ;;
    glirel_t0)
      systemd-run --user --unit=public-gliner-glirel-t0-validation --collect --no-block \
        --property=Type=exec --property=Restart=on-failure --property=RestartSec=120 \
        --working-directory="$root" \
        /usr/bin/bash -lc \
        'mkdir -p outputs/public_horizontal_validation/gliner_glirel_t0; exec >outputs/public_horizontal_validation/gliner_glirel_t0/runner.log 2>&1; exec scripts/run_gliner_glirel_t0_after_spert.sh'
      ;;
    glirel_calibration_conll04|glirel_calibration_scierc|glirel_calibration_ade)
      local dataset="${job##*_}"
      systemd-run --user --unit="public-glirel-calibration-${dataset}" --collect --no-block \
        --property=Type=exec --property=Restart=on-failure --property=RestartSec=120 \
        --property=Nice=19 --property=CPUWeight=10 \
        --working-directory="$root" \
        /usr/bin/bash -lc \
        "export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false; mkdir -p 'outputs/public_horizontal_validation/glirel_train_calibration/${dataset}'; exec >'outputs/public_horizontal_validation/glirel_train_calibration/${dataset}/runner.log' 2>&1; exec /home/xuelin/miniconda3/bin/python scripts/calibrate_glirel_train.py --dataset '${dataset}' --max-jobs 64"
      ;;
    glirel_calibrated)
      systemd-run --user --unit=public-gliner-glirel-calibrated-validation --collect --no-block \
        --property=Type=exec --property=Restart=on-failure --property=RestartSec=120 \
        --working-directory="$root" \
        /usr/bin/bash -lc \
        'mkdir -p outputs/public_horizontal_validation/gliner_glirel_calibrated; exec >outputs/public_horizontal_validation/gliner_glirel_calibrated/runner.log 2>&1; exec scripts/run_gliner_glirel_calibrated_after_t0.sh'
      ;;
    gliner_entity_only)
      systemd-run --user --unit=public-gliner-entity-only-validation --collect --no-block \
        --property=Type=exec --property=Restart=on-failure --property=RestartSec=120 \
        --property=Nice=19 --property=CPUWeight=10 \
        --working-directory="$root" \
        /usr/bin/bash -lc \
        "export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; mkdir -p outputs/public_horizontal_validation/gliner_entity_only; exec >outputs/public_horizontal_validation/gliner_entity_only/runner.log 2>&1; exec /home/xuelin/miniconda3/bin/python scripts/derive_gliner_entity_only_validation.py --overwrite"
      ;;
    post_pge)
      scripts/launch_public_post_pge_bootstrap.sh
      ;;
    validation_audit)
      systemd-run --user --unit=public-validation-audit --collect --no-block \
        --property=Type=exec --property=Restart=on-failure --property=RestartSec=120 \
        --property=Nice=19 --property=CPUWeight=10 --property=CPUQuota=100% \
        --working-directory="$root" \
        /usr/bin/bash -lc \
        "export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; mkdir -p outputs/public_validation_audit; exec >outputs/public_validation_audit/runner.log 2>&1; exec scripts/run_public_validation_audit_after_queue.sh"
      ;;
    formal_release)
      systemd-run --user --unit=public-formal-release --collect --no-block \
        --property=Type=exec --property=Restart=on-failure --property=RestartSec=120 \
        --property=Nice=10 --property=CPUWeight=25 --property=CPUQuota=800% \
        --working-directory="$root" \
        /usr/bin/bash -lc \
        'mkdir -p outputs/public_formal_matrix; exec >outputs/public_formal_matrix/release_runner.log 2>&1; exec scripts/run_public_formal_release_and_prepare.sh'
      ;;
    formal_internal)
      systemd-run --user --unit=public-formal-internal-matrix --collect --no-block \
        --property=Type=exec --property=Restart=on-failure --property=RestartSec=120 \
        --working-directory="$root" \
        /usr/bin/bash -lc \
        'mkdir -p outputs/public_formal_matrix/internal; exec >outputs/public_formal_matrix/internal/runner.log 2>&1; exec scripts/run_public_formal_internal_after_release.sh'
      ;;
    formal_qwen)
      systemd-run --user --unit=public-formal-qwen-zeroshot --collect --no-block \
        --property=Type=exec --property=Restart=on-failure --property=RestartSec=120 \
        --working-directory="$root" \
        /usr/bin/bash -lc \
        'mkdir -p outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot; exec >outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot/runner.log 2>&1; exec scripts/run_qwen_zeroshot_formal_test.sh'
      ;;
    formal_glirel)
      systemd-run --user --unit=public-formal-gliner-glirel --collect --no-block \
        --property=Type=exec --property=Restart=on-failure --property=RestartSec=120 \
        --working-directory="$root" \
        /usr/bin/bash -lc \
        'mkdir -p outputs/public_formal_matrix/horizontal/gliner_glirel_calibrated; exec >outputs/public_formal_matrix/horizontal/gliner_glirel_calibrated/runner.log 2>&1; exec scripts/run_gliner_glirel_formal_test.sh'
      ;;
    external_formal_preflight)
      systemd-run --user --unit=public-external-formal-preflight --collect --no-block \
        --property=Type=exec --property=Nice=19 --property=CPUWeight=10 \
        --property=CPUQuota=100% --working-directory="$root" \
        /usr/bin/bash -lc \
        'mkdir -p outputs/public_external_formal; exec >outputs/public_external_formal/preflight_runner.log 2>&1; exec /home/xuelin/miniconda3/bin/python scripts/preflight_public_external_formal.py'
      ;;
    oneke_canary)
      systemd-run --user --unit=public-formal-oneke-canary --collect --no-block \
        --property=Type=exec \
        --working-directory="$root" \
        /usr/bin/bash -lc \
        'mkdir -p outputs/public_external_formal/oneke; exec >outputs/public_external_formal/oneke/canary.log 2>&1; exec scripts/run_public_oneke_gpu_canary.sh'
      ;;
    oneke_formal)
      systemd-run --user --unit=public-formal-oneke-validation --collect --no-block \
        --property=Type=exec --property=Restart=on-failure --property=RestartSec=120 \
        --working-directory="$root" \
        /usr/bin/bash -lc \
        'mkdir -p outputs/public_external_formal/oneke; exec >outputs/public_external_formal/oneke/runner.log 2>&1; exec scripts/run_public_oneke_formal.sh'
      ;;
    *)
      return 2
      ;;
  esac
}

repair_managed_job() {
  local job="$1" unit="$2" marker="$3" expected="$4"
  shift 4
  local progress_paths=("$@")
  local state age=0 current_mtime=0
  marker_complete "$marker" "$expected" && return 0
  state="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
  if [[ "$state" == "active" || "$state" == "activating" ]]; then
    current_mtime="$(latest_progress_mtime "${progress_paths[@]}")"
    if (( current_mtime > 0 )); then
      age=$(( $(date +%s) - current_mtime ))
    fi
    printf '%s managed_job=%s state=%s progress_age_seconds=%s action=none\n' \
      "$timestamp" "$job" "$state" "$age" >> "$log"
    return 0
  fi
  printf '%s repair managed_job=%s state=%s marker_status=%s\n' \
    "$timestamp" "$job" "${state:-not-found}" \
    "$(marker_status "$marker" 2>/dev/null || printf missing)" >> "$log"
  if systemctl --user cat "$unit" >/dev/null 2>&1; then
    systemctl --user restart "$unit" >> "$log" 2>&1 || true
  else
    launch_managed_job "$job" >> "$log" 2>&1 || true
  fi
}

repair_download_lane() {
  local lane="$1"
  local unit="public-model-download-${lane}.service"
  local lane_log="outputs/public_baseline_downloads/${lane}.log"
  local state
  local age=0
  local current_mtime=0
  lane_complete "$lane" && return 0
  state="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
  if [[ -f "$lane_log" ]]; then
    current_mtime="$(stat -c %Y "$lane_log")"
    age=$(( $(date +%s) - current_mtime ))
  fi
  if [[ "$state" == "active" && "$current_mtime" -gt 0 && "$age" -le 2700 ]]; then
    return 0
  fi
  printf '%s repair download lane=%s state=%s log_age_seconds=%s\n' \
    "$timestamp" "$lane" "${state:-not-found}" "$age" >> "$log"
  if systemctl --user cat "$unit" >/dev/null 2>&1; then
    systemctl --user restart "$unit" >> "$log" 2>&1 || true
  else
    launch_download_lane "$lane" >> "$log" 2>&1 || true
  fi
}

experiment_status() {
  local unit="$1"
  local marker="$2"
  local expected="$3"
  local state result current_marker_status complete=no
  state="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
  result="$(systemctl --user show "$unit" -p Result --value 2>/dev/null || true)"
  current_marker_status="$(marker_status "$marker" 2>/dev/null || printf missing)"
  marker_complete "$marker" "$expected" && complete=yes
  printf '%s experiment unit=%s state=%s result=%s marker_status=%s complete=%s\n' \
    "$timestamp" "$unit" "${state:-not-found}" "${result:-unknown}" \
    "$current_marker_status" "$complete" >> "$log"
}

{
  printf '%s watchdog begin\n' "$timestamp"
  bash scripts/public_baseline_download_status.sh 2>&1
} >> "$log"

repair_download_lane hf
repair_download_lane spert
repair_download_lane repos

repair_managed_job \
  hrge public-hrge-cpu-preparation.service \
  data/processed/public_benchmarks_hrge_v1/ade/preparation_manifest.json \
  prepared_train_and_validation outputs/public_hrge_preparation/runner.log
repair_managed_job \
  qwen public-qwen-zeroshot-validation.service \
  outputs/public_horizontal_validation/qwen3_4b_zero_shot/status.json \
  validation_complete outputs/public_horizontal_validation/qwen3_4b_zero_shot/runner.log
repair_managed_job \
  gpu_queue public-gpu-validation-queue.service \
  outputs/public_gpu_validation_queue/status.json 'complete|complete_with_terminal_failures' \
  outputs/public_gpu_validation_queue/status.json
repair_managed_job \
  pge public-pge-validation.service \
  outputs/public_pge_validation_seed42/status.json complete \
  outputs/public_pge_validation_seed42/launcher_status.json \
  outputs/public_pge_validation_seed42/status.json \
  outputs/public_pge_validation_seed42/runner.log
repair_managed_job \
  spert_fresh public-spert-fresh-validation.service \
  outputs/public_horizontal_validation/spert_fresh/status.json complete \
  outputs/public_horizontal_validation/spert_fresh/launcher_status.json \
  outputs/public_horizontal_validation/spert_fresh/status.json \
  outputs/public_horizontal_validation/spert_fresh/runner.log
if glirel_t0_blocked; then
  printf '%s managed_job=glirel_t0 status=blocked_incompatible_runtime action=none\n' \
    "$timestamp" >> "$log"
else
  repair_managed_job \
    glirel_t0 public-gliner-glirel-t0-validation.service \
    outputs/public_horizontal_validation/gliner_glirel_t0/status.json complete \
    outputs/public_horizontal_validation/gliner_glirel_t0/launcher_status.json \
    outputs/public_horizontal_validation/gliner_glirel_t0/compatibility_canary.json \
    outputs/public_horizontal_validation/gliner_glirel_t0/status.json \
    outputs/public_horizontal_validation/gliner_glirel_t0/runner.log
fi
for dataset in conll04 scierc ade; do
  repair_managed_job \
    "glirel_calibration_${dataset}" "public-glirel-calibration-${dataset}.service" \
    "outputs/public_horizontal_validation/glirel_train_calibration/${dataset}/status.json" complete \
    "outputs/public_horizontal_validation/glirel_train_calibration/${dataset}/status.json" \
    "outputs/public_horizontal_validation/glirel_train_calibration/${dataset}/raw_scores.jsonl" \
    "outputs/public_horizontal_validation/glirel_train_calibration/${dataset}/runner.log"
done
repair_managed_job \
  glirel_calibrated public-gliner-glirel-calibrated-validation.service \
  outputs/public_horizontal_validation/gliner_glirel_calibrated/status.json complete \
  outputs/public_horizontal_validation/gliner_glirel_calibrated/launcher_status.json \
  outputs/public_horizontal_validation/gliner_glirel_calibrated/protocol.json \
  outputs/public_horizontal_validation/gliner_glirel_calibrated/status.json \
  outputs/public_horizontal_validation/gliner_glirel_calibrated/runner.log
if marker_complete outputs/public_horizontal_validation/gliner_glirel/status.json complete; then
  repair_managed_job \
    gliner_entity_only public-gliner-entity-only-validation.service \
    outputs/public_horizontal_validation/gliner_entity_only/status.json complete \
    outputs/public_horizontal_validation/gliner_entity_only/status.json \
    outputs/public_horizontal_validation/gliner_entity_only/runner.log
else
  printf '%s managed_job=gliner_entity_only status=waiting_for_gliner_glirel action=none\n' \
    "$timestamp" >> "$log"
fi
repair_managed_job \
  post_pge public-post-pge-bootstrap.service \
  outputs/public_post_pge_validation_seed42/status.json complete \
  outputs/public_post_pge_validation_seed42/status.json
repair_managed_job \
  validation_audit public-validation-audit.service \
  outputs/public_validation_audit/status.json complete \
  outputs/public_validation_audit/launcher_status.json \
  outputs/public_validation_audit/status.json \
  outputs/public_validation_audit/runner.log
repair_managed_job \
  external_formal_preflight public-external-formal-preflight.service \
  outputs/public_external_formal/preflight_status.json 'complete|ready_to_queue' \
  outputs/public_external_formal/preflight_status.json \
  outputs/public_external_formal/preflight_runner.log \
  outputs/public_external_formal/queue.jsonl
oneke_environment_status="$(marker_status outputs/public_external_formal/oneke/environment_status.json 2>/dev/null || true)"
oneke_download_status="$(jq -r '.components.oneke.status // empty' \
  outputs/public_baseline_downloads/integrity_status.json 2>/dev/null || true)"
oneke_canary_launcher_status="$(marker_status outputs/public_external_formal/oneke/canary_launcher_status.json 2>/dev/null || true)"
oneke_canary_launcher_stage="$(jq -r '.stage // empty' \
  outputs/public_external_formal/oneke/canary_launcher_status.json 2>/dev/null || true)"
oneke_canary_retry_ready="$(jq -r '.retry_ready // false' \
  outputs/public_external_formal/oneke/canary_launcher_status.json 2>/dev/null || printf false)"
if [[ "$oneke_canary_launcher_status" == blocked_with_reason &&
      "$oneke_canary_launcher_stage" == failed &&
      "$oneke_canary_retry_ready" != true ]]; then
  printf '%s managed_job=oneke_canary status=terminal_failure action=await_repair_rearm\n' \
    "$timestamp" >> "$log"
elif [[ "$oneke_environment_status" == complete && "$oneke_download_status" == ready ]]; then
  repair_managed_job \
    oneke_canary public-formal-oneke-canary.service \
    outputs/public_external_formal/oneke/gpu_canary.json passed \
    outputs/public_external_formal/oneke/environment_status.json \
    outputs/public_external_formal/oneke/canary_launcher_status.json \
    outputs/public_external_formal/oneke/gpu_canary.json \
    outputs/public_external_formal/oneke/canary.log
else
  printf '%s managed_job=oneke_canary environment=%s download=%s action=none\n' \
    "$timestamp" "${oneke_environment_status:-missing}" \
    "${oneke_download_status:-missing}" >> "$log"
fi
oneke_static_status="$(jq -r '.baselines.oneke.status // empty' \
  outputs/public_external_formal/preflight_status.json 2>/dev/null || true)"
if marker_complete outputs/public_external_formal/oneke/gpu_canary.json passed && \
   [[ "$oneke_static_status" == ready_to_queue ]]; then
  repair_managed_job \
    oneke_formal public-formal-oneke-validation.service \
    outputs/public_external_formal/oneke/status.json complete \
    outputs/public_external_formal/oneke/status.json \
    outputs/public_external_formal/oneke/runner.log \
    outputs/public_external_formal/oneke/conll04 \
    outputs/public_external_formal/oneke/scierc \
    outputs/public_external_formal/oneke/ade
else
  printf '%s managed_job=oneke_formal canary=%s preflight=%s action=none\n' \
    "$timestamp" \
    "$(marker_status outputs/public_external_formal/oneke/gpu_canary.json 2>/dev/null || printf missing)" \
    "${oneke_static_status:-missing}" >> "$log"
fi
formal_gate_review="$(marker_status outputs/public_formal_matrix/gate_review_status.json 2>/dev/null || true)"
formal_gate_release_allowed="$(jq -r '.formal_gpu_release_allowed // false' \
  outputs/public_formal_matrix/gate_review_status.json 2>/dev/null || printf false)"
if [[ "$formal_gate_review" != passed || "$formal_gate_release_allowed" != true ]]; then
  printf '%s managed_job=formal_release status=%s release_allowed=%s action=none\n' \
    "$timestamp" "${formal_gate_review:-missing}" "$formal_gate_release_allowed" >> "$log"
  printf '%s managed_job=formal_internal status=%s release_allowed=%s action=none\n' \
    "$timestamp" "${formal_gate_review:-missing}" "$formal_gate_release_allowed" >> "$log"
  printf '%s managed_job=formal_qwen status=%s release_allowed=%s action=none\n' \
    "$timestamp" "${formal_gate_review:-missing}" "$formal_gate_release_allowed" >> "$log"
else
  repair_managed_job \
    formal_release public-formal-release.service \
    outputs/public_formal_matrix/release_status.json complete \
    outputs/public_formal_matrix/release_status.json \
    outputs/public_formal_matrix/release_runner.log \
    data/processed/public_benchmarks_hrge_test_v1
  repair_managed_job \
    formal_internal public-formal-internal-matrix.service \
    outputs/public_formal_matrix/internal/status.json complete \
    outputs/public_formal_matrix/internal_launcher_status.json \
    outputs/public_formal_matrix/internal/status.json \
    outputs/public_formal_matrix/internal
  if marker_complete outputs/public_formal_matrix/release_status.json complete; then
    repair_managed_job \
      formal_qwen public-formal-qwen-zeroshot.service \
      outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot/status.json complete \
      outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot/status.json \
      outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot/runner.log \
      outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot
  else
    printf '%s managed_job=formal_qwen status=waiting_for_formal_release action=none\n' \
      "$timestamp" >> "$log"
  fi
  repair_managed_job \
    formal_glirel public-formal-gliner-glirel.service \
    outputs/public_formal_matrix/horizontal/gliner_glirel_calibrated/status.json complete \
    outputs/public_formal_matrix/horizontal/gliner_glirel_calibrated/status.json \
    outputs/public_formal_matrix/horizontal/gliner_glirel_calibrated
fi

# Query once per watchdog cycle, then attribute CUDA processes to the owning
# systemd cgroup. Global utilization alone is not a safe liveness signal.
refresh_gpu_snapshot
printf '%s gpu_poll query=%s compute_pids=%s interval_minutes=20\n' \
  "$timestamp" "$gpu_query_status" "${gpu_compute_pids//$'\n'/,}" >> "$log"

qwen_expected=no
qwen_phase="service_inactive_or_complete"
if [[ "$(systemctl --user is-active public-qwen-zeroshot-validation.service 2>/dev/null || true)" == active ]] && \
   ! marker_complete outputs/public_horizontal_validation/qwen3_4b_zero_shot/status.json validation_complete; then
  qwen_expected=yes
  qwen_phase="active_generation"
fi
check_gpu_liveness \
  qwen public-qwen-zeroshot-validation.service \
  outputs/public_horizontal_validation/qwen3_4b_zero_shot/status.json validation_complete \
  "$qwen_expected" "$qwen_phase" \
  outputs/public_horizontal_validation/qwen3_4b_zero_shot

queue_state="$(marker_status outputs/public_gpu_validation_queue/status.json 2>/dev/null || true)"
queue_stage="$(jq -r '.stage // empty' outputs/public_gpu_validation_queue/status.json 2>/dev/null || true)"
queue_expected=no
[[ "$queue_state" == running ]] && queue_expected=yes
check_gpu_liveness \
  gpu_queue public-gpu-validation-queue.service \
  outputs/public_gpu_validation_queue/status.json 'complete|complete_with_terminal_failures' \
  "$queue_expected" "${queue_state:-missing}:${queue_stage:-unknown}" \
  outputs/public_gpu_validation_queue outputs/public_horizontal_validation/gliner_glirel

pge_state="$(marker_status outputs/public_pge_validation_seed42/status.json 2>/dev/null || true)"
pge_stage="$(jq -r '.stage // empty' outputs/public_pge_validation_seed42/status.json 2>/dev/null || true)"
pge_expected=no
if [[ "$pge_state" == running && ( "$pge_stage" == training || "$pge_stage" == validation_inference ) ]]; then
  pge_expected=yes
fi
check_gpu_liveness \
  pge public-pge-validation.service \
  outputs/public_pge_validation_seed42/status.json complete \
  "$pge_expected" "${pge_state:-missing}:${pge_stage:-unknown}" \
  outputs/public_pge_validation_seed42

spert_phase="$(marker_status outputs/public_horizontal_validation/spert_fresh/launcher_status.json 2>/dev/null || true)"
spert_expected=no
[[ "$spert_phase" == running_spert_fresh ]] && spert_expected=yes
check_gpu_liveness \
  spert_fresh public-spert-fresh-validation.service \
  outputs/public_horizontal_validation/spert_fresh/status.json complete \
  "$spert_expected" "${spert_phase:-missing}" \
  outputs/public_horizontal_validation/spert_fresh

t0_state="$(marker_status outputs/public_horizontal_validation/gliner_glirel_t0/status.json 2>/dev/null || true)"
t0_dataset="$(jq -r '.active_dataset // empty' outputs/public_horizontal_validation/gliner_glirel_t0/status.json 2>/dev/null || true)"
t0_expected=no
[[ "$t0_state" == running ]] && t0_expected=yes
check_gpu_liveness \
  glirel_t0 public-gliner-glirel-t0-validation.service \
  outputs/public_horizontal_validation/gliner_glirel_t0/status.json complete \
  "$t0_expected" "${t0_state:-missing}:${t0_dataset:-none}" \
  outputs/public_horizontal_validation/gliner_glirel_t0

calibrated_state="$(marker_status outputs/public_horizontal_validation/gliner_glirel_calibrated/status.json 2>/dev/null || true)"
calibrated_dataset="$(jq -r '.active_dataset // empty' outputs/public_horizontal_validation/gliner_glirel_calibrated/status.json 2>/dev/null || true)"
calibrated_execution=""
if [[ -n "$calibrated_dataset" && -f outputs/public_horizontal_validation/gliner_glirel_calibrated/protocol.json ]]; then
  calibrated_execution="$(jq -r --arg dataset "$calibrated_dataset" '.datasets[$dataset].execution // empty' \
    outputs/public_horizontal_validation/gliner_glirel_calibrated/protocol.json 2>/dev/null || true)"
fi
calibrated_expected=no
if [[ "$calibrated_state" == running && "$calibrated_execution" == fresh_inference ]]; then
  calibrated_expected=yes
fi
check_gpu_liveness \
  glirel_calibrated public-gliner-glirel-calibrated-validation.service \
  outputs/public_horizontal_validation/gliner_glirel_calibrated/status.json complete \
  "$calibrated_expected" \
  "${calibrated_state:-missing}:${calibrated_dataset:-none}:${calibrated_execution:-unknown}" \
  outputs/public_horizontal_validation/gliner_glirel_calibrated

formal_internal_state="$(marker_status outputs/public_formal_matrix/internal/status.json 2>/dev/null || true)"
formal_internal_stage="$(jq -r '.stage // empty' outputs/public_formal_matrix/internal/status.json 2>/dev/null || true)"
formal_internal_expected=no
if [[ "$formal_internal_state" == running && \
      ( "$formal_internal_stage" == training || "$formal_internal_stage" == *_inference ) ]]; then
  formal_internal_expected=yes
fi
check_gpu_liveness \
  formal_internal public-formal-internal-matrix.service \
  outputs/public_formal_matrix/internal/status.json complete \
  "$formal_internal_expected" \
  "${formal_internal_state:-missing}:${formal_internal_stage:-unknown}" \
  outputs/public_formal_matrix/internal/status.json \
  outputs/public_formal_matrix/internal

formal_qwen_state="$(marker_status outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot/status.json 2>/dev/null || true)"
formal_qwen_stage="$(jq -r '.stage // empty' outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot/status.json 2>/dev/null || true)"
formal_qwen_expected=no
if [[ "$formal_qwen_state" == running && "$formal_qwen_stage" == test_inference ]]; then
  formal_qwen_expected=yes
fi
check_gpu_liveness \
  formal_qwen public-formal-qwen-zeroshot.service \
  outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot/status.json complete \
  "$formal_qwen_expected" \
  "${formal_qwen_state:-missing}:${formal_qwen_stage:-unknown}" \
  outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot/status.json \
  outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot/runner.log \
  outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot

formal_glirel_state="$(marker_status outputs/public_formal_matrix/horizontal/gliner_glirel_calibrated/status.json 2>/dev/null || true)"
formal_glirel_stage="$(jq -r '.stage // empty' outputs/public_formal_matrix/horizontal/gliner_glirel_calibrated/status.json 2>/dev/null || true)"
formal_glirel_expected=no
if [[ "$formal_glirel_state" == running && "$formal_glirel_stage" == test_inference ]]; then
  formal_glirel_expected=yes
fi
check_gpu_liveness \
  formal_glirel public-formal-gliner-glirel.service \
  outputs/public_formal_matrix/horizontal/gliner_glirel_calibrated/status.json complete \
  "$formal_glirel_expected" \
  "${formal_glirel_state:-missing}:${formal_glirel_stage:-unknown}" \
  outputs/public_formal_matrix/horizontal/gliner_glirel_calibrated/status.json \
  outputs/public_formal_matrix/horizontal/gliner_glirel_calibrated

oneke_canary_state="$(marker_status outputs/public_external_formal/oneke/canary_launcher_status.json 2>/dev/null || true)"
oneke_canary_stage="$(jq -r '.stage // empty' outputs/public_external_formal/oneke/canary_launcher_status.json 2>/dev/null || true)"
oneke_canary_expected=no
if [[ "$oneke_canary_state" == running && "$oneke_canary_stage" == gpu_canary ]]; then
  oneke_canary_expected=yes
fi
check_gpu_liveness \
  oneke_canary public-formal-oneke-canary.service \
  outputs/public_external_formal/oneke/gpu_canary.json passed \
  "$oneke_canary_expected" \
  "${oneke_canary_state:-missing}:${oneke_canary_stage:-unknown}" \
  outputs/public_external_formal/oneke/canary_launcher_status.json \
  outputs/public_external_formal/oneke/gpu_canary.json \
  outputs/public_external_formal/oneke/canary.log

oneke_formal_state="$(marker_status outputs/public_external_formal/oneke/status.json 2>/dev/null || true)"
oneke_formal_stage="$(jq -r '.stage // empty' outputs/public_external_formal/oneke/status.json 2>/dev/null || true)"
oneke_formal_expected=no
if [[ "$oneke_formal_state" == running && "$oneke_formal_stage" == inference ]]; then
  oneke_formal_expected=yes
fi
check_gpu_liveness \
  oneke_formal public-formal-oneke-validation.service \
  outputs/public_external_formal/oneke/status.json complete \
  "$oneke_formal_expected" \
  "${oneke_formal_state:-missing}:${oneke_formal_stage:-unknown}" \
  outputs/public_external_formal/oneke/status.json \
  outputs/public_external_formal/oneke/runner.log \
  outputs/public_external_formal/oneke/conll04 \
  outputs/public_external_formal/oneke/scierc \
  outputs/public_external_formal/oneke/ade

experiment_status \
  public-validation-analysis.service \
  outputs/public_full_stage1/validation_analysis/status.json complete
experiment_status \
  public-hrge-cpu-preparation.service \
  data/processed/public_benchmarks_hrge_v1/ade/preparation_manifest.json \
  prepared_train_and_validation
experiment_status \
  public-gpu-validation-queue.service \
  outputs/public_gpu_validation_queue/status.json 'complete|complete_with_terminal_failures'
experiment_status \
  public-pge-validation.service \
  outputs/public_pge_validation_seed42/status.json complete
experiment_status \
  public-spert-fresh-validation.service \
  outputs/public_horizontal_validation/spert_fresh/status.json complete
experiment_status \
  public-gliner-glirel-t0-validation.service \
  outputs/public_horizontal_validation/gliner_glirel_t0/status.json complete
for dataset in conll04 scierc ade; do
  experiment_status \
    "public-glirel-calibration-${dataset}.service" \
    "outputs/public_horizontal_validation/glirel_train_calibration/${dataset}/status.json" complete
done
experiment_status \
  public-gliner-glirel-calibrated-validation.service \
  outputs/public_horizontal_validation/gliner_glirel_calibrated/status.json complete
experiment_status \
  public-gliner-entity-only-validation.service \
  outputs/public_horizontal_validation/gliner_entity_only/status.json complete
experiment_status \
  public-post-pge-bootstrap.service \
  outputs/public_post_pge_validation_seed42/status.json complete
experiment_status \
  public-validation-audit.service \
  outputs/public_validation_audit/status.json complete
experiment_status \
  public-external-formal-preflight.service \
  outputs/public_external_formal/preflight_status.json 'complete|ready_to_queue'
experiment_status \
  public-formal-oneke-canary.service \
  outputs/public_external_formal/oneke/gpu_canary.json passed
experiment_status \
  public-formal-oneke-validation.service \
  outputs/public_external_formal/oneke/status.json complete
experiment_status \
  public-formal-release.service \
  outputs/public_formal_matrix/release_status.json complete
experiment_status \
  public-formal-internal-matrix.service \
  outputs/public_formal_matrix/internal/status.json complete
experiment_status \
  public-formal-qwen-zeroshot.service \
  outputs/public_formal_matrix/horizontal/qwen3_4b_zero_shot/status.json complete
experiment_status \
  public-formal-gliner-glirel.service \
  outputs/public_formal_matrix/horizontal/gliner_glirel_calibrated/status.json complete
experiment_status \
  public-gliner-validation.service \
  outputs/public_horizontal_validation/gliner/status.json complete
experiment_status \
  public-glirel-validation.service \
  outputs/public_horizontal_validation/gliner_glirel/status.json complete
experiment_status \
  public-qwen-zeroshot-validation.service \
  outputs/public_horizontal_validation/qwen3_4b_zero_shot/status.json validation_complete

{
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader 2>&1 || true
  df -h /ds1 /ds2 2>&1 || true
  printf '%s watchdog end\n' "$timestamp"
} >> "$log"

tail -80 "$log" > "$state_root/latest.log"
