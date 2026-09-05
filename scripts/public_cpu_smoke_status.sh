#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

status_file="outputs/public_cpu_smoke/status.json"
printf '%s\n' '--- CPU-only public smoke ---'
printf 'service=%s\n' "$(systemctl --user is-active public-cpu-smoke.service 2>/dev/null || true)"
if [[ -f "$status_file" ]]; then
  jq -r '
    if .schema_version == "public-cpu-smoke-v1" then
      "status=\(.status) stage=\(.active_stage) mode=\(.mode) run_id=\(.run_id)",
      "python=\(.environment.python) threads=\(.environment.torch_threads) cuda_available=\(.environment.torch_cuda_available)",
      "pytest=\(.pytest.status) tests=\(.pytest.tests // 0) failures=\(.pytest.failures // 0) errors=\(.pytest.errors // 0)",
      "gliner_glirel=\(.gliner_glirel.status) gpu_runtime=\(.gpu_runtime.status)",
      (.datasets | to_entries[] | "\(.key)\t\(.value.status)\tpreprocessing=\(.value.preprocessing)\tevaluation=\(.value.evaluation.status)")
    else
      "status=\(.status) current=\(.current_check // "none") device=\(.execution.device) threads=\(.execution.thread_limit)",
      (.checks[] | "\(.check)\t\(.severity)\t\(.result)\t\(.elapsed_seconds)s")
    end
  ' "$status_file"
else
  printf '%s\n' 'status=not_started'
fi
