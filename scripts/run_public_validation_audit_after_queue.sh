#!/usr/bin/env bash
# Wait without using a GPU, then run the closed-registry validation audit.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

run_root="${PUBLIC_VALIDATION_AUDIT_ROOT:-outputs/public_validation_audit}"
launcher_status="$run_root/launcher_status.json"
audit_status="$run_root/status.json"
poll_seconds="${PUBLIC_VALIDATION_AUDIT_POLL_SECONDS:-30}"
python_bin="${PUBLIC_VALIDATION_AUDIT_PYTHON:-/home/xuelin/miniconda3/bin/python}"
mkdir -p "$run_root"

if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "PUBLIC_VALIDATION_AUDIT_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Python interpreter is not executable: $python_bin" >&2
  exit 2
fi

upstream_ids=(
  stage1_analysis
  qwen_generation
  qwen_retry
  pge
  spert_fresh
  gliner_glirel_threshold_05
  gliner_glirel_canonical_t0
  gliner_glirel_train_calibrated
  gliner_entity_only
  post_pge_bootstrap
  glirel_calibration_conll04
  glirel_calibration_scierc
  glirel_calibration_ade
)
upstream_paths=(
  outputs/public_full_stage1/validation_analysis/status.json
  outputs/public_horizontal_validation/qwen3_4b_zero_shot/status.json
  outputs/public_horizontal_validation/qwen3_4b_zero_shot/retry_status.json
  outputs/public_pge_validation_seed42/status.json
  outputs/public_horizontal_validation/spert_fresh/status.json
  outputs/public_horizontal_validation/gliner_glirel/status.json
  outputs/public_horizontal_validation/gliner_glirel_t0/status.json
  outputs/public_horizontal_validation/gliner_glirel_calibrated/status.json
  outputs/public_horizontal_validation/gliner_entity_only/status.json
  outputs/public_post_pge_validation_seed42/status.json
  outputs/public_horizontal_validation/glirel_train_calibration/conll04/status.json
  outputs/public_horizontal_validation/glirel_train_calibration/scierc/status.json
  outputs/public_horizontal_validation/glirel_train_calibration/ade/status.json
)
upstream_terminal=(
  complete
  validation_complete
  'complete|complete_with_terminal_failures'
  complete
  complete
  complete
  complete
  complete
  complete
  complete
  complete
  complete
  complete
)

write_launcher_status() {
  local state="$1"
  local arguments=()
  local index
  for ((index=0; index<${#upstream_ids[@]}; index++)); do
    arguments+=(
      "${upstream_ids[$index]}"
      "${upstream_paths[$index]}"
      "${upstream_terminal[$index]}"
    )
  done
  "$python_bin" - "$launcher_status" "$state" "$poll_seconds" "${arguments[@]}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
state = sys.argv[2]
poll_seconds = int(sys.argv[3])
triples = sys.argv[4:]
if len(triples) % 3:
    raise RuntimeError("invalid upstream launcher registry")

upstreams = {}
ready = True
for offset in range(0, len(triples), 3):
    upstream_id, raw_path, expected = triples[offset : offset + 3]
    path = Path(raw_path)
    observed = None
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            observed = value.get("status") if isinstance(value, dict) else "invalid"
        except (OSError, json.JSONDecodeError):
            observed = "invalid"
    accepted = expected.split("|")
    complete = observed in accepted
    ready = ready and complete
    upstreams[upstream_id] = {
        "path": raw_path,
        "expected_status": accepted,
        "observed_status": observed,
        "ready": complete,
    }

payload = {
    "schema_version": "public-validation-audit-launcher-v1",
    "status": state,
    "selection_split": "validation",
    "formal_test_read": False,
    "test_namespace_status": "sealed_not_read",
    "gpu_used": False,
    "poll_seconds": poll_seconds,
    "ready": ready,
    "upstreams": upstreams,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
}

all_ready() {
  local index observed candidate matched
  for ((index=0; index<${#upstream_ids[@]}; index++)); do
    observed="$(jq -r '.status // empty' "${upstream_paths[$index]}" 2>/dev/null || true)"
    matched=no
    IFS='|' read -r -a candidates <<< "${upstream_terminal[$index]}"
    for candidate in "${candidates[@]}"; do
      [[ "$observed" == "$candidate" ]] && matched=yes
    done
    [[ "$matched" == yes ]] || return 1
  done
}

write_launcher_status waiting_for_upstreams
while ! all_ready; do
  sleep "$poll_seconds"
  write_launcher_status waiting_for_upstreams
done

write_launcher_status running_audit
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

if "$python_bin" scripts/audit_public_validation_results.py --output-root "$run_root"; then
  write_launcher_status complete
else
  code="$?"
  write_launcher_status failed_audit
  exit "$code"
fi
