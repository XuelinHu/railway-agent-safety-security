#!/usr/bin/env bash
# Freeze train-only calibration choices, then run after the canonical t0 arm.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

poll_seconds="${GLIREL_CALIBRATED_QUEUE_POLL_SECONDS:-30}"
t0_status="outputs/public_horizontal_validation/gliner_glirel_t0/status.json"
calibration_root="outputs/public_horizontal_validation/glirel_train_calibration"
run_root="outputs/public_horizontal_validation/gliner_glirel_calibrated"
launcher_status="$run_root/launcher_status.json"
protocol="$run_root/protocol.json"
mkdir -p "$run_root"

if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "GLIREL_CALIBRATED_QUEUE_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

write_status() {
  local state="$1"
  python3 - "$launcher_status" "$state" "$poll_seconds" "$t0_status" \
    "$calibration_root" "$run_root" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
state = sys.argv[2]
poll_seconds = int(sys.argv[3])
t0_path = Path(sys.argv[4])
calibration_root = Path(sys.argv[5])
run_root = Path(sys.argv[6])

def marker(path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status")
    except (OSError, json.JSONDecodeError):
        return "invalid"

payload = {
    "status": state,
    "queue_order": ["spert_fresh_seed42", "gliner_glirel_t0", "gliner_glirel_calibrated"],
    "waits_for": {
        "t0": f"{t0_path}:complete",
        "train_calibration": {
            dataset: f"{calibration_root / dataset / 'status.json'}:complete"
            for dataset in ("conll04", "scierc", "ade")
        },
    },
    "upstream": {
        "t0": marker(t0_path),
        "train_calibration": {
            dataset: marker(calibration_root / dataset / "status.json")
            for dataset in ("conll04", "scierc", "ade")
        },
    },
    "output_root": str(run_root),
    "poll_seconds": poll_seconds,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temporary = target.with_name(f".{target.name}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(target)
PY
}

on_exit() {
  local code="$?"
  if (( code != 0 )); then
    write_status failed || true
  fi
}
trap on_exit EXIT

write_status waiting_for_t0_and_calibration
while :; do
  t0="$(jq -r '.status // ""' "$t0_status" 2>/dev/null || true)"
  ready=yes
  [[ "$t0" == complete ]] || ready=no
  for dataset in conll04 scierc ade; do
    state="$(jq -r '.status // ""' "$calibration_root/$dataset/status.json" 2>/dev/null || true)"
    [[ "$state" == complete ]] || ready=no
  done
  [[ "$ready" == yes ]] && break
  sleep "$poll_seconds"
  write_status waiting_for_t0_and_calibration
done

python3 - "$protocol" "$calibration_root" "$run_root" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
calibration_root = Path(sys.argv[2])
run_root = Path(sys.argv[3])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

datasets = {}
for dataset in ("conll04", "scierc", "ade"):
    calibration_path = calibration_root / dataset / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("status") != "complete":
        raise RuntimeError(f"calibration is incomplete for {dataset}")
    selected = calibration["selected_configuration"]
    label_mode = selected["label_mode"]
    threshold = float(selected["threshold"])
    reuse_t0 = label_mode == "canonical" and threshold == 0.0
    datasets[dataset] = {
        "label_mode": label_mode,
        "relation_threshold": threshold,
        "entity_threshold": 0.5,
        "selection_split": "train_inner_calibration",
        "gold_ner_oracle_calibration_f1": selected["f1"],
        "calibration": str(calibration_path),
        "calibration_sha256": sha256(calibration_path),
        "execution": "reuse_canonical_t0_predictions" if reuse_t0 else "fresh_inference",
    }

frozen = {
    "schema_version": "gliner-glirel-calibrated-validation-v1",
    "selection_split": "train_inner_calibration",
    "validation_gold_used_for_selection": False,
    "test_gold_used_for_selection": False,
    "entity_threshold": 0.5,
    "top_k_after_signature_filter": 1,
    "datasets": datasets,
}
if target.is_file():
    existing = json.loads(target.read_text(encoding="utf-8"))
    comparable = {key: value for key, value in existing.items() if key != "frozen_at"}
    if comparable != frozen:
        raise RuntimeError("existing calibrated protocol differs from current train-only selection")
else:
    frozen["frozen_at"] = datetime.now(timezone.utc).isoformat()
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
PY

write_status starting_calibrated_validation
exec scripts/run_gliner_glirel_calibrated_validation.sh
