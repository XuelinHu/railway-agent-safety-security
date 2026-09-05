#!/usr/bin/env bash
# Open the audited formal-test gate and prepare leakage-safe test prompts.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

python="${PUBLIC_FORMAL_PYTHON:-/home/xuelin/miniconda3/bin/python}"
run_root="outputs/public_formal_matrix"
status_file="$run_root/release_status.json"
promotion="$run_root/promotion.json"
gate_review="$run_root/gate_review_status.json"
release_attestation="$run_root/prepared_release_attestation.json"
test_jobs_root="data/processed/public_benchmarks_hrge_test_v1"
current_stage="initializing"
mkdir -p "$run_root"

write_status() {
  local state="$1" error="${2:-}"
  "$python" - "$status_file" "$state" "$current_stage" "$promotion" \
    "$test_jobs_root" "$error" "$gate_review" "$release_attestation" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
promotion = Path(sys.argv[4])
test_root = Path(sys.argv[5])
gate_review = Path(sys.argv[7])
release_attestation = Path(sys.argv[8])
datasets = {}
for dataset in ("conll04", "scierc", "ade"):
    manifest = test_root / dataset / "preparation_manifest.json"
    status = "missing"
    if manifest.is_file():
        try:
            status = json.loads(manifest.read_text(encoding="utf-8")).get("status", "unknown")
        except Exception:
            status = "invalid"
    datasets[dataset] = {
        "preparation_status": status,
        "manifest": str(manifest),
    }
payload = {
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "promotion_status": (
        json.loads(promotion.read_text(encoding="utf-8")).get("status", "unknown")
        if promotion.is_file() else "missing"
    ),
    "gate_review_status": (
        json.loads(gate_review.read_text(encoding="utf-8")).get("status", "unknown")
        if gate_review.is_file() else "missing"
    ),
    "datasets": datasets,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
if release_attestation.is_file():
    try:
        attestation_payload = release_attestation.read_bytes()
        attestation = json.loads(attestation_payload)
        payload["canonical_prepared_release"] = {
            "path": str(release_attestation),
            "bytes": len(attestation_payload),
            "sha256": __import__("hashlib").sha256(attestation_payload).hexdigest(),
            "status": attestation.get("status"),
            "schema_version": attestation.get("schema_version"),
            "release_sha256": attestation.get("release_sha256"),
        }
    except Exception:
        payload["canonical_prepared_release"] = {
            "path": str(release_attestation), "status": "invalid"
        }
else:
    payload["canonical_prepared_release"] = {
        "path": str(release_attestation), "status": "missing"
    }
if sys.argv[6]:
    payload["error"] = sys.argv[6]
target.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(
    mode="w", encoding="utf-8", dir=target.parent,
    prefix=f".{target.name}.", delete=False,
) as stream:
    temporary = Path(stream.name)
    json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, target)
PY
}

on_exit() {
  local code="$?"
  if (( code != 0 )); then
    write_status failed "stage=${current_stage}; exit_code=${code}" || true
  fi
}
trap on_exit EXIT

current_stage="independent_review_gate"
write_status running
jq -e '.status == "passed" and .formal_gpu_release_allowed == true' \
  "$gate_review" >/dev/null

current_stage="validation_promotion"
write_status running
"$python" scripts/promote_public_validation_to_test.py --output "$promotion"

current_stage="test_input_preparation"
write_status running
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  "$python" scripts/prepare_public_test_inputs.py --threads 8 --batch-size 16

current_stage="contract_verification"
write_status running
jq -e '.status == "promoted" and .promoted_systems == ["soe", "pge"]' \
  "$promotion" >/dev/null
for dataset in conll04 scierc ade; do
  manifest="$test_jobs_root/$dataset/preparation_manifest.json"
  jq -e '.status == "prepared_test" and .test_gold_read == false' "$manifest" >/dev/null
  [[ -s "$test_jobs_root/$dataset/jobs/test_eae_jobs.jsonl" ]]
  [[ -s "$test_jobs_root/$dataset/jobs/test_hrge_jobs.jsonl" ]]
done

# The release cannot become complete until the preparation boundary has
# re-opened and canonically validated every emitted byte.  Persist that exact
# all-dataset fingerprint and immediately reproduce it once more to close the
# writer/verifier and publication windows.
attestation_tmp="$(mktemp "$run_root/.prepared_release_attestation.XXXXXX")"
"$python" scripts/prepare_public_test_inputs.py --verify-release >"$attestation_tmp"
jq -e '.status == "verified_release" and
       .schema_version == "public-formal-test-release-v2" and
       (.release_sha256 | type == "string" and length == 64)' \
  "$attestation_tmp" >/dev/null
mv "$attestation_tmp" "$release_attestation"
"$python" scripts/prepare_public_test_inputs.py --verify-release | \
  cmp -s - "$release_attestation"

current_stage="complete"
write_status complete
trap - EXIT
