#!/usr/bin/env bash
# Build the isolated, inference-only OneKE environment without exposing a GPU.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

env_root="${PUBLIC_ONEKE_ENV:-/ds2/xuelin/envs/public-oneke-formal}"
bootstrap_python="${ONEKE_BOOTSTRAP_PYTHON:-/home/xuelin/miniconda3/envs/rc-llm-comet/bin/python}"
mirror="${ONEKE_PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
status_root="outputs/public_external_formal/oneke"
status_file="$status_root/environment_status.json"
runtime_map="outputs/public_external_formal/runtime_map.json"
mkdir -p "$status_root" "$(dirname "$runtime_map")" "$(dirname "$env_root")"

write_status() {
  local state="$1" detail="$2"
  "$bootstrap_python" - "$status_file" "$state" "$detail" "$env_root" "$mirror" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": "public-oneke-environment-v1",
    "status": sys.argv[2],
    "detail": sys.argv[3],
    "environment": sys.argv[4],
    "package_index": sys.argv[5],
    "cuda_visible_devices": "",
    "gpu_process_started": False,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

on_exit() {
  local code="$?"
  if (( code != 0 )); then
    write_status failed "environment setup exited with code $code"
  fi
}
trap on_exit EXIT

write_status creating "creating isolated Python environment"
if [[ ! -x "$env_root/bin/python" ]]; then
  CUDA_VISIBLE_DEVICES='' "$bootstrap_python" -m venv "$env_root"
fi

write_status installing "installing pinned inference dependencies"
CUDA_VISIBLE_DEVICES='' "$env_root/bin/python" -m pip install \
  --disable-pip-version-check \
  --index-url "$mirror" \
  --upgrade pip wheel setuptools
CUDA_VISIBLE_DEVICES='' "$env_root/bin/python" -m pip install \
  --disable-pip-version-check \
  --index-url "$mirror" \
  --requirement requirements/oneke-formal.txt

write_status verifying "running metadata-only and CPU import checks"
CUDA_VISIBLE_DEVICES='' "$env_root/bin/python" - "$env_root" <<'PY'
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve()
for name in (
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "sentencepiece",
    "safetensors",
):
    print(name, importlib.metadata.version(name))
import torch
assert not torch.cuda.is_available()
assert torch.version.cuda
bnb_spec = importlib.util.find_spec("bitsandbytes")
assert bnb_spec and bnb_spec.submodule_search_locations
bnb_root = Path(next(iter(bnb_spec.submodule_search_locations)))
cuda_tag = torch.version.cuda.replace(".", "")
assert (bnb_root / f"libbitsandbytes_cuda{cuda_tag}.so").is_file()
PY

"$bootstrap_python" - "$runtime_map" "$env_root/bin/python" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {}
if path.is_file():
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(current, dict):
            payload.update(current)
    except (OSError, json.JSONDecodeError):
        pass
payload["oneke"] = {"python": sys.argv[2], "purpose": "public-formal-validation"}
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY

write_status complete "isolated runtime is ready for a queued GPU canary"
trap - EXIT
