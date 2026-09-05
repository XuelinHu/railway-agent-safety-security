#!/usr/bin/env bash
# Run the minimal public pipeline smoke without exposing a CUDA device.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES="void"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONUNBUFFERED=1

python="${PUBLIC_CPU_SMOKE_PYTHON:-/home/xuelin/miniconda3/bin/python}"
if [[ ! -x "$python" ]]; then
  echo "CPU smoke Python is not executable: $python" >&2
  exit 2
fi

exec "$python" scripts/run_public_cpu_smoke.py "$@"
