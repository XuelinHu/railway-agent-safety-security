#!/usr/bin/env bash
# Queue validation-only SOE-vs-PGE statistics without occupying the caller.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

unit="${PUBLIC_POST_PGE_UNIT:-public-post-pge-bootstrap}"
pge_root="${PUBLIC_PGE_RUN_ROOT:-outputs/public_pge_validation_seed42}"
run_root="${PUBLIC_POST_PGE_RUN_ROOT:-outputs/public_post_pge_validation_seed42}"
status_file="$run_root/status.json"
python_bin="${PUBLIC_POST_PGE_PYTHON:-/home/xuelin/miniconda3/bin/python}"

if [[ ! "$unit" =~ ^[A-Za-z0-9_.@-]+$ ]]; then
  echo "PUBLIC_POST_PGE_UNIT contains invalid characters" >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Python interpreter is not executable: $python_bin" >&2
  exit 2
fi
mkdir -p "$run_root"

if systemctl --user is-active --quiet "$unit.service"; then
  echo "post-PGE bootstrap is already active: $unit.service"
  exit 0
fi

systemd-run --user --unit="$unit" --collect --no-block \
  --property=Type=exec \
  --property=Restart=on-failure \
  --property=RestartSec=120 \
  --property=Nice=19 \
  --property=CPUWeight=10 \
  --property=CPUQuota=100% \
  --working-directory="$root" \
  --setenv=CUDA_VISIBLE_DEVICES= \
  --setenv=OMP_NUM_THREADS=1 \
  --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=NUMEXPR_NUM_THREADS=1 \
  --setenv=TOKENIZERS_PARALLELISM=false \
  "$python_bin" scripts/run_public_post_pge_bootstrap.py \
  --pge-root "$pge_root" \
  --output-root "$run_root"

echo "queued $unit.service; status: $status_file"
