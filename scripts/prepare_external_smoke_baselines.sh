#!/usr/bin/env bash
# CPU/network preflight only. It intentionally does not start another GPU job.

set -euo pipefail

export PATH="/home/xuelin/miniconda3/bin:$PATH"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p tools/external-baselines
mkdir -p outputs/public_smoke_40_8_8

if [[ ! -d tools/external-baselines/spert/.git ]]; then
  git clone --depth 1 https://github.com/lavis-nlp/spert.git tools/external-baselines/spert
fi
if [[ ! -d tools/external-baselines/gliner/.git ]]; then
  git clone --depth 1 https://github.com/urchade/GLiNER.git tools/external-baselines/gliner
fi
python3 -m pip install gliner
python3 - <<'PY'
from gliner import GLiNER
print('GLiNER import: ok')
PY
printf '{"status":"prepared","frameworks":["SpERT","GLiNER"],"note":"InstructUIE requires a separate schema adapter before a fair run."}\n' > outputs/public_smoke_40_8_8/external_preflight.json
