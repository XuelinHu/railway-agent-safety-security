#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
integrity_status="outputs/public_baseline_downloads/integrity_status.json"
python3 scripts/audit_public_baseline_downloads.py --output "$integrity_status" >/dev/null
echo "Services:"
for unit in public-model-download-hf public-model-download-spert public-model-download-repos; do
  printf '%-30s %s\n' "$unit" "$(systemctl --user is-active "$unit.service" 2>/dev/null || true)"
done
echo
echo "Authoritative integrity status:"
jq -r '
  "overall\t" + .status,
  (.components | to_entries[] | [.key, .value.status] | @tsv)
' "$integrity_status"
echo
echo "Component status:"
for file in outputs/public_baseline_downloads/*.status.json; do
  [[ -f "$file" ]] && jq -r '[.name,.status,.updated_at] | @tsv' "$file"
done
echo
echo "Downloaded sizes:"
du -sh \
  /ds2/xuelin/cache/huggingface/hub/models--ZWK--InstructUIE \
  /ds2/xuelin/cache/huggingface/hub/models--jackboyla--glirel-large-v0 \
  /ds2/xuelin/cache/huggingface/hub/models--zjunlp--OneKE \
  /ds2/xuelin/cache/huggingface/hub/models--allenai--scibert_scivocab_uncased \
  /ds2/xuelin/cache/huggingface/hub/models--FacebookAI--roberta-large \
  /ds2/xuelin/cache/external-baselines/mirror \
  tools/external-baselines/spert/data/models 2>/dev/null || true
