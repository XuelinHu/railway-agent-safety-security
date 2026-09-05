#!/usr/bin/env bash
# Download public comparison code and checkpoints without starting GPU jobs.
set -euo pipefail

lane="${1:?usage: $0 hf|spert|repos}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="/home/xuelin/miniconda3/bin:$PATH"
export HF_HOME="/ds2/xuelin/cache/huggingface"
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
LOG_ROOT="outputs/public_baseline_downloads"
mkdir -p "$LOG_ROOT" tools/external-baselines /ds2/xuelin/cache/external-baselines

mark() {
  local name="$1" status="$2"
  local target="$LOG_ROOT/${name}.status.json"
  local temporary="$LOG_ROOT/.${name}.status.json.tmp"
  printf '{"name":"%s","status":"%s","updated_at":"%s"}\n' \
    "$name" "$status" "$(date --iso-8601=seconds)" > "$temporary"
  mv "$temporary" "$target"
}

require_size() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || { echo "missing downloaded file: $path" >&2; return 1; }
  actual="$(stat -c %s "$path")"
  [[ "$actual" == "$expected" ]] || {
    echo "downloaded size mismatch for $path: expected=$expected actual=$actual" >&2
    return 1
  }
}

require_zip_directory() {
  local path="$1"
  zipinfo -t "$path" >/dev/null
}

hf_model() {
  local repo="$1" name="$2"
  mark "$name" downloading
  if ! HF_ENDPOINT=https://hf-mirror.com hf download "$repo" --max-workers 4; then
    echo "mirror failed for $repo; retrying official endpoint" >&2
    HF_ENDPOINT=https://huggingface.co hf download "$repo" --max-workers 4
  fi
  mark "$name" complete
}

clone_repo() {
  local url="$1" target="$2" name="$3"
  if [[ -d "$target/.git" ]]; then
    git -C "$target" fsck --no-progress --connectivity-only >/dev/null
    mark "$name" complete
    return
  fi
  mark "$name" downloading
  git clone --depth 1 "$url" "$target"
  git -C "$target" fsck --no-progress --connectivity-only >/dev/null
  mark "$name" complete
}

case "$lane" in
  hf)
    # Main recent generative baseline, lightweight relation baseline, optional
    # LLM baseline, and the two encoder backbones used by PL-Marker/SpERT.
    hf_model ZWK/InstructUIE instructuie
    hf_model jackboyla/glirel-large-v0 glirel
    hf_model zjunlp/OneKE oneke
    hf_model allenai/scibert_scivocab_uncased scibert
    hf_model FacebookAI/roberta-large roberta_large
    ;;
  spert)
    mark spert_models downloading
    cd tools/external-baselines/spert
    mkdir -p data/models
    for dataset in conll04 scierc ade; do
      wget --continue --recursive --no-host-directories --cut-dirs=100 \
        --reject 'index.html*' --no-parent \
        "https://lavis.cs.hs-rm.de/storage/spert/public/models/${dataset}/" \
        -P "$(pwd)/data/models/${dataset}"
    done
    require_size data/models/conll04/pytorch_model.bin 433380286
    require_size data/models/scierc/pytorch_model.bin 439924262
    require_size data/models/ade/pytorch_model.bin 433330110
    cd "$ROOT"
    mark spert_models complete
    ;;
  repos)
    clone_repo https://github.com/Spico197/Mirror.git tools/external-baselines/mirror mirror_code
    clone_repo https://github.com/jackboyla/GLiREL.git tools/external-baselines/glirel glirel_code
    clone_repo https://github.com/thunlp/PL-Marker.git tools/external-baselines/pl-marker pl_marker_code
    clone_repo https://github.com/zjunlp/OneKE.git tools/external-baselines/oneke oneke_code
    mark mirror_models downloading
    mirror_dir=/ds2/xuelin/cache/external-baselines/mirror
    mkdir -p "$mirror_dir"
    wget --continue --output-document="$mirror_dir/mirror_outputs.zip" https://osf.io/download/qj2d3/
    wget --continue --output-document="$mirror_dir/resources.zip" https://osf.io/download/q6utn/
    require_size "$mirror_dir/mirror_outputs.zip" 4357712150
    require_size "$mirror_dir/resources.zip" 306218565
    require_zip_directory "$mirror_dir/mirror_outputs.zip"
    require_zip_directory "$mirror_dir/resources.zip"
    mark mirror_models complete
    ;;
  *)
    echo "unknown lane: $lane" >&2
    exit 2
    ;;
esac
