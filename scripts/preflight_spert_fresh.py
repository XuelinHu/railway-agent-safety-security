#!/usr/bin/env python3
"""CPU-only structural and model-loading preflight for fresh SpERT runs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


DATASETS = ("conll04", "scierc", "ade")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_snapshot(cache_root: Path, repository: str) -> tuple[Path, str]:
    repo_root = cache_root / repository
    revision = (repo_root / "refs/main").read_text(encoding="utf-8").strip()
    snapshot = repo_root / "snapshots" / revision
    required = ("config.json", "vocab.txt")
    missing = [name for name in required if not (snapshot / name).is_file()]
    if not (snapshot / "pytorch_model.bin").is_file() and not (snapshot / "model.safetensors").is_file():
        missing.append("pytorch_model.bin|model.safetensors")
    if missing:
        raise FileNotFoundError(f"Incomplete backbone {snapshot}: {', '.join(missing)}")
    return snapshot, revision


def git_revision(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "data/processed/spert_fresh_train_v1")
    parser.add_argument("--repo", type=Path, default=root / "tools/external-baselines/spert")
    parser.add_argument("--hf-hub-root", type=Path, default=Path("/ds2/xuelin/cache/huggingface/hub"))
    parser.add_argument("--status", type=Path, default=root / "outputs/public_horizontal_validation/spert_fresh/preflight.json")
    parser.add_argument("--skip-model-load", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Hide CUDA before importing torch/transformers; this preflight must never
    # contend with an active experiment.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from run_spert_compat import HistoricalAdamW, install_compatibility

    install_compatibility()
    import torch
    import transformers
    from transformers import BertConfig, BertTokenizer

    sys.path.insert(0, str(args.repo.resolve()))
    from spert import sampling
    from spert.entities import Dataset
    from spert.input_reader import JsonInputReader
    from spert.models import SpERT

    backbones = {
        "bert-base-cased": resolve_snapshot(args.hf_hub_root, "models--google-bert--bert-base-cased"),
        "scibert": resolve_snapshot(args.hf_hub_root, "models--allenai--scibert_scivocab_uncased"),
    }
    backbone_by_dataset = {"conll04": "bert-base-cased", "scierc": "scibert", "ade": "bert-base-cased"}

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([0.25])
    optimizer = HistoricalAdamW([parameter], lr=1e-3, correct_bias=False)
    optimizer.step()

    results = {}
    for dataset in DATASETS:
        dataset_root = args.data_root / dataset
        manifest_path = dataset_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        backbone_name = backbone_by_dataset[dataset]
        backbone, revision = backbones[backbone_name]
        tokenizer = BertTokenizer.from_pretrained(str(backbone), local_files_only=True)
        reader = JsonInputReader(str(dataset_root / "types.json"), tokenizer, 100, 100, 10)
        train = reader.read(str(dataset_root / "train.json"), "train")
        validation = reader.read(str(dataset_root / "validation.json"), "validation")
        expected = manifest["rows"]
        observed = {"train": train.document_count, "validation": validation.document_count}
        if observed != expected:
            raise ValueError(f"{dataset}: parsed row mismatch: {observed} != {expected}")
        model_loaded = False
        train_forward = False
        validation_forward = False
        if not args.skip_model_load:
            config = BertConfig.from_pretrained(str(backbone), local_files_only=True)
            model = SpERT.from_pretrained(
                str(backbone),
                config=config,
                local_files_only=True,
                cls_token=tokenizer.convert_tokens_to_ids("[CLS]"),
                relation_types=reader.relation_type_count - 1,
                entity_types=reader.entity_type_count,
                max_pairs=1000,
                prop_drop=0.1,
                size_embedding=25,
                freeze_transformer=False,
            )
            if model.bert.config.hidden_size != config.hidden_size:
                raise ValueError(f"{dataset}: backbone hidden-size mismatch")
            model.eval()
            with torch.no_grad():
                train.switch_mode(Dataset.TRAIN_MODE)
                train_batch = sampling.collate_fn_padding([train[0]])
                entity_logits, relation_logits = model(
                    encodings=train_batch["encodings"],
                    context_masks=train_batch["context_masks"],
                    entity_masks=train_batch["entity_masks"],
                    entity_sizes=train_batch["entity_sizes"],
                    relations=train_batch["rels"],
                    rel_masks=train_batch["rel_masks"],
                )
                if entity_logits.shape[0] != 1 or relation_logits.shape[0] != 1:
                    raise ValueError(f"{dataset}: invalid train forward output")
                train_forward = True
                validation.switch_mode(Dataset.EVAL_MODE)
                validation_batch = sampling.collate_fn_padding([validation[0]])
                entity_scores, relation_scores, relation_pairs = model(
                    encodings=validation_batch["encodings"],
                    context_masks=validation_batch["context_masks"],
                    entity_masks=validation_batch["entity_masks"],
                    entity_sizes=validation_batch["entity_sizes"],
                    entity_spans=validation_batch["entity_spans"],
                    entity_sample_masks=validation_batch["entity_sample_masks"],
                    inference=True,
                )
                if entity_scores.shape[0] != 1 or relation_scores.shape[:2] != relation_pairs.shape[:2]:
                    raise ValueError(f"{dataset}: invalid validation forward output")
                validation_forward = True
            model_loaded = True
            del model
            gc.collect()
        results[dataset] = {
            "rows": observed,
            "manifest_sha256": sha256_file(manifest_path),
            "backbone": str(backbone),
            "backbone_revision": revision,
            "model_loaded_on_cpu": model_loaded,
            "train_forward_on_cpu": train_forward,
            "validation_forward_on_cpu": validation_forward,
        }

    status = {
        "status": "ready",
        "mode": "cpu-only",
        "test_split_access": "forbidden-and-not-materialized",
        "spert_repo": str(args.repo),
        "spert_revision": git_revision(args.repo),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "compatibility": {
            "historical_adamw_correct_bias": False,
            "legacy_pytorch_model_bin": True,
        },
        "datasets": results,
    }
    args.status.parent.mkdir(parents=True, exist_ok=True)
    args.status.write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
