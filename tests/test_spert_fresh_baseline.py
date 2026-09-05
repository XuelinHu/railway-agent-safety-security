import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def types_payload():
    return {
        "entities": {
            "Person": {"short": "Per", "verbose": "Person"},
            "Org": {"short": "Org", "verbose": "Organization"},
        },
        "relations": {
            "Works_For": {"short": "Work", "verbose": "Works for", "symmetric": False}
        },
    }


def row(identifier: int):
    return {
        "orig_id": identifier,
        "tokens": ["Alice", "joined", "Acme"],
        "entities": [
            {"type": "Person", "start": 0, "end": 1},
            {"type": "Org", "start": 2, "end": 3},
        ],
        "relations": [{"type": "Works_For", "head": 0, "tail": 1}],
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_conll_preparation_never_reads_or_materializes_test(tmp_path, monkeypatch):
    module = load_script("prepare_spert_fresh_splits.py")
    source = tmp_path / "source" / "conll04"
    write_json(source / "conll04_train.json", [row(1)])
    write_json(source / "conll04_dev.json", [row(2)])
    write_json(source / "conll04_types.json", types_payload())
    write_json(source / "conll04_test.json", {"must": "not be read"})
    accessed = []
    original = module.read_json

    def tracked(path):
        accessed.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(module, "read_json", tracked)
    args = SimpleNamespace(
        input_root=tmp_path / "source",
        output_root=tmp_path / "output",
        reference_root=None,
        ade_seed="public-full-42",
    )
    summary = module.prepare_dataset(args, "conll04")
    assert accessed == ["conll04_train.json", "conll04_types.json", "conll04_dev.json"]
    assert summary["rows"] == {"train": 1, "validation": 1}
    assert not (tmp_path / "output" / "conll04" / "test.json").exists()


def test_ade_preparation_matches_deterministic_current_split(tmp_path, monkeypatch):
    module = load_script("prepare_spert_fresh_splits.py")
    source = tmp_path / "source" / "ade"
    rows = [row(index) for index in range(20)]
    write_json(source / "ade_train.json", rows)
    write_json(source / "ade_types.json", types_payload())
    write_json(source / "ade_test.json", {"must": "not be read"})
    accessed = []
    original = module.read_json

    def tracked(path):
        accessed.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(module, "read_json", tracked)
    args = SimpleNamespace(
        input_root=tmp_path / "source",
        output_root=tmp_path / "output",
        reference_root=None,
        ade_seed="public-full-42",
    )
    summary = module.prepare_dataset(args, "ade")
    expected = sorted(
        rows,
        key=lambda item: hashlib.sha256(f"public-full-42:{item['orig_id']}".encode()).hexdigest(),
    )
    validation = json.loads((tmp_path / "output/ade/validation.json").read_text())
    train = json.loads((tmp_path / "output/ade/train.json").read_text())
    assert accessed == ["ade_train.json", "ade_types.json"]
    assert [item["orig_id"] for item in validation] == [item["orig_id"] for item in expected[:2]]
    assert [item["orig_id"] for item in train] == [item["orig_id"] for item in expected[2:]]
    assert summary["rows"] == {"train": 18, "validation": 2}
    assert not (tmp_path / "output/ade/test.json").exists()


def test_historical_adamw_preserves_no_bias_correction_semantics():
    module = load_script("run_spert_compat.py")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([0.25])
    optimizer = module.HistoricalAdamW(
        [parameter], lr=1e-3, betas=(0.9, 0.999), eps=1e-6, correct_bias=False
    )
    optimizer.step()
    exp_avg = 0.025
    exp_avg_sq = 0.0000625
    expected = 1.0 - 1e-3 * exp_avg / (exp_avg_sq**0.5 + 1e-6)
    assert parameter.item() == pytest.approx(expected, abs=1e-7)


def test_process_local_compatibility_restores_adamw_symbol():
    module = load_script("run_spert_compat.py")
    module.install_compatibility()
    import transformers

    assert transformers.AdamW is module.HistoricalAdamW
    assert getattr(transformers.PreTrainedModel.save_pretrained, "_spert_compat", False)


def test_wrapper_keeps_adamw_compatibility_across_config_dispatch(tmp_path):
    repo = tmp_path / "stub_spert"
    repo.mkdir()
    (repo / "config_reader.py").write_text(
        """import multiprocessing as mp


def process_configs(target, arg_parser):
    args, _ = arg_parser.parse_known_args()
    ctx = mp.get_context(\"spawn\")
    for run_args, _config, _repeat in _yield_configs(arg_parser, args):
        process = ctx.Process(target=target, args=(run_args,))
        process.start()
        process.join()


def _yield_configs(arg_parser, args, verbose=True):
    for _ in range(args.repeat):
        yield args, None, args.repeat
""",
        encoding="utf-8",
    )
    (repo / "spert.py").write_text(
        """import argparse
import os
from pathlib import Path

from config_reader import process_configs


def execute(args):
    from transformers import AdamW

    with Path(args.marker).open(\"a\", encoding=\"utf-8\") as stream:
        stream.write(f\"{os.getpid()}|{AdamW.__name__}|{args.value}\\n\")
    args.value = \"mutated\"
    if args.fail:
        raise RuntimeError(\"sentinel-spert-target-error\")


if __name__ == \"__main__\":
    parser = argparse.ArgumentParser()
    parser.add_argument(\"mode\")
    parser.add_argument(\"--marker\", required=True)
    parser.add_argument(\"--repeat\", type=int, default=1)
    parser.add_argument(\"--value\", default=\"alpha\")
    parser.add_argument(\"--fail\", action=\"store_true\")
    process_configs(execute, parser)
""",
        encoding="utf-8",
    )

    wrapper = ROOT / "scripts" / "run_spert_compat.py"
    marker = tmp_path / "marker.txt"
    command = [
        sys.executable,
        str(wrapper),
        "--repo",
        str(repo),
        "--",
        "train",
        "--marker",
        str(marker),
        "--repeat",
        "2",
        "--value",
        "alpha",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    stdout, stderr = process.communicate(timeout=60)
    assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    rows = [line.split("|") for line in marker.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        [str(process.pid), "HistoricalAdamW", "alpha"],
        [str(process.pid), "HistoricalAdamW", "alpha"],
    ]

    failed = subprocess.run(
        [*command[:-4], "--repeat", "1", "--value", "alpha", "--fail"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ.copy(),
    )
    assert failed.returncode != 0
    assert "sentinel-spert-target-error" in failed.stderr


def test_fresh_runner_is_resumable_locked_and_emits_project_metrics():
    runner = ROOT / "scripts" / "run_spert_fresh_baseline.sh"
    launcher = ROOT / "scripts" / "run_spert_after_pge.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    text = runner.read_text(encoding="utf-8")

    assert "flock -n 8" in text
    assert "reuse_spert_checkpoint" in text
    assert "reuse_spert_validation" in text
    assert "convert_spert_predictions.py" in text
    assert "evaluate_annotations.py" in text
    assert "evaluate_public_validation_spans.py" in text
    assert "all) all_validation" in text


def test_spert_launcher_waits_only_for_completed_pge_validation():
    launcher = (ROOT / "scripts" / "run_spert_after_pge.sh").read_text(encoding="utf-8")

    assert "public_pge_validation_seed42/status.json" in launcher
    assert '!= "complete"' in launcher
    assert "run_spert_fresh_baseline.sh all" in launcher
