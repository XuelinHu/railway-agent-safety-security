import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "calibrate_glirel_train.py"


def load_script():
    spec = importlib.util.spec_from_file_location("calibrate_glirel_train", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_threshold_scoring_matches_strict_runtime_and_pair_top_k():
    module = load_script()
    rows = [
        {
            "gold_relations": [
                {"source_id": "E1", "type": "good", "target_id": "E2"}
            ],
            "candidates": [
                {"source_id": "E1", "type": "bad", "target_id": "E2", "score": 0.7},
                {"source_id": "E1", "type": "good", "target_id": "E2", "score": 0.6},
                {"source_id": "E2", "type": "other", "target_id": "E1", "score": 0.5},
            ],
        }
    ]

    at_point_five = module.score_rows(rows, 0.5)
    assert at_point_five["true_positives"] == 0
    assert at_point_five["false_positives"] == 1
    assert at_point_five["false_negatives"] == 1
    assert at_point_five["predictions"] == 1

    above_point_six = module.score_rows(rows, 0.6)
    assert above_point_six["predictions"] == 1
    assert above_point_six["true_positives"] == 0


def test_train_subset_selection_is_deterministic_and_covers_labels():
    module = load_script()
    jobs = [{"job_id": f"demo_train_{index}_C1"} for index in range(6)]
    annotations = {
        job["job_id"]: {
            "relations": [
                {
                    "type": "a" if index % 2 == 0 else "b",
                    "source_id": "E1",
                    "target_id": "E2",
                }
            ]
        }
        for index, job in enumerate(jobs)
    }

    first = module.select_job_ids(jobs, annotations, ["a", "b"], max_jobs=4, seed=42)
    second = module.select_job_ids(jobs, annotations, ["a", "b"], max_jobs=4, seed=42)
    assert first == second
    assert len(first) == 4
    assert {annotations[job_id]["relations"][0]["type"] for job_id in first} == {"a", "b"}


def test_configuration_selection_prefers_measured_f1_then_recall():
    module = load_script()
    metrics = [
        {
            "label_mode": "canonical",
            "threshold": 0.1,
            "f1": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "predictions": 10,
        },
        {
            "label_mode": "naturalized",
            "threshold": 0.2,
            "f1": 0.5,
            "precision": 0.4,
            "recall": 0.7,
            "predictions": 12,
        },
    ]
    assert module.choose_configuration(metrics)["label_mode"] == "naturalized"


def test_calibration_source_never_mentions_validation_or_test_gold_paths():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"train_baseline_jobs.jsonl"' in source
    assert '"train_gold.jsonl"' in source
    assert '"train_index.jsonl"' in source
    assert "validation_gold.jsonl" not in source
    assert "test_gold.jsonl" not in source
