import importlib.util
import json
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_missing_predictions",
    ROOT / "scripts" / "materialize_missing_predictions.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_materializes_missing_rows_in_job_order(tmp_path):
    jobs = tmp_path / "validation_jobs.jsonl"
    predictions = tmp_path / "partial.jsonl"
    output = tmp_path / "complete.jsonl"
    summary = tmp_path / "summary.json"
    write_jsonl(
        jobs,
        [
            {"job_id": "J1", "document_id": "D1", "language": "en"},
            {"job_id": "J2", "document_id": "D2", "language": "en"},
        ],
    )
    write_jsonl(
        predictions,
        [{"job_id": "J2", "annotation": {"entities": [], "relations": []}}],
    )

    result = MODULE.run(
        Namespace(
            jobs=jobs,
            predictions=predictions,
            output=output,
            summary=summary,
            reason="terminal failure",
        )
    )

    rows = MODULE.load_jsonl(output)
    assert [row["job_id"] for row in rows] == ["J1", "J2"]
    assert rows[0]["annotation"]["review"]["notes"] == "terminal failure"
    assert result["failures_materialized_as_empty"] == 1
    assert result["gold_read"] is False


def test_rejects_unknown_prediction_job(tmp_path):
    jobs = tmp_path / "validation_jobs.jsonl"
    predictions = tmp_path / "partial.jsonl"
    output = tmp_path / "complete.jsonl"
    write_jsonl(jobs, [{"job_id": "J1", "document_id": "D1"}])
    write_jsonl(
        predictions,
        [{"job_id": "OTHER", "annotation": {"entities": [], "relations": []}}],
    )

    try:
        MODULE.run(
            Namespace(
                jobs=jobs,
                predictions=predictions,
                output=output,
                summary=None,
                reason="terminal failure",
            )
        )
    except ValueError as error:
        assert "unknown jobs" in str(error)
    else:
        raise AssertionError("unknown prediction should be rejected")
