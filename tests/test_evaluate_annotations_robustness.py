import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_annotations_robustness", ROOT / "scripts" / "evaluate_annotations.py"
)
EVALUATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(EVALUATOR)


def test_malformed_generated_relation_is_counted_but_never_matches_gold():
    entities = {"E1": {"id": "E1", "text": "Alice", "type": "Peop"}}
    malformed = {"id": "R1", "source_id": "E1", "type": "Work_For"}
    key = EVALUATOR.relation_key(malformed, entities)

    assert key[0].startswith("__invalid_relation__:")
    assert key != ("alice", "Work_For", "acme")
    result = EVALUATOR.scores({("alice", "Work_For", "acme")}, {key})
    assert result == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "gold": 1,
        "predicted": 1,
        "correct": 0,
    }


def test_malformed_generated_entity_does_not_abort_keying():
    assert EVALUATOR.entity_key({"id": "E1"}) == (
        "__invalid_entity_text__",
        "__invalid_entity_type__",
    )
    assert EVALUATOR.entity_key("not-an-object") == (
        "__invalid_entity_text__",
        "__invalid_entity_type__",
    )


def test_non_list_annotation_fields_are_treated_as_empty():
    assert EVALUATOR.annotation_items({"entities": None}, "entities") == []
    assert EVALUATOR.annotation_items({"relations": {}}, "relations") == []
