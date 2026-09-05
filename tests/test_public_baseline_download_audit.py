import importlib.util
import json
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_public_baseline_downloads.py"


def load_script():
    spec = importlib.util.spec_from_file_location("audit_public_baseline_downloads", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_archive_states_distinguish_partial_complete_and_oversize():
    audit = load_script()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "artifact.zip"
        archive.write_bytes(b"partial")
        assert audit.inspect_archive(archive, 100, None)["status"] == "downloading"

        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("model/checkpoint.pth", b"weights")
        expected = archive.stat().st_size
        ready = audit.inspect_archive(archive, expected, "checkpoint.pth")
        assert ready["status"] == "ready"
        assert ready["required_member_present"] is True

        assert audit.inspect_archive(archive, expected - 1, None)["status"] == "invalid"


def test_hf_snapshot_requires_revision_shards_and_resolved_links():
    audit = load_script()
    with tempfile.TemporaryDirectory() as directory:
        hub = Path(directory)
        repository = hub / "models--org--model"
        snapshot = repository / "snapshots" / "abc123"
        blobs = repository / "blobs"
        (repository / "refs").mkdir(parents=True)
        snapshot.mkdir(parents=True)
        blobs.mkdir()
        (repository / "refs" / "main").write_text("abc123\n", encoding="utf-8")
        for name, data in (("config.json", b"{}"), ("tokenizer.json", b"{}"), ("shard.bin", b"1234")):
            blob = blobs / name
            blob.write_bytes(data)
            (snapshot / name).symlink_to(blob)
        index = {"weight_map": {"layer": "shard.bin"}}
        index_blob = blobs / "index.json"
        index_blob.write_text(json.dumps(index), encoding="utf-8")
        (snapshot / "index.json").symlink_to(index_blob)
        specification = {
            "cache_name": "models--org--model",
            "revision": "abc123",
            "required": ("config.json", "tokenizer.json", "index.json"),
            "index": "index.json",
            "shards": 1,
            "weight_bytes": 4,
        }
        assert audit.inspect_hf_model(hub, specification)["status"] == "ready"
        (snapshot / "shard.bin").unlink()
        (snapshot / "shard.bin").symlink_to(blobs / "missing")
        invalid = audit.inspect_hf_model(hub, specification)
        assert invalid["status"] == "invalid"
        assert invalid["broken_links"] == 1
