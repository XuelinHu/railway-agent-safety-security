#!/usr/bin/env python3
"""Run the GLiNER entity-only zero-shot smoke baseline after Qwen is idle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    import yaml
    from gliner import GLiNER

    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    labels = list(ontology["entity_types"])
    model = GLiNER.from_pretrained("urchade/gliner_small-v2.1", map_location="cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for job in rows(args.jobs):
            text = "\n".join(segment["text"] for segment in job["segments"])
            found = model.predict_entities(text, labels, threshold=args.threshold)
            entities = []
            for index, item in enumerate(found, 1):
                label = item.get("label")
                if label not in ontology["entity_types"]:
                    continue
                mention = item.get("text", "")
                start = text.find(mention)
                if start < 0:
                    continue
                entities.append({"id": f"E{index}", "text": mention, "type": label, "confidence": float(item.get("score", 0.0))})
            annotation = {"schema_version": "0.1.0", "document_id": job["document_id"], "language": "en", "entities": entities, "relations": []}
            stream.write(json.dumps({"job_id": job["job_id"], "annotation": annotation}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
