#!/usr/bin/env python3
"""Select a deterministic, diverse pilot set from successfully extracted documents."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ENGLISH_CATEGORIES = [
    ("collision", re.compile(r"\bcollision|collided|struck\b", re.IGNORECASE)),
    ("derailment", re.compile(r"\bderail(?:ment|ed)?\b", re.IGNORECASE)),
    ("level_crossing", re.compile(r"\blevel crossing\b", re.IGNORECASE)),
    ("fire", re.compile(r"\bfire|burning|smoke\b", re.IGNORECASE)),
    ("signal_or_spad", re.compile(r"\bsignal|SPAD|danger aspect\b", re.IGNORECASE)),
    ("infrastructure", re.compile(r"\btrack|switch|points|bridge|landslip\b", re.IGNORECASE)),
]

CHINESE_CATEGORIES = [
    ("railway", re.compile(r"铁路|铁道|地铁|列车|轨道|道岔|信号")),
    ("fire", re.compile(r"消防|火灾|着火|自燃")),
    ("hazardous_material", re.compile(r"危化|危险化学品|危险品|燃气|泄漏")),
    ("construction", re.compile(r"施工|建筑|工程")),
    ("emergency_drill", re.compile(r"应急演练|演练方案")),
    ("management", re.compile(r"管理制度|责任制|检查表|操作规程")),
]


def classify(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> str:
    for category, pattern in patterns:
        if pattern.search(text):
            return category
    return "other"


def load_document(document_root: Path, document_id: str) -> dict[str, Any]:
    path = document_root / f"{document_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def select(args: argparse.Namespace) -> None:
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    candidates: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["extract_status"] != "success" or row["duplicate_of"]:
            continue
        document = load_document(args.document_root, row["document_id"])
        if row["source_group"] == "raib":
            category = classify(document["text"][:50000], ENGLISH_CATEGORIES)
            language = "en"
        elif row["source_group"] == "cn":
            path_parts = Path(row["relative_path"]).parts
            # The downloaded bundle root contains generic words such as "消防" in
            # every path, so only classify on inner directories and file content.
            inner_path = "/".join(path_parts[2:])
            category = classify(inner_path, CHINESE_CATEGORIES)
            language = "zh"
        else:
            continue
        candidate = dict(row)
        candidate["category"] = category
        candidate["pilot_language"] = language
        candidates[(language, category)].append(candidate)

    selected: list[dict[str, str]] = []
    for key in sorted(candidates):
        group = sorted(candidates[key], key=lambda item: (int(item["char_count"]), item["document_id"]))
        # Avoid tiny files and favor documents near the group median rather than the largest reports.
        substantive = [item for item in group if int(item["char_count"]) >= args.min_characters]
        if not substantive:
            substantive = group
        if not substantive:
            continue
        positions = []
        if args.per_category == 1:
            positions = [len(substantive) // 2]
        else:
            positions = [
                round((index + 1) * (len(substantive) - 1) / (args.per_category + 1))
                for index in range(args.per_category)
            ]
        seen: set[str] = set()
        for position in positions:
            item = substantive[position]
            if item["document_id"] in seen:
                continue
            seen.add(item["document_id"])
            selected.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "document_id",
        "pilot_language",
        "source_group",
        "category",
        "relative_path",
        "actual_format",
        "char_count",
        "sha256",
        "review_status",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in selected:
            writer.writerow({**{field: item.get(field, "") for field in fieldnames}, "review_status": "pending"})
    counts: dict[str, int] = defaultdict(int)
    for item in selected:
        counts[f"{item['pilot_language']}:{item['category']}"] += 1
    print(json.dumps({"selected": len(selected), "groups": counts}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/catalog/corpus_inventory.csv"))
    parser.add_argument("--document-root", type=Path, default=Path("data/processed/corpus/documents"))
    parser.add_argument("--output", type=Path, default=Path("data/catalog/pilot_set.csv"))
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--min-characters", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    select(parse_args())
