#!/usr/bin/env python3
"""Build a near-duplicate-aware, stratified annotation expansion manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from select_pilot_set import CHINESE_CATEGORIES, ENGLISH_CATEGORIES, classify


def load_document(root: Path, document_id: str) -> dict[str, Any]:
    return json.loads((root / f"{document_id}.json").read_text(encoding="utf-8"))


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def simhash(text: str, max_shingles: int = 20000) -> int:
    compact = normalized_text(text)
    if len(compact) < 5:
        return int.from_bytes(hashlib.blake2b(compact.encode(), digest_size=8).digest(), "big")
    shingles = {compact[index : index + 5] for index in range(0, len(compact) - 4, 3)}
    if len(shingles) > max_shingles:
        step = max(1, len(shingles) // max_shingles)
        shingles = set(sorted(shingles)[::step][:max_shingles])
    votes = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            votes[bit] += 1 if value & (1 << bit) else -1
    return sum((1 << bit) for bit, vote in enumerate(votes) if vote >= 0)


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def cluster_rows(rows: list[dict[str, str]], documents: dict[str, dict[str, Any]], threshold: int) -> tuple[dict[str, str], dict[tuple[str, str], int]]:
    fingerprints = {row["document_id"]: simhash(documents[row["document_id"]]["text"]) for row in rows}
    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for document_id, fingerprint in fingerprints.items():
        for band in range(4):
            buckets[(band, (fingerprint >> (band * 16)) & 0xFFFF)].append(document_id)

    parent = {document_id: document_id for document_id in fingerprints}

    def find(document_id: str) -> str:
        while parent[document_id] != document_id:
            parent[document_id] = parent[parent[document_id]]
            document_id = parent[document_id]
        return document_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    seen_pairs: set[tuple[str, str]] = set()
    for candidates in buckets.values():
        for index, left in enumerate(candidates):
            for right in candidates[index + 1 :]:
                pair = tuple(sorted((left, right)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                left_row = next(row for row in rows if row["document_id"] == left)
                right_row = next(row for row in rows if row["document_id"] == right)
                left_length = max(1, int(left_row.get("char_count") or 0))
                right_length = max(1, int(right_row.get("char_count") or 0))
                if min(left_length, right_length) / max(left_length, right_length) < 0.35:
                    continue
                if hamming(fingerprints[left], fingerprints[right]) <= threshold:
                    union(left, right)

    roots = {document_id: find(document_id) for document_id in fingerprints}
    root_to_cluster: dict[str, str] = {}
    for root in sorted(set(roots.values())):
        root_to_cluster[root] = f"near_{len(root_to_cluster) + 1:04d}"
    cluster_by_document = {document_id: root_to_cluster[root] for document_id, root in roots.items()}
    distances = {(left, right): hamming(fingerprints[left], fingerprints[right]) for left, right in seen_pairs if find(left) == find(right)}
    return cluster_by_document, distances


def choose(rows: list[dict[str, str]], current_ids: set[str], target_total: int, railway_target: int) -> list[dict[str, str]]:
    selected = [row for row in rows if row["document_id"] in current_ids]
    selected_ids = {row["document_id"] for row in selected}
    group_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["document_id"] in selected_ids:
            continue
        group_rows[(row["source_group"], row["category"])].append(row)
    for values in group_rows.values():
        values.sort(key=lambda row: (-int(row["char_count"] or 0), row["document_id"]))

    quotas = {"raib": max(0, railway_target - sum(row["source_group"] == "raib" for row in selected)), "cn": max(0, target_total - railway_target - sum(row["source_group"] == "cn" for row in selected))}
    cluster_used = {row["cluster_id"] for row in selected}
    while len(selected) < target_total:
        progress = False
        for source_group in ("raib", "cn"):
            if quotas[source_group] <= 0:
                continue
            candidates = [
                row for key, values in group_rows.items() if key[0] == source_group for row in values
                if row["document_id"] not in selected_ids and row["cluster_id"] not in cluster_used
            ]
            if not candidates:
                continue
            # Favor underrepresented categories, then substantive documents.
            category_counts = defaultdict(int)
            for row in selected:
                if row["source_group"] == source_group:
                    category_counts[row["category"]] += 1
            candidates.sort(key=lambda row: (category_counts[row["category"]], -int(row["char_count"] or 0), row["document_id"]))
            chosen = candidates[0]
            selected.append(chosen)
            selected_ids.add(chosen["document_id"])
            cluster_used.add(chosen["cluster_id"])
            quotas[source_group] -= 1
            progress = True
            if len(selected) >= target_total:
                break
        if not progress:
            break
    return selected


def run(args: argparse.Namespace) -> int:
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    rows = [row for row in raw_rows if row["extract_status"] == "success" and not row["duplicate_of"]]
    documents = {row["document_id"]: load_document(args.document_root, row["document_id"]) for row in rows}
    for row in rows:
        document = documents[row["document_id"]]
        if row["source_group"] == "raib":
            row["language"] = "en"
            row["category"] = classify(document["text"][:50000], ENGLISH_CATEGORIES)
        else:
            row["language"] = "zh"
            path_parts = Path(row["relative_path"]).parts
            row["category"] = classify("/".join(path_parts[2:]), CHINESE_CATEGORIES)

    cluster_by_document, distances = cluster_rows(rows, documents, args.hamming_threshold)
    for row in rows:
        row["cluster_id"] = cluster_by_document[row["document_id"]]

    current_ids: set[str] = set()
    if args.current_pilot.exists():
        with args.current_pilot.open(encoding="utf-8-sig", newline="") as stream:
            current_ids = {row["document_id"] for row in csv.DictReader(stream)}
    selected = choose(rows, current_ids, args.target_total, args.railway_target)
    selected_ids = {row["document_id"] for row in selected}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["document_id", "source_group", "language", "category", "cluster_id", "relative_path", "actual_format", "char_count", "sha256", "review_status"]
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(selected, key=lambda item: (item["source_group"], item["category"], item["document_id"])):
            writer.writerow({**{field: row.get(field, "") for field in fields}, "review_status": "pilot_accepted" if row["document_id"] in current_ids else "pending"})

    cluster_output = args.cluster_output
    cluster_output.parent.mkdir(parents=True, exist_ok=True)
    with cluster_output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["document_id", "cluster_id", "source_group", "category", "char_count"])
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["cluster_id"], item["document_id"])):
            writer.writerow({field: row.get(field, "") for field in ["document_id", "cluster_id", "source_group", "category", "char_count"]})

    print(json.dumps({"eligible_documents": len(rows), "near_duplicate_clusters": len(set(cluster_by_document.values())), "near_duplicate_pairs": len(distances), "selected": len(selected), "selected_existing_pilot": len(selected_ids & current_ids), "selected_pending": len(selected_ids - current_ids), "selected_by_source": {source: sum(row["source_group"] == source for row in selected) for source in ("raib", "cn")}, "output": str(args.output), "cluster_output": str(cluster_output)}, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/catalog/corpus_inventory.csv"))
    parser.add_argument("--document-root", type=Path, default=Path("data/processed/corpus/documents"))
    parser.add_argument("--current-pilot", type=Path, default=Path("data/catalog/pilot_set.csv"))
    parser.add_argument("--target-total", type=int, default=150)
    parser.add_argument("--railway-target", type=int, default=80)
    parser.add_argument("--hamming-threshold", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("data/catalog/annotation_expansion_set.csv"))
    parser.add_argument("--cluster-output", type=Path, default=Path("data/catalog/near_duplicate_clusters.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
