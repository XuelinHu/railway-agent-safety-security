#!/usr/bin/env python3
"""Inventory and extract the local safety corpus with source offsets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

from pypdf import PdfReader


SUPPORTED_TEXT_FORMATS = {"pdf", "docx", "doc", "txt"}
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_format(path: Path) -> str:
    with path.open("rb") as stream:
        header = stream.read(16)
    if header.startswith(b"%PDF"):
        return "pdf"
    if header.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "doc" if path.suffix.lower() in {".doc", ".docx"} else "ole"
    if header.startswith(b"PK"):
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
            if "word/document.xml" in names:
                return "docx"
            if "xl/workbook.xml" in names:
                return "xlsx"
            if "ppt/presentation.xml" in names:
                return "pptx"
            return "zip"
        except zipfile.BadZipFile:
            return "invalid_zip"
    if header.startswith(b"{\\rtf"):
        return "rtf"
    if header.startswith(b"\xff\xd8\xff"):
        return "jpg"
    return path.suffix.lower().lstrip(".") or "unknown"


def language_hint(text: str) -> str:
    letters = sum(character.isalpha() for character in text)
    chinese = sum("\u4e00" <= character <= "\u9fff" for character in text)
    if not text.strip():
        return "unknown"
    if chinese >= 10 and chinese / max(letters, 1) >= 0.20:
        return "zh"
    if letters >= 10:
        return "en"
    return "unknown"


def make_segments(parts: Iterable[tuple[str, int | None, str]]) -> tuple[str, list[dict[str, Any]]]:
    full_text_parts: list[str] = []
    segments: list[dict[str, Any]] = []
    cursor = 0
    for segment_type, page, raw_text in parts:
        text = raw_text.strip()
        if not text:
            continue
        if full_text_parts:
            separator = "\n\n"
            full_text_parts.append(separator)
            cursor += len(separator)
        start = cursor
        full_text_parts.append(text)
        cursor += len(text)
        segments.append(
            {
                "segment_id": f"S{len(segments) + 1}",
                "segment_type": segment_type,
                "page": page,
                "start": start,
                "end": cursor,
                "text": text,
            }
        )
    return "".join(full_text_parts), segments


def extract_pdf(path: Path) -> tuple[str, list[dict[str, Any]]]:
    reader = PdfReader(str(path), strict=False)
    parts = (("page", page_number, page.extract_text() or "") for page_number, page in enumerate(reader.pages, 1))
    return make_segments(parts)


def extract_docx(path: Path) -> tuple[str, list[dict[str, Any]]]:
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml_bytes)
    parts: list[tuple[str, int | None, str]] = []
    for paragraph in root.iter(f"{WORD_NS}p"):
        text_fragments: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{WORD_NS}t" and node.text:
                text_fragments.append(node.text)
            elif node.tag == f"{WORD_NS}tab":
                text_fragments.append("\t")
            elif node.tag in {f"{WORD_NS}br", f"{WORD_NS}cr"}:
                text_fragments.append("\n")
        text = "".join(text_fragments)
        if text.strip():
            parts.append(("paragraph", None, text))
    return make_segments(parts)


def extract_txt(path: Path) -> tuple[str, list[dict[str, Any]]]:
    raw = path.read_bytes()
    text = ""
    for encoding in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        text = raw.decode("utf-8", errors="replace")
    paragraphs = (part for part in text.replace("\r\n", "\n").split("\n\n"))
    return make_segments(("paragraph", None, part) for part in paragraphs)


def extract_doc(path: Path) -> tuple[str, list[dict[str, Any]]]:
    project_root = Path(__file__).resolve().parents[1]
    private_antiword = project_root / "tools/antiword/root/usr/bin/antiword"
    private_catdoc = project_root / "tools/catdoc/root/usr/bin/catdoc"
    antiword = str(private_antiword) if private_antiword.exists() else shutil.which("antiword")
    catdoc = str(private_catdoc) if private_catdoc.exists() else shutil.which("catdoc")
    antiword_home = project_root / "tools/antiword/root/usr/share/antiword"
    libreoffice = shutil.which("libreoffice") or shutil.which("soffice")
    if antiword:
        result = subprocess.run(
            [antiword, "-m", "UTF-8.txt", str(path)],
            check=False,
            capture_output=True,
            env={**os.environ, "ANTIWORDHOME": str(antiword_home)},
        )
        if result.returncode == 0 and result.stdout.strip():
            text = result.stdout.decode("utf-8", errors="replace")
            return make_segments(("paragraph", None, part) for part in text.split("\n\n"))
    if catdoc:
        result = subprocess.run([catdoc, "-d", "UTF-8", str(path)], check=False, capture_output=True)
        if result.returncode == 0 and result.stdout.strip():
            text = result.stdout.decode("utf-8", errors="replace")
            return make_segments(("paragraph", None, part) for part in text.split("\n\n"))
        catdoc_error = result.stderr.decode("utf-8", errors="replace").strip()
    else:
        catdoc_error = ""
    if libreoffice:
        with tempfile.TemporaryDirectory(prefix="safety_doc_") as temporary_directory:
            subprocess.run(
                [libreoffice, "--headless", "--convert-to", "txt:Text", "--outdir", temporary_directory, str(path)],
                check=True,
                capture_output=True,
            )
            converted = Path(temporary_directory) / f"{path.stem}.txt"
            if not converted.exists():
                raise RuntimeError("LibreOffice did not create a text file")
            return extract_txt(converted)
    if catdoc_error:
        raise RuntimeError(f"legacy_doc_extract_failed: {catdoc_error[:400]}")
    raise RuntimeError("tool_unavailable: install antiword, catdoc, or LibreOffice to extract legacy .doc files")


def extract_document(path: Path, actual_format: str) -> tuple[str, list[dict[str, Any]]]:
    if actual_format == "pdf":
        return extract_pdf(path)
    if actual_format == "docx":
        return extract_docx(path)
    if actual_format == "doc":
        return extract_doc(path)
    if actual_format == "txt":
        return extract_txt(path)
    raise RuntimeError(f"unsupported_format: {actual_format}")


def inventory_files(source_root: Path) -> list[Path]:
    return sorted(path for path in source_root.rglob("*") if path.is_file())


def build(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    document_root = output_root / "documents"
    output_root.mkdir(parents=True, exist_ok=True)
    document_root.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    paths = inventory_files(source_root)
    if args.max_documents:
        paths = paths[: args.max_documents]

    rows: list[dict[str, Any]] = []
    first_by_hash: dict[str, str] = {}
    for index, path in enumerate(paths, 1):
        relative_path = path.relative_to(source_root).as_posix()
        path_hash = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
        document_id = f"doc_{path_hash[:16]}"
        content_hash = sha256_file(path)
        duplicate_of = first_by_hash.get(content_hash, "")
        first_by_hash.setdefault(content_hash, document_id)
        actual_format = detect_format(path)
        row: dict[str, Any] = {
            "document_id": document_id,
            "source_group": relative_path.split("/", 1)[0],
            "relative_path": relative_path,
            "extension": path.suffix.lower().lstrip("."),
            "actual_format": actual_format,
            "size_bytes": path.stat().st_size,
            "sha256": content_hash,
            "duplicate_of": duplicate_of,
            "extract_status": "not_attempted",
            "language": "unknown",
            "segment_count": 0,
            "char_count": 0,
            "error": "",
        }
        if actual_format not in SUPPORTED_TEXT_FORMATS:
            row["extract_status"] = "unsupported"
            row["error"] = f"unsupported_format: {actual_format}"
        elif duplicate_of and not args.extract_duplicates:
            row["extract_status"] = "duplicate"
        else:
            try:
                text, segments = extract_document(path, actual_format)
                if not text.strip():
                    raise RuntimeError("empty_text")
                language = language_hint(text)
                document = {
                    "schema_version": "0.1.0",
                    "document_id": document_id,
                    "source_group": row["source_group"],
                    "relative_path": relative_path,
                    "sha256": content_hash,
                    "actual_format": actual_format,
                    "language": language,
                    "text": text,
                    "segments": segments,
                }
                output_path = document_root / f"{document_id}.json"
                output_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
                row.update(
                    extract_status="success",
                    language=language,
                    segment_count=len(segments),
                    char_count=len(text),
                )
            except Exception as error:  # Keep one bad source from aborting the corpus build.
                row["extract_status"] = "failed"
                row["error"] = str(error).replace("\n", " ")[:500]
        rows.append(row)
        if index % 100 == 0 or index == len(paths):
            print(f"processed {index}/{len(paths)}")

    fieldnames = list(rows[0].keys()) if rows else []
    with args.manifest.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["extract_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    print(json.dumps({"documents": len(rows), "status": status_counts}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/corpus"))
    parser.add_argument("--manifest", type=Path, default=Path("data/catalog/corpus_inventory.csv"))
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument("--extract-duplicates", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
