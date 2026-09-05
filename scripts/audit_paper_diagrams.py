#!/usr/bin/env python3
"""Audit typography, labels, names, and endpoint ports in paper diagrams."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "figures"
PORT_PATTERN = re.compile(r"(exitX|exitY|entryX|entryY)=([^;]+)")
VERSION_PATTERN = re.compile(r"\bV[123]\b", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]+>")
CARDINAL_PORTS = {
    ("0.5", "0"),
    ("0.5", "1"),
    ("0", "0.5"),
    ("1", "0.5"),
}


def audit(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    errors: list[str] = []
    port_uses: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    vertices = {
        cell.get("id"): cell
        for cell in root.iter("mxCell")
        if cell.get("vertex") == "1"
    }

    for cell in root.iter("mxCell"):
        style = cell.get("style", "")
        cell_id = cell.get("id", "<unknown>")
        value = cell.get("value", "")

        if cell.get("vertex") == "1" and cell_id != "canvas":
            if "fontSize=22" not in style:
                errors.append(f"{cell_id}: vertex font is not 22 pt")
            visible_text = HTML_PATTERN.sub(" ", value)
            if VERSION_PATTERN.search(visible_text):
                errors.append(f"{cell_id}: forbidden V1/V2/V3 name")

        if cell.get("edge") != "1":
            continue

        if "edgeStyle=orthogonalEdgeStyle" not in style or "rounded=0" not in style:
            errors.append(f"{cell_id}: edge is not forced to orthogonal routing")

        ports = dict(PORT_PATTERN.findall(style))
        source = cell.get("source")
        target = cell.get("target")
        if source:
            if not {"exitX", "exitY"}.issubset(ports):
                errors.append(f"{cell_id}: missing explicit exit port")
            else:
                port_uses[(source, "out", ports["exitX"], ports["exitY"])].append(cell_id)
                source_cell = vertices.get(source)
                if source_cell is not None and "ellipse" in source_cell.get("style", ""):
                    if (ports["exitX"], ports["exitY"]) not in CARDINAL_PORTS:
                        errors.append(f"{cell_id}: ellipse exit is not a cardinal port")
        if target:
            if not {"entryX", "entryY"}.issubset(ports):
                errors.append(f"{cell_id}: missing explicit entry port")
            else:
                port_uses[(target, "in", ports["entryX"], ports["entryY"])].append(cell_id)
                target_cell = vertices.get(target)
                if target_cell is not None and "ellipse" in target_cell.get("style", ""):
                    if (ports["entryX"], ports["entryY"]) not in CARDINAL_PORTS:
                        errors.append(f"{cell_id}: ellipse entry is not a cardinal port")

        if value:
            if "fontSize=20" not in style:
                errors.append(f"{cell_id}: edge label font is not 20 pt")
            if "labelBackgroundColor=none" not in style:
                errors.append(f"{cell_id}: edge label background is not transparent")
            geometry = cell.find("mxGeometry")
            if geometry is None or geometry.get("y") in {None, "0", "0.0"}:
                errors.append(f"{cell_id}: edge label is not offset from the line")

    for port, edge_ids in sorted(port_uses.items()):
        if len(edge_ids) > 1:
            errors.append(f"duplicate port {port}: {', '.join(edge_ids)}")

    if path.name == "system_architecture.drawio":
        module_a_outputs = [
            cell.get("id", "<unknown>")
            for cell in root.iter("mxCell")
            if cell.get("edge") == "1"
            and cell.get("source", "").startswith("a_")
            and not cell.get("target", "").startswith("a_")
        ]
        if len(module_a_outputs) < 3:
            errors.append(
                "module A must send budget, source-window, and ontology data "
                "to a downstream module"
            )
    return errors


def main() -> int:
    paths = sorted(FIGURES.glob("*.drawio"))
    if not paths:
        print(f"No Draw.io files found under {FIGURES}", file=sys.stderr)
        return 1

    failed = False
    for path in paths:
        errors = audit(path)
        if errors:
            failed = True
            print(f"FAIL {path.name}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {path.name}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
