#!/usr/bin/env python3
"""Build the editable draw.io diagrams used by the manuscript."""

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "figures"

BLUE = "#2f6f9f"
BLUE_DARK = "#17324d"
BLUE_FILL = "#eaf2fb"
GREEN = "#2f7d5b"
GREEN_DARK = "#216244"
GREEN_FILL = "#eaf6ef"
AMBER = "#b7791f"
AMBER_DARK = "#7a4e00"
AMBER_FILL = "#fff4d6"
RED = "#b84c6a"
RED_DARK = "#7c3448"
RED_FILL = "#fcecef"
PURPLE = "#7257a6"
PURPLE_DARK = "#513c7a"
PURPLE_FILL = "#f1ecfa"
GREY = "#8aa0b5"
GREY_DARK = "#334e68"
GREY_FILL = "#f7fafc"


class Diagram:
    def __init__(self, diagram_id: str, name: str, width: int, height: int):
        self.mxfile = ET.Element(
            "mxfile",
            {
                "host": "app.diagrams.net",
                "modified": "2026-09-01T02:20:00.000Z",
                "agent": "Codex",
                "version": "24.7.17",
            },
        )
        diagram = ET.SubElement(
            self.mxfile, "diagram", {"id": diagram_id, "name": name}
        )
        graph = ET.SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": str(width),
                "dy": str(height),
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(width),
                "pageHeight": str(height),
                "math": "0",
                "shadow": "0",
            },
        )
        self.root = ET.SubElement(graph, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.vertex(
            "canvas",
            "",
            0,
            0,
            width,
            height,
            fill="#ffffff",
            stroke="#ffffff",
            rounded=False,
            stroke_width=0,
        )

    @staticmethod
    def _vertex_style(
        *,
        fill: str,
        stroke: str,
        bold: bool,
        rounded: bool,
        dashed: bool,
        stroke_width: int,
        font_color: str,
        shape: str | None,
        align: str,
        extra: str,
    ) -> str:
        parts = [
            "whiteSpace=wrap",
            "html=1",
            "fontFamily=Arial",
            "fontSize=22",
            f"fontColor={font_color}",
            f"fillColor={fill}",
            f"strokeColor={stroke}",
            f"strokeWidth={stroke_width}",
            f"align={align}",
            "verticalAlign=middle",
        ]
        if shape == "text":
            parts.insert(0, "text")
        elif shape == "hexagon":
            parts.extend(
                ["shape=hexagon", "perimeter=hexagonPerimeter2", "fixedSize=1"]
            )
        elif shape == "note":
            parts.extend(["shape=note", "backgroundOutline=1"])
        elif rounded:
            parts.extend(["rounded=1", "arcSize=12"])
        if bold:
            parts.append("fontStyle=1")
        if dashed:
            parts.extend(["dashed=1", "dashPattern=8 6"])
        if extra:
            parts.extend(item for item in extra.split(";") if item)
        return ";".join(parts) + ";"

    def vertex(
        self,
        cell_id: str,
        value: str,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        fill: str = "#ffffff",
        stroke: str = GREY,
        font_color: str = BLUE_DARK,
        bold: bool = False,
        rounded: bool = True,
        dashed: bool = False,
        stroke_width: int = 2,
        shape: str | None = None,
        align: str = "center",
        extra: str = "",
    ) -> None:
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "value": value,
                "style": self._vertex_style(
                    fill=fill,
                    stroke=stroke,
                    bold=bold,
                    rounded=rounded,
                    dashed=dashed,
                    stroke_width=stroke_width,
                    font_color=font_color,
                    shape=shape,
                    align=align,
                    extra=extra,
                ),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(x),
                "y": str(y),
                "width": str(width),
                "height": str(height),
                "as": "geometry",
            },
        )

    def edge(
        self,
        cell_id: str,
        source: str,
        target: str,
        value: str = "",
        *,
        color: str = GREY_DARK,
        dashed: bool = False,
        width: int = 2,
        extra: str = "",
        points: list[tuple[int, int]] | None = None,
        label_position: float = 0.0,
        label_offset: int = -28,
    ) -> None:
        style = [
            "edgeStyle=orthogonalEdgeStyle",
            "rounded=0",
            "orthogonalLoop=1",
            "jettySize=auto",
            "html=1",
            "endArrow=block",
            "endFill=1",
            f"strokeColor={color}",
            f"strokeWidth={width}",
            "fontFamily=Arial",
            "fontSize=20",
            "fontStyle=1",
            "labelBackgroundColor=none",
        ]
        if dashed:
            style.extend(["dashed=1", "dashPattern=8 6"])
        if extra:
            style.extend(item for item in extra.split(";") if item)
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "value": value,
                "style": ";".join(style) + ";",
                "edge": "1",
                "parent": "1",
                "source": source,
                "target": target,
            },
        )
        geometry_attributes = {"relative": "1", "as": "geometry"}
        if value:
            geometry_attributes.update(
                {"x": str(label_position), "y": str(label_offset)}
            )
        geometry = ET.SubElement(cell, "mxGeometry", geometry_attributes)
        if value:
            ET.SubElement(geometry, "mxPoint", {"as": "offset"})
        if points:
            point_array = ET.SubElement(geometry, "Array", {"as": "points"})
            for x, y in points:
                ET.SubElement(point_array, "mxPoint", {"x": str(x), "y": str(y)})

    def write(self, filename: str) -> None:
        ET.indent(self.mxfile, space="  ")
        target = FIGURES / filename
        ET.ElementTree(self.mxfile).write(
            target, encoding="utf-8", xml_declaration=True
        )


def add_title(d: Diagram, title: str, subtitle: str, width: int) -> None:
    d.vertex(
        "title",
        title,
        80,
        18,
        width - 160,
        38,
        fill="none",
        stroke="none",
        bold=True,
        shape="text",
        stroke_width=0,
    )
    d.vertex(
        "subtitle",
        subtitle,
        120,
        58,
        width - 240,
        40,
        fill="none",
        stroke="none",
        font_color="#52677a",
        shape="text",
        stroke_width=0,
    )


def build_system_architecture() -> None:
    d = Diagram("system-architecture", "Experimental System Architecture", 2400, 1200)

    phases = [
        ("a", 30, 440, BLUE_FILL, BLUE, "A  DATA, SPLITS, AND ONTOLOGY"),
        ("b", 500, 500, GREEN_FILL, GREEN, "B  PROMPT-CONTEXT CONDITIONS"),
        ("c", 1030, 500, "#fff8e8", AMBER, "C  GENERATION AND EXPANSION"),
        ("d", 1580, 790, "#fbf2f6", RED, "D  VERIFICATION AND FUSION"),
    ]
    for phase_id, x, width, fill, stroke, label in phases:
        d.vertex(f"{phase_id}_panel", "", x, 30, width, 1130, fill=fill, stroke=stroke)
        d.vertex(f"{phase_id}_title", label, x + 25, 55, width - 50, 80, fill=stroke, stroke=stroke, font_color="#ffffff", bold=True)

    d.vertex("a_docs", "<b>REVIEWED DOCUMENTS</b><br>source text + approved spans", 70, 175, 360, 130, fill="#ffffff", stroke=BLUE)
    d.vertex("a_split", "<b>DOCUMENT SPLIT + BUDGETS</b><br>train | validation | sealed test<br>D10 &lt; D25 &lt; D50 &lt; D100", 70, 390, 360, 165, fill="#ffffff", stroke=BLUE)
    d.vertex("a_windows", "<b>SEMANTIC + RESCUE WINDOWS</b><br>800-character cap | 160 overlap<br>protected endpoints and evidence", 70, 645, 360, 175, fill="#ffffff", stroke=BLUE)
    d.vertex("a_policy", "<b>FROZEN ONTOLOGY</b><br>15 entity types | 15 relations<br>legal signatures | claim status", 70, 910, 360, 165, fill="#ffffff", stroke=PURPLE, font_color=PURPLE_DARK)

    d.vertex("b_rule", "<b>GOVERNED TRAIN INPUTS + SUBSET KG</b><br>budgeted windows | frozen ontology<br>exclude document-only candidate support", 540, 175, 420, 125, fill="#ffffff", stroke=GREEN, font_color=GREEN_DARK)
    d.vertex("b_source", "<b>SOURCE-ONLY EXTRACTION (SOE)</b><br>source text only", 540, 370, 420, 115, fill="#ffffff", stroke=GREY)
    d.vertex("b_exact", "<b>EXACT-ANCHOR EXTRACTION (EAE)</b><br>exact graph hints", 540, 555, 420, 130, fill="#ffffff", stroke=BLUE)
    d.vertex("b_recall", "<b>HIGH-RECALL GRAPH EXTRACTION (HRGE)</b><br>bounded graph hints", 540, 760, 420, 205, fill="#ffffff", stroke=GREEN, font_color=GREEN_DARK)

    d.vertex("c_qlora", "<b>THREE SEPARATELY TRAINED GENERATORS</b><br>SOE | EAE | HRGE<br>shared Qwen3-4B QLoRA settings", 1070, 175, 420, 180, fill="#ffffff", stroke=AMBER, font_color=AMBER_DARK)
    d.vertex("c_target", "<b>SHARED COMPACT TARGET</b><br>entity text + type<br>relation endpoints + type + status", 1070, 440, 420, 155, fill="#ffffff", stroke=AMBER, font_color=AMBER_DARK)
    d.vertex("c_expand", "<b>DETERMINISTIC SOURCE EXPANSION</b><br>exact string binding<br>global offsets + evidence spans", 1070, 680, 420, 165, fill="#ffffff", stroke=BLUE)
    d.vertex("c_raw", "<b>DIRECT PREDICTION SETS</b><br>SOE | EAE | HRGE", 1070, 930, 420, 120, fill="#ffffff", stroke=PURPLE, font_color=PURPLE_DARK)

    d.vertex("d_direct", "<b>DIRECTLY EVALUATED SYSTEMS</b><br>SOE | EAE | HRGE", 1620, 175, 710, 125, fill="#ffffff", stroke=GREY)
    d.vertex("d_verified", "<b>EVIDENCE-VERIFIED GRAPH EXTRACTION (EVGE)</b><br>verified HRGE relations", 1620, 385, 710, 140, fill=RED_FILL, stroke=RED, font_color=RED_DARK)
    d.vertex("d_gate", "<b>CONSERVATIVE ENTITY GATE</b><br>accepted HRGE entities", 1620, 610, 710, 165, fill=GREEN_FILL, stroke=GREEN, font_color=GREEN_DARK)
    d.vertex("d_fusion", "<b>CONSERVATIVE-FUSION EXTRACTION (CFE)</b><br>gated entities + raw relations", 1620, 875, 340, 165, fill=PURPLE_FILL, stroke=PURPLE, font_color=PURPLE_DARK)
    d.vertex("d_provenance", "<b>PROVENANCE-GATED EXTRACTION (PGE)</b><br>gated entities + verified relations", 1990, 875, 340, 165, fill=PURPLE_FILL, stroke=PURPLE, font_color=PURPLE_DARK)

    d.edge("a1", "a_docs", "a_split", "", color=BLUE, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0")
    d.edge("a2", "a_split", "a_windows", "", color=BLUE, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0")
    d.edge("ab_budget", "a_split", "b_rule", "", color=BLUE, width=3, extra="exitX=1;exitY=0.3;entryX=0;entryY=0.15", points=[(475, 440), (475, 194)])
    d.edge("ab_windows", "a_windows", "b_rule", "", color=BLUE, width=3, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.4", points=[(485, 733), (485, 225)])
    d.edge("ab_policy", "a_policy", "b_rule", "", color=PURPLE, width=3, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.65", points=[(495, 993), (495, 256)])
    d.edge("b0", "b_rule", "b_source", "omit graph", color=GREY_DARK, dashed=True, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0", label_offset=36)
    d.edge("b1", "b_rule", "b_exact", "", color=BLUE, extra="exitX=0;exitY=0.9;entryX=0;entryY=0.25", points=[(515, 287), (515, 587)])
    d.edge("b2", "b_rule", "b_recall", "", color=GREEN, extra="exitX=1;exitY=0.65;entryX=1;entryY=0.2", points=[(980, 256), (980, 801)])
    d.edge("bc0", "b_source", "c_qlora", "", color=GREY_DARK, width=3, extra="exitX=1;exitY=0.35;entryX=0;entryY=0.2", points=[(1010, 410), (1010, 211)])
    d.edge("bc1", "b_exact", "c_qlora", "", color=BLUE, width=3, extra="exitX=1;exitY=0.6;entryX=0;entryY=0.5", points=[(1018, 633), (1018, 265)])
    d.edge("bc2", "b_recall", "c_qlora", "", color=GREEN, width=3, extra="exitX=1;exitY=0.75;entryX=0;entryY=0.8", points=[(1025, 914), (1025, 319)])
    d.edge("c1", "c_qlora", "c_target", "", color=AMBER, width=3, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0")
    d.edge("c2", "c_target", "c_expand", "", color=AMBER, width=3, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0")
    d.edge("c3", "c_expand", "c_raw", "", color=BLUE, width=3, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0")
    d.edge("cd", "c_raw", "d_direct", "", color=PURPLE, width=3, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.25", points=[(1545, 990), (1545, 206)])
    d.edge("d1", "d_direct", "d_verified", "verify", color=RED, dashed=True, extra="exitX=0.7;exitY=1;entryX=0.7;entryY=0", label_offset=-42)
    d.edge("d2", "d_direct", "d_gate", "", color=GREEN, extra="exitX=0;exitY=0.75;entryX=0;entryY=0.5", points=[(1565, 269), (1565, 692)])
    d.edge("d3", "d_verified", "d_gate", "endpoints", color=GREEN, extra="exitX=0.7;exitY=1;entryX=0.7;entryY=0", label_offset=42)
    d.edge("d4", "d_gate", "d_fusion", "raw", color=PURPLE, extra="exitX=0.25;exitY=1;entryX=0.35;entryY=0", label_offset=-38)
    d.edge("d5", "d_gate", "d_provenance", "verified", color=PURPLE, extra="exitX=0.75;exitY=1;entryX=0.65;entryY=0", label_offset=38)
    d.write("system_architecture.drawio")


def build_model_innovation() -> None:
    d = Diagram("model-innovation", "Conservative Fusion and Verification", 2200, 1350)
    d.vertex("window", "<b>EVIDENCE-PRESERVING SOURCE WINDOW</b><br>source evidence + stable segment identifiers", 250, 35, 1700, 110, fill=GREY_FILL, stroke="#9fb3c8")
    d.vertex("generator", "<b>HRGE CANDIDATE GENERATOR</b><br>Qwen3-4B + QLoRA", 650, 225, 900, 145, fill=AMBER_FILL, stroke=AMBER, shape="hexagon", stroke_width=3)
    d.vertex("raw", "<b>HRGE CANDIDATES</b><br>E_HRGE entities  |  R_HRGE relations", 300, 455, 1600, 125, fill=AMBER_FILL, stroke=AMBER, font_color=AMBER_DARK)

    d.vertex("entity_panel", "", 40, 665, 1030, 480, fill=GREEN_FILL, stroke=GREEN)
    d.vertex("entity_title", "ENTITY ACCEPTANCE GATE  A(e)", 75, 690, 960, 65, fill=GREEN, stroke=GREEN, font_color="#ffffff", bold=True)
    d.vertex("sig_agree", "<b>EAE / HRGE AGREEMENT</b><br>same normalized text + type", 80, 795, 285, 130, fill="#ffffff", stroke=BLUE)
    d.vertex("sig_anchor", "<b>SOURCE ANCHOR</b><br>same source text + type", 395, 795, 285, 130, fill="#ffffff", stroke=GREEN, font_color=GREEN_DARK)
    d.vertex("sig_endpoint", "<b>VERIFIED ENDPOINT</b><br>entity occurs in R_EVGE", 710, 795, 285, 130, fill="#ffffff", stroke=PURPLE, font_color=PURPLE_DARK)
    d.vertex("any_signal", "PASS: ANY SIGNAL", 400, 970, 280, 70, fill="#ffffff", stroke=GREEN, font_color=GREEN_DARK, bold=True, shape="hexagon")
    d.vertex("e_fusion", "<b>PASS: E_CFE</b><br>accepted, deduplicated HRGE entities", 225, 1060, 630, 70, fill=PURPLE_FILL, stroke=PURPLE, font_color=PURPLE_DARK)

    d.vertex("relation_panel", "", 1130, 665, 1030, 480, fill=RED_FILL, stroke=RED)
    d.vertex("relation_title", "RELATION COMPLIANCE GATE  V(r)", 1165, 690, 960, 65, fill=RED, stroke=RED, font_color="#ffffff", bold=True)
    d.vertex("checks", "<b>ALL CHECKS MUST PASS</b><br>valid references + non-self relation<br>ontology signature + claim status<br>evidence + endpoint co-occurrence", 1170, 795, 950, 180, fill="#ffffff", stroke=RED, font_color=RED_DARK)
    d.vertex("r_verified", "<b>PASS: R_EVGE</b><br>accepted HRGE relations", 1190, 1020, 405, 95, fill=GREEN_FILL, stroke=GREEN, font_color=GREEN_DARK)
    d.vertex("r_rejected", "<b>FAIL: REJECTED</b><br>reason recorded", 1710, 1020, 390, 95, fill="#fff5f7", stroke=RED, font_color=RED_DARK, dashed=True)

    d.vertex("fusion_output", "<b>CONSERVATIVE-FUSION EXTRACTION (CFE)</b><br>E_CFE + endpoint-filtered R_HRGE", 220, 1210, 700, 110, fill=PURPLE_FILL, stroke=PURPLE, font_color=PURPLE_DARK)
    d.vertex("provenance_output", "<b>PROVENANCE-GATED EXTRACTION (PGE)</b><br>E_CFE + endpoint-filtered R_EVGE", 1280, 1210, 700, 110, fill=PURPLE_FILL, stroke=PURPLE, font_color=PURPLE_DARK)

    d.edge("w1", "window", "generator", "", color=BLUE, width=3, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0")
    d.edge("w2", "generator", "raw", "", color=AMBER, width=3, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0")
    d.edge("r_to_entity", "raw", "entity_panel", "E_HRGE", color=GREEN, width=3, extra="exitX=0.3;exitY=1;entryX=0.5;entryY=0", label_position=0.15, label_offset=-34)
    d.edge("r_to_relation", "raw", "relation_panel", "R_HRGE", color=RED, width=3, extra="exitX=0.7;exitY=1;entryX=0.5;entryY=0", label_position=0.15, label_offset=34)
    d.edge("e1", "sig_agree", "any_signal", "", color=GREEN, extra="exitX=0.5;exitY=1;entryX=0.3;entryY=0")
    d.edge("e2", "sig_anchor", "any_signal", "", color=GREEN, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0")
    d.edge("e3", "sig_endpoint", "any_signal", "", color=GREEN, extra="exitX=0.5;exitY=1;entryX=0.7;entryY=0")
    d.edge("e4", "any_signal", "e_fusion", "", color=PURPLE, width=3, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0")
    d.edge("rv1", "checks", "r_verified", "", color=GREEN, width=3, extra="exitX=0.3;exitY=1;entryX=0.5;entryY=0")
    d.edge("rv2", "checks", "r_rejected", "", color=RED, dashed=True, extra="exitX=0.8;exitY=1;entryX=0.5;entryY=0")
    d.edge("rv3", "r_verified", "sig_endpoint", "", color=PURPLE, extra="exitX=0;exitY=0.45;entryX=1;entryY=0.6", points=[(1100, 1063), (1100, 873)])
    d.edge("out1", "e_fusion", "fusion_output", "", color=PURPLE, width=3, extra="exitX=0.35;exitY=1;entryX=0.5;entryY=0")
    d.edge("out2", "e_fusion", "provenance_output", "", color=PURPLE, width=3, extra="exitX=1;exitY=0.55;entryX=0.25;entryY=0", points=[(1090, 1098), (1090, 1180), (1455, 1180)])
    d.edge("out3", "r_verified", "provenance_output", "R_EVGE", color=RED, width=3, extra="exitX=0.5;exitY=1;entryX=0.75;entryY=0", points=[(1392, 1160), (1805, 1160)], label_position=0.2, label_offset=-34)
    d.edge("raw_out", "raw", "fusion_output", "", color=AMBER, width=2, extra="exitX=0;exitY=0.5;entryX=0;entryY=0.5", points=[(20, 517), (20, 1265)])
    d.write("model_innovation.drawio")


def build_kg_construction() -> None:
    d = Diagram("kg-construction", "Leakage-Controlled KG Context", 2400, 1120)
    phases = [
        ("a", 30, 510, BLUE_FILL, BLUE, "A  TRAIN-ONLY GRAPH SUPPORT"),
        ("b", 580, 510, GREEN_FILL, GREEN, "B  SOURCE MATCHING"),
        ("c", 1130, 570, "#fff8e8", AMBER, "C  CONTEXT BUILDERS"),
        ("d", 1740, 630, "#fbf2f6", RED, "D  BOUNDED PROMPTS"),
    ]
    for phase_id, x, width, fill, stroke, label in phases:
        d.vertex(f"{phase_id}_panel", "", x, 30, width, 1060, fill=fill, stroke=stroke)
        d.vertex(f"{phase_id}_title", label, x + 25, 55, width - 50, 80, fill=stroke, stroke=stroke, font_color="#ffffff", bold=True)

    d.vertex("a_budget", "<b>REVIEWED TRAINING BUDGET</b><br>D10 | D25 | D50 | D100", 70, 180, 430, 130, fill="#ffffff", stroke=BLUE)
    d.vertex("a_docout", "<b>PROVENANCE ISOLATION GATE</b><br>exclude candidates supported only by d", 70, 395, 430, 150, fill="#ffffff", stroke=RED, font_color=RED_DARK)
    d.vertex("a_graph", "<b>PROVENANCE-BEARING SUBSET KG</b><br>concepts + mentions + typed edges<br>source document identifiers", 70, 630, 430, 180, fill="#ffffff", stroke=GREEN, font_color=GREEN_DARK)

    d.vertex("b_window", "<b>SEMANTIC / RESCUE WINDOW</b><br>current source text remains authoritative", 620, 180, 430, 140, fill="#ffffff", stroke=BLUE)
    d.vertex("b_exact", "<b>EXACT SOURCE MATCHING</b><br>mention anchors + endpoint co-occurrence", 620, 455, 430, 145, fill="#ffffff", stroke=GREEN, font_color=GREEN_DARK)
    d.vertex("b_embed", "<b>SEMANTIC RETRIEVAL</b><br>frozen BGE-M3", 620, 740, 430, 140, fill="#ffffff", stroke=PURPLE, font_color=PURPLE_DARK)

    d.vertex("c_exact", "<b>EXACT-ANCHOR CONTEXT</b><br>balanced, exactly matched concept hints", 1170, 165, 490, 145, fill="#ffffff", stroke=BLUE)
    d.vertex("c_recall_anchor", "<b>HIGH-RECALL SOURCE ANCHORS</b><br>empirical type purity &gt;= 0.8", 1170, 385, 490, 130, fill="#ffffff", stroke=GREEN, font_color=GREEN_DARK)
    d.vertex("c_recall_edge", "<b>HIGH-RECALL TYPED EDGE PRIORS</b><br>observed outside d; endpoints in source", 1170, 600, 490, 140, fill="#ffffff", stroke=AMBER, font_color=AMBER_DARK)
    d.vertex("c_recall_sem", "<b>HIGH-RECALL SEMANTIC PATTERNS</b><br>BGE-M3 similarity &gt;= 0.72", 1170, 820, 490, 130, fill="#ffffff", stroke=PURPLE, font_color=PURPLE_DARK)

    d.vertex("d_caps", "<b>FROZEN HIGH-RECALL CAPS</b><br>12 anchors; max 2 per type<br>6 edge priors; 4 semantic patterns", 1780, 165, 550, 170, fill="#ffffff", stroke=RED, font_color=RED_DARK)
    d.vertex("d_exact", "<b>EAE PROMPT</b><br>exact conservative hints", 1780, 425, 550, 125, fill=BLUE_FILL, stroke=BLUE)
    d.vertex("d_recall", "<b>HRGE PROMPT</b><br>bounded high-recall context", 1780, 650, 550, 125, fill=GREEN_FILL, stroke=GREEN, font_color=GREEN_DARK)
    d.vertex("d_use", "<b>SEPARATE QLoRA RUNS</b><br>hints are not facts<br>unsupported output is forbidden", 1780, 870, 550, 150, fill="#ffffff", stroke=AMBER, font_color=AMBER_DARK)

    d.edge("a1", "a_budget", "a_docout", "", color=RED, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0")
    d.edge("a2", "a_docout", "a_graph", "", color=GREEN, width=3, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0")
    d.edge("b1", "b_window", "b_exact", "", color=BLUE, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0")
    d.edge("b2", "b_window", "b_embed", "", color=PURPLE, extra="exitX=1;exitY=0.65;entryX=0.8;entryY=0", points=[(1070, 271), (1070, 700), (964, 700)])
    d.edge("ab", "a_graph", "b_exact", "", color=GREEN, width=3, extra="exitX=1;exitY=0.45;entryX=0;entryY=0.3", points=[(555, 711), (555, 498)])
    d.edge("bc1", "b_exact", "c_exact", "", color=BLUE, width=3, extra="exitX=1;exitY=0.2;entryX=0;entryY=0.5", points=[(1110, 484), (1110, 237)])
    d.edge("bc2", "b_exact", "c_recall_anchor", "", color=GREEN, width=3, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5")
    d.edge("bc3", "b_exact", "c_recall_edge", "", color=AMBER, width=3, extra="exitX=1;exitY=0.8;entryX=0;entryY=0.5", points=[(1115, 571), (1115, 670)])
    d.edge("bc4", "b_embed", "c_recall_sem", "", color=PURPLE, width=3, extra="exitX=1;exitY=0.6;entryX=0;entryY=0.5")
    d.edge("cd1", "c_exact", "d_exact", "", color=BLUE, width=3, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.3", points=[(1720, 237), (1720, 462)])
    d.edge("cd2", "c_recall_anchor", "d_recall", "", color=GREEN, width=3, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.2", points=[(1710, 450), (1710, 675)])
    d.edge("cd3", "c_recall_edge", "d_recall", "", color=AMBER, width=3, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5")
    d.edge("cd4", "c_recall_sem", "d_recall", "", color=PURPLE, width=3, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.8", points=[(1730, 885), (1730, 750)])
    d.edge("d1", "d_caps", "d_recall", "", color=RED, extra="exitX=1;exitY=0.5;entryX=1;entryY=0.3", points=[(2350, 250), (2350, 687)])
    d.edge("d2", "d_exact", "d_use", "", color=BLUE, extra="exitX=0.3;exitY=1;entryX=0.3;entryY=0", points=[(1760, 585), (1760, 835), (1945, 835)])
    d.edge("d3", "d_recall", "d_use", "", color=GREEN, extra="exitX=0.7;exitY=1;entryX=0.7;entryY=0")
    d.write("kg_construction_process.drawio")


def build_representation_flow() -> None:
    d = Diagram(
        "representation-flow",
        "Graph Structure and Representation Flow",
        2500,
        1320,
    )
    phases = [
        ("a", 30, 560, BLUE_FILL, BLUE, "A  PROVENANCE-BEARING DIRECTED KG"),
        ("b", 620, 600, GREEN_FILL, GREEN, "B  EMBEDDING REPRESENTATION"),
        ("c", 1250, 560, "#fff8e8", AMBER, "C  RETRIEVAL AND EVIDENCE GATING"),
        ("d", 1840, 630, "#fbf2f6", RED, "D  EVIDENCE-GATED DIRECTED OUTPUT"),
    ]
    for phase_id, x, width, fill, stroke, label in phases:
        d.vertex(
            f"{phase_id}_panel",
            "",
            x,
            30,
            width,
            1260,
            fill=fill,
            stroke=stroke,
        )
        d.vertex(
            f"{phase_id}_title",
            label,
            x + 25,
            55,
            width - 50,
            80,
            fill=stroke,
            stroke=stroke,
            font_color="#ffffff",
            bold=True,
        )

    # Phase A: a directed, typed graph whose records retain source provenance.
    d.vertex("kg_hazard", "<b>HAZARD</b><br>low adhesion", 80, 195, 190, 100, fill="#ffffff", stroke=RED, font_color=RED_DARK, rounded=False, extra="ellipse")
    d.vertex("kg_failure", "<b>FAILURE</b><br>wheel slide", 350, 195, 190, 100, fill="#ffffff", stroke=AMBER, font_color=AMBER_DARK, rounded=False, extra="ellipse")
    d.vertex("kg_event", "<b>EVENT</b><br>signal overrun", 215, 430, 190, 100, fill="#ffffff", stroke=BLUE, rounded=False, extra="ellipse")
    d.vertex("kg_control", "<b>CONTROL</b><br>sand application", 60, 680, 210, 100, fill="#ffffff", stroke=GREEN, font_color=GREEN_DARK, rounded=False, extra="ellipse")
    d.vertex("kg_consequence", "<b>CONSEQUENCE</b><br>collision risk", 340, 680, 210, 100, fill="#ffffff", stroke=PURPLE, font_color=PURPLE_DARK, rounded=False, extra="ellipse")
    d.vertex("kg_records", "<b>TYPED TRIPLES + PROVENANCE</b><br>(head, relation, tail, status)<br>document ID + segment ID + source span", 100, 960, 420, 175, fill="#ffffff", stroke=GREEN, font_color=GREEN_DARK)

    d.edge("kg_e1", "kg_hazard", "kg_event", "contributes-to", color=RED, width=3, extra="exitX=0.5;exitY=1;entryX=0;entryY=0.5", points=[(175, 360), (175, 480)], label_position=-0.1, label_offset=-34)
    d.edge("kg_e2", "kg_failure", "kg_event", "causes", color=AMBER, width=3, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0", points=[(445, 390), (310, 390)], label_position=0.1, label_offset=34)
    d.edge("kg_e3", "kg_control", "kg_hazard", "mitigates", color=GREEN, width=3, extra="exitX=0;exitY=0.5;entryX=0;entryY=0.5", points=[(45, 730), (45, 245)], label_position=0.15, label_offset=-60)
    d.edge("kg_e4", "kg_event", "kg_consequence", "leads-to", color=PURPLE, width=3, extra="exitX=1;exitY=0.5;entryX=1;entryY=0.5", points=[(570, 480), (570, 730)], label_position=0.2, label_offset=34)
    d.edge("kg_e5", "kg_event", "kg_records", "evidence spans", color=BLUE, width=3, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0", label_position=-0.65, label_offset=-80)

    # Phase B: the graph records and current source are encoded separately.
    d.vertex("graph_text", "<b>GRAPH TEXT RECORDS</b><br>typed relations + provenance", 660, 200, 230, 145, fill="#ffffff", stroke=GREEN, font_color=GREEN_DARK)
    d.vertex("source_window", "<b>CURRENT SOURCE WINDOW</b><br>authoritative report text", 920, 200, 230, 145, fill="#ffffff", stroke=BLUE)
    d.vertex("encoder", "<b>FROZEN BGE-M3 ENCODER</b><br>multilingual vectorization", 760, 445, 320, 140, fill=PURPLE_FILL, stroke=PURPLE, font_color=PURPLE_DARK, shape="hexagon", stroke_width=3)
    d.vertex("matrix_h", "<b>RELATION EMBEDDING MATRIX</b><br>H in R^(m x d)<br>[ h11  ...  h1d ]<br>[  ...  ...  ... ]<br>[ hm1  ...  hmd ]", 660, 720, 260, 260, fill="#ffffff", stroke=PURPLE, font_color=PURPLE_DARK)
    d.vertex("query_q", "<b>QUERY VECTOR</b><br>q in R^d<br>[ q1  ...  qd ]^T", 950, 750, 220, 200, fill="#ffffff", stroke=BLUE)

    d.edge("ab_records", "kg_records", "graph_text", "", color=GREEN, width=3, extra="exitX=1;exitY=0.25;entryX=0;entryY=0.75", points=[(600, 1004), (600, 309)])
    d.edge("enc_graph", "graph_text", "encoder", "", color=GREEN, width=3, extra="exitX=0.5;exitY=1;entryX=0.3;entryY=0")
    d.edge("enc_source", "source_window", "encoder", "", color=BLUE, width=3, extra="exitX=0.5;exitY=1;entryX=0.7;entryY=0")
    d.edge("enc_matrix", "encoder", "matrix_h", "", color=PURPLE, width=3, extra="exitX=0.3;exitY=1;entryX=0.5;entryY=0")
    d.edge("enc_query", "encoder", "query_q", "", color=BLUE, width=3, extra="exitX=0.7;exitY=1;entryX=0.5;entryY=0")

    # Phase C: similarity ranking remains subordinate to current-source evidence.
    d.vertex("score", "<b>SIMILARITY TRANSFORMATION</b><br>s = Hq  in R^m", 1300, 210, 460, 135, fill="#ffffff", stroke=PURPLE, font_color=PURPLE_DARK)
    d.vertex("topk", "<b>TOP-k RELATION PATTERNS</b><br>I_k = TopK(s; k)", 1300, 455, 460, 135, fill="#ffffff", stroke=AMBER, font_color=AMBER_DARK)
    d.vertex("source_evidence", "<b>SOURCE EVIDENCE</b><br>segment IDs + spans", 1290, 720, 220, 150, fill="#ffffff", stroke=BLUE)
    d.vertex("evidence_gate", "<b>SOURCE-EVIDENCE FILTER</b><br>eligible graph hints", 1540, 720, 230, 150, fill=GREEN_FILL, stroke=GREEN, font_color=GREEN_DARK)
    d.vertex("bounded_context", "<b>BOUNDED HRGE CONTEXT</b><br>ranked patterns are hints; source text is authoritative", 1300, 1020, 460, 140, fill="#ffffff", stroke=AMBER, font_color=AMBER_DARK)

    d.edge("h_score", "matrix_h", "score", "", color=PURPLE, width=3, extra="exitX=1;exitY=0.3;entryX=0;entryY=0.35", points=[(935, 798), (935, 380), (1190, 380), (1190, 257)])
    d.edge("q_score", "query_q", "score", "", color=BLUE, width=3, extra="exitX=1;exitY=0.65;entryX=0;entryY=0.75", points=[(1270, 880), (1270, 311)])
    d.edge("rank", "score", "topk", "", color=PURPLE, width=3, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0")
    d.edge("source_to_evidence", "source_window", "source_evidence", "", color=BLUE, width=3, extra="exitX=0.9;exitY=1;entryX=0;entryY=0.5", points=[(1210, 345), (1210, 795)])
    d.edge("patterns_to_gate", "topk", "evidence_gate", "", color=AMBER, width=3, extra="exitX=0.65;exitY=1;entryX=0.5;entryY=0", points=[(1599, 655), (1655, 655)])
    d.edge("evidence_to_gate", "source_evidence", "evidence_gate", "", color=GREEN, width=3, extra="exitX=1;exitY=0.65;entryX=0;entryY=0.65")
    d.edge("gate_to_context", "evidence_gate", "bounded_context", "", color=GREEN, width=3, extra="exitX=0.5;exitY=1;entryX=0.7;entryY=0")

    # Phase D: PGE returns a directed risk graph with traceable evidence IDs.
    d.vertex("pge", "<b>PGE EXTRACTOR</b><br>evidence-gated objects", 1880, 190, 550, 145, fill=PURPLE_FILL, stroke=PURPLE, font_color=PURPLE_DARK, shape="hexagon", stroke_width=3)
    d.vertex("out_hazard", "<b>HAZARD</b><br>low adhesion", 1850, 500, 210, 100, fill="#ffffff", stroke=RED, font_color=RED_DARK, rounded=False, extra="ellipse")
    d.vertex("out_event", "<b>EVENT</b><br>signal overrun", 2190, 500, 210, 100, fill="#ffffff", stroke=BLUE, rounded=False, extra="ellipse")
    d.vertex("out_control", "<b>CONTROL</b><br>sand application", 1850, 760, 210, 100, fill="#ffffff", stroke=GREEN, font_color=GREEN_DARK, rounded=False, extra="ellipse")
    d.vertex("out_consequence", "<b>CONSEQUENCE</b><br>collision risk", 2190, 760, 210, 100, fill="#ffffff", stroke=PURPLE, font_color=PURPLE_DARK, rounded=False, extra="ellipse")
    d.vertex("out_evidence", "<b>TRACEABLE OUTPUT</b><br>document ID + segment ID<br>evidence spans + claim status", 1960, 1040, 400, 145, fill="#ffffff", stroke=GREEN, font_color=GREEN_DARK)

    d.edge("context_to_pge", "bounded_context", "pge", "", color=GREEN, width=3, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.75", points=[(1825, 1090), (1825, 299)])
    d.edge("pge_entities", "pge", "out_hazard", "", color=PURPLE, width=3, extra="exitX=0.3;exitY=1;entryX=0.5;entryY=0")
    d.edge("pge_relations", "pge", "out_event", "", color=RED, width=3, extra="exitX=0.75;exitY=1;entryX=0.5;entryY=0")
    d.edge("out_e1", "out_hazard", "out_event", "contributes-to", color=RED, width=3, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5", label_offset=-34)
    d.edge("out_e2", "out_control", "out_hazard", "mitigates", color=GREEN, width=3, extra="exitX=0.5;exitY=0;entryX=0.5;entryY=1", label_offset=34)
    d.edge("out_e3", "out_event", "out_consequence", "leads-to", color=PURPLE, width=3, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0", label_offset=-34)
    d.edge("out_e4", "out_event", "out_evidence", "evidence", color=BLUE, width=3, extra="exitX=1;exitY=0.5;entryX=1;entryY=0.5", points=[(2445, 550), (2445, 1112)], label_position=0.2, label_offset=-44)
    d.write("representation_flow.drawio")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    build_system_architecture()
    build_model_innovation()
    build_kg_construction()
    build_representation_flow()


if __name__ == "__main__":
    main()
