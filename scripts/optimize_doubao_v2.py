from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path


FIGURES = Path(__file__).resolve().parents[1] / "paper" / "figures"
TARGET = FIGURES / "archive/doubao-architecture-v2.drawio"


def parse_style(raw: str | None) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for token in (raw or "").split(";"):
        if not token:
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
        else:
            result[token] = None
    return result


def dump_style(style: dict[str, str | None]) -> str:
    return ";".join(key if value is None else f"{key}={value}" for key, value in style.items()) + ";"


def update_style(cell: ET.Element, **updates: str | None) -> None:
    style = parse_style(cell.get("style"))
    for key, value in updates.items():
        if value is None:
            style.pop(key, None)
        else:
            style[key] = value
    cell.set("style", dump_style(style))


def geometry(cell: ET.Element, x: int, y: int, width: int, height: int) -> None:
    node = cell.find("mxGeometry")
    if node is None:
        node = ET.SubElement(cell, "mxGeometry")
    # A vertex should not retain an old edge-point array.
    for child in list(node):
        node.remove(child)
    node.attrib.clear()
    node.set("x", str(x))
    node.set("y", str(y))
    node.set("width", str(width))
    node.set("height", str(height))
    node.set("as", "geometry")


def edge_geometry(
    cell: ET.Element,
    points: list[tuple[int, int]] | None = None,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    label_y: int = -16,
) -> None:
    node = cell.find("mxGeometry")
    if node is None:
        node = ET.SubElement(cell, "mxGeometry")
    for child in list(node):
        node.remove(child)
    node.attrib.clear()
    node.set("relative", "1")
    # In mxGeometry, x is a normalized position along the edge (-1..1), not
    # a pixel offset. Pixel nudges belong on the child mxPoint instead.
    node.set("x", "0")
    node.set("y", str(label_y))
    node.set("as", "geometry")
    offset = ET.SubElement(node, "mxPoint")
    if offset_x:
        offset.set("x", str(offset_x))
    if offset_y:
        offset.set("y", str(offset_y))
    offset.set("as", "offset")
    if points:
        array = ET.SubElement(node, "Array")
        array.set("as", "points")
        for px, py in points:
            point = ET.SubElement(array, "mxPoint")
            point.set("x", str(px))
            point.set("y", str(py))


def set_value(cell: ET.Element, value: str) -> None:
    # Values are stored as HTML in an XML attribute. ElementTree performs the
    # XML escaping on write; keeping the HTML unescaped here avoids double escapes.
    cell.set("value", value)


def format_html(value: str) -> str:
    """Use 22 px for bold/main fragments and 19 px for supporting text."""
    text = html.unescape(value or "")
    # 14/15 px remnants are subtitle sizes from the hand-edited source. Keep
    # deliberate 17 px sizes for dense technical lists so they stay inside the
    # enlarged boxes; ordinary supporting text is 19 px.
    text = re.sub(r"font-size\s*:\s*(?:14|15)px", "font-size:19px", text, flags=re.I)
    text = re.sub(
        r'color\s*=\s*"(?:#ffffff|#5b6573|#4f5965|#9aa4b2|#6b7280)"',
        'color="#1f2937"',
        text,
        flags=re.I,
    )
    # The source values use simple <b> fragments. Wrapping the fragments keeps
    # the title line at 22 px while the inherited body remains 19 px.
    text = text.replace("<b>", '<font style="font-size:22px"><b>').replace("</b>", "</b></font>")
    return text


def apply_palette(cell: ET.Element, fill: str, stroke: str, font: str) -> None:
    update_style(cell, fillColor=fill, strokeColor=stroke, fontColor=font, fontFamily="Arial")


def main() -> None:
    tree = ET.parse(TARGET)
    root = tree.getroot()
    cells = {cell.get("id"): cell for cell in root.findall(".//mxCell") if cell.get("id")}

    # Deliberately compact copy edits keep the enlarged boxes readable at 19 px.
    values = {
        "source_document": "<b>SOURCE DOCUMENT</b><br>token sequence + local evidence spans",
        "subset_kg": "<b>TRAIN-SPLIT KG</b><br>typed entities, relations, evidence &amp; provenance<br><font style=\"font-size:19px\">no validation/test gold</font>",
        "exact_context": "<b>EXACT-ANCHOR CONTEXT</b><br>exact source match + type balance<br>conservative EAE hints",
        "bge_encoder": "<b>FROZEN BGE-M3</b><br>semantic encoder",
        "semantic_retrieval": "<b>DENSE RETRIEVAL</b><br>normalized embeddings → cosine Top-k<br>source/evidence gates",
        "hrge_context": "<b>BOUNDED HIGH-RECALL KG CONTEXT</b><br>exact anchors + edge priors + relation patterns<br><font style=\"font-size:19px\">caps: 12 anchors / 6 edges / 4 patterns</font>",
        "retrieval_note": "<b>OFFLINE RETRIEVAL ONLY</b><br>BGE-M3 is not a GNN<br>no entity/relation decoding",
        "soe_title": "<b>SOE BRANCH</b><br><font style=\"font-size:19px\">Source-only extraction · baseline</font>",
        "soe_prompt": "<b>Source-only prompt</b><br>source + ontology · no KG context",
        "soe_embed": "Token Embedding",
        "soe_backbone": "<b>FROZEN QWEN3-4B</b><br><font style=\"font-size:17px\">36 blocks · RMSNorm · GQA/RoPE · SwiGLU</font>",
        "soe_adapter": "<b>SOE QLoRA ADAPTER</b><br>q/k/v/o + gate/up/down",
        "soe_head": "LM Head → structured sequence",
        "soe_expand": "Deterministic span expansion",
        "soe_output": "<b>E_SOE + R_SOE</b>",
        "baseline_eval": "<b>BASELINE EVALUATION</b><br>excluded from PGE fusion",
        "eae_title": "<b>EAE BRANCH</b><br><font style=\"font-size:19px\">Exact-anchor extraction</font>",
        "eae_prompt": "<b>Prompt serialization</b><br>source + ontology + exact anchors",
        "eae_embed": "Token Embedding",
        "eae_backbone": "<b>FROZEN QWEN3-4B</b><br><font style=\"font-size:17px\">36 blocks · RMSNorm · GQA/RoPE · SwiGLU</font>",
        "eae_adapter": "<b>EAE QLoRA ADAPTER</b><br>q/k/v/o + gate/up/down",
        "eae_head": "LM Head → structured sequence",
        "eae_expand": "Deterministic span expansion",
        "eae_output": "<b>E_EAE + R_EAE</b><br><font style=\"font-size:19px\">standalone comparison only</font>",
        "hrge_title": "<b>HRGE BRANCH</b><br><font style=\"font-size:19px\">High-recall graph extraction</font>",
        "hrge_prompt": "<b>Prompt serialization</b><br>source + ontology + bounded KG context",
        "hrge_embed": "Token Embedding",
        "hrge_backbone": "<b>FROZEN QWEN3-4B</b><br><font style=\"font-size:17px\">36 blocks · RMSNorm · GQA/RoPE · SwiGLU</font>",
        "hrge_adapter": "<b>HRGE QLoRA ADAPTER</b><br>q/k/v/o + gate/up/down",
        "hrge_head": "LM Head → structured sequence",
        "hrge_expand": "Deterministic span expansion",
        "hrge_output": "<b>E_HRGE + R_HRGE</b>",
        "r_hrge": "<b>R_HRGE</b><br>high-recall relation candidates",
        "e_eae": "<b>E_EAE</b><br>exact-anchor entity candidates",
        "e_hrge": "<b>E_HRGE</b><br>high-recall entity candidates",
        "gate_header": "<b>DETERMINISTIC DUAL-GATE COMPOSITION</b><br><font style=\"font-size:19px\">auditable Boolean rules · no learned gate</font>",
        "relation_gate_title": "<b>RELATION COMPLIANCE GATE · V(r)</b>",
        "relation_checks": "<b>ALL SIX CHECKS</b><br><font style=\"font-size:17px\">① distinct endpoint references<br>② known relation type<br>③ legal endpoint types<br>④ permitted claim status<br>⑤ well-formed evidence<br>⑥ local co-occurrence</font>",
        "relation_and": "<b>AND</b>",
        "r_evge": "<b>R_EVGE</b><br>evidence-verified relations",
        "entity_gate_title": "<b>ENTITY ACCEPTANCE GATE · A(e)</b>",
        "agree_signal": "<b>EAE ↔ HRGE</b><br><font style=\"font-size:17px\">same text + type</font>",
        "anchor_signal": "<b>SOURCE ANCHOR</b><br><font style=\"font-size:17px\">same-type, source-gated</font>",
        "endpoint_signal": "<b>VERIFIED ENDPOINT</b><br><font style=\"font-size:17px\">in an R_EVGE relation</font>",
        "entity_or": "<b>ANY / OR</b>",
        "accepted_entities": "<b>E_CFE · ACCEPTED ENTITIES</b><br><font style=\"font-size:17px\">normalized text–type dedup.</font>",
        "final_title": "<b>FINAL PGE PREDICTION</b>",
        "pge_compose": "<b>ENDPOINT-FILTERED COMPOSITION</b><br>retain R_EVGE only when both endpoints survive",
        "pge_formula": "<b>PGE = E_CFE + endpoint-filtered R_EVGE</b><br>directed, typed, evidence-linked graph",
        "legend_title": "<b>FIGURE LEGEND</b>",
        "decoder_title": "<b>QWEN3-4B DECODER-ONLY TRANSFORMER · EXPANDED NEURAL BLOCK</b><br><font style=\"font-size:19px\">shared by SOE, EAE and HRGE</font>",
        "detail_embed": "Token<br>Embedding",
        "detail_stack": "<b>36 × Decoder Blocks</b><br>hidden size 2560",
        "detail_finalnorm": "Final<br>RMSNorm",
        "detail_lmhead": "LM Head",
        "detail_sequence": "Autoregressive structured output<br><font style=\"font-size:17px\">[(span,type)] + [(head,relation,tail)]</font>",
        "block_label": "ONE REPEATED DECODER BLOCK",
        "block_input": "Hidden<br>states",
        "block_norm1": "RMSNorm",
        "block_attn": "<b>CAUSAL GQA + RoPE</b><br><font style=\"font-size:17px\">32 query / 8 KV heads<br><font color=\"#2f1b4c\">LoRA: q, k, v, o</font></font>",
        "block_add1": "+<br>Residual",
        "block_norm2": "RMSNorm",
        "block_mlp": "<b>SWIGLU MLP</b><br><font style=\"font-size:17px\">SiLU(gate) ⊙ up → down<br>FFN size 9728<br><font color=\"#2f1b4c\">LoRA: gate, up, down</font></font>",
        "block_add2": "+<br>Residual",
        "qlora_note": "<b>QLoRA:</b> <font style=\"font-size:17px\">4-bit NF4 · double quant. · BF16 · r=8 · α=16 · dropout=.05</font>",
        "cfe_title": "<b>CFE ABLATION PATH</b><br><font style=\"font-size:19px\">deterministic composition · no extra network</font>",
        "cfe_e": "<b>E_CFE</b><br>accepted entities",
        "cfe_r": "<b>RAW R_HRGE</b><br>unverified relations",
        "cfe_filter": "<b>ENDPOINT FILTER</b><br><font style=\"font-size:17px\">retain raw relation when both endpoints are in E_CFE</font>",
        "cfe_output": "<b>CFE = E_CFE + endpoint-filtered raw R_HRGE</b>",
        "truth_panel": "<b>ARCHITECTURE SEMANTICS</b><br><br><font style=\"font-size:17px\">• SOE, EAE and HRGE are separate Qwen3-4B + QLoRA models.<br><br>• KG records are serialized into prompts; there is no graph neural network.<br><br>• EVGE, CFE and PGE are deterministic filters/compositions.<br><br>• Logical parallelism is shown; one-GPU execution is sequential.</font>",
    }
    for ident, value in values.items():
        if ident in cells:
            set_value(cells[ident], format_html(value))

    # Enlarged geometry: the page remains 3000×1700, while narrow labels get
    # more breathing room and the main regions retain their original ordering.
    vertices = {
        "source_document": (50, 95, 340, 130),
        "subset_kg": (430, 90, 400, 150),
        "exact_context": (455, 245, 350, 95),
        "bge_encoder": (900, 95, 360, 115),
        "semantic_retrieval": (1320, 90, 380, 135),
        "hrge_context": (1760, 90, 460, 150),
        "retrieval_note": (2250, 95, 460, 130),
        "soe_panel": (40, 350, 430, 700),
        "eae_panel": (500, 350, 430, 700),
        "hrge_panel": (960, 350, 430, 700),
        "soe_title": (65, 370, 380, 70),
        "soe_prompt": (75, 460, 360, 80),
        "soe_embed": (75, 555, 360, 60),
        "soe_backbone": (65, 635, 380, 175),
        "soe_adapter": (95, 710, 320, 80),
        "soe_head": (75, 825, 360, 60),
        "soe_expand": (75, 900, 360, 55),
        "soe_output": (85, 970, 340, 55),
        "baseline_eval": (75, 1055, 360, 60),
        "eae_title": (525, 370, 380, 70),
        "eae_prompt": (520, 460, 390, 80),
        "eae_embed": (520, 555, 390, 60),
        "eae_backbone": (525, 635, 380, 175),
        "eae_adapter": (555, 710, 320, 80),
        "eae_head": (520, 825, 390, 60),
        "eae_expand": (520, 900, 390, 55),
        "eae_output": (520, 970, 390, 55),
        "hrge_title": (985, 370, 380, 70),
        "hrge_prompt": (980, 460, 390, 80),
        "hrge_embed": (980, 555, 390, 60),
        "hrge_backbone": (985, 635, 380, 175),
        "hrge_adapter": (1015, 710, 320, 80),
        "hrge_head": (980, 825, 390, 60),
        "hrge_expand": (980, 900, 390, 55),
        "hrge_output": (980, 970, 390, 55),
        "r_hrge": (1420, 440, 260, 95),
        "e_eae": (1420, 690, 260, 95),
        "e_hrge": (1420, 815, 260, 95),
        "gate_panel": (1690, 350, 710, 700),
        "gate_header": (1720, 370, 660, 75),
        "relation_gate_panel": (1720, 465, 660, 260),
        "relation_gate_title": (1740, 482, 620, 45),
        "relation_checks": (1745, 540, 420, 170),
        "relation_and": (2185, 585, 70, 70),
        "r_evge": (2270, 560, 110, 115),
        "entity_gate_panel": (1720, 745, 660, 285),
        "entity_gate_title": (1740, 762, 620, 45),
        "agree_signal": (1745, 825, 190, 110),
        "anchor_signal": (1950, 825, 190, 110),
        "endpoint_signal": (2155, 825, 190, 110),
        "entity_or": (1830, 965, 155, 50),
        "accepted_entities": (2030, 955, 320, 70),
        "final_panel": (2420, 350, 530, 700),
        "final_title": (2450, 370, 470, 65),
        "pge_compose": (2465, 465, 440, 100),
        "graph_boundary": (2500, 600, 370, 300),
        "graph_n1": (2560, 635, 68, 68),
        "graph_n2": (2750, 635, 68, 68),
        "graph_n3": (2658, 720, 86, 86),
        "graph_n4": (2525, 760, 68, 68),
        "graph_n5": (2795, 750, 68, 68),
        "graph_n6": (2595, 825, 68, 68),
        "graph_n7": (2735, 825, 68, 68),
        "pge_formula": (2455, 925, 460, 90),
        "legend_panel": (40, 1120, 430, 465),
        "legend_title": (65, 1140, 380, 55),
        "decoder_panel": (500, 1080, 1225, 535),
        "decoder_title": (525, 1100, 1175, 70),
        "detail_embed": (530, 1195, 170, 80),
        "detail_stack": (735, 1195, 235, 80),
        "detail_finalnorm": (1005, 1195, 170, 80),
        "detail_lmhead": (1200, 1195, 170, 80),
        "detail_sequence": (1395, 1195, 280, 80),
        "block_label": (530, 1310, 1155, 48),
        "block_input": (530, 1385, 120, 120),
        "block_norm1": (670, 1385, 120, 120),
        "block_attn": (810, 1375, 230, 130),
        "block_add1": (1060, 1385, 80, 120),
        "block_norm2": (1160, 1385, 120, 120),
        "block_mlp": (1300, 1375, 215, 130),
        "block_add2": (1535, 1385, 80, 120),
        "qlora_note": (530, 1530, 1155, 70),
        "cfe_panel": (1740, 1080, 640, 535),
        "cfe_title": (1765, 1100, 590, 70),
        "cfe_e": (1780, 1215, 245, 80),
        "cfe_r": (2090, 1215, 245, 80),
        "cfe_filter": (1810, 1360, 445, 100),
        "cfe_output": (1785, 1515, 495, 70),
        "truth_panel": (2420, 1080, 530, 535),
    }
    for ident, values_ in vertices.items():
        if ident in cells:
            geometry(cells[ident], *values_)

    # Color groups: red top, purple middle, green gates/output, blue bottom.
    top = ("#fde7ea", "#b4233c", "#641b2e")
    top_light = ("#fff5f6", "#b4233c", "#641b2e")
    purple = ("#eee4f7", "#7a5aa6", "#2f1b4c")
    purple_panel = ("#f7f1fb", "#7a5aa6", "#2f1b4c")
    purple_title = ("#d9c8ee", "#7a5aa6", "#2f1b4c")
    green = ("#e8f5ec", "#2f7d5b", "#174a32")
    green_panel = ("#f0f8f2", "#2f7d5b", "#174a32")
    green_title = ("#d7eedf", "#2f7d5b", "#174a32")
    blue = ("#e5f0f9", "#4b78a8", "#17324d")
    blue_panel = ("#f4f9fd", "#6c91b3", "#17324d")
    blue_title = ("#c3ddef", "#4b78a8", "#17324d")

    top_ids = {"source_document", "subset_kg", "bge_encoder", "semantic_retrieval", "hrge_context"}
    top_light_ids = {"exact_context", "retrieval_note"}
    for ident in top_ids:
        apply_palette(cells[ident], *top)
    for ident in top_light_ids:
        apply_palette(cells[ident], *top_light)

    middle_panels = {"soe_panel", "eae_panel", "hrge_panel"}
    middle_titles = {"soe_title", "eae_title", "hrge_title"}
    middle_children = {
        "soe_prompt", "soe_embed", "soe_backbone", "soe_adapter", "soe_head", "soe_expand", "soe_output",
        "eae_prompt", "eae_embed", "eae_backbone", "eae_adapter", "eae_head", "eae_expand", "eae_output",
        "hrge_prompt", "hrge_embed", "hrge_backbone", "hrge_adapter", "hrge_head", "hrge_expand", "hrge_output",
        "r_hrge", "e_eae", "e_hrge",
    }
    for ident in middle_panels:
        apply_palette(cells[ident], *purple_panel)
    for ident in middle_titles:
        apply_palette(cells[ident], *purple_title)
    for ident in middle_children:
        apply_palette(cells[ident], *purple)

    gate_panels = {"gate_panel", "relation_gate_panel", "entity_gate_panel", "final_panel"}
    gate_titles = {"gate_header", "relation_gate_title", "entity_gate_title", "final_title"}
    gate_children = {
        "relation_checks", "relation_and", "r_evge", "agree_signal", "anchor_signal", "endpoint_signal",
        "entity_or", "accepted_entities", "pge_compose", "pge_formula", "graph_boundary",
        "graph_n1", "graph_n2", "graph_n3", "graph_n4", "graph_n5", "graph_n6", "graph_n7",
    }
    for ident in gate_panels:
        apply_palette(cells[ident], *green_panel)
    for ident in gate_titles:
        apply_palette(cells[ident], *green_title)
    for ident in gate_children:
        apply_palette(cells[ident], *green)

    bottom_panels = {"legend_panel", "decoder_panel", "cfe_panel", "truth_panel"}
    bottom_titles = {"legend_title", "decoder_title", "cfe_title"}
    bottom_children = {
        "detail_embed", "detail_stack", "detail_finalnorm", "detail_lmhead", "detail_sequence", "block_label",
        "block_input", "block_norm1", "block_attn", "block_add1", "block_norm2", "block_mlp", "block_add2",
        "qlora_note", "cfe_e", "cfe_r", "cfe_filter", "cfe_output",
    }
    for ident in bottom_panels:
        apply_palette(cells[ident], *blue_panel)
    for ident in bottom_titles:
        apply_palette(cells[ident], *blue_title)
    for ident in bottom_children:
        apply_palette(cells[ident], *blue)

    apply_palette(cells["baseline_eval"], "#e5f0f9", "#6c91b3", "#17324d")
    for ident in {"lgt1", "lgt2", "lgt3", "lgt4", "lgt5", "lgt6"}:
        if ident in cells:
            update_style(cells[ident], fontColor="#17324d", fontFamily="Arial")
    # Keep the legend swatches semantically distinct, but make their labels dark.

    # 19 px is the common explanatory size. Short, single-line primary labels
    # are promoted to 22 px without enlarging dense multi-line bodies.
    single_line_22 = {
        "soe_embed", "eae_embed", "hrge_embed", "soe_head", "eae_head", "hrge_head",
        "soe_expand", "eae_expand", "hrge_expand", "soe_output", "relation_and", "entity_or",
        "legend_title", "detail_embed", "detail_finalnorm", "detail_lmhead", "block_label",
        "block_input", "block_norm1", "block_norm2", "block_add1", "block_add2", "graph_n1",
        "graph_n2", "graph_n3", "graph_n4", "graph_n5", "graph_n6", "graph_n7",
    }
    for cell in cells.values():
        if cell.get("vertex") == "1" and cell.get("value"):
            update_style(cell, fontSize="22" if cell.get("id") in single_line_22 else "19", fontFamily="Arial")
            # Remove any remaining pale text from inherited/manual styles.
            style = parse_style(cell.get("style"))
            if style.get("fontColor", "").lower() in {"#ffffff", "#5b6573", "#4f5965", "#9aa4b2", "#6b7280"}:
                update_style(cell, fontColor="#17324d")

    # All edge labels use one transparent, dark 14 px treatment. The line color
    # still reflects the region, but the typography no longer changes.
    labelled_edges = {
        ident for ident, cell in cells.items() if cell.get("edge") == "1" and cell.get("value")
    }
    short_labels = {
        # The destination box already states these details; leaving the
        # short connector unlabeled prevents a 14 px caption from sitting on
        # the 5–10 px gap between the top boxes.
        "kg_to_exact": "",
        "source_to_exact": "",
        "kg_to_bge": "eligible",
        "source_to_bge": "segments",
        "bge_to_retrieval": "vectors",
        "retrieval_to_context": "selected",
        "eae_to_entities": "entities",
        "hrge_to_relations": "relations",
        "hrge_to_entities": "entities",
        "evge_to_endpoint": "verified",
        "evge_to_pge": "R_EVGE",
        "entities_to_pge": "E_CFE",
        "tower_to_detail": "expanded",
        "accepted_to_cfe": "same set",
        "raw_to_cfe": "raw R",
    }
    for ident in labelled_edges:
        cell = cells[ident]
        if ident in short_labels:
            cell.set("value", short_labels[ident])
        update_style(
            cell,
            fontFamily="Arial",
            fontSize="14",
            fontColor="#1f2937",
            labelBackgroundColor="none",
            labelBorderColor="none",
            verticalAlign="middle",
        )

    # Explicit lanes keep converging arrows separated. Labels are offset from
    # the line so transparent text does not sit directly on a stroke.
    edge_points = {
        "kg_to_exact": ([(630, 245)], 165, -30),
        "source_to_exact": ([(410, 190), (410, 288)], -40, -20),
        "kg_to_bge": ([], 0, -18),
        "source_to_bge": ([(220, 70), (1070, 70)], 0, -18),
        "bge_to_retrieval": ([], 0, -18),
        "retrieval_to_context": ([], 0, -18),
        "note_to_bge": ([(2230, 160)], 0, 0),
        "source_to_soe": ([(25, 250), (25, 500)], 0, 0),
        "source_to_eae": ([(445, 250), (445, 500)], 0, 0),
        "source_to_hrge": ([(920, 250), (920, 500)], 0, 0),
        "exact_to_eae": ([(825, 288), (825, 500)], 0, 0),
        "context_to_hrge": ([(1735, 250), (1735, 300), (1400, 300), (1400, 500)], 0, 0),
        "soe_eval_edge": ([], -35, 0),
        "eae_to_entities": ([(690, 1040), (1370, 1040), (1370, 738)], 0, -18),
        "hrge_to_relations": ([(1375, 990), (1395, 990), (1395, 488)], 0, -18),
        "hrge_to_entities": ([(1375, 1010), (1410, 1010), (1410, 862)], 0, -18),
        "eae_to_agree": ([(1695, 738), (1710, 738), (1710, 860), (1745, 860)], 0, -18),
        "hrge_to_agree": ([(1695, 862), (1725, 862), (1725, 900), (1745, 900)], 0, -18),
        "hrge_to_anchor": ([(1695, 945), (1930, 945), (1930, 880)], 0, -18),
        "evge_to_endpoint": ([(2325, 700), (2245, 700)], 0, 12),
        "sig1_to_or": ([(1840, 948), (1860, 965)], 0, 0),
        "sig2_to_or": ([(2045, 952), (1905, 952), (1905, 965)], 0, 0),
        "sig3_to_or": ([(2250, 944), (1980, 944), (1980, 965)], 0, 0),
        "or_to_entities": ([], 0, -18),
        "evge_to_pge": ([], 0, -18),
        "entities_to_pge": ([(2410, 990), (2410, 540)], 0, -18),
        "compose_to_graph": ([], 0, 0),
        "tower_to_detail": ([(925, 810), (925, 1060), (1100, 1060)], 0, -18),
        "accepted_to_cfe": ([(2350, 1040), (2405, 1040), (2405, 1190), (1902, 1190)], 0, -18),
        "raw_to_cfe": ([(1540, 1045), (1710, 1045), (1710, 1190), (2212, 1190)], 0, -18),
    }
    for ident, (points, label_x, label_y) in edge_points.items():
        if ident in cells:
            edge_geometry(cells[ident], points, offset_x=label_x, label_y=label_y)

    # Stable vertical ports for all sequential flows and bottom pipelines.
    vertical_edges = {
        "soe_1", "soe_2", "soe_3", "soe_4", "soe_5", "soe_eval_edge",
        "eae_1", "eae_2", "eae_3", "eae_4", "eae_5",
        "hrge_1", "hrge_2", "hrge_3", "hrge_4", "hrge_5",
        "checks_to_and", "and_to_evge", "cfe_e_to_filter", "cfe_r_to_filter", "cfe_filter_to_output",
        "dp1", "dp2", "dp3", "dp4", "db1", "db2", "db3", "db4", "db5", "db6",
    }
    for ident in vertical_edges:
        if ident not in cells:
            continue
        cell = cells[ident]
        update_style(cell, exitX="0.5", exitY="1", entryX="0.5", entryY="0")
        if ident in {"soe_eval_edge"}:
            update_style(cell, exitX="0.5", exitY="1", entryX="0.5", entryY="0")
        if ident not in edge_points:
            edge_geometry(cell, [], label_y=-16)

    # Region stroke colors for unlabeled edges; dashed explanatory paths remain
    # dashed/open while their labels share the same typography.
    edge_groups = {
        "top": {"kg_to_exact", "source_to_exact", "kg_to_bge", "source_to_bge", "bge_to_retrieval", "retrieval_to_context", "note_to_bge", "source_to_soe", "source_to_eae", "source_to_hrge", "exact_to_eae", "context_to_hrge"},
        "purple": {"soe_1", "soe_2", "soe_3", "eae_1", "eae_2", "eae_3", "hrge_1", "hrge_2", "hrge_3", "eae_to_entities", "hrge_to_relations", "hrge_to_entities"},
        "green": {"soe_4", "eae_4", "hrge_4", "r_to_checks", "checks_to_and", "and_to_evge", "eae_to_agree", "hrge_to_agree", "hrge_to_anchor", "evge_to_endpoint", "sig1_to_or", "sig2_to_or", "sig3_to_or", "or_to_entities", "evge_to_pge", "entities_to_pge", "compose_to_graph"},
        "blue": {"soe_5", "eae_5", "hrge_5", "dp1", "dp2", "dp3", "db1", "db2", "db4", "db5", "tower_to_detail", "cfe_e_to_filter", "cfe_r_to_filter", "cfe_filter_to_output"},
    }
    stroke = {"top": "#b4233c", "purple": "#7a5aa6", "green": "#2f7d5b", "blue": "#4b78a8"}
    for group, ids in edge_groups.items():
        for ident in ids:
            if ident in cells:
                update_style(cells[ident], strokeColor=stroke[group])
    for ident in {"soe_eval_edge", "accepted_to_cfe", "raw_to_cfe", "db_res1", "db_res2"}:
        if ident in cells:
            update_style(cells[ident], strokeColor="#6c91b3", dashed="1", dashPattern="7 5", endArrow="open", endFill="0")
    # Short explanatory links are deliberately shown as dashed callouts so
    # their labels can sit in the surrounding whitespace without competing
    # with the solid data-flow arrows.
    for ident in {"kg_to_exact", "kg_to_bge", "bge_to_retrieval", "retrieval_to_context"}:
        if ident in cells:
            update_style(cells[ident], dashed="1", dashPattern="6 4", endArrow="open", endFill="0")
    for ident in {"db_res1", "db_res2"}:
        if ident in cells:
            edge_geometry(
                cells[ident],
                [(590, 1515), (1120, 1515)] if ident == "db_res1" else [(1120, 1360), (1580, 1360)],
                label_y=-16,
            )

    # Ensure the legend's six swatches remain visible and the XML stays UTF-8.
    tree.write(TARGET, encoding="utf-8", xml_declaration=False)


if __name__ == "__main__":
    main()
