#!/usr/bin/env python3
"""Shared Draw.io XML builder for active paper diagrams."""

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
