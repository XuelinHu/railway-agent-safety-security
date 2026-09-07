#!/usr/bin/env python3
"""Render and crop the numbered figures from the three reference papers."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "paper" / "ref"
OUT = REF / "extracted_figures"
TMP = ROOT / "tmp" / "pdfs" / "reference_figure_pages"


@dataclass(frozen=True)
class FigureCrop:
    paper_glob: str
    filename: str
    figure: int
    page: int
    # Fractional crop box: left, top, right, bottom.
    crop: tuple[float, float, float, float]
    description: str


FIGURES = (
    FigureCrop("00653*.pdf", "01-dataset-graph-example", 1, 1, (0.455, 0.238, 0.827, 0.402), "数据转换为小型知识图谱的布局参考"),
    FigureCrop("2026-AAAI-MKGC*.pdf", "02-model-paradigm-comparison", 1, 2, (0.082, 0.055, 0.440, 0.278), "传统方法与本文方法的范式对比参考"),
    FigureCrop("00653*.pdf", "03-compact-architecture-reference", 2, 3, (0.075, 0.055, 0.930, 0.330), "只展开创新模块的紧凑总览参考"),
    FigureCrop("2026-AAAI-MKGC*.pdf", "04-hierarchical-fusion-reference", 2, 3, (0.075, 0.050, 0.925, 0.302), "分层输入、融合和推理模块布局参考"),
    FigureCrop("26-AAAI-Sample-specific*.pdf", "05-detailed-module-layout-reference", 1, 3, (0.075, 0.055, 0.925, 0.365), "复杂模块图参考，仅用于判断哪些细节应从主图删除"),
)


def render_page(pdf: Path, page: int, target: Path) -> None:
    prefix = target.with_suffix("")
    subprocess.run(
        [
            "pdftoppm",
            "-f",
            str(page),
            "-l",
            str(page),
            "-r",
            "300",
            "-png",
            "-singlefile",
            str(pdf),
            str(prefix),
        ],
        check=True,
    )


def main() -> None:
    if shutil.which("pdftoppm") is None:
        raise SystemExit("pdftoppm is required to extract vector and raster figures")

    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    index_rows = []

    for spec in FIGURES:
        matches = list(REF.glob(spec.paper_glob))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one match for {spec.paper_glob}, found {len(matches)}")

        pdf = matches[0]
        page_image = TMP / f"{spec.filename}-page-{spec.page}.png"
        if not page_image.exists():
            render_page(pdf, spec.page, page_image)

        with Image.open(page_image) as image:
            width, height = image.size
            left, top, right, bottom = spec.crop
            pixels = (
                round(left * width),
                round(top * height),
                round(right * width),
                round(bottom * height),
            )
            figure = image.crop(pixels).convert("RGB")
            target = OUT / f"{spec.filename}.png"
            figure.save(target, dpi=(300, 300), optimize=True)

        index_rows.append(
            f"| `{target.name}` | {pdf.name} | Figure {spec.figure}, p. {spec.page} | {spec.description} |"
        )

    readme = "\n".join(
        [
            "# Extracted reference figures",
            "",
            "These PNG files are 300-dpi crops rendered from the reference PDFs. They are for layout study and citation-aware comparison; copyright remains with the original authors/publishers.",
            "",
            "| File | Source paper | Location | Content |",
            "|---|---|---:|---|",
            *index_rows,
            "",
            "## Most useful architecture references",
            "",
            "1. `01-dataset-graph-example.png`: data-to-graph example layout.",
            "2. `02-model-paradigm-comparison.png`: compact traditional-versus-proposed paradigm comparison.",
            "3. `03-compact-architecture-reference.png`: left-to-right pipeline that expands only the proposed mechanism.",
            "4. `04-hierarchical-fusion-reference.png`: module-level overview that treats the pre-trained MLLM as one box.",
            "5. `05-detailed-module-layout-reference.png`: a cautionary detail-level reference; do not copy its density into the main figure.",
            "",
        ]
    )
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"Extracted {len(FIGURES)} figures to {OUT}")


if __name__ == "__main__":
    main()
