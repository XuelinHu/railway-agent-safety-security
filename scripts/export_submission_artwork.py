#!/usr/bin/env python3
"""Export only the figures referenced by the current two-benchmark manuscript.

Requires a working DISPLAY for Draw.io Desktop, Poppler, Pillow, and pypdf.
Set DISPLAY/XAUTHORITY for the local desktop or run under an Xvfb session.
"""
import re
import subprocess
from pathlib import Path

from PIL import Image
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper/figures"
OUT = ROOT / "output/pdf/ade-conll04/figures"
TMP = ROOT / "tmp/pdfs"


def run(*args):
    subprocess.run([str(x) for x in args], check=True, cwd=ROOT)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    text = (ROOT / "paper/cas-dc/manuscript.tex").read_text()
    paths = re.findall(r"\\includegraphics\[[^]]*\]\{\.\./figures/([^}]+)\}", text)
    for name in paths:
        stem = str(Path(name).with_suffix(''))
        if not (FIG / f"{stem}.drawio").is_file():
            continue
        run("drawio", "--disable-gpu", "--export", "--format", "pdf", "--crop",
            "--output", FIG / f"{stem}.pdf", FIG / f"{stem}.drawio")
        run("drawio", "--disable-gpu", "--export", "--format", "svg", "--theme", "light",
            "--output", FIG / f"{stem}.svg", FIG / f"{stem}.drawio")
        run("pdftoppm", "-scale-to", "3600", "-singlefile", "-png",
            FIG / f"{stem}.pdf", FIG / stem)
    # Numbered PDFs are physically sized to a nominal 145 mm production width.
    # The archival full diagram retains its native large vector canvas.
    width_pt = 145 / 25.4 * 72
    for index, name in enumerate(paths, 1):
        target = OUT / f"fig{index:02d}-{Path(name).stem}"
        page = PdfReader(FIG / name).pages[0]
        page.scale_by(width_pt / float(page.mediabox.width))
        writer = PdfWriter()
        writer.add_page(page)
        writer.add_metadata({"/Title": f"Figure {index}", "/Author": ""})
        with target.with_suffix(".pdf").open("wb") as handle:
            writer.write(handle)
        # Cairo outlines text in EPS, avoiding font substitution at production.
        run("pdftocairo", "-eps", target.with_suffix(".pdf"), target.with_suffix(".eps"))
        if (FIG / name).with_suffix('.drawio').is_file():
            raster = TMP / f"ade-conll04-fig{index:02d}-1000dpi"
            run("pdftoppm", "-r", "1000", "-singlefile", "-png",
                target.with_suffix(".pdf"), raster)
            with Image.open(raster.with_suffix(".png")) as im:
                im.convert("RGB").save(target.with_suffix(".tiff"),
                                       compression="tiff_lzw", dpi=(1000, 1000))
    print(f"Exported {len(paths)} numbered figures to {OUT}")


if __name__ == "__main__":
    main()
