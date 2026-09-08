#!/usr/bin/env python3
"""Render every delivered manuscript page and make labeled layout contact sheets."""
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/pdfs/ade-conll04-review"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for stem in ("manuscript", "title-page"):
        directory = OUT / stem
        directory.mkdir(exist_ok=True)
        subprocess.run(["pdftoppm", "-scale-to", "1300", "-png",
                        str(ROOT / "output/pdf/ade-conll04" / f"{stem}.pdf"),
                        str(directory / "page")], check=True)
        page_count = len(PdfReader(ROOT / "output/pdf/ade-conll04" / f"{stem}.pdf").pages)
        pages = sorted(p for p in directory.glob("page-*.png")
                       if int(p.stem.split("-")[-1]) <= page_count)
        assert len(pages) == page_count
        for start in range(0, len(pages), 8):
            sheet = Image.new("RGB", (1440, 990), "#dddddd")
            draw = ImageDraw.Draw(sheet)
            for offset, path in enumerate(pages[start:start+8]):
                with Image.open(path) as im:
                    im.thumbnail((350, 465))
                    x, y = (offset % 4) * 360, (offset // 4) * 495
                    sheet.paste(im, (x, y + 22))
                    draw.text((x + 8, y + 5), f"{stem}: p{start+offset+1}", fill="black")
            sheet.save(OUT / f"{stem}-sheet-{start//8+1}.png")
        print(stem, len(pages), "pages rendered")


if __name__ == "__main__":
    main()
