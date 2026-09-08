#!/usr/bin/env python3
"""Check and package the ADE/CoNLL04 manuscript, figures, and generated tables."""
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper/cas-dc"
OUT = ROOT / "output/pdf/ade-conll04"
EXPECTED_FIGURES = [
    "../figures/methodology_detailed_draft.pdf",
]


def captions(text):
    """Extract figure captions, respecting nested TeX braces."""
    result = []
    for figure in re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", text, re.S):
        start = figure.index(r"\caption{") + len(r"\caption{")
        depth, end = 1, start
        while depth:
            char = figure[end]
            if char == "{" and figure[end-1] != "\\":
                depth += 1
            elif char == "}" and figure[end-1] != "\\":
                depth -= 1
            end += 1
        result.append(figure[start:end-1])
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"datasets": ["ade", "conll04"], "study_uses_released_test_results": True,
              "packaging_reads_raw_test_data": False, "documents": {}}
    for stem, target in (("manuscript", "manuscript"),
                         ("title-page", "title-page")):
        log = (TEX / f"{stem}.log").read_text()
        for problem in ("Overfull", "There were undefined references",
                        "There were undefined citations", "! LaTeX Error"):
            if problem in log:
                raise RuntimeError(f"{stem}: {problem}")
        source = TEX / f"{stem}.tex"
        if (TEX / f"{stem}.pdf").stat().st_mtime < source.stat().st_mtime:
            raise RuntimeError(f"Recompile {stem} before packaging")
        document = PdfReader(TEX / f"{stem}.pdf")
        plain = "\n".join(p.extract_text() or "" for p in document.pages)
        if "??" in plain or "\ufffd" in plain:
            raise RuntimeError(f"Unresolved PDF text in {stem}")
        if stem != "title-page":
            text = source.read_text()
            labels = re.findall(r"\\label\{([^}]+)\}", text)
            refs = re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", text)
            assert len(labels) == len(set(labels)), "duplicate labels"
            assert set(refs) <= set(labels), "unresolved labels"
            for obsolete in ('SciERC', 'D100', 'D25', 'D10', '13.71', '3.45'):
                assert obsolete not in plain, f"obsolete manuscript content: {obsolete}"
            for identity in ("huxuelinai@gmail.com", "static@zut.edu.cn", "262102211055"):
                assert identity not in plain, "author information in anonymous PDF"
        dest = OUT / f"{target}.pdf"
        shutil.copy2(TEX / f"{stem}.pdf", dest)
        report["documents"][stem] = {"pages": len(document.pages),
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}

    text = (TEX / "manuscript.tex").read_text()
    figure_paths = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", text)
    if figure_paths != EXPECTED_FIGURES:
        raise RuntimeError(
            "Manuscript figures do not match the numbered figure inventory: "
            f"{figure_paths}"
        )
    figure_captions = captions(text)
    assert len(figure_paths) == len(figure_captions) == 1
    caption_source = "\n\n".join(
        f"\\noindent\\textbf{{Figure {i}.}} {caption}\\par"
        for i, caption in enumerate(figure_captions, 1))
    (OUT / "figure-captions.tex").write_text(caption_source + "\n")
    with zipfile.ZipFile(OUT / "review-source.zip", "w",
                         compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("manuscript.tex", "manuscript.bbl",
                     "cas-dc.cls", "cas-common.sty", "cas-model2-names.bst"):
            archive.write(TEX / name, f"paper/cas-dc/{name}")
        cited = {key.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}",text)
                 for key in group.split(',')}
        entries = re.split(r"(?=^@\w+\{)",(TEX/'references.bib').read_text(),flags=re.M)
        selected=[]
        for entry in entries:
            match=re.match(r"@\w+\{([^,]+),",entry)
            if match and match.group(1) in cited:
                selected.append(entry)
        assert len(selected)==len(cited), 'missing bibliography entry'
        archive.writestr('paper/cas-dc/references.bib',''.join(selected))
        for rel in figure_paths:
            path = (TEX / rel).resolve()
            archive.write(path, str(path.relative_to(ROOT)))
            editable = path.with_suffix(".drawio")
            if editable.exists():
                archive.write(editable, str(editable.relative_to(ROOT)))
        for rel in re.findall(r"\\input\{([^}]+)\}", text):
            path = (TEX / rel).resolve()
            archive.write(path, str(path.relative_to(ROOT)))
        for name in ('dataset_statistics.csv','label_distribution.csv','results_snapshot.json',
                     'evaluator_reconciliation.json','paired_test_bootstrap.json','overlap_sensitivity.json',
                     'source_hashes.json','test_evidence.csv','protocol_snapshot.json',
                     'repeated_run_stability.json','training_loss_availability.md'):
            path = ROOT / 'paper/results/ade_conll04' / name
            archive.write(path,str(path.relative_to(ROOT)))
        archive.write(OUT / "figure-captions.tex", "figure-captions.tex")
        archive.writestr("BUILD.txt", "cd paper/cas-dc\nlatexmk -pdf manuscript.tex\n"
            "Anonymous manuscript source only. Upload the title page separately.\n"
            "ADE and CoNLL04 only; main results plus two repeated-run stability checks.\n"
            "No raw dataset text or model weights are included.\n")
    word_out = OUT / "submission-word"
    word_out.mkdir(parents=True, exist_ok=True)
    for source in sorted((TEX / "words").glob("*.docx")):
        shutil.copy2(source, word_out / source.name)
    figure_out = OUT / "figures"
    figure_out.mkdir(parents=True, exist_ok=True)
    for suffix in (".drawio", ".pdf", ".png", ".svg"):
        source = ROOT / "paper/figures" / f"methodology_detailed_draft{suffix}"
        if source.exists():
            shutil.copy2(source, figure_out / source.name)
    report["figure_count"] = len(figure_paths)
    report["table_count"] = len(re.findall(r"\\begin\{table\}", text))
    report["visual_review_required"] = True
    (OUT / "submission-build-checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
