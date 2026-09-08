#!/usr/bin/env python3
"""Build the editable submission-side Word documents from verified metadata."""

from datetime import date
from pathlib import Path
import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "submission" / "submission-word"
MANUSCRIPT = ROOT / "paper" / "cas-dc" / "manuscript.tex"
SOURCE = MANUSCRIPT.read_text()
TITLE = re.search(r"\\title\[mode = title\]\{([^}]+)\}", SOURCE).group(1).replace("--", "-")
JOURNAL = "Journal of Safety Science and Resilience"


def setup(document: Document, heading: str) -> None:
    section = document.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)
    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    style.paragraph_format.space_after = Pt(6)
    style.paragraph_format.line_spacing = 1.05
    style.paragraph_format.widow_control = True
    for name in ("Heading 1", "Heading 2"):
        heading_style = document.styles[name]
        heading_style.font.name = "Times New Roman"
        heading_style.font.size = Pt(12)
        heading_style.font.color.rgb = RGBColor(0, 0, 0)
        heading_style.paragraph_format.space_before = Pt(8)
        heading_style.paragraph_format.space_after = Pt(4)
        heading_style.paragraph_format.keep_with_next = True
    document.core_properties.title = heading
    document.core_properties.subject = TITLE
    document.core_properties.author = ""
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(heading)
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(14)


def save(document: Document, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    document.save(OUT / name)


def highlights() -> None:
    document = Document()
    setup(document, "Highlights")
    block = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", SOURCE, re.S).group(1)
    for text in re.findall(r"\\item\s+([^\n]+)", block):
        assert len(text) <= 85, f"Highlight exceeds 85 characters: {text}"
        document.add_paragraph(text, style="List Bullet")
    save(document, "Highlights.docx")


def title_page() -> None:
    document = Document()
    setup(document, "Title Page")
    document.add_heading("Article title", level=1)
    document.add_paragraph(TITLE)
    document.add_heading("Authors", level=1)
    document.add_paragraph(
        "Xuelin Hu¹; Xiaoqin Fu¹; Youjing Fu²; Jingchao Wang³,*; "
        "Ruishen Liu⁴; Charles Jumaa Katila⁵; Pengming Hu³"
    )
    document.add_heading("Affiliations", level=1)
    affiliations = (
        "¹ Liuzhou Railway Vocational Technical College, No. 2 Wenyuan Road, "
        "Yufeng District, Liuzhou 545000, Guangxi Zhuang Autonomous Region, China.",
        "² College of Earth and Environmental Sciences, Lanzhou University, "
        "Lanzhou 730000, China.",
        "³ School of Computer Science, Zhongyuan University of Technology, "
        "Zhengzhou 450007, China.",
        "⁴ School of Artificial Intelligence, Luoyang Normal University, "
        "Luoyang 471000, China.",
        "⁵ School of Computing and Mathematics, The Co-operative University of Kenya, Kenya.",
    )
    for item in affiliations:
        document.add_paragraph(item)
    document.add_heading("Corresponding author", level=1)
    document.add_paragraph("Jingchao Wang; email: static@zut.edu.cn")
    document.add_heading("Author email addresses", level=1)
    for item in (
        "Xuelin Hu: huxuelinai@gmail.com; huxl@ltzy.edu.cn",
        "Xiaoqin Fu: xiaoqin.fu@qq.com; fuxiaoqin@ltzy.edu.cn",
        "Youjing Fu: fuyj2025@lzu.edu.cn",
        "Jingchao Wang: static@zut.edu.cn",
        "Ruishen Liu: liuruishen@lynu.edu.cn",
        "Charles Jumaa Katila: ckatila@cuk.ac.ke",
        "Pengming Hu: hupengming@zut.edu.cn",
    ):
        document.add_paragraph(item)
    document.add_heading("ORCID identifiers", level=1)
    document.add_paragraph(
        "Xuelin Hu: 0000-0002-4475-3034; Xiaoqin Fu: 0009-0003-5123-8393; "
        "Youjing Fu: 0009-0004-1369-7879."
    )
    document.add_heading("Funding", level=1)
    document.add_paragraph(
        "This work was supported by the Science and Technology Research Project of "
        "Henan Province (grant no. 262102211055) and the Key Scientific Research "
        "Project of Colleges and Universities in Henan Province (grant no. 27AQ520018)."
    )
    save(document, "Title Page.docx")


def author_statement() -> None:
    document = Document()
    setup(document, "CRediT Author Statement - Draft for Author Confirmation")
    document.add_paragraph(TITLE)
    document.add_paragraph(
        "Proposed allocation for author review, not a verified record of contributions. "
        "Each author must confirm or correct the roles below to reflect work actually "
        "performed before journal submission."
    )
    for name, roles in (
        ("Xuelin Hu", "Conceptualization; Methodology; Software; Formal analysis; "
         "Investigation; Writing - original draft."),
        ("Xiaoqin Fu", "Data curation; Investigation; Validation; Visualization; "
         "Writing - original draft; Writing - review & editing."),
        ("Youjing Fu", "Methodology; Formal analysis; Investigation; Validation; "
         "Writing - original draft; Writing - review & editing."),
        ("Jingchao Wang", "Conceptualization; Methodology; Supervision; "
         "Project administration; Writing - review & editing."),
        ("Ruishen Liu", "Data curation; Validation; Writing - review & editing."),
        ("Charles Jumaa Katila", "Validation; Writing - review & editing."),
        ("Pengming Hu", "Software; Validation; Writing - review & editing."),
    ):
        paragraph = document.add_paragraph()
        paragraph.add_run(f"{name}: ").bold = True
        paragraph.add_run(roles)
    save(document, "author statement.docx")


def cover_letter() -> None:
    document = Document()
    setup(document, "Cover Letter")
    document.add_paragraph(date.today().strftime("%B %d, %Y"))
    document.add_paragraph("Dear Editors,")
    document.add_paragraph(
        f'Please consider our manuscript entitled "{TITLE}" for publication in the {JOURNAL}.'
    )
    document.add_paragraph(
        "The study addresses two connected questions in graph-augmented extraction: "
        "which retrieved associations are eligible to guide generation, and which "
        "generated candidates should enter the final output. A provenance-constrained "
        "context policy uses training annotations only, excludes current-example-only "
        "support, and quarantines exact cross-split text overlaps. A separate evidence-gated "
        "acceptance mechanism combines source-span reconstruction, entity selection, "
        "relation verification, and endpoint consistency with inspectable decisions."
    )
    document.add_paragraph(
        "Under a common strict-span evaluation, the proposed system improves relation F1 "
        "over matched source-only generation by 5.18 percentage points on ADE and "
        "4.65 points on CoNLL04. The relation gains recur in two additional training "
        "runs and persist after exact CoNLL04 train-test overlaps are excluded. "
        "Development-set ablations distinguish candidate-recovery benefits from the "
        "precision and coverage effects of deterministic acceptance."
    )
    document.add_paragraph(
        "The connection to safety science is the construction of inspectable knowledge "
        "from source text: ADE provides a drug-safety-related extraction task, while "
        "CoNLL04 tests general relational structure. The framework exposes source links "
        "and acceptance decisions for analyst review. The study evaluates extraction and "
        "rule compliance, not clinical truth or measured downstream safety outcomes."
    )
    document.add_paragraph("Thank you for considering our manuscript.")
    document.add_paragraph("Sincerely,")
    document.add_paragraph(
        "Jingchao Wang\nCorresponding author\nSchool of Computer Science, "
        "Zhongyuan University of Technology\nZhengzhou 450007, China\n"
        "Email: static@zut.edu.cn"
    )
    save(document, "cover letter.docx")


def competing_interests() -> None:
    document = Document()
    setup(document, "Declaration of Competing Interests")
    document.add_paragraph(TITLE)
    document.add_paragraph(
        "The authors declare that they have no known competing financial interests or "
        "personal relationships that could have appeared to influence the work reported "
        "in this paper."
    )
    save(document, "declaration-of-competing-interests.docx")


def main() -> None:
    highlights()
    title_page()
    author_statement()
    cover_letter()
    competing_interests()


if __name__ == "__main__":
    main()
