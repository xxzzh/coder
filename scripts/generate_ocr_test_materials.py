#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate OCR and layout extraction test materials with ground truth."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile
from xml.sax.saxutils import escape
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "test_materials" / "ocr"
GROUND_TRUTH = OUT_DIR / "ground_truth.jsonl"


def find_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_dirs = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts"),
        Path("/System/Library/Fonts"),
    ]
    for font_dir in font_dirs:
        for name in candidates:
            path = font_dir / name
            if path.exists():
                return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_LATIN = find_font(["arial.ttf", "calibri.ttf", "segoeui.ttf", "DejaVuSans.ttf"], 52)
FONT_LATIN_SMALL = find_font(["arial.ttf", "calibri.ttf", "segoeui.ttf", "DejaVuSans.ttf"], 38)
FONT_CJK = find_font(["msyh.ttc", "simsun.ttc", "simhei.ttf", "NotoSansCJK-Regular.ttc"], 52)
FONT_CJK_SMALL = find_font(["msyh.ttc", "simsun.ttc", "simhei.ttf", "NotoSansCJK-Regular.ttc"], 38)


def ensure_out() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_image_pdf(path: Path, image: Image.Image) -> None:
    image.convert("RGB").save(path, "PDF", resolution=300)


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str], font: ImageFont.ImageFont, fill: str) -> None:
    x, y = xy
    line_height = int((getattr(font, "size", 40) or 40) * 1.45)
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height


def basic_english_scan() -> dict[str, object]:
    path = OUT_DIR / "scan_english_basic.pdf"
    lines = [
        "Scanned OCR Evaluation - English",
        "ACID means Atomicity, Consistency, Isolation, and Durability.",
        "Ohm's law states that voltage equals current multiplied by resistance.",
        "Water cycle processes include evaporation, condensation, and precipitation.",
    ]
    image = Image.new("RGB", (2400, 1800), "white")
    draw = ImageDraw.Draw(image)
    draw_wrapped(draw, (150, 180), lines, FONT_LATIN, "black")
    save_image_pdf(path, image)
    return record(path, lines, "image_only_pdf", "Clean English scanned PDF.")


def basic_chinese_scan() -> dict[str, object]:
    path = OUT_DIR / "scan_chinese_basic.pdf"
    lines = [
        "中文 OCR 评测材料",
        "水循环包括蒸发、凝结、降水、径流和地下水补给。",
        "数据库 ACID 指原子性、一致性、隔离性和持久性。",
        "光合作用受到光照强度、二氧化碳浓度、温度和水分条件影响。",
    ]
    image = Image.new("RGB", (2600, 1900), "white")
    draw = ImageDraw.Draw(image)
    draw_wrapped(draw, (150, 180), lines, FONT_CJK, "black")
    save_image_pdf(path, image)
    required = [
        "中文 OCR 评测材料",
        "水循环",
        "蒸发",
        "凝结",
        "降水",
        "径流",
        "地下水补给",
        "数据库",
        "ACID",
        "原子性",
        "一致性",
        "隔离性",
        "持久性",
        "光合作用",
        "光照强度",
        "二氧化碳",
        "温度",
        "水分条件",
    ]
    return record(path, lines, "image_only_pdf", "Clean Chinese scanned PDF.", required)


def mixed_table_scan() -> dict[str, object]:
    path = OUT_DIR / "scan_mixed_table.pdf"
    title = "Scanned Table OCR Evaluation"
    rows = [
        ["Concept", "Key fact", "Value"],
        ["ACID", "Atomicity Consistency Isolation Durability", "4 properties"],
        ["Ohm", "Voltage equals current times resistance", "V = I x R"],
        ["DNA", "A pairs with T and C pairs with G", "base pairing"],
        ["水循环", "蒸发 凝结 降水 径流", "四个过程"],
    ]
    image = Image.new("RGB", (2600, 1900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((150, 120), title, fill="black", font=FONT_LATIN)
    x0, y0 = 150, 260
    col_widths = [440, 1320, 560]
    row_h = 190
    for row_index, row in enumerate(rows):
        y = y0 + row_index * row_h
        x = x0
        for col_index, cell in enumerate(row):
            font = FONT_CJK_SMALL if any("\u4e00" <= ch <= "\u9fff" for ch in cell) else FONT_LATIN_SMALL
            draw.text((x + 24, y + 58), cell, fill="black", font=font)
            x += col_widths[col_index]
    save_image_pdf(path, image)
    expected = [title] + [" | ".join(row) for row in rows]
    required = [
        title,
        "Concept",
        "Key fact",
        "Value",
        "ACID",
        "Atomicity Consistency Isolation Durability",
        "4 properties",
        "Ohm",
        "Voltage equals current times resistance",
        "DNA",
        "A pairs with T and C pairs with G",
        "base pairing",
        "水循环",
        "蒸发",
        "凝结",
        "降水",
        "四个过程",
    ]
    return record(path, expected, "image_table_pdf", "Scanned table with English and Chinese cells.", required)


def low_contrast_scan() -> dict[str, object]:
    path = OUT_DIR / "scan_low_contrast.pdf"
    lines = [
        "Low Contrast OCR Evaluation",
        "Weather is short-term atmospheric condition.",
        "Climate is the long-term pattern of weather statistics.",
        "Indexes speed up many database queries but add write overhead.",
    ]
    image = Image.new("RGB", (2400, 1800), "#f2f2f2")
    draw = ImageDraw.Draw(image)
    draw_wrapped(draw, (150, 180), lines, FONT_LATIN, "#565656")
    save_image_pdf(path, image)
    return record(path, lines, "image_only_pdf", "Low contrast scanned PDF.")


def rotated_scan() -> dict[str, object]:
    path = OUT_DIR / "scan_rotated_slightly.pdf"
    lines = [
        "Rotated Scan OCR Evaluation",
        "Newton's second law states that net force equals mass times acceleration.",
        "Momentum equals mass times velocity and is conserved in isolated systems.",
    ]
    image = Image.new("RGB", (2400, 1800), "white")
    draw = ImageDraw.Draw(image)
    draw_wrapped(draw, (160, 260), lines, FONT_LATIN, "black")
    image = image.rotate(2.0, expand=True, fillcolor="white")
    save_image_pdf(path, image)
    return record(path, lines, "image_only_pdf", "Slightly rotated scanned PDF.")


def two_column_digital_pdf() -> dict[str, object]:
    path = OUT_DIR / "complex_two_column_layout.pdf"
    left = [
        "Complex Layout Evaluation",
        "Algorithms organize procedures for solving problems.",
        "Big O notation describes growth in time or space use.",
        "Hash tables map keys to values for fast lookup.",
    ]
    right = [
        "Database Notes",
        "A transaction groups operations into a logical unit of work.",
        "Normalization reduces avoidable duplication.",
        "Durability means committed data should survive crashes.",
    ]
    image = Image.new("RGB", (2600, 1900), "white")
    draw = ImageDraw.Draw(image)
    draw_wrapped(draw, (140, 160), left, FONT_LATIN_SMALL, "black")
    draw.line((1300, 130, 1300, 1600), fill="#999999", width=3)
    draw_wrapped(draw, (1400, 160), right, FONT_LATIN_SMALL, "black")
    save_image_pdf(path, image)
    return record(path, left + right, "two_column_image_pdf", "Two-column image-based layout.")


def create_docx_with_image(source_pdf: Path) -> dict[str, object]:
    path = OUT_DIR / "docx_with_embedded_image.docx"
    image = Image.new("RGB", (1200, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 80), "Embedded image OCR phrase: nitrogen cycle and database index", fill="black", font=FONT_LATIN_SMALL)
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    paragraphs = [
        "DOCX Embedded Image Evaluation",
        "This document intentionally contains text and an embedded image reference.",
        "Visible DOCX text should be extracted from XML.",
        "Image-only content should be extracted by embedded image OCR.",
    ]
    paragraph_xml = "".join(f"<w:p><w:r><w:t>{escape(paragraph)}</w:t></w:r></w:p>" for paragraph in paragraphs)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>""",
        )
        zf.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>""",
        )
        zf.writestr(
            "word/document.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{paragraph_xml}<w:sectPr/></w:body></w:document>""",
        )
        zf.writestr("word/media/embedded_note.png", image_bytes.getvalue())
    expected = [*paragraphs, "Embedded image OCR phrase: nitrogen cycle and database index"]
    return record(path, expected, "docx_with_image", "DOCX with embedded image OCR test.")


def record(
    path: Path,
    expected_lines: list[str],
    kind: str,
    description: str,
    required_phrases: list[str] | None = None,
) -> dict[str, object]:
    return {
        "file": path.name,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "kind": kind,
        "description": description,
        "expected_text": "\n".join(expected_lines),
        "required_phrases": required_phrases or expected_lines,
    }


def main() -> None:
    ensure_out()
    records = [
        basic_english_scan(),
        basic_chinese_scan(),
        mixed_table_scan(),
        low_contrast_scan(),
        rotated_scan(),
        two_column_digital_pdf(),
    ]
    records.append(create_docx_with_image(OUT_DIR / "scan_english_basic.pdf"))

    with GROUND_TRUTH.open("w", encoding="utf-8") as out:
        for item in records:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps({"output_dir": str(OUT_DIR), "materials": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
