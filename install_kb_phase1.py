#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create sample phase-1 knowledge-base files and build the local index."""

from __future__ import annotations

import subprocess
import sys
import zipfile
import zlib
from pathlib import Path
from xml.sax.saxutils import escape


TARGET = Path(__file__).resolve().parent

SCIENCE_MD = """# 光合作用与碳循环知识说明

## 摘要
光合作用是绿色植物、藻类和部分细菌利用光能将二氧化碳和水转化为有机物的过程。
该过程释放氧气，并把太阳能转化为化学能，是陆地生态系统碳固定的重要来源。

## 关键机制
- 光反应发生在类囊体膜上，主要作用是产生 ATP 和 NADPH。
- 暗反应又称卡尔文循环，主要作用是固定二氧化碳并合成糖类。
- 影响光合作用速率的因素包括光照强度、二氧化碳浓度、温度和水分条件。

## 可回答的问题
- 什么是光合作用？
- 光反应和暗反应有什么区别？
- 哪些因素会影响植物吸收二氧化碳？
"""

WORD_PARAGRAPHS = [
    "疫苗免疫学基础",
    "疫苗通过向人体呈递抗原或抗原编码信息，诱导免疫系统形成特异性免疫记忆。",
    "体液免疫主要由 B 细胞产生抗体完成。抗体可以中和病原体、阻止其进入细胞，或帮助免疫细胞清除病原体。",
    "细胞免疫主要由 T 细胞参与。辅助性 T 细胞协调免疫反应，细胞毒性 T 细胞可以识别并清除被感染的细胞。",
    "加强针的作用是再次刺激免疫系统，提高抗体水平并改善免疫记忆的持久性。",
    "疫苗效果受抗原设计、接种剂量、接种间隔、个体年龄和基础健康状况等因素影响。",
]

EXCEL_ROWS = [
    ["实验编号", "材料", "温度_C", "压力_kPa", "观察结果", "科学解释"],
    ["EXP-001", "水", "25", "101.3", "液态稳定", "常温常压下水分子间氢键维持液态结构"],
    ["EXP-002", "水", "100", "101.3", "开始沸腾", "蒸气压接近外界压强时发生沸腾"],
    ["EXP-003", "乙醇", "78", "101.3", "开始沸腾", "乙醇分子间作用力弱于水，沸点较低"],
    ["EXP-004", "二氧化碳", "-78.5", "101.3", "升华", "干冰在常压下由固态直接变为气态"],
]

PDF_LINES = [
    "Seismic Waves and Earth's Internal Structure",
    "An earthquake releases energy and generates seismic waves.",
    "P waves are longitudinal waves. They travel through solids, liquids, and gases.",
    "S waves are transverse waves. They travel only through solids.",
    "Because S waves cannot pass through the liquid outer core, scientists infer that the outer core is liquid.",
    "Arrival times, travel paths, and shadow zones reveal Earth's crust, mantle, outer core, and inner core.",
]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def create_docx(path: Path, paragraphs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    paragraph_xml = "".join(f"<w:p><w:r><w:t>{escape(paragraph)}</w:t></w:r></w:p>" for paragraph in paragraphs)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>""",
        )
        zf.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>""",
        )
        zf.writestr(
            "word/document.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{paragraph_xml}<w:sectPr/></w:body></w:document>""",
        )


def col_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def create_xlsx(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    strings: list[str] = []
    indexes: dict[str, int] = {}
    for row in rows:
        for value in row:
            if value not in indexes:
                indexes[value] = len(strings)
                strings.append(value)

    rows_xml: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for col_index, value in enumerate(row, start=1):
            cells.append(f'<c r="{col_name(col_index)}{row_index}" t="s"><v>{indexes[value]}</v></c>')
        rows_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    shared_xml = "".join(f"<si><t>{escape(value)}</t></si>" for value in strings)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>""",
        )
        zf.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>""",
        )
        zf.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="物质状态实验" sheetId="1" r:id="rId1"/></sheets></workbook>""",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/></Relationships>""",
        )
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{''.join(rows_xml)}</sheetData></worksheet>""",
        )
        zf.writestr(
            "xl/sharedStrings.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{len(strings)}" uniqueCount="{len(strings)}">{shared_xml}</sst>""",
        )


def create_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream_lines = ["BT", "/F1 12 Tf", "72 760 Td"]
    for line in PDF_LINES:
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_lines.extend([f"({safe}) Tj", "0 -18 Td"])
    stream_lines.append("ET")
    compressed = zlib.compress("\n".join(stream_lines).encode("latin-1"))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(compressed) + compressed + b"\nendstream",
    ]
    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref_offset = sum(len(chunk) for chunk in chunks)
    xref = [f"xref\n0 {len(objects) + 1}\n".encode("ascii"), b"0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    path.write_bytes(b"".join(chunks + xref + [trailer]))


def run(command: list[str]) -> None:
    print("> " + " ".join(command))
    subprocess.run(command, cwd=str(TARGET), check=True)


def main() -> None:
    write_text(TARGET / "knowledge_base/raw/markdown/photosynthesis_carbon_cycle.md", SCIENCE_MD)
    create_docx(TARGET / "knowledge_base/raw/word/vaccine_immunology.docx", WORD_PARAGRAPHS)
    create_xlsx(TARGET / "knowledge_base/raw/excel/phase_change_experiments.xlsx", EXCEL_ROWS)
    create_pdf(TARGET / "knowledge_base/raw/pdf/seismic_waves_earth_structure.pdf")

    run([sys.executable, "scripts/ingest_knowledge_base.py", "knowledge_base/raw"])
    print(f"Phase-1 sample knowledge base is ready: {TARGET}")


if __name__ == "__main__":
    main()
