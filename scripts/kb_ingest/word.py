"""Word documents: .docx text and images, .doc conversion through Word."""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from kb_ingest.ocr import ocr_image_bytes
from kb_ingest.reviews import extraction_report


W_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def w_tag(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1]


def w_value(node: ET.Element | None, path: str) -> str | None:
    found = node.find(path.replace("w:", W_NAMESPACE)) if node is not None else None
    return found.get(W_NAMESPACE + "val") if found is not None else None


def style_heading_levels(styles: ET.Element | None) -> dict[str, int]:
    """Paragraph style id -> Markdown heading level: the Title style is 1, Heading N (标题 N in a
    Chinese Word, same internal name) and styles with outline level N-1 are N + 1."""
    if styles is None:
        return {}
    defined: dict[str, tuple[str, str | None, str | None]] = {}
    for style in styles.iter(W_NAMESPACE + "style"):
        if style.get(W_NAMESPACE + "type") == "paragraph" and style.get(W_NAMESPACE + "styleId"):
            name = (w_value(style, "w:name") or "").strip().lower()
            defined[style.get(W_NAMESPACE + "styleId")] = (name, w_value(style, "w:pPr/w:outlineLvl"), w_value(style, "w:basedOn"))

    def level(style_id: str, depth: int = 0) -> int:
        if style_id not in defined or depth > 10:
            return 0
        name, outline, based_on = defined[style_id]
        if name == "title":
            return 1
        heading = re.fullmatch(r"heading (\d)", name)
        if heading:
            return min(6, int(heading.group(1)) + 1)
        if outline is not None and outline.isdigit() and int(outline) < 9:
            return min(6, int(outline) + 2)
        return level(based_on, depth + 1) if based_on else 0

    return {style_id: found for style_id in defined if (found := level(style_id))}


def style_lists(styles: ET.Element | None) -> dict[str, tuple[str, str]]:
    """Paragraph style id -> (numId, level) of the list numbering the style itself carries (List
    Number, List Bullet)."""
    found: dict[str, tuple[str, str]] = {}
    for style in styles.iter(W_NAMESPACE + "style") if styles is not None else []:
        num_id = w_value(style, "w:pPr/w:numPr/w:numId")
        if style.get(W_NAMESPACE + "styleId") and num_id:
            found[style.get(W_NAMESPACE + "styleId")] = (num_id, w_value(style, "w:pPr/w:numPr/w:ilvl") or "0")
    return found


def list_formats(numbering: ET.Element | None) -> dict[tuple[str, str], str]:
    """(numId, level) -> number format of Word's list numbering (decimal, bullet, ...)."""
    if numbering is None:
        return {}
    abstract: dict[str, dict[str, str]] = {}
    for item in numbering.iter(W_NAMESPACE + "abstractNum"):
        abstract[item.get(W_NAMESPACE + "abstractNumId", "")] = {
            lvl.get(W_NAMESPACE + "ilvl", "0"): w_value(lvl, "w:numFmt") or "decimal" for lvl in item.iter(W_NAMESPACE + "lvl")
        }
    formats: dict[tuple[str, str], str] = {}
    for num in numbering.iter(W_NAMESPACE + "num"):
        for level, number_format in abstract.get(w_value(num, "w:abstractNumId") or "", {}).items():
            formats[(num.get(W_NAMESPACE + "numId", ""), level)] = number_format
    return formats


def paragraph_text(paragraph: ET.Element) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        tag = w_tag(node)
        if tag == "t" and node.text:
            parts.append(node.text)
        elif tag == "tab":
            parts.append("\t")
        elif tag in {"br", "cr"}:
            parts.append("\n")
    return "".join(parts).strip()


def table_lines(table: ET.Element) -> list[str]:
    """A Word table as Markdown rows, its first row the header. A one-column table is a text box
    drawn as a table: its paragraphs are plain lines. Cells inside a merged cell stay empty."""
    rows: list[list[str]] = []
    for row in table.findall(W_NAMESPACE + "tr"):
        cells: list[str] = []
        for cell in row.findall(W_NAMESPACE + "tc"):
            text = " ".join(line for line in (paragraph_text(p) for p in cell.iter(W_NAMESPACE + "p")) if line)
            cells.append(re.sub(r"\s+", " ", text).replace("|", "/").strip())
            span = w_value(cell, "w:tcPr/w:gridSpan")
            cells.extend([""] * (int(span) - 1 if span and span.isdigit() else 0))
        if any(cells):
            rows.append(cells)
    if not rows:
        return []
    if max(len(cells) for cells in rows) == 1:
        return [cells[0] for cells in rows if cells[0]]
    width = max(len(cells) for cells in rows)
    rows = [cells + [""] * (width - len(cells)) for cells in rows]
    return ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |", *("| " + " | ".join(cells) + " |" for cells in rows[1:])]


def read_docx(path: Path) -> str:
    """The document's text in reading order: heading styles become Markdown headings (sections
    are then split and cited like a Markdown file's), list paragraphs keep their numbers or
    bullets, tables become Markdown rows instead of one line per cell."""
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        root = ET.fromstring(zf.read("word/document.xml"))
        styles = ET.fromstring(zf.read("word/styles.xml")) if "word/styles.xml" in names else None
        numbering = ET.fromstring(zf.read("word/numbering.xml")) if "word/numbering.xml" in names else None
    headings = style_heading_levels(styles)
    style_numbering = style_lists(styles)
    formats = list_formats(numbering)
    counters: dict[tuple[str, str], int] = {}
    blocks: list[list[str]] = [[]]

    def add_paragraph(paragraph: ET.Element) -> None:
        text = paragraph_text(paragraph)
        if not text:
            return
        properties = paragraph.find(W_NAMESPACE + "pPr")
        outline = w_value(properties, "w:outlineLvl")
        style = w_value(properties, "w:pStyle") or ""
        level = headings.get(style, 0)
        if outline is not None and outline.isdigit() and int(outline) < 9:
            level = min(6, int(outline) + 2)
        if level and len(text) <= 120:
            blocks.extend([["#" * level + " " + re.sub(r"\s+", " ", text)], []])
            counters.clear()
            return
        num_id, list_level = style_numbering.get(style, (None, "0"))
        num_id = w_value(properties, "w:numPr/w:numId") or num_id
        list_level = w_value(properties, "w:numPr/w:ilvl") or list_level
        if num_id and num_id != "0":
            number_format = formats.get((num_id, list_level), "decimal")
            if number_format == "bullet":
                text = f"- {text}"
            elif number_format != "none":
                for key in [key for key in counters if key[0] == num_id and key[1] > list_level]:
                    del counters[key]
                counters[(num_id, list_level)] = counters.get((num_id, list_level), 0) + 1
                text = f"{counters[(num_id, list_level)]}. {text}"
        blocks[-1].append(text)

    def walk(parent: ET.Element) -> None:
        for child in parent:
            tag = w_tag(child)
            if tag == "p":
                add_paragraph(child)
            elif tag == "tbl":
                lines = table_lines(child)
                if lines:
                    blocks.extend([lines, []])
            elif tag in {"sdt", "sdtContent", "customXml", "smartTag"}:
                walk(child)

    body = root.find(W_NAMESPACE + "body")
    walk(body if body is not None else root)
    return "\n\n".join("\n".join(lines) for lines in blocks if lines)


def find_winword() -> str | None:
    found = shutil.which("WINWORD.EXE")
    if found:
        return found
    candidates: list[Path] = []
    for office_version in ("16", "15", "14"):
        candidates.extend(
            (
                Path(f"C:/Program Files/Microsoft Office/root/Office{office_version}/WINWORD.EXE"),
                Path(f"C:/Program Files (x86)/Microsoft Office/root/Office{office_version}/WINWORD.EXE"),
                Path(f"C:/Program Files/Microsoft Office/Office{office_version}/WINWORD.EXE"),
                Path(f"C:/Program Files (x86)/Microsoft Office/Office{office_version}/WINWORD.EXE"),
            )
        )
    if os.name == "nt":
        try:
            import winreg

            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                for office_version in ("16.0", "15.0", "14.0"):
                    for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
                        try:
                            key = winreg.OpenKey(
                                hive,
                                rf"SOFTWARE\Microsoft\Office\{office_version}\Common\InstallRoot",
                                0,
                                winreg.KEY_READ | view,
                            )
                            install_root, _ = winreg.QueryValueEx(key, "Path")
                            candidates.append(Path(install_root) / "WINWORD.EXE")
                            winreg.CloseKey(key)
                        except OSError:
                            continue
        except ImportError:
            pass
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def convert_doc_to_docx(path: Path, target: Path) -> None:
    if os.name != "nt":
        raise RuntimeError("旧版 .doc 提取目前需要在 Windows 上运行。")
    powershell = shutil.which("powershell") or shutil.which("powershell.exe")
    if not powershell:
        raise RuntimeError("旧版 .doc 提取需要 Windows PowerShell。")
    script = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$source = [IO.Path]::GetFullPath($env:LKA_DOC_SOURCE)
$target = [IO.Path]::GetFullPath($env:LKA_DOC_TARGET)
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    try { $word.AutomationSecurity = 3 } catch {}
    $document = $word.Documents.Open($source, $false, $true)
    $document.SaveAs2($target, 16)
} finally {
    if ($document) {
        $document.Close(0)
        [Runtime.InteropServices.Marshal]::ReleaseComObject($document) | Out-Null
    }
    if ($word) {
        $word.Quit()
        [Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
"""
    encoded_script = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    env = os.environ.copy()
    env["LKA_DOC_SOURCE"] = str(path.resolve())
    env["LKA_DOC_TARGET"] = str(target.resolve())
    completed = subprocess.run(
        [powershell, "-NoProfile", "-STA", "-EncodedCommand", encoded_script],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        env=env,
    )
    if completed.returncode != 0 or not target.exists():
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise RuntimeError(f"Microsoft Word 转换 .doc 失败：{detail}")


def read_doc_with_report(path: Path) -> tuple[str, dict[str, Any]]:
    report = extraction_report(path, "doc")
    with tempfile.TemporaryDirectory(prefix="lka_doc_") as temp:
        converted = Path(temp) / f"{path.stem}.docx"
        convert_doc_to_docx(path, converted)
        text, converted_report = read_docx_with_report(converted)
    report["method"] = ["word_com_doc_to_docx", *converted_report["method"]]
    report["warnings"].extend(converted_report["warnings"])
    report["requires_review"] = converted_report["requires_review"]
    report["text_chars"] = len(text)
    return text, report


def read_docx_with_report(path: Path) -> tuple[str, dict[str, Any]]:
    report = extraction_report(path, "docx")
    text = read_docx(path)
    image_texts: list[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            media_files = [name for name in zf.namelist() if name.startswith("word/media/")]
            for media_file in media_files:
                media_text = ocr_image_bytes(media_file, zf.read(media_file))
                if media_text:
                    image_texts.append(f"[embedded image {Path(media_file).name}]\n{media_text}")
    except zipfile.BadZipFile:
        media_files = []
    if media_files:
        if image_texts:
            text = "\n\n".join(part for part in [text, *image_texts] if part).strip()
            report["method"].append("embedded_image_ocr")
        else:
            report["warnings"].append(
                f"DOCX contains {len(media_files)} embedded image(s), but OCR returned no image text."
            )
            report["requires_review"] = True
    report["text_chars"] = len(text)
    return text, report
