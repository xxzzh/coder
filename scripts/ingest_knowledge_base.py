#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Import local documents and build a SQLite FTS5 + local embedding index."""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


SUPPORTED = {".md", ".txt", ".doc", ".docx", ".pdf", ".xlsx"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path("knowledge_base/index/knowledge.db")
PROCESSED_DIR = Path("knowledge_base/processed")
REVIEW_APPROVALS_FILE = "review_approvals.json"
PROJECT_TESSDATA_DIR = PROJECT_ROOT / "tools" / "tessdata"
EMBEDDING_DIMS = 384
CHUNK_MAX_CHARS = 1100
CHUNK_MIN_CHARS = 180
SEMANTIC_BREAK_SIMILARITY = 0.08
FAQ_LIMIT = 200
OCR_LANGUAGES = "eng+chi_sim"
MIN_EXTRACTED_TEXT_CHARS = 80
OCR_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def is_ignored_raw_file(path: Path) -> bool:
    """Ignore transient Office lock files created while documents are open."""
    return path.name.startswith("~$")


class IngestAlreadyRunningError(RuntimeError):
    """Raised when another process is already updating the same index."""


def ingest_lock_path(db_path: Path = DB_PATH) -> Path:
    return db_path.with_suffix(".ingest.lock")


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if process:
                ctypes.windll.kernel32.CloseHandle(process)
                return True
            return False
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def ingest_in_progress(db_path: Path = DB_PATH) -> bool:
    lock_path = ingest_lock_path(db_path)
    if not lock_path.exists():
        return False
    try:
        lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(lock_data.get("pid", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        pid = 0
    if _pid_is_running(pid):
        return True
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    return False


@contextmanager
def ingestion_lock(db_path: Path):
    lock_path = ingest_lock_path(db_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if ingest_in_progress(db_path):
        raise IngestAlreadyRunningError("知识库索引正在更新，请稍后再试。")
    try:
        handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise IngestAlreadyRunningError("知识库索引正在更新，请稍后再试。") from exc
    try:
        payload = json.dumps({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        os.write(handle, payload.encode("utf-8"))
        os.close(handle)
        yield
    finally:
        try:
            os.close(handle)
        except OSError:
            pass
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def read_plain_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def unsupported_raw_files(raw_dir: Path) -> list[str]:
    if not raw_dir.exists():
        return []
    return [
        str(path.relative_to(raw_dir)).replace("\\", "/")
        for path in sorted(raw_dir.rglob("*"))
        if path.is_file() and not is_ignored_raw_file(path) and path.suffix.lower() not in SUPPORTED
    ]


def raw_file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_review_approvals(processed_dir: Path = PROCESSED_DIR) -> dict[str, dict[str, str]]:
    approvals_path = processed_dir / REVIEW_APPROVALS_FILE
    if not approvals_path.exists():
        return {}
    try:
        payload = json.loads(approvals_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    files = payload.get("files", {})
    return files if isinstance(files, dict) else {}


def write_review_approvals(approvals: dict[str, dict[str, str]], processed_dir: Path = PROCESSED_DIR) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    approvals_path = processed_dir / REVIEW_APPROVALS_FILE
    payload = {"files": approvals}
    approvals_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def review_approval_matches(approvals: dict[str, dict[str, str]], rel_path: str, path: Path) -> bool:
    approval = approvals.get(rel_path)
    return bool(approval and approval.get("fingerprint") == raw_file_fingerprint(path))


def extraction_reports(processed_dir: Path = PROCESSED_DIR) -> list[dict[str, Any]]:
    report_path = processed_dir / "extraction_report.jsonl"
    if not report_path.exists():
        return []
    reports: list[dict[str, Any]] = []
    with report_path.open("r", encoding="utf-8") as report_in:
        for line in report_in:
            if not line.strip():
                continue
            try:
                reports.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return reports


def pending_review_reports(raw_dir: Path, processed_dir: Path = PROCESSED_DIR) -> list[dict[str, Any]]:
    approvals = load_review_approvals(processed_dir)
    pending: list[dict[str, Any]] = []
    for report in extraction_reports(processed_dir):
        rel_path = str(report.get("file_path", ""))
        path = (raw_dir / rel_path).resolve()
        if not report.get("requires_review") or not path.exists():
            continue
        if review_approval_matches(approvals, rel_path, path):
            continue
        pending.append(report)
    return pending


def approve_review_file(raw_dir: Path, processed_dir: Path, rel_path: str) -> dict[str, str]:
    rel_path = rel_path.replace("\\", "/").strip("/")
    raw_root = raw_dir.resolve()
    path = (raw_root / rel_path).resolve()
    if raw_root not in path.parents or not path.is_file():
        raise ValueError("待复核文件不存在。")
    pending_paths = {str(report.get("file_path", "")) for report in pending_review_reports(raw_dir, processed_dir)}
    if rel_path not in pending_paths:
        raise ValueError("该文件当前不需要复核，或文件内容已经变化，请刷新列表。")
    approvals = load_review_approvals(processed_dir)
    approval = {
        "fingerprint": raw_file_fingerprint(path),
        "approved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    approvals[rel_path] = approval
    write_review_approvals(approvals, processed_dir)
    return {"file_path": rel_path, **approval}


def read_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", ns):
        parts: list[str] = []
        for node in paragraph.iter():
            tag = node.tag.rsplit("}", 1)[-1]
            if tag == "t" and node.text:
                parts.append(node.text)
            elif tag == "tab":
                parts.append("\t")
            elif tag in {"br", "cr"}:
                parts.append("\n")
        line = "".join(parts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


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


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall(".//m:si", ns):
        strings.append("".join(node.text or "" for node in item.findall(".//m:t", ns)))
    return strings


def _xlsx_column_index(cell_reference: str) -> int:
    match = re.match(r"([A-Z]+)", cell_reference.upper())
    if not match:
        return 1
    index = 0
    for letter in match.group(1):
        index = index * 26 + ord(letter) - ord("A") + 1
    return index


def _xlsx_sheets(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    workbook_path = "xl/workbook.xml"
    relationships_path = "xl/_rels/workbook.xml.rels"
    if workbook_path not in zf.namelist() or relationships_path not in zf.namelist():
        sheets = sorted(
            (name for name in zf.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)),
            key=lambda name: int(re.search(r"sheet(\d+)", name).group(1)),
        )
        return [(f"sheet{index}", sheet) for index, sheet in enumerate(sheets, start=1)]

    relationships_root = ET.fromstring(zf.read(relationships_path))
    targets = {
        relationship.attrib["Id"]: relationship.attrib["Target"]
        for relationship in relationships_root.findall(f"{{{package_rel_ns}}}Relationship")
    }
    workbook_root = ET.fromstring(zf.read(workbook_path))
    sheets: list[tuple[str, str]] = []
    for sheet in workbook_root.findall(f".//{{{main_ns}}}sheet"):
        relation_id = sheet.attrib.get(f"{{{rel_ns}}}id")
        target = targets.get(str(relation_id), "")
        if not target:
            continue
        normalized_target = target.lstrip("/")
        if not normalized_target.startswith("xl/"):
            normalized_target = f"xl/{normalized_target}"
        sheets.append((sheet.attrib.get("name", f"sheet{len(sheets) + 1}"), normalized_target))
    return sheets


def read_xlsx(path: Path) -> str:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    lines: list[str] = []

    with zipfile.ZipFile(path) as zf:
        shared = _xlsx_shared_strings(zf)
        sheets = _xlsx_sheets(zf)

        for sheet_name, sheet in sheets:
            root = ET.fromstring(zf.read(sheet))
            lines.append(f"工作表 {sheet_name}")
            for row in root.findall(".//m:row", ns):
                cells: dict[int, str] = {}
                for cell in row.findall("m:c", ns):
                    cell_type = cell.attrib.get("t")
                    value = ""
                    if cell_type == "inlineStr":
                        value = "".join(node.text or "" for node in cell.findall(".//m:t", ns))
                    else:
                        value_node = cell.find("m:v", ns)
                        if value_node is not None:
                            value = value_node.text or ""
                            if cell_type == "s" and value.isdigit():
                                index = int(value)
                                if 0 <= index < len(shared):
                                    value = shared[index]
                    cells[_xlsx_column_index(cell.attrib.get("r", "A1"))] = re.sub(r"\s*\r?\n\s*", " ", value).strip()
                if any(cell.strip() for cell in cells.values()):
                    lines.append(" | ".join(cells.get(index, "") for index in range(1, max(cells) + 1)))
    return "\n".join(lines)


def read_xlsx_with_report(path: Path) -> tuple[str, dict[str, Any]]:
    report = extraction_report(path, "xlsx")
    text = read_xlsx(path)
    image_texts: list[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            drawings = [
                name
                for name in zf.namelist()
                if name.startswith("xl/media/") or name.startswith("xl/drawings/") or name.startswith("xl/charts/")
            ]
            media_files = [name for name in zf.namelist() if name.startswith("xl/media/")]
            for media_file in media_files:
                media_text = ocr_image_bytes(media_file, zf.read(media_file))
                if media_text:
                    image_texts.append(f"[embedded image {Path(media_file).name}]\n{media_text}")
    except zipfile.BadZipFile:
        drawings = []
        media_files = []
    if image_texts:
        text = "\n\n".join(part for part in [text, *image_texts] if part).strip()
        report["method"].append("embedded_image_ocr")
    if drawings:
        non_media_drawings = [name for name in drawings if not name.startswith("xl/media/")]
        if non_media_drawings:
            report["warnings"].append(
                "XLSX contains drawings or charts; cell values are extracted but visual-only chart labels may need review."
            )
            report["requires_review"] = True
        elif media_files and not image_texts:
            report["warnings"].append("XLSX contains embedded image(s), but OCR returned no image text.")
            report["requires_review"] = True
    report["text_chars"] = len(text)
    return text, report


def _read_pdf_with_library(path: Path) -> str | None:
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        reader = module.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(page.strip() for page in pages if page.strip())
        if text.strip():
            return text
    return None


def _read_pdf_with_pdfplumber(path: Path) -> str | None:
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return None

    pieces: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pieces.append(f"[page {page_index}]\n{text.strip()}")
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            for table_index, table in enumerate(tables, start=1):
                rows = [
                    " | ".join((cell or "").strip() for cell in row)
                    for row in table
                    if row and any((cell or "").strip() for cell in row)
                ]
                if rows:
                    pieces.append(f"[page {page_index} table {table_index}]\n" + "\n".join(rows))
    text = "\n\n".join(pieces).strip()
    return text or None


def pdf_likely_has_images(path: Path) -> bool:
    data = path.read_bytes()
    return b"/Subtype" in data and b"/Image" in data


def find_tesseract() -> str | None:
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in (
        Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
        Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


def find_tessdata_dir() -> Path | None:
    if PROJECT_TESSDATA_DIR.exists():
        return PROJECT_TESSDATA_DIR
    for candidate in (
        Path("C:/Program Files/Tesseract-OCR/tessdata"),
        Path("C:/Program Files (x86)/Tesseract-OCR/tessdata"),
    ):
        if candidate.exists():
            return candidate
    return None


def available_ocr_languages() -> str:
    tessdata_dir = find_tessdata_dir()
    if not tessdata_dir:
        return "eng"
    desired = OCR_LANGUAGES.split("+")
    available = [lang for lang in desired if (tessdata_dir / f"{lang}.traineddata").exists()]
    return "+".join(available or ["eng"])


def render_pdf_pages_for_ocr(path: Path, temp_dir: Path) -> list[Path]:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        return []
    output_prefix = temp_dir / "page"
    subprocess.run(
        [pdftoppm, "-r", "300", "-png", str(path), str(output_prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return sorted(temp_dir.glob("page-*.png"))


def ocr_image(path: Path) -> str:
    tesseract = find_tesseract()
    if not tesseract:
        return ""
    tessdata_dir = find_tessdata_dir()
    best_text = ""
    for psm in ("6", "11", "4", "3"):
        command = [tesseract, str(path), "stdout", "-l", available_ocr_languages(), "--psm", psm]
        if tessdata_dir:
            command.extend(["--tessdata-dir", str(tessdata_dir)])
        proc = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="ignore",
        )
        if proc.returncode == 0 and len(proc.stdout.strip()) > len(best_text):
            best_text = proc.stdout.strip()
    return best_text


def ocr_image_bytes(name: str, data: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix not in OCR_IMAGE_EXTENSIONS:
        return ""
    with tempfile.TemporaryDirectory(prefix="lka_image_ocr_") as temp:
        image_path = Path(temp) / f"image{suffix}"
        image_path.write_bytes(data)
        return ocr_image(image_path)


def ocr_pdf(path: Path) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if not find_tesseract():
        warnings.append("OCR skipped: tesseract executable is not installed or not on PATH.")
        return "", warnings
    if not shutil.which("pdftoppm"):
        warnings.append("OCR skipped: pdftoppm executable is not installed or not on PATH.")
        return "", warnings

    pieces: list[str] = []
    with tempfile.TemporaryDirectory(prefix="lka_ocr_") as temp:
        try:
            images = render_pdf_pages_for_ocr(path, Path(temp))
        except Exception as exc:
            warnings.append(f"OCR render failed: {exc}")
            return "", warnings
        for page_index, image_path in enumerate(images, start=1):
            page_text = ocr_image(image_path)
            if page_text:
                pieces.append(f"[ocr page {page_index}]\n{page_text}")
            else:
                warnings.append(f"OCR returned no text for page {page_index}.")
    return "\n\n".join(pieces).strip(), warnings


def _decode_pdf_literal(raw: bytes) -> str:
    raw = raw.replace(rb"\\(", b"(").replace(rb"\\)", b")").replace(rb"\\\\", b"\\")
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="ignore")


def read_pdf(path: Path) -> str:
    text = _read_pdf_with_pdfplumber(path)
    if text is not None:
        return text

    text = _read_pdf_with_library(path)
    if text is not None:
        return text

    data = path.read_bytes()
    pieces: list[str] = []
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, flags=re.S):
        raw = match.group(1)
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass
        for text_match in re.finditer(rb"\((.*?)\)\s*Tj", raw, flags=re.S):
            pieces.append(_decode_pdf_literal(text_match.group(1)))
        for array_match in re.finditer(rb"\[(.*?)\]\s*TJ", raw, flags=re.S):
            for text_match in re.finditer(rb"\((.*?)\)", array_match.group(1), flags=re.S):
                pieces.append(_decode_pdf_literal(text_match.group(1)))
    return "\n".join(piece.strip() for piece in pieces if piece.strip())


def extraction_report(path: Path, file_type: str) -> dict[str, Any]:
    return {
        "file": path.name,
        "file_path": str(path),
        "file_type": file_type,
        "method": [],
        "text_chars": 0,
        "warnings": [],
        "requires_review": False,
    }


def read_pdf_with_report(path: Path) -> tuple[str, dict[str, Any]]:
    report = extraction_report(path, "pdf")
    pieces: list[str] = []

    pdfplumber_text = _read_pdf_with_pdfplumber(path)
    if pdfplumber_text:
        pieces.append(pdfplumber_text)
        report["method"].append("pdfplumber")

    library_text = _read_pdf_with_library(path)
    if library_text and library_text not in pieces:
        pieces.append(library_text)
        report["method"].append("pypdf_or_pypdf2")

    fallback_text = ""
    if not pieces:
        fallback_text = read_pdf(path)
        if fallback_text:
            pieces.append(fallback_text)
            report["method"].append("pdf_stream_literals")

    has_images = pdf_likely_has_images(path)
    text = "\n\n".join(pieces).strip()
    if has_images or len(text) < MIN_EXTRACTED_TEXT_CHARS:
        ocr_text, ocr_warnings = ocr_pdf(path)
        report["warnings"].extend(ocr_warnings)
        if ocr_text:
            text = "\n\n".join(part for part in [text, ocr_text] if part).strip()
            report["method"].append("ocr_tesseract")
        elif has_images:
            report["warnings"].append("PDF appears to contain images; OCR text was not available.")
            report["requires_review"] = True

    if not text:
        report["warnings"].append("No text could be extracted from PDF.")
        report["requires_review"] = True
    report["text_chars"] = len(text)
    return text, report


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return read_plain_text(path)
    if suffix == ".doc":
        return read_doc_with_report(path)[0]
    if suffix == ".docx":
        return read_docx(path)
    if suffix == ".xlsx":
        return read_xlsx(path)
    if suffix == ".pdf":
        return read_pdf(path)
    raise ValueError(f"Unsupported file type: {path}")


def read_document_with_report(path: Path) -> tuple[str, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        text = read_plain_text(path)
        report = extraction_report(path, suffix.lstrip("."))
    elif suffix == ".doc":
        text, report = read_doc_with_report(path)
    elif suffix == ".docx":
        text, report = read_docx_with_report(path)
    elif suffix == ".xlsx":
        text, report = read_xlsx_with_report(path)
    elif suffix == ".pdf":
        text, report = read_pdf_with_report(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    report["text_chars"] = len(text)
    if len(normalize_text(text)) < MIN_EXTRACTED_TEXT_CHARS and suffix not in {".md", ".txt"}:
        report["warnings"].append("Extracted text is very short; source may be image-only, encrypted, or layout-heavy.")
        report["requires_review"] = True
    if not report["method"]:
        report["method"].append("plain_text" if suffix in {".md", ".txt"} else "office_xml")
    return text, report


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int = CHUNK_MAX_CHARS, overlap: int = 0) -> list[str]:
    """Split by semantic boundaries instead of fixed windows.

    The splitter keeps headings with nearby content, starts new chunks when
    adjacent units drift semantically, and only falls back to character limits
    for unusually long units.
    """
    text = normalize_text(text)
    if not text:
        return []

    units = semantic_units(text)
    chunks: list[str] = []
    current_units: list[str] = []
    current_embedding: dict[int, float] = {}

    for unit in units:
        unit_chunks = _split_long_text(unit, size, overlap)
        if len(unit_chunks) > 1:
            if current_units:
                chunks.append("\n\n".join(current_units).strip())
                current_units = []
                current_embedding = {}
            chunks.extend(unit_chunks)
            continue

        projected = "\n\n".join([*current_units, unit]).strip()
        similarity = cosine_dicts(current_embedding, local_embedding_dict(unit)) if current_units else 1.0
        starts_new_section = bool(re.match(r"^\s{0,3}#{1,6}\s+", unit))
        should_break = (
            current_units
            and len(projected) > CHUNK_MIN_CHARS
            and (len(projected) > size or starts_new_section or similarity < SEMANTIC_BREAK_SIMILARITY)
        )
        if should_break:
            chunks.append("\n\n".join(current_units).strip())
            current_units = [unit]
        else:
            current_units.append(unit)
        current_embedding = local_embedding_dict("\n\n".join(current_units))

    if current_units:
        chunks.append("\n\n".join(current_units).strip())
    return [chunk for chunk in chunks if chunk]


def semantic_units(text: str) -> list[str]:
    blocks = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    units: list[str] = []
    pending_heading: str | None = None

    for block in blocks:
        if re.match(r"^\s{0,3}#{1,6}\s+", block):
            if pending_heading:
                units.append(pending_heading)
            pending_heading = block
            continue

        split_blocks = split_semantic_block(block)
        if pending_heading and split_blocks:
            split_blocks[0] = f"{pending_heading}\n{split_blocks[0]}"
            pending_heading = None
        units.extend(split_blocks)

    if pending_heading:
        units.append(pending_heading)
    return units


def split_semantic_block(block: str) -> list[str]:
    if len(block) <= CHUNK_MAX_CHARS:
        return [block]

    if "|" in block and "\n" in block:
        rows = [row.strip() for row in block.splitlines() if row.strip()]
        if rows:
            return rows

    sentences = re.split(r"(?<=[。！？；?!;])\s*|(?<=\.)\s+|\n+", block)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _split_long_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        breakpoint = max(text.rfind("\n", start, end), text.rfind("。", start, end), text.rfind(".", start, end))
        if breakpoint > start + size // 2:
            end = breakpoint + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def searchable_text(text: str) -> str:
    chinese_terms: list[str] = []
    for block in re.findall(r"[\u4e00-\u9fff]+", text):
        for size in (2, 3, 4):
            if len(block) < size:
                continue
            chinese_terms.extend(block[index : index + size] for index in range(len(block) - size + 1))
    return f"{text}\n{' '.join(chinese_terms)}"


def content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", "", text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def embedding_terms(text: str) -> list[str]:
    terms: list[str] = []
    lowered = text.lower()
    terms.extend(re.findall(r"[a-z0-9_]{2,}", lowered))
    for block in re.findall(r"[\u4e00-\u9fff]+", text):
        for size in (2, 3, 4):
            if len(block) < size:
                continue
            terms.extend(block[index : index + size] for index in range(len(block) - size + 1))
    return terms


def local_embedding(text: str, dims: int = EMBEDDING_DIMS) -> list[list[float]]:
    vector: dict[int, float] = {}
    for term in embedding_terms(text):
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dims
        sign = 1.0 if (value >> 8) & 1 else -1.0
        weight = 1.0 + min(len(term), 8) / 8.0
        vector[index] = vector.get(index, 0.0) + sign * weight

    norm = math.sqrt(sum(value * value for value in vector.values()))
    if not norm:
        return []
    return [[index, round(value / norm, 6)] for index, value in sorted(vector.items()) if abs(value) > 1e-9]


def local_embedding_dict(text: str, dims: int = EMBEDDING_DIMS) -> dict[int, float]:
    return {int(index): float(value) for index, value in local_embedding(text, dims)}


def cosine_dicts(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


def init_db(db_path: Path, reset: bool = False) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    if reset:
        conn.executescript(
            """
            DROP TABLE IF EXISTS query_cache;
            DROP TABLE IF EXISTS metadata;
            DROP TABLE IF EXISTS chunks_fts;
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS documents;
            """
        )
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            modified_at TEXT NOT NULL,
            indexed_at TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            duplicate_of TEXT,
            text TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
        );

        CREATE TABLE IF NOT EXISTS query_cache (
            cache_key TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            answer_json TEXT NOT NULL,
            hit_count INTEGER NOT NULL DEFAULT 1,
            index_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED,
            doc_id UNINDEXED,
            file_name UNINDEXED,
            file_path UNINDEXED,
            file_type UNINDEXED,
            text,
            search_text,
            tokenize='unicode61'
        );
        """
    )
    return conn


def doc_id_for_path(rel_path: str) -> str:
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def clear_document(conn: sqlite3.Connection, doc_id: str) -> None:
    chunk_ids = [row[0] for row in conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()]
    for chunk_id in chunk_ids:
        conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))


def remove_deleted_documents(conn: sqlite3.Connection, current_paths: set[str]) -> int:
    removed = 0
    rows = conn.execute("SELECT doc_id, file_path FROM documents").fetchall()
    for row in rows:
        if row["file_path"] not in current_paths:
            clear_document(conn, row["doc_id"])
            removed += 1
    return removed


def metadata_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def faq_record(chunk: dict[str, Any]) -> dict[str, Any]:
    text = normalize_text(chunk["text"])
    first_line = next((line.strip("# -\t ") for line in text.splitlines() if line.strip()), "")
    return {
        "question": first_line[:120] or chunk["file_name"],
        "answer": text[:700],
        "file": chunk["file_name"],
        "file_path": chunk["file_path"],
        "file_type": chunk["file_type"],
        "chunk_id": chunk["chunk_id"],
    }


def write_processed_files(conn: sqlite3.Connection, processed_dir: Path) -> int:
    processed_dir.mkdir(parents=True, exist_ok=True)
    doc_rows = conn.execute(
        """
        SELECT doc_id, file_name, file_path, file_type, file_size, modified_at, indexed_at,
               content_hash, duplicate_of, text
        FROM documents
        ORDER BY file_path
        """
    ).fetchall()
    chunk_rows = conn.execute(
        """
        SELECT chunks.chunk_id, chunks.doc_id, chunks.chunk_index, documents.file_name,
               documents.file_path, documents.file_type, chunks.text
        FROM chunks
        JOIN documents ON documents.doc_id = chunks.doc_id
        ORDER BY documents.file_path, chunks.chunk_index
        """
    ).fetchall()

    with (processed_dir / "documents.jsonl").open("w", encoding="utf-8") as docs_out:
        for row in doc_rows:
            docs_out.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    faq_count = 0
    with (processed_dir / "chunks.jsonl").open("w", encoding="utf-8") as chunks_out, (
        processed_dir / "faq.jsonl"
    ).open("w", encoding="utf-8") as faq_out:
        for row in chunk_rows:
            record = dict(row)
            record["embedding_model"] = f"local-hash-ngram-{EMBEDDING_DIMS}d"
            chunks_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if faq_count < FAQ_LIMIT:
                faq_out.write(json.dumps(faq_record(record), ensure_ascii=False) + "\n")
                faq_count += 1
    return faq_count


def write_extraction_report(
    processed_dir: Path,
    reports: list[dict[str, Any]],
    current_paths: set[str] | None = None,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_path = processed_dir / "extraction_report.jsonl"
    merged: dict[str, dict[str, Any]] = {}
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as report_in:
            for line in report_in:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("file_path"):
                    merged[str(item["file_path"])] = item
    for report in reports:
        if report.get("file_path"):
            merged[str(report["file_path"])] = report
    if current_paths is not None:
        merged = {file_path: report for file_path, report in merged.items() if file_path in current_paths}
    with report_path.open("w", encoding="utf-8") as report_out:
        for report in (merged[file_path] for file_path in sorted(merged)):
            report_out.write(json.dumps(report, ensure_ascii=False) + "\n")


def source_fingerprint(raw_dir: Path) -> str:
    items: list[str] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and not is_ignored_raw_file(path) and path.suffix.lower() in SUPPORTED:
            rel_path = str(path.relative_to(raw_dir)).replace("\\", "/")
            stat = path.stat()
            items.append(f"{rel_path}:{stat.st_size}:{int(stat.st_mtime)}")
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def _ingest_unlocked(
    raw_dir: Path,
    db_path: Path = DB_PATH,
    processed_dir: Path = PROCESSED_DIR,
    reset: bool = False,
    strict_extraction: bool = False,
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    conn = init_db(db_path, reset=reset)
    conn.row_factory = sqlite3.Row
    indexed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    previous_indexed_at = conn.execute("SELECT value FROM metadata WHERE key = ?", ("indexed_at",)).fetchone()
    metadata_set(conn, "embedding_model", f"local-hash-ngram-{EMBEDDING_DIMS}d")
    metadata_set(conn, "chunk_strategy", "semantic-boundary-local-embedding")

    doc_count = 0
    changed_doc_count = 0
    chunk_count = 0
    skipped_count = 0
    unchanged_count = 0
    duplicate_count = 0
    duplicate_files: list[dict[str, str]] = []
    extraction_reports: list[dict[str, Any]] = []
    current_paths: set[str] = set()
    approvals = load_review_approvals(processed_dir)
    approved_review_count = 0
    seen_hashes = {
        row["content_hash"]: row["file_path"]
        for row in conn.execute("SELECT content_hash, file_path FROM documents WHERE duplicate_of IS NULL").fetchall()
    }

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or is_ignored_raw_file(path) or path.suffix.lower() not in SUPPORTED:
            continue

        doc_count += 1
        rel_path = str(path.relative_to(raw_dir)).replace("\\", "/")
        current_paths.add(rel_path)
        stat = path.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds")
        existing = conn.execute(
            "SELECT modified_at, file_size FROM documents WHERE file_path = ?",
            (rel_path,),
        ).fetchone()
        if existing and existing["modified_at"] == modified_at and existing["file_size"] == stat.st_size:
            unchanged_count += 1
            continue

        try:
            text, extract_report = read_document_with_report(path)
            extract_report["file_path"] = rel_path
            if extract_report.get("requires_review") and review_approval_matches(approvals, rel_path, path):
                extract_report["requires_review"] = False
                extract_report["review_approved"] = True
                approved_review_count += 1
            extraction_reports.append(extract_report)
            if strict_extraction and extract_report.get("requires_review"):
                skipped_count += 1
                print(f"skip {path}: extraction requires review")
                continue
            text = normalize_text(text)
        except Exception as exc:
            skipped_count += 1
            print(f"skip {path}: {exc}")
            continue
        if not text:
            skipped_count += 1
            continue

        changed_doc_count += 1
        doc_id = doc_id_for_path(rel_path)
        file_type = path.suffix.lower().lstrip(".")
        hash_value = content_hash(text)
        duplicate_of = seen_hashes.get(hash_value)
        if duplicate_of == rel_path:
            duplicate_of = None
        if duplicate_of:
            duplicate_count += 1
            duplicate_files.append({"file_path": rel_path, "duplicate_of": duplicate_of})
        else:
            seen_hashes[hash_value] = rel_path

        clear_document(conn, doc_id)
        conn.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                path.name,
                rel_path,
                file_type,
                stat.st_size,
                modified_at,
                indexed_at,
                hash_value,
                duplicate_of,
                text,
            ),
        )

        if duplicate_of:
            continue

        for index, chunk in enumerate(chunk_text(text), start=1):
            chunk_count += 1
            chunk_id = f"{doc_id}_{index:04d}"
            embedding = json.dumps(local_embedding(chunk), separators=(",", ":"))
            conn.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)", (chunk_id, doc_id, index, chunk, embedding))
            conn.execute(
                "INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?, ?, ?)",
                (chunk_id, doc_id, path.name, rel_path, file_type, chunk, searchable_text(chunk)),
            )

    removed_count = remove_deleted_documents(conn, current_paths)
    if changed_doc_count or removed_count:
        conn.execute("DELETE FROM query_cache")
    if changed_doc_count or removed_count or previous_indexed_at is None:
        metadata_set(conn, "indexed_at", indexed_at)
    else:
        indexed_at = previous_indexed_at["value"]
    metadata_set(conn, "source_fingerprint", source_fingerprint(raw_dir))
    write_extraction_report(processed_dir, extraction_reports, current_paths)
    faq_count = write_processed_files(conn, processed_dir)
    totals = conn.execute(
        """
        SELECT
            COUNT(*) AS documents,
            SUM(CASE WHEN duplicate_of IS NULL THEN 1 ELSE 0 END) AS indexed_documents
        FROM documents
        """
    ).fetchone()
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    conn.commit()
    conn.close()
    warning_reports = [report for report in extraction_reports if report.get("warnings")]
    review_reports = [report for report in extraction_reports if report.get("requires_review")]
    return {
        "documents": int(totals["documents"] or 0),
        "indexed_documents": int(totals["indexed_documents"] or 0),
        "changed_documents": changed_doc_count,
        "unchanged_documents": unchanged_count,
        "removed_documents": removed_count,
        "chunks": int(total_chunks or 0),
        "changed_chunks": chunk_count,
        "skipped": skipped_count,
        "duplicates": duplicate_count,
        "duplicate_files": duplicate_files,
        "extraction_warnings": len(warning_reports),
        "requires_review": len(review_reports),
        "approved_reviews": approved_review_count,
        "extraction_report": str(processed_dir / "extraction_report.jsonl"),
        "faq_entries": faq_count,
        "indexed_at": indexed_at,
        "embedding_model": f"local-hash-ngram-{EMBEDDING_DIMS}d",
        "chunk_strategy": "semantic-boundary-local-embedding",
        "unsupported_files": unsupported_raw_files(raw_dir),
    }


def ingest(
    raw_dir: Path,
    db_path: Path = DB_PATH,
    processed_dir: Path = PROCESSED_DIR,
    reset: bool = False,
    strict_extraction: bool = False,
) -> dict[str, Any]:
    with ingestion_lock(db_path):
        return _ingest_unlocked(raw_dir, db_path, processed_dir, reset, strict_extraction)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import local documents and build a SQLite FTS5 + embedding index.")
    parser.add_argument("raw_dir", nargs="?", default="knowledge_base/raw")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--processed", default=str(PROCESSED_DIR))
    parser.add_argument("--rebuild", action="store_true", help="Drop generated index tables before importing.")
    parser.add_argument(
        "--strict-extraction",
        action="store_true",
        help="Skip files whose extraction report requires manual review.",
    )
    args = parser.parse_args()

    try:
        result = ingest(
            Path(args.raw_dir),
            Path(args.db),
            Path(args.processed),
            reset=args.rebuild,
            strict_extraction=args.strict_extraction,
        )
    except IngestAlreadyRunningError as exc:
        result = {"updated": False, "skipped": True, "reason": "update_in_progress", "message": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
