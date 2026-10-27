"""PDF text: pypdf, pdfplumber and pdftotext readers, with OCR for scanned pages."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Any

from kb_ingest.ocr import ocr_pdf
from kb_ingest.reviews import extraction_report


MIN_EXTRACTED_TEXT_CHARS = 80


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


def find_pdftotext() -> str | None:
    found = shutil.which("pdftotext")
    if found:
        return found
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        for suffix in (".cmd", ".exe", ""):
            candidate = Path(pdftoppm).with_name(f"pdftotext{suffix}")
            if candidate.exists():
                return str(candidate)
    return None


def _read_pdf_with_pdftotext(path: Path) -> str | None:
    pdftotext = find_pdftotext()
    if not pdftotext:
        return None
    with tempfile.TemporaryDirectory(prefix="lka_pdftotext_") as temp:
        output_path = Path(temp) / "out.txt"
        proc = subprocess.run(
            [pdftotext, "-layout", "-enc", "UTF-8", str(path), str(output_path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if proc.returncode != 0 or not output_path.exists():
            return None
        text = output_path.read_text(encoding="utf-8", errors="ignore").strip()
    return text or None


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

    text = _read_pdf_with_pdftotext(path)
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

    pdftotext_text = _read_pdf_with_pdftotext(path)
    if pdftotext_text and pdftotext_text not in pieces:
        pieces.append(pdftotext_text)
        report["method"].append("pdftotext")

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
