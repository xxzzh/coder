"""Read any supported document, with an extraction report, by dispatching on its format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kb_ingest.chunking import normalize_text
from kb_ingest.files import read_plain_text
from kb_ingest.pdf import MIN_EXTRACTED_TEXT_CHARS, read_pdf, read_pdf_with_report
from kb_ingest.reviews import extraction_report
from kb_ingest.word import read_doc_with_report, read_docx, read_docx_with_report
from kb_ingest.xlsx import read_xlsx, read_xlsx_with_report


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
