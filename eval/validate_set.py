#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the evaluation set against the original raw documents.

The raw files are read with independent readers (openpyxl / plain text /
zip XML), never with the project's own extractor, so the ground truth cannot
silently inherit an extraction bug. Checks:

- schema: unique ids, known category, required fields
- every gold keyword of a positive item exists verbatim in one of its gold docs
- every absent term of a negative item exists in none of the raw docs
- optional (--db): gold keywords that exist in the raw doc but in no indexed
  chunk, i.e. content lost during ingestion
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from eval_lib import (
    CATEGORIES,
    DB_PATH,
    DEFAULT_SET,
    NEGATIVE_CATEGORIES,
    POSITIVE_CATEGORIES,
    RAW_DIR,
    contains,
    load_eval_set,
    norm,
    raw_files_by_name,
    use_utf8_stdout,
    write_json,
)


_TEXT_CACHE: dict[Path, str | None] = {}


def _xlsx_text(path: Path) -> str | None:
    try:
        import openpyxl  # type: ignore
    except ImportError:
        return None
    import warnings

    cells: list[str] = []
    with warnings.catch_warnings():
        # openpyxl warns about unsupported data-validation extensions while streaming rows.
        warnings.simplefilter("ignore")
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for sheet in workbook.worksheets:
            cells.append(sheet.title)
            for row in sheet.iter_rows(values_only=True):
                cells.extend(str(value) for value in row if value not in (None, ""))
        workbook.close()
    # Join with a separator so a keyword never matches across two cells.
    return "␟".join(norm(cell) for cell in cells)


def _docx_text(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
    except (KeyError, zipfile.BadZipFile, ET.ParseError):
        return None
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs = ["".join(node.text or "" for node in para.iter(f"{namespace}t")) for para in root.iter(f"{namespace}p")]
    return "␟".join(norm(paragraph) for paragraph in paragraphs)


def _pdf_text(path: Path) -> str | None:
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return None
    with pdfplumber.open(path) as pdf:
        return norm("\n".join(page.extract_text() or "" for page in pdf.pages))


def raw_text(path: Path) -> str | None:
    """Normalized text of a raw document, or None when no independent reader is available."""
    if path in _TEXT_CACHE:
        return _TEXT_CACHE[path]
    suffix = path.suffix.lower()
    text: str | None
    if suffix in {".md", ".txt"}:
        text = norm(path.read_text(encoding="utf-8", errors="ignore"))
    elif suffix == ".xlsx":
        text = _xlsx_text(path)
    elif suffix == ".docx":
        text = _docx_text(path)
    elif suffix == ".pdf":
        text = _pdf_text(path)
    else:
        text = None
    _TEXT_CACHE[path] = text
    return text


def indexed_text_by_file(db_path: Path) -> dict[str, str]:
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT documents.file_name, chunks.text
            FROM chunks JOIN documents ON documents.doc_id = chunks.doc_id
            ORDER BY documents.file_name, chunks.chunk_index
            """
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    joined: dict[str, list[str]] = {}
    for file_name, text in rows:
        joined.setdefault(str(file_name), []).append(norm(text))
    return {name: "␟".join(parts) for name, parts in joined.items()}


def validate(eval_set: list[dict[str, Any]], raw_dir: Path = RAW_DIR, db_path: Path | None = None) -> dict[str, Any]:
    raw_files = raw_files_by_name(raw_dir)
    indexed = indexed_text_by_file(db_path) if db_path else {}
    errors: list[str] = []
    warnings: list[str] = []
    items: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()

    for item in eval_set:
        item_id = str(item.get("id", "")).strip()
        status = "ok"
        problems: list[str] = []
        if not item_id:
            errors.append(f"item without id: {item.get('question')!r}")
            continue
        if item_id in seen_ids:
            errors.append(f"{item_id}: duplicate id")
        seen_ids.add(item_id)
        category = item.get("category")
        if category not in CATEGORIES:
            errors.append(f"{item_id}: unknown category {category!r}")
            continue
        if not str(item.get("question", "")).strip():
            errors.append(f"{item_id}: empty question")

        extraction_loss: list[str] = []
        if category in POSITIVE_CATEGORIES:
            docs = list(item.get("gold_doc_ids") or [])
            keywords = list(item.get("gold_chunk_keywords") or [])
            if not docs or not keywords:
                errors.append(f"{item_id}: positive item needs gold_doc_ids and gold_chunk_keywords")
                continue
            missing_docs = [doc for doc in docs if doc not in raw_files]
            if missing_docs:
                status = "skipped_doc_missing"
                warnings.append(f"{item_id}: gold doc not in raw dir, item skipped: {missing_docs}")
            else:
                texts = {doc: raw_text(raw_files[doc]) for doc in docs}
                unreadable = [doc for doc, text in texts.items() if text is None]
                if unreadable:
                    warnings.append(f"{item_id}: no independent reader for {unreadable}, keywords not verified")
                readable = [text for text in texts.values() if text is not None]
                if readable:
                    for keyword in keywords:
                        if not any(contains(text, keyword) for text in readable):
                            problems.append(f"gold keyword not found in gold docs: {keyword!r}")
                if indexed:
                    for keyword in keywords:
                        in_raw = any(contains(text, keyword) for text in readable)
                        in_index = any(contains(indexed.get(doc, ""), keyword) for doc in docs)
                        if in_raw and not in_index:
                            extraction_loss.append(keyword)
        else:
            terms = list(item.get("absent_terms") or [])
            if not terms:
                errors.append(f"{item_id}: negative item needs absent_terms")
                continue
            for name, path in raw_files.items():
                text = raw_text(path)
                if text is None:
                    continue
                for term in terms:
                    if contains(text, term):
                        problems.append(f"absent term {term!r} actually appears in {name}")

        if problems:
            status = "invalid"
            errors.extend(f"{item_id}: {problem}" for problem in problems)
        items[item_id] = {"status": status, "problems": problems, "extraction_loss": extraction_loss}
        if extraction_loss:
            warnings.append(f"{item_id}: in raw doc but missing from index: {extraction_loss}")

    runnable = sorted(item_id for item_id, info in items.items() if info["status"] == "ok")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "items": items,
        "runnable_ids": runnable,
        "raw_files": sorted(raw_files),
    }


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser(description="Validate eval_set.jsonl against raw documents.")
    parser.add_argument("--set", default=str(DEFAULT_SET))
    parser.add_argument("--raw", default=str(RAW_DIR))
    parser.add_argument("--db", default=str(DB_PATH), help="Also report keywords lost during ingestion.")
    parser.add_argument("--out", help="Write the validation report as JSON.")
    args = parser.parse_args()

    report = validate(load_eval_set(Path(args.set)), Path(args.raw), Path(args.db) if args.db else None)
    for line in report["errors"]:
        print("ERROR  ", line)
    for line in report["warnings"]:
        print("WARN   ", line)
    print(f"items: {len(report['items'])}, runnable: {len(report['runnable_ids'])}, ok: {report['ok']}")
    if args.out:
        write_json(Path(args.out), report)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
