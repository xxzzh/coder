#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Health checks for Local Knowledge Base Agent."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "knowledge_base" / "raw"
DB_PATH = ROOT / "knowledge_base" / "index" / "knowledge.db"
EXTRACTION_REPORT = ROOT / "knowledge_base" / "processed" / "extraction_report.jsonl"
PROJECT_TESSDATA_DIR = ROOT / "tools" / "tessdata"
SUPPORTED = {".md", ".docx", ".pdf", ".xlsx"}


def sqlite_has_fts5() -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE test_fts USING fts5(text)")
        conn.close()
        return True
    except sqlite3.Error:
        return False


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def db_stats() -> dict[str, object]:
    if not DB_PATH.exists():
        return {"exists": False}
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        metadata = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM metadata").fetchall()}
        documents = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()
        return {
            "exists": True,
            "documents": documents,
            "chunks": chunks,
            "indexed_at": metadata.get("indexed_at"),
            "chunk_strategy": metadata.get("chunk_strategy"),
            "embedding_model": metadata.get("embedding_model"),
        }
    except sqlite3.Error as exc:
        return {"exists": True, "error": str(exc)}


def optional_extractors() -> dict[str, object]:
    modules: dict[str, bool] = {}
    for module_name in ("pypdf", "PyPDF2", "pdfplumber", "PIL", "pytesseract", "openpyxl"):
        try:
            __import__(module_name)
            modules[module_name] = True
        except ImportError:
            modules[module_name] = False
    tesseract = shutil.which("tesseract")
    if not tesseract:
        for candidate in (
            Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
            Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
        ):
            if candidate.exists():
                tesseract = str(candidate)
                break
    tessdata_dir: Path | None = None
    for candidate in (
        PROJECT_TESSDATA_DIR,
        Path("C:/Program Files/Tesseract-OCR/tessdata"),
        Path("C:/Program Files (x86)/Tesseract-OCR/tessdata"),
    ):
        if candidate.exists():
            tessdata_dir = candidate
            break
    languages = []
    if tessdata_dir:
        languages = sorted(path.stem for path in tessdata_dir.glob("*.traineddata"))
    return {
        "modules": modules,
        "tesseract": tesseract,
        "pdftoppm": shutil.which("pdftoppm"),
        "tessdata_dir": str(tessdata_dir) if tessdata_dir else None,
        "ocr_languages": languages,
        "ocr_ready": bool(tesseract and shutil.which("pdftoppm") and {"eng", "chi_sim"}.issubset(set(languages))),
    }


def extraction_report_stats() -> dict[str, object]:
    if not EXTRACTION_REPORT.exists():
        return {"exists": False}
    total = 0
    warnings = 0
    requires_review = 0
    examples: list[dict[str, object]] = []
    with EXTRACTION_REPORT.open("r", encoding="utf-8") as report_in:
        for line in report_in:
            if not line.strip():
                continue
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("warnings"):
                warnings += 1
            if item.get("requires_review"):
                requires_review += 1
                if len(examples) < 5:
                    examples.append(
                        {
                            "file_path": item.get("file_path"),
                            "warnings": item.get("warnings"),
                        }
                    )
    return {
        "exists": True,
        "files_reported": total,
        "warnings": warnings,
        "requires_review": requires_review,
        "review_examples": examples,
    }


def main() -> int:
    env = load_env(ROOT / ".env")
    raw_files = [path for path in RAW_DIR.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED]
    counts: dict[str, int] = {}
    for path in raw_files:
        counts[path.suffix.lower()] = counts.get(path.suffix.lower(), 0) + 1

    stats = db_stats()
    checks = {
        "python_version": sys.version.split()[0],
        "sqlite_fts5": sqlite_has_fts5(),
        "raw_dir_exists": RAW_DIR.exists(),
        "raw_supported_files": len(raw_files),
        "raw_counts": counts,
        "index": stats,
        "optional_extractors": optional_extractors(),
        "extraction_report": extraction_report_stats(),
        "api_configured": env.get("LKA_USE_API", "false").lower() == "true" and bool(env.get("LKA_API_KEY")),
        "api_provider": env.get("LKA_API_PROVIDER", "none"),
    }
    ok = (
        checks["sqlite_fts5"]
        and checks["raw_dir_exists"]
        and stats.get("exists") is True
        and not stats.get("error")
        and int(stats.get("documents", 0) or 0) > 0
        and int(stats.get("chunks", 0) or 0) > 0
        and int(checks["extraction_report"].get("requires_review", 0) or 0) == 0
    )
    checks["ok"] = ok
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
