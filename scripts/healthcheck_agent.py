#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Health checks for Local Knowledge Base Agent."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

from api_providers import infer_provider, token_plan_rejected
from ingest_knowledge_base import extraction_reports, find_winword, is_ignored_raw_file, pending_review_reports


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "knowledge_base" / "raw"
DB_PATH = ROOT / "knowledge_base" / "index" / "knowledge.db"
EXTRACTION_REPORT = ROOT / "knowledge_base" / "processed" / "extraction_report.jsonl"
PROJECT_TESSDATA_DIR = ROOT / "tools" / "tessdata"
SUPPORTED = {".md", ".txt", ".doc", ".docx", ".pdf", ".xlsx"}


def unsupported_raw_files() -> list[str]:
    if not RAW_DIR.exists():
        return []
    return [
        str(path.relative_to(RAW_DIR)).replace("\\", "/")
        for path in sorted(RAW_DIR.rglob("*"))
        if path.is_file() and not is_ignored_raw_file(path) and path.suffix.lower() not in SUPPORTED
    ]


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


def api_backend_allowed(env: dict[str, str]) -> bool:
    base_url = env.get("LKA_API_BASE_URL", "").lower()
    api_key = env.get("LKA_API_KEY", "").lower()
    return not token_plan_rejected(base_url, api_key)


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
        "word_doc_converter": find_winword(),
        "tessdata_dir": str(tessdata_dir) if tessdata_dir else None,
        "ocr_languages": languages,
        "ocr_ready": bool(tesseract and shutil.which("pdftoppm") and {"eng", "chi_sim"}.issubset(set(languages))),
    }


def offline_translation_status() -> dict[str, object]:
    try:
        from argostranslate import translate  # type: ignore
    except ImportError:
        return {"package_installed": False, "en_to_zh_ready": False}
    installed = translate.get_installed_languages()
    english = next((language for language in installed if language.code == "en"), None)
    ready = bool(english and any(item.to_lang.code == "zh" for item in english.translations_from))
    return {"package_installed": True, "en_to_zh_ready": ready}


def extraction_report_stats() -> dict[str, object]:
    if not EXTRACTION_REPORT.exists():
        return {"exists": False}
    reports = extraction_reports(EXTRACTION_REPORT.parent)
    pending = pending_review_reports(RAW_DIR, EXTRACTION_REPORT.parent)
    warnings = 0
    approved = 0
    for item in reports:
        if item.get("warnings"):
            warnings += 1
        if item.get("review_approved"):
            approved += 1
    examples = [
        {
            "file_path": item.get("file_path"),
            "warnings": item.get("warnings"),
        }
        for item in pending[:5]
    ]
    return {
        "exists": True,
        "files_reported": len(reports),
        "warnings": warnings,
        "requires_review": len(pending),
        "approved_reviews": approved,
        "review_examples": examples,
    }


def main() -> int:
    env = load_env(ROOT / ".env")
    raw_files = [
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file() and not is_ignored_raw_file(path) and path.suffix.lower() in SUPPORTED
    ]
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
        "raw_unsupported_files": unsupported_raw_files(),
        "index": stats,
        "optional_extractors": optional_extractors(),
        "offline_translation": offline_translation_status(),
        "extraction_report": extraction_report_stats(),
        "api_configured": (
            env.get("LKA_USE_API", "false").lower() == "true"
            and bool(env.get("LKA_API_KEY"))
            and bool(env.get("LKA_API_MODEL"))
        ),
        "api_backend_usable": api_backend_allowed(env),
        "api_provider": infer_provider(
            env.get("LKA_API_BASE_URL", ""),
            env.get("LKA_API_KEY", ""),
            env.get("LKA_API_PROVIDER", ""),
        ),
    }
    ok = (
        checks["sqlite_fts5"]
        and checks["raw_dir_exists"]
        and stats.get("exists") is True
        and not stats.get("error")
        and int(stats.get("documents", 0) or 0) > 0
        and int(stats.get("chunks", 0) or 0) > 0
        and int(checks["extraction_report"].get("requires_review", 0) or 0) == 0
        and not checks["raw_unsupported_files"]
    )
    checks["ok"] = ok
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
