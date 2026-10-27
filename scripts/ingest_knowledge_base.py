#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Import local documents and build a SQLite FTS5 + local embedding index.

This module is the main flow: one ingest run (_ingest_unlocked), the locked ingest() entry point and
the command line. The building blocks live in kb_ingest/: files (raw files, fingerprints, the ingest
lock), reviews (extraction reports and review approvals), ocr, word, xlsx, pdf, documents (format
dispatch), chunking (chunks and the local embedding), store (the SQLite index), reports (processed
files and the retrieval quality report) and vectors (model embeddings and the FAISS index).
Their names are re-exported here, so existing callers keep working.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Any

from kb_ingest.files import (  # noqa: F401  (re-exported for callers of this module)
    DB_PATH,
    IngestAlreadyRunningError,
    PROCESSED_DIR,
    PROJECT_ROOT,
    SMALL_FILE_HASH_BYTES,
    SUPPORTED,
    _pid_is_running,
    ingest_in_progress,
    ingest_lock_path,
    ingestion_lock,
    is_ignored_raw_file,
    portable_error_message,
    raw_file_fingerprint,
    raw_file_stat_fingerprint,
    read_plain_text,
    source_fingerprint,
    unsupported_raw_files,
)
from kb_ingest.chunking import (  # noqa: F401  (re-exported for callers of this module)
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    EMBEDDING_DIMS,
    EMBEDDING_LSH_BUCKETS,
    EMBEDDING_LSH_PAIR_BUCKETS,
    SEMANTIC_BREAK_SIMILARITY,
    _split_long_text,
    chunk_text,
    content_hash,
    cosine_dicts,
    embedding_lsh_buckets,
    embedding_terms,
    local_embedding,
    local_embedding_dict,
    normalize_text,
    parse_embedding,
    searchable_text,
    semantic_units,
    split_semantic_block,
)
from kb_ingest.reviews import (  # noqa: F401  (re-exported for callers of this module)
    REVIEW_APPROVALS_FILE,
    approve_review_file,
    extraction_report,
    extraction_reports,
    load_review_approvals,
    pending_review_reports,
    review_approval_matches,
    write_extraction_report,
    write_review_approvals,
)
from kb_ingest.ocr import (  # noqa: F401  (re-exported for callers of this module)
    OCR_IMAGE_EXTENSIONS,
    OCR_LANGUAGES,
    OCR_RENDER_DPI,
    PROJECT_TESSDATA_DIR,
    available_ocr_languages,
    find_tessdata_dir,
    find_tesseract,
    ocr_image,
    ocr_image_bytes,
    ocr_pdf,
    pdf_page_renderer,
    render_pdf_pages_for_ocr,
    tsv_mean_confidence,
)
from kb_ingest.word import (  # noqa: F401  (re-exported for callers of this module)
    convert_doc_to_docx,
    find_winword,
    read_doc_with_report,
    read_docx,
    read_docx_with_report,
)
from kb_ingest.xlsx import (  # noqa: F401  (re-exported for callers of this module)
    CHART_POINTS_MAX,
    CHUNK_TEXT_REPORT_KEY,
    OFFICE_RELATIONSHIP_NS,
    PACKAGE_PROG_IDS,
    PACKAGE_RELATIONSHIP_NS,
    XLSX_DRAWING_NS,
    XLSX_GROUP_HEADING_MAX_CHARS,
    XLSX_HEADER_CELL_MAX_CHARS,
    XLSX_HEADER_SCAN_ROWS,
    XLSX_OVERVIEW_DESCRIPTION_CHARS,
    XLSX_OVERVIEW_ITEM_MAX_CHARS,
    _chart_values,
    _drawingml_text,
    _xlsx_cell_value,
    _xlsx_chart_text,
    _xlsx_column_index,
    _xlsx_drawing_contents,
    _xlsx_header_labels,
    _xlsx_header_score,
    _xlsx_item_columns,
    _xlsx_ref,
    _xlsx_shared_strings,
    _xlsx_sheet_grid,
    _xlsx_sheets,
    _zip_part_relationships,
    read_xlsx,
    read_xlsx_records,
    read_xlsx_with_report,
)
from kb_ingest.pdf import (  # noqa: F401  (re-exported for callers of this module)
    MIN_EXTRACTED_TEXT_CHARS,
    _decode_pdf_literal,
    _read_pdf_with_library,
    _read_pdf_with_pdfplumber,
    _read_pdf_with_pdftotext,
    find_pdftotext,
    pdf_likely_has_images,
    read_pdf,
    read_pdf_with_report,
)
from kb_ingest.documents import (  # noqa: F401  (re-exported for callers of this module)
    read_document,
    read_document_with_report,
)
from kb_ingest.store import (  # noqa: F401  (re-exported for callers of this module)
    backfill_embedding_lsh,
    clear_document,
    doc_id_for_path,
    ensure_schema,
    init_db,
    insert_embedding_lsh,
    metadata_set,
    remove_deleted_documents,
)
from kb_ingest.reports import (  # noqa: F401  (re-exported for callers of this module)
    FAQ_LIMIT,
    chunk_information_quality,
    faq_record,
    fts_quality_queries,
    write_processed_files,
    write_retrieval_quality_report,
)
from kb_ingest.vectors import (  # noqa: F401  (re-exported for callers of this module)
    build_optional_vector_index,
    embed_records_with_cache,
    seed_embedding_cache_from_index,
)


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
    metadata_set(conn, "semantic_candidate_strategy", "exact-scan-small-lsh-large")

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
        file_source_fingerprint = raw_file_stat_fingerprint(path, raw_dir)
        modified_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds")
        existing = conn.execute(
            "SELECT modified_at, file_size, source_fingerprint FROM documents WHERE file_path = ?",
            (rel_path,),
        ).fetchone()
        existing_fingerprint = existing["source_fingerprint"] if existing else ""
        if existing and existing_fingerprint == file_source_fingerprint:
            unchanged_count += 1
            continue

        try:
            text, extract_report = read_document_with_report(path)
            structured_chunk_text = extract_report.pop(CHUNK_TEXT_REPORT_KEY, None)
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
            """
            INSERT INTO documents(
                doc_id, file_name, file_path, file_type, file_size, modified_at,
                indexed_at, source_fingerprint, content_hash, duplicate_of, text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                path.name,
                rel_path,
                file_type,
                stat.st_size,
                modified_at,
                indexed_at,
                file_source_fingerprint,
                hash_value,
                duplicate_of,
                text,
            ),
        )

        if duplicate_of:
            continue

        chunk_source = normalize_text(structured_chunk_text) if structured_chunk_text else text
        for index, chunk in enumerate(chunk_text(chunk_source), start=1):
            chunk_count += 1
            chunk_id = f"{doc_id}_{index:04d}"
            embedding_values = local_embedding(chunk)
            embedding = json.dumps(embedding_values, separators=(",", ":"))
            conn.execute(
                "INSERT INTO chunks(chunk_id, doc_id, chunk_index, text, embedding, indexed_at) VALUES (?, ?, ?, ?, ?, ?)",
                (chunk_id, doc_id, index, chunk, embedding, indexed_at),
            )
            insert_embedding_lsh(conn, chunk_id, doc_id, embedding_values, indexed_at)
            conn.execute(
                "INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?, ?, ?)",
                (chunk_id, doc_id, path.name, rel_path, file_type, chunk, searchable_text(chunk)),
            )

    removed_count = remove_deleted_documents(conn, current_paths)
    lsh_backfilled = backfill_embedding_lsh(conn)
    if changed_doc_count or removed_count:
        conn.execute("DELETE FROM query_cache")
    if changed_doc_count or removed_count or previous_indexed_at is None:
        metadata_set(conn, "indexed_at", indexed_at)
    else:
        indexed_at = previous_indexed_at["value"]
    source_hash = source_fingerprint(raw_dir)
    metadata_set(conn, "source_fingerprint", source_hash)
    vector_index = build_optional_vector_index(
        conn,
        indexed_at,
        source_hash,
        force=bool(changed_doc_count or removed_count or reset),
    )
    metadata_set(conn, "vector_backend", str(vector_index.get("backend", "faiss")))
    metadata_set(conn, "vector_index_status", "ready" if vector_index.get("ready") else "degraded")
    metadata_set(conn, "vector_index_reason", str(vector_index.get("reason", "")))
    if vector_index.get("model"):
        metadata_set(conn, "vector_embedding_model", str(vector_index.get("model")))
    if vector_index.get("dimension"):
        metadata_set(conn, "vector_embedding_dimension", str(vector_index.get("dimension")))
    write_extraction_report(processed_dir, extraction_reports, current_paths)
    faq_count = write_processed_files(conn, processed_dir)
    quality_summary = write_retrieval_quality_report(conn, processed_dir)
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
        "lsh_backfilled_chunks": lsh_backfilled,
        "skipped": skipped_count,
        "duplicates": duplicate_count,
        "duplicate_files": duplicate_files,
        "extraction_warnings": len(warning_reports),
        "requires_review": len(review_reports),
        "approved_reviews": approved_review_count,
        "extraction_report": str(processed_dir / "extraction_report.jsonl"),
        "faq_entries": faq_count,
        "quality_report": str(processed_dir / "retrieval_quality_report.json"),
        "quality_summary": quality_summary,
        "vector_index": vector_index,
        "indexed_at": indexed_at,
        "embedding_model": f"local-hash-ngram-{EMBEDDING_DIMS}d",
        "vector_backend": vector_index.get("backend", "faiss"),
        "vector_index_status": "ready" if vector_index.get("ready") else "degraded",
        "vector_embedding_model": vector_index.get("model"),
        "vector_embedding_dimension": vector_index.get("dimension"),
        "chunk_strategy": "semantic-boundary-local-embedding",
        "semantic_candidate_strategy": "exact-scan-small-lsh-large",
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
