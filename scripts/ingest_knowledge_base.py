#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Import local documents and build a SQLite FTS5 + local embedding index.

This module is the main flow: one ingest run (_ingest_unlocked), the locked ingest() entry point and
the command line. The building blocks live in kb_ingest/: files (raw files, fingerprints, the ingest
lock), reviews (extraction reports and review approvals), ocr, word, xlsx, pdf, documents (format
dispatch), chunking (chunks and the local embedding), store (the SQLite index), reports (processed
files and the retrieval quality report), vectors (model embeddings and the FAISS index) and
progress (what a running ingest reports to the Web UI).
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
    RAW_DIR,
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
    embed_before_indexing,
    embed_records_with_cache,
    seed_embedding_cache_from_index,
)
from kb_ingest import progress

# A file read without error that gave no text at all.
NO_TEXT_READ_ERROR = "no text extracted"


def read_failure_report(path: Path, rel_path: str, error: str) -> dict[str, Any]:
    """The extraction report of a file that could not be read: kept with the others, so the document
    list can say why, and stamped with the file's size and time, so a newer copy is not marked failed."""
    stat = path.stat()
    return {
        **extraction_report(path, path.suffix.lower().lstrip(".")),
        "file_path": rel_path,
        "read_error": error[:300],
        "file_size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
    }


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
    try:
        return _ingest_with_connection(conn, raw_dir, processed_dir, reset, strict_extraction)
    finally:
        conn.close()  # an ingest that failed half-way leaves nothing written and no lock held


def _ingest_with_connection(
    conn: sqlite3.Connection,
    raw_dir: Path,
    processed_dir: Path,
    reset: bool,
    strict_extraction: bool,
) -> dict[str, Any]:
    """One ingest run in two parts, so questions can be asked while it runs.

    Reading the files (OCR included), chunking and embedding take minutes and write nothing to the
    index: questions asked meanwhile read the previous index, complete. Only then is everything
    written, in one short transaction. A write transaction open from the first file on would make
    every question wait for the lock on its answer-cache entry and then fail.
    """
    indexed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    previous_indexed_at = conn.execute("SELECT value FROM metadata WHERE key = ?", ("indexed_at",)).fetchone()

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
    # Changed documents, read and chunked, waiting to be written.
    prepared: list[dict[str, Any]] = []
    seen_hashes = {
        row["content_hash"]: row["file_path"]
        for row in conn.execute("SELECT content_hash, file_path FROM documents WHERE duplicate_of IS NULL").fetchall()
    }

    candidates = [
        path
        for path in sorted(raw_dir.rglob("*"))
        if path.is_file() and not is_ignored_raw_file(path) and path.suffix.lower() in SUPPORTED
    ]
    for position, path in enumerate(candidates):
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

        progress.report("reading", done=position, total=len(candidates), file=rel_path)
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
            extraction_reports.append(read_failure_report(path, rel_path, str(exc)))
            continue
        if not text:
            skipped_count += 1
            extraction_reports.append(read_failure_report(path, rel_path, NO_TEXT_READ_ERROR))
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

        chunks: list[tuple[str, list[list[float]]]] = []
        if not duplicate_of:
            chunk_source = normalize_text(structured_chunk_text) if structured_chunk_text else text
            chunks = [(chunk, local_embedding(chunk)) for chunk in chunk_text(chunk_source)]
            chunk_count += len(chunks)
        prepared.append(
            {
                "doc_id": doc_id,
                "file_name": path.name,
                "file_path": rel_path,
                "file_type": file_type,
                "file_size": stat.st_size,
                "modified_at": modified_at,
                "source_fingerprint": file_source_fingerprint,
                "content_hash": hash_value,
                "duplicate_of": duplicate_of,
                "text": text,
                "chunks": chunks,
            }
        )

    # Still nothing written: the vectors of the new chunks, and what the write will need.
    indexed_documents = {str(row["file_path"]): str(row["doc_id"]) for row in conn.execute("SELECT doc_id, file_path FROM documents")}
    removed_doc_ids = {doc_id for file_path, doc_id in indexed_documents.items() if file_path not in current_paths}
    source_hash = source_fingerprint(raw_dir)
    early_embedding = embed_before_indexing(
        conn,
        {document["doc_id"] for document in prepared} | removed_doc_ids,
        [chunk for document in prepared for chunk, _ in document["chunks"]],
        force=bool(prepared or removed_doc_ids or reset),
    )

    # From here on the index is written, in one transaction that ends with the commit below.
    progress.report("indexing")
    metadata_set(conn, "embedding_model", f"local-hash-ngram-{EMBEDDING_DIMS}d")
    metadata_set(conn, "chunk_strategy", "semantic-boundary-local-embedding")
    metadata_set(conn, "semantic_candidate_strategy", "exact-scan-small-lsh-large")
    for document in prepared:
        doc_id = document["doc_id"]
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
                document["file_name"],
                document["file_path"],
                document["file_type"],
                document["file_size"],
                document["modified_at"],
                indexed_at,
                document["source_fingerprint"],
                document["content_hash"],
                document["duplicate_of"],
                document["text"],
            ),
        )
        for index, (chunk, embedding_values) in enumerate(document["chunks"], start=1):
            chunk_id = f"{doc_id}_{index:04d}"
            conn.execute(
                "INSERT INTO chunks(chunk_id, doc_id, chunk_index, text, embedding, indexed_at) VALUES (?, ?, ?, ?, ?, ?)",
                (chunk_id, doc_id, index, chunk, json.dumps(embedding_values, separators=(",", ":")), indexed_at),
            )
            insert_embedding_lsh(conn, chunk_id, doc_id, embedding_values, indexed_at)
            conn.execute(
                "INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?, ?, ?)",
                (chunk_id, doc_id, document["file_name"], document["file_path"], document["file_type"], chunk, searchable_text(chunk)),
            )

    removed_count = remove_deleted_documents(conn, current_paths)
    lsh_backfilled = backfill_embedding_lsh(conn)
    if changed_doc_count or removed_count:
        conn.execute("DELETE FROM query_cache")
    if changed_doc_count or removed_count or previous_indexed_at is None:
        metadata_set(conn, "indexed_at", indexed_at)
    else:
        indexed_at = previous_indexed_at["value"]
    metadata_set(conn, "source_fingerprint", source_hash)
    vector_index = build_optional_vector_index(
        conn,
        indexed_at,
        source_hash,
        force=bool(changed_doc_count or removed_count or reset),
        prepared=early_embedding,
    )
    metadata_set(conn, "vector_backend", str(vector_index.get("backend", "faiss")))
    metadata_set(conn, "vector_index_status", "ready" if vector_index.get("ready") else "degraded")
    metadata_set(conn, "vector_index_reason", str(vector_index.get("reason", "")))
    if vector_index.get("model"):
        metadata_set(conn, "vector_embedding_model", str(vector_index.get("model")))
    if vector_index.get("dimension"):
        metadata_set(conn, "vector_embedding_dimension", str(vector_index.get("dimension")))
    conn.commit()

    # The reports only read the index, so they are written after the commit, outside the transaction.
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
    parser.add_argument("raw_dir", nargs="?", default=str(RAW_DIR))
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
