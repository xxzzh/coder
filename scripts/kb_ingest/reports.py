"""Processed outputs written after a build: chunk/FAQ files and the retrieval quality report."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from kb_ingest.chunking import EMBEDDING_DIMS, normalize_text


FAQ_LIMIT = 200


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
               source_fingerprint, content_hash, duplicate_of, text
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


def chunk_information_quality(text: str) -> float:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return 0.0
    if len(compact) < 40:
        return 0.35
    pipe_count = compact.count("|")
    alpha_numeric = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", compact))
    if pipe_count >= 8 and alpha_numeric / max(len(compact), 1) < 0.45:
        return 0.45
    if len(set(compact)) <= 8 and len(compact) > 80:
        return 0.3
    return 1.0


def fts_quality_queries(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT file_name, text
        FROM documents
        WHERE duplicate_of IS NULL
        ORDER BY file_path
        LIMIT 8
        """
    ).fetchall()
    queries: list[str] = []
    for row in rows:
        stem = Path(str(row["file_name"])).stem
        if stem:
            queries.append(stem[:40])
        words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,8}", str(row["text"])[:1200])
        if words:
            queries.append(words[0])
    return list(dict.fromkeys(query for query in queries if query))[:12]


def write_retrieval_quality_report(conn: sqlite3.Connection, processed_dir: Path) -> dict[str, Any]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    doc_rows = conn.execute(
        """
        SELECT documents.doc_id, documents.file_name, documents.file_path, documents.duplicate_of,
               COUNT(chunks.chunk_id) AS chunk_count,
               AVG(LENGTH(chunks.text)) AS avg_chunk_length,
               SUM(CASE WHEN TRIM(chunks.text) = '' THEN 1 ELSE 0 END) AS empty_chunks
        FROM documents
        LEFT JOIN chunks ON chunks.doc_id = documents.doc_id
        GROUP BY documents.doc_id
        ORDER BY documents.file_path
        """
    ).fetchall()
    chunk_rows = conn.execute(
        """
        SELECT chunks.chunk_id, documents.file_path, chunks.text
        FROM chunks
        JOIN documents ON documents.doc_id = chunks.doc_id
        """
    ).fetchall()
    low_info = [
        {
            "chunk_id": row["chunk_id"],
            "file_path": row["file_path"],
            "length": len(str(row["text"])),
            "quality_score": chunk_information_quality(str(row["text"])),
        }
        for row in chunk_rows
        if chunk_information_quality(str(row["text"])) < 0.6
    ]
    quality_queries = fts_quality_queries(conn)
    fts_tests: list[dict[str, Any]] = []
    fts_hits = 0
    for query_text in quality_queries:
        try:
            row = conn.execute(
                """
                SELECT documents.file_path
                FROM chunks_fts
                JOIN chunks ON chunks.chunk_id = chunks_fts.chunk_id
                JOIN documents ON documents.doc_id = chunks.doc_id
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT 1
                """,
                (f'"{query_text.replace(chr(34), chr(34) + chr(34))}"',),
            ).fetchone()
        except sqlite3.OperationalError:
            row = None
        if row:
            fts_hits += 1
        fts_tests.append({"query": query_text, "hit": bool(row), "top_source": row["file_path"] if row else None})
    top_sources = conn.execute(
        """
        SELECT documents.file_path, COUNT(chunks.chunk_id) AS chunk_count
        FROM documents
        JOIN chunks ON chunks.doc_id = documents.doc_id
        GROUP BY documents.doc_id
        ORDER BY chunk_count DESC, documents.file_path
        LIMIT 10
        """
    ).fetchall()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "documents": [
            {
                "file_path": row["file_path"],
                "file_name": row["file_name"],
                "chunk_count": int(row["chunk_count"] or 0),
                "avg_chunk_length": round(float(row["avg_chunk_length"] or 0), 1),
                "empty_chunks": int(row["empty_chunks"] or 0),
                "duplicate_of": row["duplicate_of"],
            }
            for row in doc_rows
        ],
        "summary": {
            "documents": len(doc_rows),
            "chunks": len(chunk_rows),
            "avg_chunk_length": round(
                sum(len(str(row["text"])) for row in chunk_rows) / max(len(chunk_rows), 1),
                1,
            ),
            "empty_chunks": sum(1 for row in chunk_rows if not str(row["text"]).strip()),
            "duplicate_documents": sum(1 for row in doc_rows if row["duplicate_of"]),
            "low_information_chunks": len(low_info),
            "fts_hit_rate": round(fts_hits / max(len(fts_tests), 1), 3),
        },
        "low_information_examples": low_info[:30],
        "fts_hit_tests": fts_tests,
        "top_query_hit_sources": [dict(row) for row in top_sources],
        "vector_index_recommendation": (
            "Current SQLite scan is acceptable."
            if len(chunk_rows) < 10000
            else "Chunk count is above 10000; evaluate FAISS, sqlite-vec, Qdrant, or pgvector."
        ),
    }
    (processed_dir / "retrieval_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report["summary"]
