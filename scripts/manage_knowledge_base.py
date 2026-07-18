#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convenience commands for the local knowledge base."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from ingest_knowledge_base import DB_PATH, PROCESSED_DIR, IngestAlreadyRunningError, ingest
import vector_store


RAW_DIR = Path("knowledge_base/raw")


def metadata(conn: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = conn.execute("SELECT key, value FROM metadata ORDER BY key").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {row["key"]: row["value"] for row in rows}


def sources(db_path: Path = DB_PATH) -> dict[str, Any]:
    if not db_path.exists():
        return {
            "indexed": False,
            "message": "本地知识库索引不存在，请先运行 ingest 或 rebuild。",
            "sources": [],
        }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        meta = metadata(conn)
        rows = conn.execute(
            """
            SELECT documents.doc_id, documents.file_name, documents.file_path, documents.file_type,
                   documents.file_size, documents.modified_at, documents.indexed_at,
                   documents.content_hash, documents.duplicate_of,
                   COUNT(chunks.chunk_id) AS chunk_count
            FROM documents
            LEFT JOIN chunks ON chunks.doc_id = documents.doc_id
            GROUP BY documents.doc_id
            ORDER BY documents.file_path
            """
        ).fetchall()
    finally:
        conn.close()

    items = [dict(row) for row in rows]
    duplicate_items = [item for item in items if item.get("duplicate_of")]
    return {
        "indexed": True,
        "index_updated_at": meta.get("indexed_at"),
        "embedding_model": meta.get("embedding_model"),
        "chunk_strategy": meta.get("chunk_strategy"),
        "semantic_candidate_strategy": meta.get("semantic_candidate_strategy"),
        "vector_backend": meta.get("vector_backend"),
        "vector_index_status": meta.get("vector_index_status"),
        "vector_index_reason": meta.get("vector_index_reason"),
        "vector_embedding_model": meta.get("vector_embedding_model"),
        "vector_embedding_dimension": meta.get("vector_embedding_dimension"),
        "vector_index": vector_store.vector_status(sqlite_chunk_count=sum(int(item.get("chunk_count", 0) or 0) for item in items)),
        "documents": len(items),
        "indexed_documents": len([item for item in items if not item.get("duplicate_of")]),
        "duplicates": len(duplicate_items),
        "sources": items,
        "duplicate_files": [
            {"file_path": item["file_path"], "duplicate_of": item["duplicate_of"]} for item in duplicate_items
        ],
    }


def safe_remove_generated(db_path: Path, processed_dir: Path) -> None:
    targets = [
        db_path,
        processed_dir / "documents.jsonl",
        processed_dir / "chunks.jsonl",
        processed_dir / "faq.jsonl",
        processed_dir / "extraction_report.jsonl",
        processed_dir / "retrieval_quality_report.json",
    ]
    allowed_roots = [
        (Path.cwd() / "knowledge_base" / "index").resolve(),
        (Path.cwd() / "knowledge_base" / "processed").resolve(),
    ]
    for target in targets:
        resolved = (Path.cwd() / target).resolve() if not target.is_absolute() else target.resolve()
        if not any(resolved == root or root in resolved.parents for root in allowed_roots):
            raise RuntimeError(f"Refusing to remove generated file outside knowledge_base: {resolved}")
        if resolved.exists() and resolved.is_file():
            resolved.unlink()


def rebuild(
    raw_dir: Path = RAW_DIR,
    db_path: Path = DB_PATH,
    processed_dir: Path = PROCESSED_DIR,
    strict_extraction: bool = False,
) -> dict[str, Any]:
    safe_remove_generated(db_path, processed_dir)
    vector_store.reset_vector_index()
    result = ingest(raw_dir, db_path, processed_dir, reset=True, strict_extraction=strict_extraction)
    return {"rebuilt": True, **result}


def update(
    raw_dir: Path = RAW_DIR,
    db_path: Path = DB_PATH,
    processed_dir: Path = PROCESSED_DIR,
    strict_extraction: bool = False,
) -> dict[str, Any]:
    result = ingest(raw_dir, db_path, processed_dir, reset=False, strict_extraction=strict_extraction)
    return {"updated": True, **result}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Manage the local knowledge base.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sources_parser = subparsers.add_parser("sources", help="列出当前索引中的知识来源。")
    sources_parser.add_argument("--db", default=str(DB_PATH))

    rebuild_parser = subparsers.add_parser("rebuild", help="删除生成索引并重新导入 raw 目录。")
    rebuild_parser.add_argument("raw_dir", nargs="?", default=str(RAW_DIR))
    rebuild_parser.add_argument("--db", default=str(DB_PATH))
    rebuild_parser.add_argument("--processed", default=str(PROCESSED_DIR))
    rebuild_parser.add_argument("--strict-extraction", action="store_true")

    update_parser = subparsers.add_parser("update", help="增量更新 raw 目录中变化的知识文件。")
    update_parser.add_argument("raw_dir", nargs="?", default=str(RAW_DIR))
    update_parser.add_argument("--db", default=str(DB_PATH))
    update_parser.add_argument("--processed", default=str(PROCESSED_DIR))
    update_parser.add_argument("--strict-extraction", action="store_true")

    args = parser.parse_args()
    try:
        if args.command == "sources":
            output = sources(Path(args.db))
        elif args.command == "rebuild":
            output = rebuild(Path(args.raw_dir), Path(args.db), Path(args.processed), args.strict_extraction)
        elif args.command == "update":
            output = update(Path(args.raw_dir), Path(args.db), Path(args.processed), args.strict_extraction)
        else:
            parser.error(f"Unknown command: {args.command}")
    except IngestAlreadyRunningError as exc:
        output = {"updated": False, "skipped": True, "reason": "update_in_progress", "message": str(exc)}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
