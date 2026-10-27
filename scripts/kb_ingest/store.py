"""The SQLite index: schema, document rows, embedding LSH buckets and metadata."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from kb_ingest.chunking import embedding_lsh_buckets, parse_embedding


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
            DROP TABLE IF EXISTS embedding_lsh;
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
            source_fingerprint TEXT NOT NULL DEFAULT '',
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
            indexed_at TEXT NOT NULL DEFAULT '',
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

        CREATE TABLE IF NOT EXISTS embedding_lsh (
            bucket TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            indexed_at TEXT NOT NULL,
            PRIMARY KEY(bucket, chunk_id),
            FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
        );

        CREATE INDEX IF NOT EXISTS idx_embedding_lsh_chunk ON embedding_lsh(chunk_id);

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
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    document_columns = {row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
    if "source_fingerprint" not in document_columns:
        conn.execute("ALTER TABLE documents ADD COLUMN source_fingerprint TEXT NOT NULL DEFAULT ''")
    chunk_columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    if "indexed_at" not in chunk_columns:
        conn.execute("ALTER TABLE chunks ADD COLUMN indexed_at TEXT NOT NULL DEFAULT ''")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS embedding_lsh (
            bucket TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            indexed_at TEXT NOT NULL,
            PRIMARY KEY(bucket, chunk_id),
            FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
        );
        CREATE INDEX IF NOT EXISTS idx_embedding_lsh_chunk ON embedding_lsh(chunk_id);
        """
    )


def doc_id_for_path(rel_path: str) -> str:
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def clear_document(conn: sqlite3.Connection, doc_id: str) -> None:
    chunk_ids = [row[0] for row in conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()]
    for chunk_id in chunk_ids:
        conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
        conn.execute("DELETE FROM embedding_lsh WHERE chunk_id = ?", (chunk_id,))
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


def insert_embedding_lsh(
    conn: sqlite3.Connection,
    chunk_id: str,
    doc_id: str,
    embedding: list[list[float]],
    indexed_at: str,
) -> None:
    for bucket in embedding_lsh_buckets(embedding):
        conn.execute(
            "INSERT OR IGNORE INTO embedding_lsh(bucket, chunk_id, doc_id, indexed_at) VALUES (?, ?, ?, ?)",
            (bucket, chunk_id, doc_id, indexed_at),
        )


def backfill_embedding_lsh(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT chunks.chunk_id, chunks.doc_id, chunks.embedding, chunks.indexed_at
        FROM chunks
        LEFT JOIN embedding_lsh ON embedding_lsh.chunk_id = chunks.chunk_id
        WHERE embedding_lsh.chunk_id IS NULL
        """
    ).fetchall()
    for row in rows:
        insert_embedding_lsh(
            conn,
            str(row["chunk_id"]),
            str(row["doc_id"]),
            parse_embedding(str(row["embedding"])),
            str(row["indexed_at"] or ""),
        )
    return len(rows)


def metadata_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
