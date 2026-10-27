"""Model embeddings with a reusable cache, and the optional FAISS vector index."""

from __future__ import annotations

import sqlite3
from typing import Any

import model_capabilities
import vector_store


def embed_records_with_cache(
    records: list[dict[str, Any]],
    embedding: dict[str, Any],
    config: dict[str, str],
) -> dict[str, Any]:
    """Vectors for every record, embedding only chunk texts the cache has not seen for this model."""
    cache = vector_store.EmbeddingCache(model_capabilities.embedding_model_key(embedding))
    try:
        hashes = [vector_store.text_hash(str(record["text"])) for record in records]
        texts_by_hash = dict(zip(hashes, (str(record["text"]) for record in records)))
        vectors_by_hash = cache.get_many(hashes)
        missing = [digest for digest in dict.fromkeys(hashes) if digest not in vectors_by_hash]
        if missing:
            embedded = model_capabilities.embed_texts([texts_by_hash[digest] for digest in missing], config)
            if not embedded.get("ok"):
                return {"ok": False, "error": embedded.get("error")}
            fresh = dict(zip(missing, embedded["vectors"]))
            dimensions = {len(vector) for vector in fresh.values()} | {len(vector) for vector in vectors_by_hash.values()}
            if len(dimensions) > 1:
                # Cached vectors of another dimension (e.g. an API that changed its output size): start over.
                embedded = model_capabilities.embed_texts(list(texts_by_hash.values()), config)
                if not embedded.get("ok"):
                    return {"ok": False, "error": embedded.get("error")}
                fresh = dict(zip(texts_by_hash.keys(), embedded["vectors"]))
                missing = list(fresh)
                vectors_by_hash = {}
            cache.put_many(fresh.items())
            vectors_by_hash.update(fresh)
        cache.prune(hashes)
        missing_set = set(missing)
        return {
            "ok": True,
            "vectors": [vectors_by_hash[digest] for digest in hashes],
            "embedded_chunks": sum(1 for digest in hashes if digest in missing_set),
            "reused_chunks": sum(1 for digest in hashes if digest not in missing_set),
        }
    finally:
        cache.close()


def seed_embedding_cache_from_index(
    conn: sqlite3.Connection,
    embedding: dict[str, Any],
    indexed_at: str,
    index_metadata: dict[str, Any],
) -> int:
    """Store the vectors of an index built before the cache existed, so the next rebuild reuses them.

    Only runs when the index was built from exactly the current chunks: same model, same
    indexed_at as the database, and every mapped chunk still present.
    """
    if index_metadata.get("model") != embedding.get("model") or index_metadata.get("provider") != embedding.get("provider"):
        return 0
    if index_metadata.get("indexed_at") != indexed_at:
        return 0
    cache = vector_store.EmbeddingCache(model_capabilities.embedding_model_key(embedding))
    try:
        if cache.count() >= int(index_metadata.get("chunk_count", 0) or 0):
            return 0
        mapping, vectors = vector_store.index_vectors()
        texts = {str(chunk_id): str(text) for chunk_id, text in conn.execute("SELECT chunk_id, text FROM chunks")}
        if len(mapping) != len(vectors) or any(str(row["chunk_id"]) not in texts for row in mapping):
            return 0
        return cache.put_many(
            (vector_store.text_hash(texts[str(row["chunk_id"])]), vector) for row, vector in zip(mapping, vectors)
        )
    except Exception:  # noqa: BLE001 - seeding is an optimization; never block indexing
        return 0
    finally:
        cache.close()


def build_optional_vector_index(
    conn: sqlite3.Connection,
    indexed_at: str,
    source_hash: str,
    force: bool = False,
) -> dict[str, Any]:
    config = model_capabilities.runtime_config()
    embedding = model_capabilities.embedding_config(config)
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    existing_status = vector_store.vector_status(sqlite_chunk_count=int(chunk_count or 0))
    if not force and existing_status.get("ready"):
        metadata = existing_status.get("metadata", {})
        seeded = seed_embedding_cache_from_index(conn, embedding, indexed_at, metadata) if embedding["usable"] else 0
        return {
            "ready": True,
            "reason": "unchanged",
            "backend": "faiss",
            "chunk_count": int(metadata.get("chunk_count", 0) or 0),
            "dimension": metadata.get("dimension"),
            "model": metadata.get("model"),
            "embedding_cache_seeded": seeded,
        }
    if not embedding["enabled"]:
        vector_store.reset_vector_index()
        payload = vector_store.write_unavailable_metadata(
            "embedding_disabled",
            chunk_count=int(chunk_count or 0),
            indexed_at=indexed_at,
            source_fingerprint=source_hash,
        )
        return payload
    if not embedding["usable"]:
        vector_store.reset_vector_index()
        payload = vector_store.write_unavailable_metadata(
            "embedding_config_unusable",
            chunk_count=int(chunk_count or 0),
            indexed_at=indexed_at,
            source_fingerprint=source_hash,
            embedding_status=model_capabilities.embedding_status(config),
        )
        return payload

    rows = conn.execute(
        """
        SELECT chunks.chunk_id, chunks.doc_id, chunks.chunk_index, chunks.text,
               documents.file_name, documents.file_path, documents.file_type
        FROM chunks
        JOIN documents ON documents.doc_id = chunks.doc_id
        ORDER BY documents.file_path, chunks.chunk_index
        """
    ).fetchall()
    records = [dict(row) for row in rows]
    embedded = embed_records_with_cache(records, embedding, config)
    if not embedded.get("ok"):
        vector_store.reset_vector_index()
        payload = vector_store.write_unavailable_metadata(
            "embedding_api_unavailable",
            chunk_count=len(records),
            indexed_at=indexed_at,
            source_fingerprint=source_hash,
            error=embedded.get("error"),
            embedding_status=model_capabilities.embedding_status(config),
        )
        return payload
    payload = vector_store.build_vector_index(
        records,
        embedded["vectors"],
        {
            "indexed_at": indexed_at,
            "source_fingerprint": source_hash,
            "provider": embedding.get("provider"),
            "model": embedding.get("model"),
            "requested_dimensions": embedding.get("requested_dimensions"),
            "embedded_chunks": embedded["embedded_chunks"],
            "reused_chunks": embedded["reused_chunks"],
        },
    )
    return payload
