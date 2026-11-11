"""Model embeddings with a reusable cache, and the optional FAISS vector index."""

from __future__ import annotations

import sqlite3
from typing import Any

import model_capabilities
import vector_store

from kb_ingest import progress

# New chunks are embedded this many at a time, so a large document reports its progress.
EMBED_PROGRESS_SLICE = 64


def embed_texts_with_progress(texts: list[str], config: dict[str, str]) -> dict[str, Any]:
    vectors: list[list[float]] = []
    progress.report("embedding", done=0, total=len(texts))
    for start in range(0, len(texts), EMBED_PROGRESS_SLICE):
        embedded = model_capabilities.embed_texts(texts[start : start + EMBED_PROGRESS_SLICE], config)
        if not embedded.get("ok"):
            return embedded
        vectors.extend(embedded["vectors"])
        progress.report("embedding", done=len(vectors), total=len(texts))
    return {"ok": True, "vectors": vectors}


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
            embedded = embed_texts_with_progress([texts_by_hash[digest] for digest in missing], config)
            if not embedded.get("ok"):
                return {"ok": False, "error": embedded.get("error")}
            fresh = dict(zip(missing, embedded["vectors"]))
            dimensions = {len(vector) for vector in fresh.values()} | {len(vector) for vector in vectors_by_hash.values()}
            if len(dimensions) > 1:
                # Cached vectors of another dimension (e.g. an API that changed its output size): start over.
                embedded = embed_texts_with_progress(list(texts_by_hash.values()), config)
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


def embed_before_indexing(
    conn: sqlite3.Connection,
    replaced_doc_ids: set[str],
    new_texts: list[str],
    force: bool,
) -> dict[str, Any]:
    """Embed the chunk texts the vector index is about to hold, before the index is written.

    Embedding a large document takes minutes with a local model. Done here, with no write
    transaction open, questions asked meanwhile still read the complete previous index; the write
    that follows finds every vector in the embedding cache and only assembles the index.

    replaced_doc_ids: documents whose chunks are about to be replaced or removed. new_texts: the
    chunks about to be written. force: as for build_optional_vector_index.
    Returns {"error": why embedding failed, or None, "embedded_hashes": the texts embedded here}.
    """
    outcome: dict[str, Any] = {"error": None, "embedded_hashes": set()}
    config = model_capabilities.runtime_config()
    embedding = model_capabilities.embedding_config(config)
    if not embedding["enabled"] or not embedding["usable"]:
        return outcome
    chunk_count = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] or 0)
    if not force and vector_store.vector_status(sqlite_chunk_count=chunk_count).get("ready"):
        return outcome  # the index stays as it is: nothing will be embedded
    kept = [str(text) for doc_id, text in conn.execute("SELECT doc_id, text FROM chunks") if str(doc_id) not in replaced_doc_ids]
    texts_by_hash = {vector_store.text_hash(text): text for text in [*kept, *new_texts]}
    if not texts_by_hash:
        return outcome
    cache = vector_store.EmbeddingCache(model_capabilities.embedding_model_key(embedding))
    try:
        missing = cache.missing(texts_by_hash)
        if not missing:
            return outcome
        embedded = embed_texts_with_progress([texts_by_hash[digest] for digest in missing], config)
        if not embedded.get("ok"):
            outcome["error"] = str(embedded.get("error") or "embedding failed")
            return outcome
        if len(cache.dimensions() | {len(vector) for vector in embedded["vectors"]}) > 1:
            # Cached vectors of another size (an API that changed its output size): start over, as
            # embed_records_with_cache does, but here rather than inside the write transaction.
            missing = list(texts_by_hash)
            embedded = embed_texts_with_progress([texts_by_hash[digest] for digest in missing], config)
            if not embedded.get("ok"):
                outcome["error"] = str(embedded.get("error") or "embedding failed")
                return outcome
            cache.prune([])  # every vector of the old size, so the sizes are not mixed again
        cache.put_many(zip(missing, embedded["vectors"]))
        outcome["embedded_hashes"] = set(missing)
        return outcome
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
    prepared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """prepared: what embed_before_indexing() returned for this run. Its error means embedding
    already failed, so it is not tried again here, inside the write transaction."""
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
    if prepared and prepared.get("error"):
        embedded: dict[str, Any] = {"ok": False, "error": prepared["error"]}
    else:
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
    # Chunks embedded just before this write count as embedded in this run, not as reused.
    early = (prepared or {}).get("embedded_hashes") or set()
    embedded_early = sum(1 for record in records if vector_store.text_hash(str(record["text"])) in early) if early else 0
    payload = vector_store.build_vector_index(
        records,
        embedded["vectors"],
        {
            "indexed_at": indexed_at,
            "source_fingerprint": source_hash,
            "provider": embedding.get("provider"),
            "model": embedding.get("model"),
            "requested_dimensions": embedding.get("requested_dimensions"),
            "embedded_chunks": embedded["embedded_chunks"] + embedded_early,
            "reused_chunks": embedded["reused_chunks"] - embedded_early,
        },
    )
    return payload
