"""Retrieval: vector, full-text and keyword candidates, section expansion, hybrid rerank and search."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
import threading
from typing import Any

import model_capabilities
import vector_store

from kb import config as kb_config
from kb.config import DB_PATH
from kb.terms import candidate_terms, enumeration_or_list_question, fft_manual_function_question, fts_query, iterative_retrieval_question, retrieval_question_with_aliases, section_context_question
from kb.text import first_markdown_heading_level, human_text, markdown_table_of_contents, sentence_score


EMBEDDING_DIMS = 384

VECTOR_INDEX_THRESHOLD_CHUNKS = 10_000

VECTOR_CANDIDATE_MULTIPLIER = 80

VECTOR_CANDIDATE_MIN = 400

VECTOR_CANDIDATE_FALLBACK_MIN = 80

EMBEDDING_LSH_BUCKETS = 12

EMBEDDING_LSH_PAIR_BUCKETS = 4

SECTION_CONTEXT_FORWARD_CHUNKS = 4

SECTION_CONTEXT_MAX_CHARS = 3200

# Excel import writes per-group item lists as "<sheet> 概览 | <group>：" chunks.
TABLE_OVERVIEW_MARKER = " 概览 | "

OVERVIEW_LIST_BOOST = 1.1

# A question that names a document by its file name (Release Table里…, 装备测试计划里…) asks about that
# document: its chunks rank above other documents' chunks that only mention the same things.
NAMED_DOCUMENT_BOOST = 1.15

# Hybrid rerank weights, chosen with eval/tune_weights.py (two-fold validation on the local eval set,
# with the corrected bm25 normalisation): MRR 0.759 -> 0.813, Hit@1 0.727 -> 0.764, Hit@5 0.836 -> 0.927
# against the original 0.48 / 0.34 / 0.18.
RERANK_WEIGHTS = {"semantic": 0.80, "keyword": 0.15, "fts": 0.05}

ITERATIVE_RETRIEVAL_LIMIT = 12

EMBEDDING_CACHE: dict[str, Any] = {"key": None, "rows": []}

EMBEDDING_CACHE_LOCK = threading.Lock()


def embedding_terms(text: str) -> list[str]:
    terms: list[str] = []
    lowered = text.lower()
    terms.extend(re.findall(r"[a-z0-9_]{2,}", lowered))
    for block in re.findall(r"[\u4e00-\u9fff]+", text):
        for size in (2, 3, 4):
            if len(block) < size:
                continue
            terms.extend(block[index : index + size] for index in range(len(block) - size + 1))
    return terms


def local_embedding(text: str, dims: int = EMBEDDING_DIMS) -> dict[int, float]:
    vector: dict[int, float] = {}
    for term in embedding_terms(text):
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dims
        sign = 1.0 if (value >> 8) & 1 else -1.0
        weight = 1.0 + min(len(term), 8) / 8.0
        vector[index] = vector.get(index, 0.0) + sign * weight

    norm = math.sqrt(sum(value * value for value in vector.values()))
    if not norm:
        return {}
    return {index: value / norm for index, value in vector.items() if abs(value) > 1e-9}


def parse_embedding(raw: str) -> dict[int, float]:
    if not raw:
        return {}
    try:
        return {int(index): float(value) for index, value in json.loads(raw)}
    except Exception:
        return {}


def cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


def embedding_lsh_buckets(embedding: dict[int, float]) -> list[str]:
    top = sorted(embedding.items(), key=lambda item: abs(item[1]), reverse=True)[:EMBEDDING_LSH_BUCKETS]
    buckets = [f"d:{index}:{1 if value >= 0 else 0}" for index, value in top]
    pair_dims = top[:EMBEDDING_LSH_PAIR_BUCKETS]
    for left_index in range(len(pair_dims)):
        for right_index in range(left_index + 1, len(pair_dims)):
            left = pair_dims[left_index]
            right = pair_dims[right_index]
            buckets.append(
                "p:"
                + ":".join(
                    [
                        str(min(left[0], right[0])),
                        str(max(left[0], right[0])),
                        str(1 if left[1] >= 0 else 0),
                        str(1 if right[1] >= 0 else 0),
                    ]
                )
            )
    return list(dict.fromkeys(buckets))


def embedding_cache_key(conn: sqlite3.Connection, metadata: dict[str, str]) -> str:
    try:
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    except sqlite3.OperationalError:
        chunk_count = 0
    try:
        db_file = conn.execute("PRAGMA database_list").fetchone()[2]
    except Exception:
        db_file = str(DB_PATH)
    return f"{Path(db_file or ':memory:').resolve()}:{metadata.get('indexed_at', '')}:{chunk_count}"


def cached_embedding_rows(conn: sqlite3.Connection, metadata: dict[str, str]) -> list[dict[str, Any]]:
    key = embedding_cache_key(conn, metadata)
    with EMBEDDING_CACHE_LOCK:
        if EMBEDDING_CACHE.get("key") == key:
            return [row.copy() for row in EMBEDDING_CACHE.get("rows", [])]

    rows = conn.execute(
        """
        SELECT chunks.chunk_id, chunks.doc_id, chunks.chunk_index, documents.file_name,
               documents.file_path, documents.file_type, chunks.text, chunks.embedding,
               NULL AS fts_rank
        FROM chunks
        JOIN documents ON documents.doc_id = chunks.doc_id
        """
    ).fetchall()
    cached_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["embedding_vector"] = parse_embedding(item.get("embedding", ""))
        cached_rows.append(item)

    with EMBEDDING_CACHE_LOCK:
        EMBEDDING_CACHE["key"] = key
        EMBEDDING_CACHE["rows"] = cached_rows
    return [row.copy() for row in cached_rows]


def get_metadata(conn: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = conn.execute("SELECT key, value FROM metadata").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {row["key"]: row["value"] for row in rows}


# Full-text candidates are taken best-first, but one document may fill at most this share of them:
# a workbook with hundreds of rows naming an item otherwise crowds out the one overview that
# answers what that item's status is.
FTS_PER_DOCUMENT_SHARE = 0.4


def fts_candidates(conn: sqlite3.Connection, question: str, limit: int) -> list[dict[str, Any]]:
    wanted = max(limit * 20, 100)
    per_document = max(1, int(wanted * FTS_PER_DOCUMENT_SHARE))
    try:
        rows = conn.execute(
            """
            SELECT chunks.chunk_id, chunks.doc_id, chunks.chunk_index, documents.file_name,
                   documents.file_path, documents.file_type, chunks.text, chunks.embedding,
                   bm25(chunks_fts) AS fts_rank
            FROM chunks_fts
            JOIN chunks ON chunks.chunk_id = chunks_fts.chunk_id
            JOIN documents ON documents.doc_id = chunks.doc_id
            WHERE chunks_fts MATCH ?
            ORDER BY fts_rank
            LIMIT ?
            """,
            (fts_query(question), wanted * 5),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    kept: list[dict[str, Any]] = []
    per_doc: dict[str, int] = {}
    for row in rows:
        doc_id = str(row["doc_id"])
        if per_doc.get(doc_id, 0) >= per_document:
            continue
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        kept.append(dict(row))
        if len(kept) >= wanted:
            break
    return kept


def semantic_candidates(
    conn: sqlite3.Connection,
    question: str,
    limit: int,
    metadata: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    query_embedding = local_embedding(question)
    if not query_embedding:
        return []

    try:
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    except sqlite3.OperationalError:
        chunk_count = 0
    if chunk_count >= VECTOR_INDEX_THRESHOLD_CHUNKS:
        lsh_rows = lsh_semantic_candidates(conn, query_embedding, limit)
        if len(lsh_rows) >= min(VECTOR_CANDIDATE_FALLBACK_MIN, max(limit * 5, 20)):
            return lsh_rows

    rows = cached_embedding_rows(conn, metadata or get_metadata(conn))
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        item = dict(row)
        item["semantic_score"] = cosine(query_embedding, item.get("embedding_vector") or parse_embedding(item.get("embedding", "")))
        if item["semantic_score"] > 0:
            scored.append((item["semantic_score"], item))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scored[: max(limit * 20, 100)]]


def lsh_semantic_candidates(
    conn: sqlite3.Connection,
    query_embedding: dict[int, float],
    limit: int,
) -> list[dict[str, Any]]:
    buckets = embedding_lsh_buckets(query_embedding)
    if not buckets:
        return []
    placeholders = ",".join("?" for _ in buckets)
    candidate_limit = max(limit * VECTOR_CANDIDATE_MULTIPLIER, VECTOR_CANDIDATE_MIN)
    try:
        rows = conn.execute(
            f"""
            SELECT chunks.chunk_id, chunks.doc_id, chunks.chunk_index, documents.file_name,
                   documents.file_path, documents.file_type, chunks.text, chunks.embedding,
                   NULL AS fts_rank,
                   COUNT(embedding_lsh.bucket) AS vector_bucket_hits
            FROM embedding_lsh
            JOIN chunks ON chunks.chunk_id = embedding_lsh.chunk_id
            JOIN documents ON documents.doc_id = chunks.doc_id
            WHERE embedding_lsh.bucket IN ({placeholders})
            GROUP BY chunks.chunk_id
            ORDER BY vector_bucket_hits DESC
            LIMIT ?
            """,
            (*buckets, candidate_limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        item = dict(row)
        item["semantic_candidate_mode"] = "lsh"
        item["semantic_score"] = cosine(query_embedding, parse_embedding(item.get("embedding", "")))
        if item["semantic_score"] > 0:
            scored.append((item["semantic_score"], item))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scored[: max(limit * 20, 100)]]


def query_vector(conn: sqlite3.Connection, question: str, config: dict[str, str]) -> list[float] | None:
    """The question's embedding when the vector index is ready, else None."""
    embedding_cfg = model_capabilities.embedding_config(config)
    if embedding_cfg.get("backend") != "faiss" or not embedding_cfg.get("usable"):
        return None
    try:
        sqlite_chunk_count = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] or 0)
    except sqlite3.OperationalError:
        sqlite_chunk_count = None
    if not vector_store.vector_status(sqlite_chunk_count=sqlite_chunk_count).get("ready"):
        return None
    embedded = model_capabilities.embed_query(question, config)
    if not embedded.get("ok") or not embedded.get("vectors"):
        return None
    return embedded["vectors"][0]


def score_with_vectors(rows: list[dict[str, Any]], vector: list[float] | None) -> list[dict[str, Any]]:
    """Give every candidate the vector similarity, whichever search found it. A chunk that full-text
    search finds by a rare name (the one overview row naming an item) would otherwise be ranked by the
    rough local hash embedding, far below the vector search's own candidates."""
    if vector is None:
        return rows
    scores = vector_store.vector_scores(vector, [str(row["chunk_id"]) for row in rows if row.get("vector_score") is None])
    for row in rows:
        score = scores.get(str(row["chunk_id"]))
        if score is not None:
            row["vector_score"] = score
            row["semantic_score"] = max(0.0, score)
    return rows


def external_vector_candidates(
    conn: sqlite3.Connection,
    question: str,
    limit: int,
    config: dict[str, str],
    vector: list[float] | None = None,
) -> list[dict[str, Any]]:
    vector = vector if vector is not None else query_vector(conn, question, config)
    if vector is None:
        return []
    vector_rows = vector_store.search_vector_index(vector, max(limit * 20, 100))
    if not vector_rows.get("ok") or not vector_rows.get("rows"):
        return []
    score_by_chunk = {str(row["chunk_id"]): row for row in vector_rows["rows"]}
    chunk_ids = list(score_by_chunk.keys())
    placeholders = ",".join("?" for _ in chunk_ids)
    try:
        rows = conn.execute(
            f"""
            SELECT chunks.chunk_id, chunks.doc_id, chunks.chunk_index, documents.file_name,
                   documents.file_path, documents.file_type, chunks.text, chunks.embedding,
                   NULL AS fts_rank
            FROM chunks
            JOIN documents ON documents.doc_id = chunks.doc_id
            WHERE chunks.chunk_id IN ({placeholders})
            """,
            chunk_ids,
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    ranked: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        vector_row = score_by_chunk.get(str(item["chunk_id"]), {})
        item["semantic_candidate_mode"] = "faiss"
        item["vector_score"] = float(vector_row.get("vector_score", 0.0) or 0.0)
        item["semantic_score"] = float(vector_row.get("semantic_score", item["vector_score"]) or 0.0)
        ranked.append(item)
    ranked.sort(key=lambda item: float(item.get("vector_score", 0.0) or 0.0), reverse=True)
    return ranked


def fallback_like_candidates(conn: sqlite3.Connection, question: str, limit: int) -> list[dict[str, Any]]:
    terms = candidate_terms(question)
    if not terms:
        return []

    try:
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    except sqlite3.OperationalError:
        chunk_count = 0
    if chunk_count >= VECTOR_INDEX_THRESHOLD_CHUNKS:
        return sql_like_candidates(conn, terms, limit)

    rows = conn.execute(
        """
        SELECT chunks.chunk_id, chunks.doc_id, chunks.chunk_index, documents.file_name,
               documents.file_path, documents.file_type, chunks.text, chunks.embedding,
               NULL AS fts_rank
        FROM chunks
        JOIN documents ON documents.doc_id = chunks.doc_id
        """
    ).fetchall()
    scored: list[tuple[int, dict[str, Any]]] = []
    for row in rows:
        item = dict(row)
        score = sentence_score(item["text"], terms)
        if score > 0:
            item["term_score"] = score
            scored.append((score, item))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scored[: max(limit * 12, 60)]]


def sql_like_candidates(conn: sqlite3.Connection, terms: list[str], limit: int) -> list[dict[str, Any]]:
    selected_terms = terms[:10]
    if not selected_terms:
        return []
    where_clause = " OR ".join("chunks.text LIKE ?" for _ in selected_terms)
    params = [f"%{term}%" for term in selected_terms]
    try:
        rows = conn.execute(
            f"""
            SELECT chunks.chunk_id, chunks.doc_id, chunks.chunk_index, documents.file_name,
                   documents.file_path, documents.file_type, chunks.text, chunks.embedding,
                   NULL AS fts_rank
            FROM chunks
            JOIN documents ON documents.doc_id = chunks.doc_id
            WHERE {where_clause}
            LIMIT ?
            """,
            (*params, max(limit * 40, 240)),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    scored: list[tuple[int, dict[str, Any]]] = []
    for row in rows:
        item = dict(row)
        score = sentence_score(item["text"], terms)
        if score > 0:
            item["term_score"] = score
            scored.append((score, item))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scored[: max(limit * 12, 60)]]


def chunk_information_quality(text: str) -> float:
    compact = human_text(text)
    if not compact:
        return 0.0
    if markdown_table_of_contents(text):
        return 0.3
    if re.search(r"\b(enum|LogMsgType|Enum[A-Za-z0-9_]+)\b", compact, flags=re.IGNORECASE) and (
        "枚举" in compact or " 值 " in f" {compact} " or "| 值 |" in text
    ):
        return 1.25
    header_markers = ("站位 | 测试项目", "测试项 | 测试子项", "工具名称 | 测试门限", "卡关/判定方式")
    header_hits = sum(marker in compact for marker in header_markers)
    detail_markers = ("工序名称：", "岗位资源：", "1.", "1、", "PASS", "FAIL", "MES", "测试LOG")
    detail_hits = sum(marker in compact for marker in detail_markers)
    if header_hits >= 2 and detail_hits <= 1:
        return 0.35
    if header_hits and len(compact) < 180 and detail_hits <= 1:
        return 0.55
    return 1.0


def expand_fft_manual_table_context(
    conn: sqlite3.Connection,
    ranked: list[dict[str, Any]],
    question: str,
) -> list[dict[str, Any]]:
    if not fft_manual_function_question(question):
        return ranked
    expanded: list[dict[str, Any]] = []
    expanded_once = False
    for row in ranked:
        item = row.copy()
        if not expanded_once and str(item.get("file_type", "")).lower() == "xlsx":
            anchor = None
            if "人工功能测试" in str(item.get("text", "")) or "IR CameraTest" in str(item.get("text", "")):
                anchor = item
            else:
                anchor_row = conn.execute(
                    """
                    SELECT chunk_id, doc_id, chunk_index, text
                    FROM chunks
                    WHERE doc_id = ? AND text LIKE '%人工功能测试%MES%FFT%' AND text LIKE '%IR CameraTest%'
                    ORDER BY chunk_index
                    LIMIT 1
                    """,
                    (item["doc_id"],),
                ).fetchone()
                if anchor_row:
                    anchor = {**item, **dict(anchor_row)}
            if anchor:
                neighbors = conn.execute(
                    """
                    SELECT chunk_id, chunk_index, text
                    FROM chunks
                    WHERE doc_id = ? AND chunk_index >= ? AND chunk_index <= ?
                    ORDER BY chunk_index
                    """,
                    (anchor["doc_id"], anchor["chunk_index"], anchor["chunk_index"] + 5),
                ).fetchall()
                context_chunks = [str(neighbor["text"]) for neighbor in neighbors]
                if context_chunks:
                    item.update(anchor)
                    item["text"] = "\n\n".join(context_chunks)
                    item["context_expanded"] = True
                    item["context_chunk_ids"] = [str(neighbor["chunk_id"]) for neighbor in neighbors]
                    expanded_once = True
        expanded.append(item)
    return expanded


def expand_section_context(
    conn: sqlite3.Connection,
    ranked: list[dict[str, Any]],
    question: str,
) -> list[dict[str, Any]]:
    if not section_context_question(question):
        return ranked

    expanded: list[dict[str, Any]] = []
    for row in ranked:
        item = row.copy()
        heading_level = first_markdown_heading_level(str(item.get("text", "")))
        if (
            str(item.get("file_type", "")).lower() not in {"md", "markdown"}
            or heading_level is None
            or markdown_table_of_contents(str(item.get("text", "")))
        ):
            expanded.append(item)
            continue

        neighbors = conn.execute(
            """
            SELECT chunk_id, chunk_index, text
            FROM chunks
            WHERE doc_id = ? AND chunk_index > ? AND chunk_index <= ?
            ORDER BY chunk_index
            """,
            (
                item["doc_id"],
                item["chunk_index"],
                item["chunk_index"] + SECTION_CONTEXT_FORWARD_CHUNKS,
            ),
        ).fetchall()
        context_chunks = [str(item["text"])]
        context_chunk_ids = [str(item["chunk_id"])]
        context_length = len(context_chunks[0])
        for neighbor in neighbors:
            neighbor_text = str(neighbor["text"])
            neighbor_level = first_markdown_heading_level(neighbor_text)
            if neighbor_level is not None and neighbor_level <= heading_level:
                break
            if context_length + len(neighbor_text) > SECTION_CONTEXT_MAX_CHARS:
                break
            context_chunks.append(neighbor_text)
            context_chunk_ids.append(str(neighbor["chunk_id"]))
            context_length += len(neighbor_text)
        if len(context_chunks) > 1:
            item["text"] = "\n\n".join(context_chunks)
            item["context_expanded"] = True
            item["context_chunk_ids"] = context_chunk_ids
        expanded.append(item)
    return expanded


# A chunk that ends announcing what follows («主要在三个方面：») was cut before its list.
CONTINUATION_ENDING = re.compile(r"(?:[:：]|如下)\s*$")

CONTINUATION_MAX_CHARS = 1500


def attach_continuations(conn: sqlite3.Connection, ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join the next chunk of the same document to a chunk that ends with a colon, so the list it
    introduces is part of the evidence; that next chunk then no longer appears on its own."""
    attached: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in ranked:
        if str(row.get("chunk_id")) in attached:
            continue
        item = row
        if CONTINUATION_ENDING.search(str(row.get("text", ""))) and row.get("doc_id") is not None:
            last_index = int(row.get("chunk_index", 0)) + len(row.get("context_chunk_ids") or [None]) - 1
            following = conn.execute(
                "SELECT chunk_id, text FROM chunks WHERE doc_id = ? AND chunk_index = ?",
                (row["doc_id"], last_index + 1),
            ).fetchone()
            if following is not None and len(str(following["text"])) <= CONTINUATION_MAX_CHARS:
                item = row.copy()
                item["text"] = f"{row['text']}\n{following['text']}"
                item["continuation_chunk_id"] = str(following["chunk_id"])
                attached.add(str(following["chunk_id"]))
        result.append(item)
    return result


def named_documents(question: str, file_names: set[str]) -> set[str]:
    """File names the question names: two consecutive words of the name (Release Table), or one of
    its Chinese runs of at least four characters as a whole (装备测试计划)."""
    lowered = re.sub(r"[\s_\-]+", " ", question.lower())
    compact = lowered.replace(" ", "")
    named = set()
    for file_name in file_names:
        stem = Path(file_name).stem
        words = [word.lower() for word in re.findall(r"[A-Za-z]{2,}", stem)]
        pairs = [f"{left} {right}" for left, right in zip(words, words[1:])]
        runs = re.findall(r"[\u4e00-\u9fff]{4,}", stem)
        if any(pair in lowered or pair.replace(" ", "") in compact for pair in pairs) or any(run in question for run in runs):
            named.add(file_name)
    return named


def rerank(rows: list[dict[str, Any]], question: str, limit: int, config: dict[str, str] | None = None) -> list[dict[str, Any]]:
    terms = candidate_terms(question)
    query_embedding = local_embedding(question)
    list_question = enumeration_or_list_question(question)
    named = named_documents(question, {str(row.get("file_name", "")) for row in rows})
    merged: dict[str, dict[str, Any]] = {}

    for row in rows:
        chunk_id = row["chunk_id"]
        item = merged.setdefault(chunk_id, row.copy())
        if row.get("fts_rank") is not None:
            item["fts_rank"] = row["fts_rank"]
        if row.get("semantic_score") is not None:
            item["semantic_score"] = max(item.get("semantic_score", 0.0), row["semantic_score"])
        if row.get("semantic_candidate_mode") == "faiss":
            item["semantic_candidate_mode"] = "faiss"
        if row.get("vector_score") is not None:
            item["vector_score"] = max(float(item.get("vector_score", 0.0) or 0.0), float(row["vector_score"]))
        if row.get("term_score") is not None:
            item["term_score"] = max(item.get("term_score", 0), row["term_score"])

    ranked: list[dict[str, Any]] = []
    max_term = 1
    for item in merged.values():
        item["term_score"] = item.get("term_score", sentence_score(item["text"], terms))
        max_term = max(max_term, item["term_score"])
    # SQLite bm25() is negative and more negative means a better match, so score each match
    # relative to the best one for this query (best = 1.0). The former 1 / (1 + |bm25|) gave the
    # best matches the lowest scores.
    best_bm25 = max((abs(float(item["fts_rank"])) for item in merged.values() if item.get("fts_rank") is not None), default=0.0)

    for item in merged.values():
        if "semantic_score" not in item:
            item["semantic_score"] = cosine(query_embedding, parse_embedding(item.get("embedding", "")))
        fts_rank = item.get("fts_rank")
        fts_score = abs(float(fts_rank)) / best_bm25 if fts_rank is not None and best_bm25 else 0.0
        term_score = item["term_score"] / max_term
        semantic_score = max(0.0, float(item.get("semantic_score", 0.0)))
        quality_score = chunk_information_quality(item["text"])
        item["rerank_score"] = round(
            (
                RERANK_WEIGHTS["semantic"] * semantic_score
                + RERANK_WEIGHTS["keyword"] * term_score
                + RERANK_WEIGHTS["fts"] * fts_score
            )
            * quality_score,
            6,
        )
        if list_question and TABLE_OVERVIEW_MARKER in str(item["text"]):
            # A table overview lists every item of a group in one chunk; for "which items" questions
            # it beats any single row of that group.
            item["rerank_score"] = round(item["rerank_score"] * OVERVIEW_LIST_BOOST, 6)
        if str(item.get("file_name", "")) in named:
            item["rerank_score"] = round(item["rerank_score"] * NAMED_DOCUMENT_BOOST, 6)
        item["semantic_score"] = round(semantic_score, 6)
        item["keyword_score"] = round(term_score, 6)
        item["fts_score"] = round(fts_score, 6)
        item["information_quality_score"] = quality_score
        if item["rerank_score"] > 0:
            ranked.append(item)

    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    llm_rerank = model_capabilities.rerank_chunks(question, ranked, config)
    if llm_rerank.get("ok") and llm_rerank.get("ordered_chunk_ids"):
        by_id = {str(item["chunk_id"]): item for item in ranked}
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for chunk_id in llm_rerank["ordered_chunk_ids"]:
            item = by_id.get(str(chunk_id))
            if not item or str(chunk_id) in seen:
                continue
            item = item.copy()
            item["llm_rerank_used"] = True
            ordered.append(item)
            seen.add(str(chunk_id))
        ordered.extend(item for item in ranked if str(item["chunk_id"]) not in seen)
        ranked = ordered
    return ranked[:limit]


def search(
    db_path: Path,
    question: str,
    limit: int,
    config: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    config = config or kb_config.runtime_config()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        metadata = get_metadata(conn)
        search_question = retrieval_question_with_aliases(question)
        route = model_capabilities.classify_query(question, config)
        search_limit = max(limit, ITERATIVE_RETRIEVAL_LIMIT) if iterative_retrieval_question(question) else limit
        vector = query_vector(conn, search_question, config)
        rows = [
            *external_vector_candidates(conn, search_question, search_limit, config, vector),
            *fts_candidates(conn, search_question, search_limit),
            *semantic_candidates(conn, search_question, search_limit, metadata),
            *fallback_like_candidates(conn, search_question, search_limit),
        ]
        rows = score_with_vectors(rows, vector)
        ranked = rerank(rows, search_question, search_limit, config)
        ranked = expand_fft_manual_table_context(conn, ranked, question)
        ranked = expand_section_context(conn, ranked, question)
        ranked = attach_continuations(conn, ranked)
        metadata = {
            **metadata,
            "query_route": json.dumps(route, ensure_ascii=False),
            "vector_retrieval_used": "true" if any(row.get("semantic_candidate_mode") == "faiss" for row in ranked) else "false",
        }
        return ranked, metadata
    finally:
        conn.close()
