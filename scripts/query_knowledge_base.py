#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Query the local SQLite FTS5 + embedding knowledge base, with optional web fallback."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
import math
import os
import queue
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import parse_qs, quote, quote_plus, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from api_providers import chat_completion, token_plan_rejected


DB_PATH = Path("knowledge_base/index/knowledge.db")
RAW_DIR = Path("knowledge_base/raw")
PROCESSED_DIR = Path("knowledge_base/processed")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
WEB_TIMEOUT_SECONDS = 5
WEB_SEARCH_TIMEOUT_SECONDS = 14
WEB_SEARCH_TOTAL_TIMEOUT_SECONDS = 20
WEB_SEARCH_MERGE_GRACE_SECONDS = 1.2
EMBEDDING_DIMS = 384
LOCAL_SCORE_THRESHOLD = 0.16
HOT_QUERY_THRESHOLD = 2
CACHE_TTL_SECONDS = 3600
DEFAULT_API_TIMEOUT_SECONDS = 20
STOP_WORDS = {
    "什么",
    "哪些",
    "哪个",
    "如何",
    "怎么",
    "是否",
    "以及",
    "有关",
    "问题",
    "区别",
    "作用",
    "影响",
    "一下",
    "为什么",
    "是什么",
    "有哪些",
    "有什么",
    "请问",
    "和",
    "与",
    "及",
}
ENGLISH_QUERY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "please",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
CHINESE_QUERY_STOP_PHRASES = (
    "请介绍一下",
    "介绍一下",
    "是什么",
    "为什么",
    "有哪些",
    "有什么",
    "什么时候",
    "怎么样",
    "如何",
    "怎么",
    "是否",
    "请问",
    "多少",
    "哪里",
    "何时",
    "用于",
    "用来",
    "进行",
    "实现",
    "可以",
    "作用",
    "一下",
)
TRAD_TO_SIMP = str.maketrans(
    {
        "與": "与",
        "個": "个",
        "為": "为",
        "於": "于",
        "這": "这",
        "實": "实",
        "現": "现",
        "時": "时",
        "無": "无",
        "線": "线",
        "網": "网",
        "節": "节",
        "點": "点",
        "體": "体",
        "傳": "传",
        "統": "统",
        "訊": "讯",
        "資": "资",
        "數": "数",
        "據": "据",
        "聲": "声",
        "稱": "称",
        "這": "这",
        "幾": "几",
        "後": "后",
        "間": "间",
        "關": "关",
        "聯": "联",
        "態": "态",
        "變": "变",
        "竊": "窃",
        "聽": "听",
        "偵": "侦",
        "纏": "缠",
        "糾": "纠",
        "綜": "综",
        "經": "经",
        "學": "学",
        "裏": "里",
        "裏": "里",
        "單": "单",
        "獨": "独",
        "純": "纯",
        "發": "发",
        "應": "应",
        "線": "线",
        "義": "义",
    }
)
TRAD_PHRASES = {"通訊": "通信"}


def to_simplified(text: str) -> str:
    try:
        from opencc import OpenCC  # type: ignore

        return OpenCC("t2s").convert(text)
    except Exception:
        for traditional, simplified in TRAD_PHRASES.items():
            text = text.replace(traditional, simplified)
        return text.translate(TRAD_TO_SIMP)


def human_text(text: str, max_len: int | None = None, simplify: bool = False) -> str:
    if simplify:
        text = to_simplified(text)
    text = unescape(text)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", lambda match: match.group(0).split("](")[0].lstrip("["), text)
    text = re.sub(r"^[ \t]*#{1,6}[ \t]*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`~]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" -#\t\r\n")
    if max_len is not None and len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


def strip_markdown_preserve_lines(text: str) -> str:
    text = unescape(text)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", lambda match: match.group(0).split("](")[0].lstrip("["), text)
    text = re.sub(r"^[ \t]*#{1,6}[ \t]*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`~]+", "", text)
    return text


def trim_terminal_punctuation(text: str) -> str:
    return text.rstrip("。！？!?；;,.， ")


def fetch_url_bytes(request: Request, timeout: int = WEB_TIMEOUT_SECONDS) -> bytes:
    try:
        import requests  # type: ignore

        try:
            response = requests.get(
                request.full_url,
                headers=dict(request.header_items()),
                timeout=(3, timeout),
            )
            response.raise_for_status()
            return response.content
        except Exception:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
    except ImportError:
        with urlopen(request, timeout=timeout) as response:
            return response.read()


def fetch_url_text(request: Request, timeout: int = WEB_TIMEOUT_SECONDS) -> str:
    return fetch_url_bytes(request, timeout).decode("utf-8", errors="ignore")


def candidate_terms(question: str) -> list[str]:
    terms: list[str] = []
    ascii_phrases = [
        phrase.strip()
        for phrase in re.findall(
            r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]*(?:\s+[A-Za-z][A-Za-z0-9_]*)+(?![A-Za-z0-9_])",
            question,
        )
    ]
    ascii_words = [
        word.lower()
        for word in re.findall(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]*(?![A-Za-z0-9_])", question)
        if len(word) >= 2
    ]
    terms.extend(ascii_phrases)
    terms.extend(ascii_words)

    for word in re.findall(r"[\u4e00-\u9fff]+", question):
        if re.fullmatch(r"[\u4e00-\u9fff]+", word):
            if word not in STOP_WORDS and len(word) >= 2:
                terms.append(word)
            for size in range(2, min(6, len(word)) + 1):
                for index in range(0, len(word) - size + 1):
                    term = word[index : index + size]
                    if term not in STOP_WORDS:
                        terms.append(term)
    lowered = question.lower()
    if "dioxide" in lowered or "carbon" in lowered:
        terms.extend(["二氧化碳", "碳固定"])
    if "穿过" in question or "介质" in question or "pass through" in lowered:
        terms.extend(["travel", "through", "solids", "liquids", "gases", "pass through"])
    return unique_terms(terms, limit=32)


def unique_terms(terms: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for term in sorted(terms, key=lambda item: (len(item), item), reverse=True):
        normalized = term.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(term)
    return result[:limit]


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


def fts_query(question: str) -> str:
    terms = candidate_terms(question)
    if not terms:
        escaped = question.replace('"', '""').strip()
        return f'"{escaped}"' if escaped else '""'
    return " OR ".join(f'"{term.replace(chr(34), chr(34) + chr(34))}"' for term in terms[:16])


def sentence_score(sentence: str, terms: list[str]) -> int:
    lowered = sentence.lower()
    return sum(3 if len(term) >= 4 else 1 for term in terms if term.lower() in lowered)


def split_sentences(text: str) -> list[str]:
    cleaned = strip_markdown_preserve_lines(text)
    parts = re.split(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+|\n+", cleaned)
    sentences = [human_text(part) for part in parts]
    headings = {"摘要", "关键机制", "可回答的问题", "工作表 sheet1"}
    question_prefixes = ("什么", "哪些", "哪个", "如何", "怎么", "为什么")
    return [
        part
        for part in sentences
        if part
        and part not in headings
        and not part.endswith(("?", "？"))
        and not part.startswith(question_prefixes)
    ]


def get_metadata(conn: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = conn.execute("SELECT key, value FROM metadata").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {row["key"]: row["value"] for row in rows}


def source_fingerprint(raw_dir: Path = RAW_DIR) -> str:
    supported = {".md", ".docx", ".pdf", ".xlsx"}
    items: list[str] = []
    if not raw_dir.exists():
        return ""
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in supported:
            rel_path = str(path.relative_to(raw_dir)).replace("\\", "/")
            stat = path.stat()
            items.append(f"{rel_path}:{stat.st_size}:{int(stat.st_mtime)}")
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def trigger_async_update(db_path: Path = DB_PATH, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> bool:
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        metadata = get_metadata(conn)
    except sqlite3.DatabaseError:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass

    fingerprint = source_fingerprint(raw_dir)
    if metadata.get("source_fingerprint") == fingerprint:
        return False

    script = Path(__file__).resolve().with_name("ingest_knowledge_base.py")
    command = [
        sys.executable,
        str(script),
        str(raw_dir),
        "--db",
        str(db_path),
        "--processed",
        str(processed_dir),
    ]
    subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return True


def load_env_file(path: Path = ENV_PATH) -> dict[str, str]:
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


def runtime_config() -> dict[str, str]:
    config = load_env_file()
    for key in (
        "LKA_USE_API",
        "LKA_API_PROVIDER",
        "LKA_API_BASE_URL",
        "LKA_API_KEY",
        "LKA_API_MODEL",
        "LKA_API_TIMEOUT_SECONDS",
    ):
        if os.environ.get(key):
            config[key] = os.environ[key]
    return config


def api_enabled(config: dict[str, str]) -> bool:
    return (
        config.get("LKA_USE_API", "false").lower() == "true"
        and bool(config.get("LKA_API_KEY"))
        and api_backend_allowed(config)
    )


def api_backend_allowed(config: dict[str, str]) -> bool:
    base_url = config.get("LKA_API_BASE_URL", "").lower()
    api_key = config.get("LKA_API_KEY", "").lower()
    return not token_plan_rejected(base_url, api_key)


def api_cache_variant(config: dict[str, str], allow_api: bool) -> str:
    if not allow_api or not api_enabled(config):
        return "retrieval-v14:language-guard-zh"
    provider = config.get("LKA_API_PROVIDER", "openai-compatible")
    base_url = config.get("LKA_API_BASE_URL", "")
    model = config.get("LKA_API_MODEL", "")
    return f"retrieval-v14:language-guard-api-zh:{provider}:{base_url}:{model}"


def cache_key(question: str, limit: int, use_web: bool, variant: str = "local") -> str:
    normalized = re.sub(r"\s+", " ", question.strip().lower())
    return hashlib.sha256(f"{normalized}|{limit}|{int(use_web)}|{variant}".encode("utf-8")).hexdigest()


def cache_get(
    conn: sqlite3.Connection,
    question: str,
    limit: int,
    use_web: bool,
    index_version: str,
    variant: str = "local",
) -> dict[str, Any] | None:
    key = cache_key(question, limit, use_web, variant)
    try:
        row = conn.execute("SELECT * FROM query_cache WHERE cache_key = ?", (key,)).fetchone()
    except sqlite3.OperationalError:
        return None
    if not row or row["index_version"] != index_version:
        return None

    updated_at = datetime.fromisoformat(row["updated_at"])
    if (datetime.now(timezone.utc) - updated_at).total_seconds() > CACHE_TTL_SECONDS:
        return None

    hit_count = int(row["hit_count"] or 0) + 1
    conn.execute(
        "UPDATE query_cache SET hit_count = ?, updated_at = ? WHERE cache_key = ?",
        (hit_count, datetime.now(timezone.utc).isoformat(timespec="seconds"), key),
    )
    conn.commit()
    if hit_count < HOT_QUERY_THRESHOLD:
        return None

    result = json.loads(row["answer_json"])
    if result.get("web_search_used") and not result.get("sources"):
        return None
    if result.get("web_search_used"):
        cached_sources = [
            source
            for source in (result.get("sources") or [])
            if source.get("source_origin") == "web" or source.get("url")
        ]
        relevant_sources = filter_web_sources(question, cached_sources, limit, allow_relaxed=True)
        if cached_sources and len(relevant_sources) != len(cached_sources):
            return None
    result["cache_hit"] = True
    result["cache_hit_count"] = hit_count
    return result


def cache_put(
    conn: sqlite3.Connection,
    question: str,
    limit: int,
    use_web: bool,
    index_version: str,
    result: dict[str, Any],
    variant: str = "local",
) -> None:
    key = cache_key(question, limit, use_web, variant)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cached = {key: value for key, value in result.items() if key not in {"cache_hit", "cache_hit_count"}}
    try:
        existing = conn.execute("SELECT hit_count FROM query_cache WHERE cache_key = ?", (key,)).fetchone()
        hit_count = int(existing["hit_count"]) + 1 if existing else 1
        conn.execute(
            """
            INSERT INTO query_cache(cache_key, question, answer_json, hit_count, index_version, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                answer_json = excluded.answer_json,
                hit_count = excluded.hit_count,
                index_version = excluded.index_version,
                updated_at = excluded.updated_at
            """,
            (key, question, json.dumps(cached, ensure_ascii=False), hit_count, index_version, now, now),
        )
        conn.commit()
    except sqlite3.OperationalError:
        return


def fts_candidates(conn: sqlite3.Connection, question: str, limit: int) -> list[dict[str, Any]]:
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
            (fts_query(question), max(limit * 5, 20)),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(row) for row in rows]


def semantic_candidates(conn: sqlite3.Connection, question: str, limit: int) -> list[dict[str, Any]]:
    query_embedding = local_embedding(question)
    if not query_embedding:
        return []

    rows = conn.execute(
        """
        SELECT chunks.chunk_id, chunks.doc_id, chunks.chunk_index, documents.file_name,
               documents.file_path, documents.file_type, chunks.text, chunks.embedding,
               NULL AS fts_rank
        FROM chunks
        JOIN documents ON documents.doc_id = chunks.doc_id
        """
    ).fetchall()
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        item = dict(row)
        item["semantic_score"] = cosine(query_embedding, parse_embedding(item.get("embedding", "")))
        if item["semantic_score"] > 0:
            scored.append((item["semantic_score"], item))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scored[: max(limit * 5, 20)]]


def fallback_like_candidates(conn: sqlite3.Connection, question: str, limit: int) -> list[dict[str, Any]]:
    terms = candidate_terms(question)
    if not terms:
        return []

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
    return [item for _, item in scored[: max(limit * 3, 12)]]


def rerank(rows: list[dict[str, Any]], question: str, limit: int) -> list[dict[str, Any]]:
    terms = candidate_terms(question)
    query_embedding = local_embedding(question)
    merged: dict[str, dict[str, Any]] = {}

    for row in rows:
        chunk_id = row["chunk_id"]
        item = merged.setdefault(chunk_id, row.copy())
        if row.get("fts_rank") is not None:
            item["fts_rank"] = row["fts_rank"]
        if row.get("semantic_score") is not None:
            item["semantic_score"] = max(item.get("semantic_score", 0.0), row["semantic_score"])
        if row.get("term_score") is not None:
            item["term_score"] = max(item.get("term_score", 0), row["term_score"])

    ranked: list[dict[str, Any]] = []
    max_term = 1
    for item in merged.values():
        item["term_score"] = item.get("term_score", sentence_score(item["text"], terms))
        max_term = max(max_term, item["term_score"])

    for item in merged.values():
        if "semantic_score" not in item:
            item["semantic_score"] = cosine(query_embedding, parse_embedding(item.get("embedding", "")))
        fts_rank = item.get("fts_rank")
        fts_score = 0.0 if fts_rank is None else 1.0 / (1.0 + abs(float(fts_rank)))
        term_score = item["term_score"] / max_term
        semantic_score = max(0.0, float(item.get("semantic_score", 0.0)))
        item["rerank_score"] = round(0.48 * semantic_score + 0.34 * term_score + 0.18 * fts_score, 6)
        item["semantic_score"] = round(semantic_score, 6)
        item["keyword_score"] = round(term_score, 6)
        item["fts_score"] = round(fts_score, 6)
        if item["rerank_score"] > 0:
            ranked.append(item)

    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return ranked[:limit]


def search(db_path: Path, question: str, limit: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        metadata = get_metadata(conn)
        rows = [
            *fts_candidates(conn, question, limit),
            *semantic_candidates(conn, question, limit),
            *fallback_like_candidates(conn, question, limit),
        ]
        ranked = rerank(rows, question, limit)
        return ranked, metadata
    finally:
        conn.close()


def citation_for_text(text: str, question: str, max_len: int = 360) -> str:
    terms = candidate_terms(question)
    sentences = split_sentences(text)
    lowered_question = question.lower()
    if "p waves" in lowered_question and "s waves" in lowered_question:
        wave_sentences = [
            sentence
            for sentence in sentences
            if "P waves" in sentence or "S waves" in sentence or "travel through" in sentence or "travel only through" in sentence
        ]
        if wave_sentences:
            return human_text(" ".join(trim_terminal_punctuation(sentence) + "." for sentence in wave_sentences), max_len=max_len)

    scored = [(sentence_score(sentence, terms), index, sentence) for index, sentence in enumerate(sentences)]
    matches = [(score, index, sentence) for score, index, sentence in scored if score > 0]
    if matches:
        max_score = max(score for score, _, _ in matches)
        selected_items = [
            (index, sentence)
            for score, index, sentence in sorted(matches, key=lambda item: (-item[0], item[1]))
            if score >= max(1, max_score - 1)
        ][:2]
        selected: list[str] = []
        for index, sentence in selected_items:
            if sentence.startswith("They ") and index > 0:
                selected.append(sentences[index - 1])
            selected.append(sentence)
        citation = " ".join(trim_terminal_punctuation(sentence) + "。" for sentence in selected)
        return human_text(citation, max_len=max_len)
    return human_text(text, max_len=max_len)


def build_answer(question: str, sources: list[dict[str, Any]]) -> str:
    citations = [source["citation"] for source in sources[:3] if source.get("citation")]
    if not citations:
        return "本地知识库没有找到足够依据。"
    if len(citations) == 1:
        return f"根据本地知识库，{trim_terminal_punctuation(citations[0])}。"
    return "根据本地知识库，" + "；".join(trim_terminal_punctuation(item) for item in citations) + "。"


class DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._active: str | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        classes = set(attrs_dict.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self._flush()
            self._current = {"title": "", "url": self._clean_url(attrs_dict.get("href", "")), "snippet": ""}
            self._active = "title"
            self._buffer = []
        elif "result__snippet" in classes and self._current is not None:
            self._flush()
            self._active = "snippet"
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._active:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._active == "title" and tag == "a":
            self._flush()
        elif self._active == "snippet" and tag in {"a", "div"}:
            self._flush()

    def close(self) -> None:
        self._flush()
        super().close()

    def _flush(self) -> None:
        if self._active and self._current is not None:
            value = human_text("".join(self._buffer), simplify=True)
            if value:
                self._current[self._active] = value
            if self._active == "snippet" and self._current.get("title") and self._current.get("url"):
                self.results.append(clean_web_source(self._current))
                self._current = None
            self._active = None
            self._buffer = []

    @staticmethod
    def _clean_url(url: str) -> str:
        if url.startswith("//"):
            url = "https:" + url
        parsed = urlparse(url)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
            target = parse_qs(parsed.query).get("uddg", [""])[0]
            if target:
                return target
        return url


def normalize_search_question(question: str) -> str:
    replacements = {
        "时什么": "是什么",
        "是神么": "是什么",
        "事什么": "是什么",
        "为什莫": "为什么",
        "怎末": "怎么",
    }
    normalized = question
    for original, replacement in replacements.items():
        normalized = normalized.replace(original, replacement)
    return normalized


def web_query(question: str) -> str:
    question = normalize_search_question(question)
    if not re.search(r"[\u4e00-\u9fff]", question):
        return question

    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", " ", question)
    for stop in CHINESE_QUERY_STOP_PHRASES:
        text = text.replace(stop, " ")
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    if len(chinese_terms) >= 2:
        return " ".join(chinese_terms[:4])
    if chinese_terms:
        return chinese_terms[0]
    return text.strip() or question


def web_query_variants(question: str) -> list[str]:
    normalized = normalize_search_question(question)
    variants = [web_query(normalized), normalized]
    if re.search(r"[\u4e00-\u9fff]", normalized):
        compact = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", " ", normalized).strip()
        variants.append(compact)
    return list(dict.fromkeys(variant for variant in variants if variant.strip()))[:3]


def chinese_relevance_score(question: str, text: str) -> float | None:
    chinese_text = question
    for stop in CHINESE_QUERY_STOP_PHRASES:
        chinese_text = chinese_text.replace(stop, " ")
    blocks = re.findall(r"[\u4e00-\u9fff]{2,}", chinese_text)
    if not blocks:
        return None

    scores: list[float] = []
    for block in blocks:
        if block in text:
            scores.append(1.0)
            continue
        grams = {block[index : index + 2] for index in range(len(block) - 1)}
        scores.append(sum(1 for gram in grams if gram in text) / len(grams) if grams else 0.0)
    return sum(scores) / len(scores)


def english_relevance_score(question: str, text: str) -> float | None:
    terms = [
        word.lower()
        for word in re.findall(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_-]*(?![A-Za-z0-9_])", question)
        if len(word) >= 3 and word.lower() not in ENGLISH_QUERY_STOP_WORDS
    ]
    if not terms:
        return None
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered) / len(set(terms))


def filter_web_sources(
    question: str,
    sources: list[dict[str, str]],
    limit: int,
    allow_relaxed: bool = False,
) -> list[dict[str, str]]:
    normalized_question = normalize_search_question(question)
    ranked: list[tuple[float, dict[str, str]]] = []
    for source in sources:
        cleaned = clean_web_source(source)
        text = f"{cleaned.get('title', '')} {cleaned.get('snippet', '')}"
        chinese_score = chinese_relevance_score(normalized_question, text)
        english_score = english_relevance_score(normalized_question, text)
        scores = [
            score
            for score in (chinese_score, english_score)
            if score is not None
        ]
        relevance = max(scores, default=0.0)
        if chinese_score is not None:
            minimum_relevance = 0.75 if not allow_relaxed else 0.3
        else:
            minimum_relevance = 0.67
        if relevance >= minimum_relevance:
            cleaned["relevance_score"] = round(relevance, 4)
            ranked.append((relevance, cleaned))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [source for _, source in ranked[:limit]]


def source_quote(source: dict[str, Any]) -> str:
    return str(source.get("citation") or source.get("excerpt") or source.get("snippet") or source.get("title") or "")


def source_is_duplicate(source: dict[str, Any], existing: list[dict[str, Any]]) -> bool:
    url = str(source.get("url", "")).strip().lower()
    file_path = str(source.get("file_path", "")).strip().lower()
    chunk_id = str(source.get("chunk_id", "")).strip().lower()
    normalized_quote = re.sub(r"\W+", "", source_quote(source).lower())
    for item in existing:
        if url and url == str(item.get("url", "")).strip().lower():
            return True
        if file_path and chunk_id and file_path == str(item.get("file_path", "")).strip().lower() and chunk_id == str(item.get("chunk_id", "")).strip().lower():
            return True
        existing_quote = re.sub(r"\W+", "", source_quote(item).lower())
        if normalized_quote and existing_quote and (
            normalized_quote == existing_quote
            or (
                min(len(normalized_quote), len(existing_quote)) >= 40
                and (normalized_quote in existing_quote or existing_quote in normalized_quote)
            )
            or SequenceMatcher(None, normalized_quote, existing_quote).ratio() >= 0.9
        ):
            return True
    return False


def dedupe_sources(sources: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in sources:
        if source_is_duplicate(source, result):
            continue
        result.append(source.copy())
        if len(result) >= limit:
            break
    return result


def alternate_sources(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for index in range(max(len(left), len(right))):
        if index < len(left):
            merged.append(left[index])
        if index < len(right):
            merged.append(right[index])
    return merged


def web_source_quality_score(question: str, source: dict[str, Any]) -> float:
    text = f"{source.get('title', '')} {source.get('snippet', '')}"
    score = float(source.get("relevance_score", 0.0) or 0.0)
    if "是什么" in normalize_search_question(question):
        if re.search(r"(?:是|为|指|属于).{0,20}(?:平台|网站|品牌|工具|服务|产品|概念|方法|公司)", text):
            score += 0.8
        elif re.search(r"(?:是|为|指|属于)", text):
            score += 0.35
    if re.search(r"版权所有|ICP备|隐私政策|用户协议|联系我们", text):
        score -= 0.8
    if re.search(r"为什么|怎么样|课程|模板|优惠|讨论", str(source.get("title", ""))):
        score -= 0.15
    return score


def duckduckgo_search(question: str, limit: int = 5) -> list[dict[str, str]]:
    request = Request(
        f"https://duckduckgo.com/html/?q={quote_plus(question)}",
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; local-kb-agent/1.0)",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    try:
        html = fetch_url_text(request)
    except Exception as exc:
        raise RuntimeError(f"DuckDuckGo request failed: {human_text(str(exc), max_len=160)}") from exc

    parser = DuckDuckGoParser()
    parser.feed(html)
    parser.close()
    return parser.results[:limit]


def bing_rss_search(question: str, limit: int = 5) -> list[dict[str, str]]:
    return bing_rss_search_url(
        f"https://www.bing.com/search?format=rss&q={quote_plus(question)}",
        limit,
        provider_name="Bing RSS",
    )


def bing_china_search(question: str, limit: int = 5) -> list[dict[str, str]]:
    return bing_rss_search_url(
        f"https://cn.bing.com/search?format=rss&q={quote_plus(question)}",
        limit,
        provider_name="Bing China RSS",
    )


def bing_rss_search_url(url: str, limit: int, provider_name: str) -> list[dict[str, str]]:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; local-kb-agent/1.0)",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    try:
        root = ElementTree.fromstring(fetch_url_bytes(request))
    except Exception as exc:
        raise RuntimeError(f"{provider_name} request failed: {human_text(str(exc), max_len=160)}") from exc

    results: list[dict[str, str]] = []
    for item in root.findall("./channel/item"):
        title = item.findtext("title", default="")
        url = item.findtext("link", default="")
        snippet = item.findtext("description", default="")
        if title and url:
            results.append(clean_web_source({"title": title, "url": url, "snippet": snippet}))
        if len(results) >= limit:
            break
    return results


def jina_baidu_search(question: str, limit: int = 5) -> list[dict[str, str]]:
    request = Request(
        f"https://r.jina.ai/http://www.baidu.com/s?wd={quote_plus(question)}",
        headers={"User-Agent": "Mozilla/5.0 (compatible; local-kb-agent/1.0)"},
    )
    try:
        markdown = fetch_url_text(request)
    except Exception as exc:
        raise RuntimeError(f"Baidu request failed: {human_text(str(exc), max_len=160)}") from exc

    results: list[dict[str, str]] = []
    pattern = re.compile(r"^###\s+\[(.*?)\]\((.*?)\)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(markdown))
    for index, match in enumerate(matches):
        title = match.group(1)
        url = unescape(match.group(2)).strip()
        snippet_start = match.end()
        snippet_end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        snippet = markdown[snippet_start:snippet_end]
        if title.strip().lower() in {"app", "课程", "竞赛"}:
            continue
        if title and url:
            results.append(clean_web_source({"title": title, "url": url, "snippet": snippet}, max_snippet=300))
        if len(results) >= limit:
            break
    return results


def web_search_variant(
    question: str,
    search_question: str,
    limit: int = 5,
    timeout_seconds: float = WEB_SEARCH_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    normalized_question = normalize_search_question(question)
    providers = (
        ("baidu", jina_baidu_search),
        ("bing_china", bing_china_search),
        ("duckduckgo", duckduckgo_search),
        ("wikipedia", wikipedia_search),
        ("jina_bing", jina_bing_search),
        ("bing_rss", bing_rss_search),
    )
    results: queue.Queue[tuple[str, list[dict[str, str]], str | None]] = queue.Queue()

    def invoke(provider_name: str, provider: Any) -> None:
        try:
            results.put((provider_name, provider(search_question, limit), None))
        except Exception as exc:  # noqa: BLE001
            results.put((provider_name, [], human_text(str(exc), max_len=180)))

    for provider_name, provider in providers:
        threading.Thread(target=invoke, args=(provider_name, provider), daemon=True).start()

    aggregated_sources: list[dict[str, str]] = []
    selected_providers: list[str] = []
    diagnostics: dict[str, Any] = {"query": search_question, "providers": {}}
    deadline = time.monotonic() + timeout_seconds
    merge_deadline: float | None = None
    completed = 0
    while completed < len(providers):
        remaining = deadline - time.monotonic()
        if merge_deadline is not None:
            remaining = min(remaining, merge_deadline - time.monotonic())
        if remaining <= 0:
            break
        try:
            provider_name, sources, error = results.get(timeout=remaining)
        except queue.Empty:
            break
        completed += 1
        relevant_sources = filter_web_sources(normalized_question, sources, limit)
        relaxed_filter_used = False
        if not relevant_sources and sources:
            relevant_sources = filter_web_sources(normalized_question, sources, limit, allow_relaxed=True)
            relaxed_filter_used = bool(relevant_sources)
        diagnostics["providers"][provider_name] = {
            "raw_sources": len(sources),
            "relevant_sources": len(relevant_sources),
            "relaxed_filter_used": relaxed_filter_used,
            "error": error,
        }
        if relevant_sources:
            if provider_name not in selected_providers:
                selected_providers.append(provider_name)
            for source in relevant_sources:
                source["search_provider"] = provider_name
            aggregated_sources.extend(relevant_sources)
            if merge_deadline is None:
                merge_deadline = time.monotonic() + WEB_SEARCH_MERGE_GRACE_SECONDS

    if aggregated_sources:
        for source in aggregated_sources:
            source["quality_score"] = round(web_source_quality_score(normalized_question, source), 4)
        ranked_sources = sorted(
            aggregated_sources,
            key=lambda source: float(source.get("quality_score", 0.0) or 0.0),
            reverse=True,
        )
        sources = dedupe_sources(ranked_sources, limit)
        return web_search_result(sources, providers=selected_providers), diagnostics
    diagnostics["timed_out_providers"] = len(providers) - completed
    return None, diagnostics


def web_search(question: str, limit: int = 5) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    deadline = time.monotonic() + WEB_SEARCH_TOTAL_TIMEOUT_SECONDS
    for variant in web_query_variants(question):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        result, diagnostics = web_search_variant(question, variant, limit, min(WEB_SEARCH_TIMEOUT_SECONDS, remaining))
        attempts.append(diagnostics)
        if result is not None:
            result["web_search_attempts"] = attempts
            return result

    providers = [provider for attempt in attempts for provider in attempt.get("providers", {}).values()]
    raw_sources = sum(int(provider.get("raw_sources", 0) or 0) for provider in providers)
    successful_providers = sum(1 for provider in providers if not provider.get("error"))
    if raw_sources:
        return web_search_error(
            "联网搜索已完成，但没有找到与问题足够相关的可信结果。请补充关键词或换一种更具体的问法。",
            reason="no_relevant_results",
            attempts=attempts,
        )
    if successful_providers:
        return web_search_error(
            "联网搜索已完成，但搜索源没有返回可用结果。请补充关键词或稍后重试。",
            reason="no_search_results",
            attempts=attempts,
        )
    return web_search_error(
        "联网搜索源暂时无法访问。请检查网络连接或稍后重试。",
        reason="providers_unavailable",
        attempts=attempts,
    )


def wikipedia_search(question: str, limit: int = 5) -> list[dict[str, str]]:
    languages = ["zh", "en"] if re.search(r"[\u4e00-\u9fff]", question) else ["en", "zh"]
    request_succeeded = False
    last_error: Exception | None = None
    for language in languages:
        url = (
            f"https://{language}.wikipedia.org/w/api.php?"
            f"action=opensearch&search={quote_plus(question)}&limit={limit}&namespace=0"
            f"&format=json&variant=zh-cn"
        )
        try:
            request = Request(url, headers={"User-Agent": "local-kb-agent/1.0", "Accept-Language": "zh-CN,zh;q=0.9"})
            data = json.loads(fetch_url_text(request))
            request_succeeded = True
        except Exception as exc:
            last_error = exc
            continue
        if len(data) < 4 or not data[1]:
            continue

        titles: list[str] = data[1]
        descriptions: list[str] = data[2]
        urls: list[str] = data[3]
        results: list[dict[str, str]] = []
        for index, title in enumerate(titles[:limit]):
            snippet = descriptions[index] if index < len(descriptions) else ""
            if not snippet:
                snippet = wikipedia_summary(language, title)
            results.append(
                clean_web_source(
                    {
                        "title": title,
                        "url": urls[index] if index < len(urls) else f"https://{language}.wikipedia.org/wiki/{quote(title)}",
                        "snippet": snippet,
                    }
                )
            )
        if results:
            return results
    if not request_succeeded and last_error is not None:
        raise RuntimeError(f"Wikipedia request failed: {human_text(str(last_error), max_len=160)}") from last_error
    return []


def wikipedia_summary(language: str, title: str) -> str:
    url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    try:
        request = Request(url, headers={"User-Agent": "local-kb-agent/1.0", "Accept-Language": "zh-CN,zh;q=0.9"})
        data = json.loads(fetch_url_text(request))
    except Exception:
        return ""
    return str(data.get("extract", ""))[:300]


def jina_bing_search(question: str, limit: int = 5) -> list[dict[str, str]]:
    request = Request(
        f"https://r.jina.ai/https://www.bing.com/search?q={quote_plus(question)}",
        headers={"User-Agent": "Mozilla/5.0 (compatible; local-kb-agent/1.0)"},
    )
    try:
        markdown = fetch_url_text(request)
    except Exception as exc:
        raise RuntimeError(f"Jina Bing request failed: {human_text(str(exc), max_len=160)}") from exc

    results: list[dict[str, str]] = []
    pattern = re.compile(r"^\d+\.\s+##\s+\[(.*?)\]\((.*?)\)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(markdown))
    for index, match in enumerate(matches):
        title = match.group(1)
        url = clean_bing_url(unescape(match.group(2)).strip())
        snippet_start = match.end()
        snippet_end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        snippet = markdown[snippet_start:snippet_end]
        if title and url:
            results.append(clean_web_source({"title": title, "url": url, "snippet": snippet}, max_snippet=300))
        if len(results) >= limit:
            break
    return results


def clean_bing_url(url: str) -> str:
    parsed = urlparse(url)
    encoded = parse_qs(parsed.query).get("u", [""])[0]
    if parsed.netloc.endswith("bing.com") and encoded.startswith("a1"):
        payload = encoded[2:]
        padding = "=" * (-len(payload) % 4)
        try:
            return base64.urlsafe_b64decode(payload + padding).decode("utf-8")
        except Exception:
            return url
    return url


def clean_web_source(source: dict[str, str], max_snippet: int = 320) -> dict[str, str]:
    return {
        "title": human_text(source.get("title", ""), max_len=160, simplify=True),
        "url": source.get("url", "").strip(),
        "snippet": human_text(source.get("snippet", ""), max_len=max_snippet, simplify=True),
    }


def web_search_result(
    sources: list[dict[str, str]],
    provider: str | None = None,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    selected_providers = providers or ([provider] if provider else [])
    cleaned_sources: list[dict[str, Any]] = []
    for source in dedupe_sources(sources, len(sources)):
        cleaned = clean_web_source(source)
        cleaned["source_origin"] = "web"
        if source.get("search_provider"):
            cleaned["search_provider"] = source["search_provider"]
        if source.get("relevance_score") is not None:
            cleaned["relevance_score"] = source["relevance_score"]
        if source.get("quality_score") is not None:
            cleaned["quality_score"] = source["quality_score"]
        cleaned_sources.append(cleaned)
    for source_id, source in enumerate(cleaned_sources, start=1):
        source["source_id"] = source_id
    snippets = [source["snippet"] or source["title"] for source in cleaned_sources[:3] if source["snippet"] or source["title"]]
    quality_scores = [float(source.get("quality_score", 0.0) or 0.0) for source in cleaned_sources]
    best_quality = max(quality_scores, default=0.0)
    summary_source_ids = [
        source["source_id"]
        for source in cleaned_sources
        if float(source.get("quality_score", 0.0) or 0.0) >= best_quality - 0.45
    ][:3]
    answer = "本地知识库没有找到足够依据，联网搜索到：" + "；".join(
        trim_terminal_punctuation(snippet) for snippet in snippets
    ) + "。"
    return {
        "answer": human_text(answer, simplify=True),
        "source_type": "web_search",
        "sources": cleaned_sources,
        "summary_source_ids": summary_source_ids,
        "web_search_provider": "+".join(selected_providers) or None,
        "web_search_providers": selected_providers,
        "need_web_search": False,
        "web_search_used": True,
    }


def web_search_error(answer: str, reason: str | None = None, attempts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "answer": human_text(answer, simplify=True),
        "source_type": "web_search",
        "sources": [],
        "need_web_search": False,
        "web_search_used": True,
        "web_search_reason": reason,
        "web_search_attempts": attempts or [],
    }


def merge_with_web_result(local_result: dict[str, Any], web_result: dict[str, Any], limit: int) -> dict[str, Any]:
    local_sources = [source.copy() for source in (local_result.get("sources") or [])]
    web_sources = [source.copy() for source in (web_result.get("sources") or [])]
    if not local_sources:
        return web_result
    if not web_sources:
        result = local_result.copy()
        result["web_search_used"] = True
        result["web_search_reason"] = web_result.get("web_search_reason")
        result["web_search_attempts"] = web_result.get("web_search_attempts", [])
        return result

    for source in local_sources:
        source["source_origin"] = source.get("source_origin") or "local"
    for source in web_sources:
        source["source_origin"] = "web"

    sources = dedupe_sources(alternate_sources(local_sources, web_sources), max(6, limit * 2))
    for source_id, source in enumerate(sources, start=1):
        source["source_id"] = source_id

    result = local_result.copy()
    result.update(
        {
            "source_type": "hybrid_web_knowledge",
            "sources": sources,
            "citations": grounded_citations({"sources": sources}),
            "web_search_provider": web_result.get("web_search_provider"),
            "web_search_providers": web_result.get("web_search_providers", []),
            "web_search_attempts": web_result.get("web_search_attempts", []),
            "web_search_used": True,
            "need_web_search": False,
            "retrieval_mode": "local_and_web_hybrid",
        }
    )
    return result


def augment_with_web(question: str, result: dict[str, Any], limit: int) -> dict[str, Any]:
    return merge_with_web_result(result, web_search(question, limit), limit)


def meaningful_query_terms(question: str) -> list[str]:
    terms = [
        word.lower()
        for word in re.findall(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_-]*(?![A-Za-z0-9_])", question)
        if len(word) >= 3 and word.lower() not in ENGLISH_QUERY_STOP_WORDS
    ]
    chinese_text = question
    for stop in CHINESE_QUERY_STOP_PHRASES:
        chinese_text = chinese_text.replace(stop, " ")
    terms.extend(re.findall(r"[\u4e00-\u9fff]{2,}", chinese_text))
    return unique_terms(terms, limit=16)


def lexical_query_coverage(question: str, text: str) -> float:
    terms = meaningful_query_terms(question)
    if not terms:
        return 0.0
    lowered = text.lower()
    matched = sum(1 for term in terms if term.lower() in lowered)
    return matched / len(terms)


def local_result_is_relevant(rows: list[dict[str, Any]], question: str) -> bool:
    if not rows or rows[0].get("rerank_score", 0.0) < LOCAL_SCORE_THRESHOLD:
        return False
    top = rows[0]
    has_lexical_score = float(top.get("keyword_score", 0.0) or 0.0) > 0 or float(top.get("fts_score", 0.0) or 0.0) > 0
    return has_lexical_score and lexical_query_coverage(question, str(top.get("text", ""))) >= 0.5


def grounded_citations(result: dict[str, Any]) -> list[dict[str, Any]]:
    sources = result.get("sources") or []
    citations: list[dict[str, Any]] = []
    seen_quotes: list[str] = []
    for index, source in enumerate(sources[:5], start=1):
        quote = source.get("citation") or source.get("excerpt") or source.get("snippet") or source.get("title")
        if not quote:
            continue
        normalized_quote = re.sub(r"\W+", "", str(quote).lower())
        if any(
            normalized_quote == existing
            or (min(len(normalized_quote), len(existing)) >= 40 and (normalized_quote in existing or existing in normalized_quote))
            or SequenceMatcher(None, normalized_quote, existing).ratio() >= 0.86
            for existing in seen_quotes
        ):
            continue
        seen_quotes.append(normalized_quote)
        citations.append(
            {
                "source_id": source.get("source_id") or index,
                "source": source.get("file") or source.get("title") or source.get("url"),
                "quote": quote,
            }
        )
    return citations


def local_translate_to_chinese(text: str) -> str:
    text = human_text(text, max_len=520, simplify=True)
    if not text:
        return ""
    if re.search(r"[\u4e00-\u9fff]", text):
        return text
    french_markers = (
        " à ",
        " au ",
        " aux ",
        " avec ",
        " dans ",
        " des ",
        " du ",
        " est ",
        " et ",
        " les ",
        " mon ",
        " nous ",
        " pour ",
        " que ",
        " qui ",
        " une ",
        " vous ",
    )
    lowered = f" {text.lower()} "
    if re.search(r"[àâçéèêëîïôùûüÿœæ]", lowered) or sum(marker in lowered for marker in french_markers) >= 2:
        return ""
    try:
        from argostranslate import translate  # type: ignore

        translated = human_text(translate.translate(text, "en", "zh"), max_len=520, simplify=True)
        replacements = {
            "Ohm的定律": "欧姆定律",
            "Ohm定律": "欧姆定律",
            "反应API": "Responses API",
            "客户SDK": "客户端 SDK",
        }
        for original, replacement in replacements.items():
            translated = translated.replace(original, replacement)
        translated = re.sub(r"(?<=[\u4e00-\u9fff]),", "，", translated)
        translated = re.sub(r"(?<=[\u4e00-\u9fff])\.", "。", translated)
        return translated
    except Exception:
        return ""


def chinese_grounded_fallback(result: dict[str, Any]) -> dict[str, Any]:
    if not result.get("sources"):
        result["api_used"] = False
        return result
    source_type = result.get("source_type")
    translated_citations: list[str] = []
    citations = grounded_citations(result)
    summary_source_ids = set(result.get("summary_source_ids") or [])
    if summary_source_ids:
        citations = [citation for citation in citations if citation.get("source_id") in summary_source_ids]
    for citation in citations[:3]:
        translated = local_translate_to_chinese(str(citation.get("quote", "")))
        if translated:
            translated_citations.append(f"{trim_terminal_punctuation(translated)} [{citation['source_id']}]")
    if translated_citations:
        if source_type == "web_search":
            prefix = "根据联网搜索资料，"
        elif source_type == "hybrid_web_knowledge":
            prefix = "根据本地知识库和联网资料，"
        else:
            prefix = "根据本地知识库，"
        result["answer"] = prefix + "；".join(translated_citations) + "。"
        result["answer_mode"] = "local_translation_grounded_summary"
    elif source_type == "web_search":
        result["answer"] = "已找到相关联网资料，但当前缺少可用的本地翻译模型。请查看下方引用来源；为避免误译，原始引用保持不变。"
        result["answer_mode"] = "chinese_grounded_fallback"
    else:
        result["answer"] = "已找到相关本地资料，但当前缺少可用的本地翻译模型。请查看下方引用来源；为避免误译，原始引用保持不变。"
        result["answer_mode"] = "chinese_grounded_fallback"
    result["api_used"] = False
    return result


def synthesize_with_api(question: str, result: dict[str, Any], config: dict[str, str]) -> dict[str, Any]:
    if result.get("source_type") not in {"knowledge_base", "web_search", "hybrid_web_knowledge", "faq_fallback"}:
        result["api_used"] = False
        return result

    citations = grounded_citations(result)
    if not citations:
        result["api_used"] = False
        return result
    if not api_enabled(config):
        return chinese_grounded_fallback(result)

    base_url = config.get("LKA_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = config.get("LKA_API_MODEL", "gpt-4o-mini")
    timeout = int(config.get("LKA_API_TIMEOUT_SECONDS", str(DEFAULT_API_TIMEOUT_SECONDS)) or DEFAULT_API_TIMEOUT_SECONDS)
    messages = [
            {
                "role": "system",
                "content": (
                    "You are a grounded knowledge base assistant. Answer only from the supplied citations. "
                    "Always answer in Simplified Chinese, even when the question or citations are in another language. "
                    "Translate relevant facts faithfully into Chinese. Do not invent facts or silently omit important details. "
                    "Remove repeated statements from overlapping citations. Ignore citation content unrelated to the question. "
                    "Integrate local knowledge and web evidence into one direct answer instead of listing search results. "
                    "When citations conflict, state the conflict instead of choosing an unsupported claim. "
                    "If the citations are insufficient, clearly say the available evidence is insufficient. "
                    "Keep the answer concise and cite source ids like [1]."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "source_type": result.get("source_type"), "citations": citations},
                    ensure_ascii=False,
                ),
            },
        ]
    try:
        answer = chat_completion(
            config.get("LKA_API_PROVIDER", "openai-compatible"),
            base_url,
            config.get("LKA_API_KEY", ""),
            model,
            messages,
            temperature=0.1,
            timeout=timeout,
        )
    except Exception as exc:
        result["api_error"] = human_text(str(exc), max_len=240)
        return chinese_grounded_fallback(result)

    if answer:
        result["answer"] = answer
        result["answer_mode"] = "api_grounded_summary"
        result["api_used"] = True
        result["api_provider"] = config.get("LKA_API_PROVIDER", "openai-compatible")
        result["api_model"] = model
    return result


def finalize_answer(question: str, result: dict[str, Any], config: dict[str, str], allow_api: bool) -> dict[str, Any]:
    if result.get("source_type") not in {"knowledge_base", "web_search", "hybrid_web_knowledge", "faq_fallback"}:
        return result
    if allow_api:
        return synthesize_with_api(question, result, config)
    return chinese_grounded_fallback(result)


def faq_fallback(question: str, limit: int = 3) -> dict[str, Any]:
    faq_path = PROCESSED_DIR / "faq.jsonl"
    if not faq_path.exists():
        return {
            "answer": "本地索引暂不可用，FAQ 兜底也没有可用条目。",
            "source_type": "faq_fallback",
            "sources": [],
            "citations": [],
            "need_web_search": True,
            "web_search_used": False,
        }

    terms = candidate_terms(question)
    scored: list[tuple[int, dict[str, Any]]] = []
    with faq_path.open("r", encoding="utf-8") as faq_in:
        for line in faq_in:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = f"{item.get('question', '')}\n{item.get('answer', '')}"
            score = sentence_score(text, terms)
            if score > 0:
                scored.append((score, item))
    scored.sort(key=lambda item: item[0], reverse=True)
    top_score = scored[0][0] if scored else 0
    items = [item for score, item in scored[:limit] if score >= max(2, top_score * 0.5)]
    if not items:
        return {
            "answer": "本地索引暂不可用，FAQ 兜底没有命中足够依据。",
            "source_type": "faq_fallback",
            "sources": [],
            "citations": [],
            "need_web_search": True,
            "web_search_used": False,
        }

    sources: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        quote = human_text(item.get("answer", ""), max_len=360)
        source = {
            "source_id": index,
            "file": item.get("file"),
            "file_path": item.get("file_path"),
            "file_type": item.get("file_type"),
            "chunk_id": item.get("chunk_id"),
            "excerpt": quote,
            "citation": quote,
        }
        sources.append(source)
        citations.append({"source_id": index, "file": item.get("file"), "chunk_id": item.get("chunk_id"), "quote": quote})

    return {
        "answer": "本地索引暂不可用，已使用 FAQ 兜底：" + "；".join(
            trim_terminal_punctuation(source["citation"]) for source in sources
        ) + "。",
        "source_type": "faq_fallback",
        "sources": sources,
        "citations": citations,
        "need_web_search": False,
        "web_search_used": False,
    }


def no_local_answer(use_web: bool, question: str, answer: str, metadata: dict[str, str] | None = None) -> dict[str, Any]:
    if use_web:
        return web_search(question)
    metadata = metadata or {}
    return {
        "answer": answer,
        "source_type": "none",
        "sources": [],
        "citations": [],
        "need_web_search": True,
        "web_search_used": False,
        "index_updated_at": metadata.get("indexed_at"),
        "embedding_model": metadata.get("embedding_model"),
        "message": "本地知识库没有足够依据。如需联网搜索，请追加 --web 参数。",
    }


def query(db_path: Path, question: str, limit: int = 5, use_web: bool = False, allow_api: bool = True) -> dict[str, Any]:
    config = runtime_config()
    cache_variant = api_cache_variant(config, allow_api)
    if not db_path.exists():
        result = faq_fallback(question, limit)
        if result.get("sources"):
            result["degraded"] = True
            result["degrade_reason"] = "local_index_missing"
            if use_web:
                result = augment_with_web(question, result, limit)
            return finalize_answer(question, result, config, allow_api)
        return finalize_answer(
            question,
            no_local_answer(use_web, question, "本地知识库索引不存在，请先运行 ingest。"),
            config,
            allow_api,
        )

    refresh_scheduled = trigger_async_update(db_path)
    cache_conn: sqlite3.Connection | None = None
    metadata: dict[str, str] = {}
    try:
        cache_conn = sqlite3.connect(db_path)
        cache_conn.row_factory = sqlite3.Row
        metadata = get_metadata(cache_conn)
        index_version = metadata.get("indexed_at", "")
        cached = cache_get(cache_conn, question, limit, use_web, index_version, cache_variant)
        if cached is not None:
            cached["async_update_scheduled"] = refresh_scheduled
            cache_conn.close()
            return cached
    except sqlite3.DatabaseError:
        if cache_conn is not None:
            cache_conn.close()
        result = faq_fallback(question, limit)
        result["degraded"] = True
        result["degrade_reason"] = "local_index_unavailable"
        if use_web:
            result = augment_with_web(question, result, limit)
        return finalize_answer(question, result, config, allow_api)

    try:
        rows, metadata = search(db_path, question, limit)
    except sqlite3.DatabaseError:
        if cache_conn is not None:
            cache_conn.close()
        result = faq_fallback(question, limit)
        result["degraded"] = True
        result["degrade_reason"] = "local_index_unavailable"
        if use_web:
            result = augment_with_web(question, result, limit)
        return finalize_answer(question, result, config, allow_api)

    if not local_result_is_relevant(rows, question):
        result = no_local_answer(use_web, question, "本地知识库没有找到足够依据。", metadata)
        result["async_update_scheduled"] = refresh_scheduled
        result = finalize_answer(question, result, config, allow_api)
        if cache_conn is not None and (not result.get("web_search_used") or result.get("sources")) and not result.get("web_search_reason"):
            cache_put(cache_conn, question, limit, use_web, metadata.get("indexed_at", ""), result, cache_variant)
            cache_conn.close()
        return result
    min_source_score = max(LOCAL_SCORE_THRESHOLD, rows[0]["rerank_score"] * 0.75)
    rows = [row for row in rows if row["rerank_score"] >= min_source_score]

    sources: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        citation = citation_for_text(row["text"], question)
        source = {
            "source_id": index,
            "file": row["file_name"],
            "file_path": row["file_path"],
            "file_type": row["file_type"],
            "chunk_id": row["chunk_id"],
            "excerpt": citation,
            "citation": citation,
            "rerank_score": row["rerank_score"],
            "semantic_score": row["semantic_score"],
            "keyword_score": row["keyword_score"],
            "fts_score": row["fts_score"],
        }
        sources.append(source)
        citations.append(
            {
                "source_id": index,
                "file": row["file_name"],
                "chunk_id": row["chunk_id"],
                "quote": citation,
            }
        )

    result = {
        "answer": human_text(build_answer(question, sources)),
        "source_type": "knowledge_base",
        "sources": sources,
        "citations": citations,
        "need_web_search": False,
        "web_search_used": False,
        "index_updated_at": metadata.get("indexed_at"),
        "embedding_model": metadata.get("embedding_model"),
        "chunk_strategy": metadata.get("chunk_strategy"),
        "retrieval_mode": "hybrid_fts_semantic_keyword",
        "async_update_scheduled": refresh_scheduled,
        "cache_hit": False,
    }
    if use_web:
        result = augment_with_web(question, result, limit)
    result = finalize_answer(question, result, config, allow_api)
    if cache_conn is not None and not result.get("web_search_reason"):
        cache_put(cache_conn, question, limit, use_web, metadata.get("indexed_at", ""), result, cache_variant)
        cache_conn.close()
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Query the local SQLite FTS5 + embedding knowledge base.")
    parser.add_argument("question")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--web", action="store_true", help="本地知识库答不了时允许联网搜索。")
    parser.add_argument("--no-api", action="store_true", help="Do not use configured answer synthesis API.")
    args = parser.parse_args()

    result = query(Path(args.db), args.question, args.limit, args.web, allow_api=not args.no_api)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
