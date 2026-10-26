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
from ipaddress import ip_address
import json
import math
import os
import queue
import re
import socket
import sqlite3
import subprocess
import sys
from pathlib import Path
import threading
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote, quote_plus, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
from xml.etree import ElementTree

import agentic_retrieval
from api_providers import chat_completion, chat_completion_detailed, token_plan_rejected
from ingest_knowledge_base import ingest_in_progress, is_ignored_raw_file, raw_file_stat_fingerprint
import model_capabilities
import project_runtime
from retrievers import abbreviation as abbreviation_retriever
from retrievers import excel_table as excel_table_retriever
from retrievers import station_test as station_test_retriever
from retrievers.base import DomainRetriever
import vector_store
from kb.text import (  # noqa: F401  (re-exported for callers of this module)
    TRAD_PHRASES,
    TRAD_TO_SIMP,
    first_markdown_heading_level,
    human_text,
    markdown_table_of_contents,
    sentence_score,
    split_sentences,
    strip_markdown_emphasis,
    strip_markdown_preserve_lines,
    to_simplified,
    trim_terminal_punctuation,
    unique_terms,
)
from kb.terms import (  # noqa: F401  (re-exported for callers of this module)
    CANDIDATE_TERMS_LIMIT,
    CHINESE_QUERY_STOP_PHRASES,
    ENGLISH_QUERY_STOP_WORDS,
    FTS_QUERY_MAX_TERMS,
    QUERY_ALIAS_PHRASES,
    QUERY_TRAILING_PHRASES,
    QUESTION_FUNCTION_PHRASES,
    RETRIEVAL_STOP_PHRASES,
    STOP_WORDS,
    abbreviation_query_term,
    candidate_terms,
    collapse_repeated_query_terms,
    enumeration_or_list_question,
    fft_manual_function_question,
    fts_query,
    iterative_retrieval_question,
    lexical_query_coverage,
    meaningful_query_terms,
    normalize_search_question,
    normalized_retrieval_question,
    process_question,
    query_term_variants,
    requested_station_name,
    retrieval_question_with_aliases,
    section_context_question,
    station_question_includes_schemes,
    station_test_item_question,
    strip_query_trailing_phrases,
)
from kb.config import (  # noqa: F401  (re-exported for callers of this module)
    DB_PATH,
    DEFAULT_API_TIMEOUT_SECONDS,
    ENV_PATH,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    agent_enabled,
    agent_max_rounds,
    api_backend_allowed,
    api_enabled,
    api_thinking,
    load_env_file,
    runtime_config,
)
from kb.retrieval import (  # noqa: F401  (re-exported for callers of this module)
    EMBEDDING_CACHE,
    EMBEDDING_CACHE_LOCK,
    EMBEDDING_DIMS,
    EMBEDDING_LSH_BUCKETS,
    EMBEDDING_LSH_PAIR_BUCKETS,
    ITERATIVE_RETRIEVAL_LIMIT,
    OVERVIEW_LIST_BOOST,
    RERANK_WEIGHTS,
    SECTION_CONTEXT_FORWARD_CHUNKS,
    SECTION_CONTEXT_MAX_CHARS,
    TABLE_OVERVIEW_MARKER,
    VECTOR_CANDIDATE_FALLBACK_MIN,
    VECTOR_CANDIDATE_MIN,
    VECTOR_CANDIDATE_MULTIPLIER,
    VECTOR_INDEX_THRESHOLD_CHUNKS,
    cached_embedding_rows,
    chunk_information_quality,
    cosine,
    embedding_cache_key,
    embedding_lsh_buckets,
    embedding_terms,
    expand_fft_manual_table_context,
    expand_section_context,
    external_vector_candidates,
    fallback_like_candidates,
    fts_candidates,
    get_metadata,
    local_embedding,
    lsh_semantic_candidates,
    parse_embedding,
    rerank,
    search,
    semantic_candidates,
    sql_like_candidates,
)
from kb.relevance import (  # noqa: F401  (re-exported for callers of this module)
    KNOWLEDGE_TEXT_CACHE,
    LOCAL_CONTENT_COVERAGE_THRESHOLD,
    LOCAL_SCORE_THRESHOLD,
    VERSIONED_NAME_PATTERN,
    ascii_term_pattern,
    content_coverage,
    knowledge_text,
    local_result_is_relevant,
    query_content_units,
    unknown_names,
)
from kb.citations import (  # noqa: F401  (re-exported for callers of this module)
    PRIMARY_CITATION_MAX_CHARS,
    PROCESS_SECONDARY_BLOCK_MAX_CHARS,
    SECONDARY_BLOCK_MAX_CHARS,
    SECTION_SCAN_MAX_CHUNKS,
    TABLE_ROW_MAX_CHARS,
    apply_citation_validation,
    block_citation,
    citation_for_retrieval_row,
    citation_for_text,
    citation_validation,
    evidence_lines,
    evidence_terms,
    focus_table_rows,
    full_section_text,
    grounded_citations,
    primary_citation,
    prune_unreferenced_output_sources,
    strip_native_reference_markers,
    synthesis_citations,
)
from kb.answer_text import (  # noqa: F401  (re-exported for callers of this module)
    MACHINE_TRANSLATION_ARTIFACT,
    OFFLINE_TRANSLATION_SETUP_HINT,
    build_answer,
    chinese_grounded_fallback,
    local_translate_to_chinese,
    looks_like_code,
    polish_internal_answer,
    polish_internal_answer_text,
)
from kb.structured import (  # noqa: F401  (re-exported for callers of this module)
    _explicit_station_table_records,
    _heading_station_table_records,
    _station_record,
    dictionary_entry_rows,
    extract_fft_manual_items,
    extract_fft_station_items,
    extract_station_table_items,
    extract_station_table_records,
    extract_station_table_records_with_excerpt,
    legacy_xlsx_region,
    n972_best_record,
    n972_best_records,
    n972_excel_field_intent,
    n972_excel_records,
    n972_record_from_cells,
    n972_value_is_useful,
    n972_worksheet_sections,
    normalize_excel_key,
    structured_abbreviation_result,
    structured_cell_text,
    structured_fft_manual_function_answer,
    structured_fft_station_result,
    structured_local_answer,
    structured_n972_excel_result,
    structured_station_result,
)


project_runtime.apply_project_runtime_env()
WEB_TIMEOUT_SECONDS = 5
WEB_SEARCH_TIMEOUT_SECONDS = 14
WEB_SEARCH_TOTAL_TIMEOUT_SECONDS = 20
WEB_SEARCH_MERGE_GRACE_SECONDS = 1.2
WEB_RESEARCH_SOURCE_LIMIT = 3
WEB_RESEARCH_TOTAL_TIMEOUT_SECONDS = 12
WEB_PAGE_FETCH_TIMEOUT_SECONDS = 7
WEB_PAGE_MAX_BYTES = 750_000
WEB_PAGE_MAX_TEXT_CHARS = 60_000
WEB_PAGE_CHUNK_CHARS = 900
HOT_QUERY_THRESHOLD = 2
CACHE_TTL_SECONDS = 3600
NEGATIVE_CACHE_TTL_SECONDS = 120
CACHE_VARIANT_VERSION = "retrieval-v56:process-sections-offline-translation"


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


def source_fingerprint(raw_dir: Path = RAW_DIR) -> str:
    supported = {".md", ".txt", ".doc", ".docx", ".pdf", ".xlsx"}
    items: list[str] = []
    if not raw_dir.exists():
        return ""
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and not is_ignored_raw_file(path) and path.suffix.lower() in supported:
            items.append(raw_file_stat_fingerprint(path, raw_dir))
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def trigger_async_update(db_path: Path = DB_PATH, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> bool:
    if not db_path.exists():
        return False
    if ingest_in_progress(db_path):
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


def apply_index_status(result: dict[str, Any], refresh_scheduled: bool) -> dict[str, Any]:
    result["async_update_scheduled"] = bool(refresh_scheduled)
    result["index_status"] = "updating" if refresh_scheduled else result.get("index_status", "ready")
    if refresh_scheduled:
        result["index_message"] = "资料已变化，正在后台更新，稍后重试可得到最新结果。"
    return result


def api_cache_variant(config: dict[str, str], allow_api: bool) -> str:
    if not allow_api:
        return f"{CACHE_VARIANT_VERSION}:no-api"
    if not api_enabled(config):
        return f"{CACHE_VARIANT_VERSION}:api-disabled"
    provider = config.get("LKA_API_PROVIDER", "openai-compatible")
    base_url = config.get("LKA_API_BASE_URL", "")
    model = config.get("LKA_API_MODEL", "")
    agent = (
        f":{agentic_retrieval.AGENT_VERSION}:rounds-{agent_max_rounds(config)}:thinking-{api_thinking(config)}"
        if agent_enabled(config)
        else ""
    )
    return f"{CACHE_VARIANT_VERSION}:api-refined:{provider}:{base_url}:{model}{agent}"


def query_cache_variant(config: dict[str, str], allow_api: bool, use_web: bool) -> str:
    web_part = "web-enabled" if use_web else "local-only"
    embedding_part = "embedding-disabled"
    if model_capabilities.enabled(config, "LKA_EMBEDDING_ENABLED"):
        embedding_part = (
            "embedding:"
            + ":".join(
                [
                    config.get("LKA_EMBEDDING_PROVIDER", ""),
                    config.get("LKA_EMBEDDING_BASE_URL", ""),
                    config.get("LKA_EMBEDDING_MODEL", ""),
                    config.get("LKA_EMBEDDING_DIMENSIONS", ""),
                    config.get("LKA_VECTOR_BACKEND", ""),
                ]
            )
        )
    rerank_part = "rerank-on" if model_capabilities.enabled(config, "LKA_RERANK_ENABLED") else "rerank-off"
    return f"{web_part}:{api_cache_variant(config, allow_api)}:{embedding_part}:{rerank_part}"


def cache_key(question: str, limit: int, use_web: bool, variant: str = "local") -> str:
    normalized = canonical_question_for_cache(normalized_retrieval_question(question))
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

    result = json.loads(row["answer_json"])
    ttl_seconds = NEGATIVE_CACHE_TTL_SECONDS if (
        result.get("source_type") == "none" or not result.get("sources")
    ) else CACHE_TTL_SECONDS
    updated_at = datetime.fromisoformat(row["updated_at"])
    if (datetime.now(timezone.utc) - updated_at).total_seconds() > ttl_seconds:
        return None

    hit_count = int(row["hit_count"] or 0) + 1
    conn.execute(
        "UPDATE query_cache SET hit_count = ?, updated_at = ? WHERE cache_key = ?",
        (hit_count, datetime.now(timezone.utc).isoformat(timespec="seconds"), key),
    )
    conn.commit()
    if hit_count < HOT_QUERY_THRESHOLD:
        return None

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


def canonical_question_for_cache(question: str) -> str:
    normalized = normalize_search_question(normalized_retrieval_question(question))
    abbreviation = abbreviation_query_term(normalized)
    if abbreviation:
        return f"abbreviation:{abbreviation.lower()}"
    if fft_manual_function_question(normalized):
        return "test-item:fft-manual-function"
    station_name = requested_station_name(normalized)
    if station_name and station_test_item_question(normalized):
        detail = "schemes" if station_question_includes_schemes(normalized) else "items"
        return f"station:{station_name.lower()}:{detail}"
    return re.sub(r"\s+", " ", normalized.strip().lower())


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
    normalized_question = normalize_search_question(question)
    if "是什么" in normalized_question or "是谁" in normalized_question:
        if re.search(r"(?:是|为|指|属于|出生|创始人|董事长|首席执行官|CEO).{0,30}(?:平台|网站|品牌|工具|服务|产品|概念|方法|公司|企业家|投资人|软件工程师|代表)", text):
            score += 0.8
        elif re.search(r"(?:是|为|指|属于|出生|创始人|董事长|首席执行官|CEO)", text):
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


def freshness_sensitive_question(question: str) -> bool:
    lowered = question.lower()
    markers = (
        "最新",
        "当前",
        "目前",
        "今天",
        "即将",
        "停止使用",
        "弃用",
        "下线",
        "过期",
        "截止",
        "何时",
        "什么时候",
        "latest",
        "current",
        "today",
        "deprecated",
        "deprecation",
        "sunset",
        "end of life",
        "eol",
    )
    return any(marker in lowered for marker in markers)


def html_visible_text(html: str) -> str:
    text = re.sub(r"(?is)<script[^>]*>.*?</script>|<style[^>]*>.*?</style>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(text).replace("\x00", "")).strip()


class WebPageTextParser(HTMLParser):
    BLOCK_TAGS = {
        "article",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "p",
        "pre",
        "section",
        "td",
        "th",
        "tr",
    }
    SKIP_TAGS = {"aside", "canvas", "footer", "form", "header", "nav", "noscript", "script", "style", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.blocks: list[str] = []
        self._block_parts: list[str] = []
        self._title_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if self._skip_depth:
            self._skip_depth += 1
            return
        if tag in self.SKIP_TAGS:
            self._flush_block()
            self._skip_depth = 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in self.BLOCK_TAGS:
            self._flush_block()

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self.title = human_text(" ".join(self._title_parts), max_len=180, simplify=True)
            self._title_parts = []
            self._in_title = False
            return
        if tag in self.BLOCK_TAGS:
            self._flush_block()

    def handle_data(self, data: str) -> None:
        value = re.sub(r"\s+", " ", unescape(data).replace("\x00", " ")).strip()
        if not value or self._skip_depth:
            return
        if self._in_title:
            self._title_parts.append(value)
        else:
            self._block_parts.append(value)

    def close(self) -> None:
        self._flush_block()
        super().close()

    def text(self) -> str:
        self._flush_block()
        blocks: list[str] = []
        for block in self.blocks:
            if block and (not blocks or block != blocks[-1]):
                blocks.append(block)
        return "\n".join(blocks)

    def _flush_block(self) -> None:
        block = human_text(" ".join(self._block_parts), simplify=True)
        if block:
            self.blocks.append(block)
        self._block_parts = []


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def validate_public_web_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Only public http/https webpage URLs are allowed.")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
        raise ValueError("Local webpage URLs are not allowed.")
    try:
        addresses = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError(f"Cannot resolve webpage host: {hostname}") from exc
    if not addresses:
        raise ValueError(f"Cannot resolve webpage host: {hostname}")
    for address in addresses:
        value = ip_address(address[4][0].split("%", 1)[0])
        if not value.is_global:
            raise ValueError("Non-public webpage addresses are not allowed.")
    return url


def decode_webpage(data: bytes, charset: str | None = None) -> str:
    candidates = [charset]
    head = data[:4096].decode("ascii", errors="ignore")
    match = re.search(r"(?i)charset\s*=\s*[\"']?\s*([a-z0-9._-]+)", head)
    if match:
        candidates.append(match.group(1))
    candidates.extend(["utf-8", "gb18030"])
    best = ""
    best_replacements = WEB_PAGE_MAX_BYTES
    for encoding in candidates:
        if not encoding:
            continue
        try:
            decoded = data.decode(encoding, errors="replace")
        except LookupError:
            continue
        replacements = decoded.count("\ufffd")
        if replacements < best_replacements:
            best = decoded
            best_replacements = replacements
        if replacements == 0:
            break
    return best


def fetch_public_webpage_text(url: str, timeout: int = WEB_PAGE_FETCH_TIMEOUT_SECONDS) -> tuple[str, str, str]:
    opener = build_opener(NoRedirectHandler())
    current_url = url
    for _ in range(4):
        validate_public_web_url(current_url)
        request = Request(
            current_url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; local-kb-agent/1.0)",
                "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )
        try:
            response = opener.open(request, timeout=timeout)
        except HTTPError as exc:
            location = exc.headers.get("Location", "")
            if exc.code in {301, 302, 303, 307, 308} and location:
                current_url = urljoin(current_url, location)
                continue
            raise
        with response:
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
                raise ValueError(f"Unsupported webpage content type: {content_type}")
            data = response.read(WEB_PAGE_MAX_BYTES + 1)
            if len(data) > WEB_PAGE_MAX_BYTES:
                data = data[:WEB_PAGE_MAX_BYTES]
            return decode_webpage(data, response.headers.get_content_charset()), current_url, content_type
    raise ValueError("Too many webpage redirects.")


def extract_webpage_text(page_text: str, content_type: str = "text/html") -> tuple[str, str]:
    if content_type == "text/plain":
        return "", human_text(page_text, max_len=WEB_PAGE_MAX_TEXT_CHARS)
    parser = WebPageTextParser()
    parser.feed(page_text)
    parser.close()
    text = parser.text()
    if not text:
        text = html_visible_text(page_text)
    return parser.title, text[:WEB_PAGE_MAX_TEXT_CHARS]


def webpage_chunks(text: str, max_chars: int = WEB_PAGE_CHUNK_CHARS) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0
    paragraphs = [human_text(part) for part in re.split(r"\n+", text) if human_text(part)]
    for paragraph in paragraphs:
        parts = split_sentences(paragraph) if len(paragraph) > max_chars else [paragraph]
        for part in parts:
            if not part:
                continue
            if current and current_chars + len(part) + 1 > max_chars:
                chunks.append(" ".join(current))
                current = []
                current_chars = 0
            current.append(part)
            current_chars += len(part) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def webpage_chunk_score(question: str, chunk: str) -> float:
    lexical = sentence_score(chunk, candidate_terms(question))
    semantic = max(0.0, cosine(local_embedding(question), local_embedding(chunk)))
    relevance = max(
        score or 0.0
        for score in (
            chinese_relevance_score(normalize_search_question(question), chunk),
            english_relevance_score(question, chunk),
        )
    )
    return lexical + semantic * 4.0 + relevance * 3.0


def web_evidence_quality_score(question: str, source: dict[str, Any]) -> float:
    quote = source_quote(source)
    score = max(
        float(source.get("quality_score", 0.0) or 0.0),
        float(source.get("relevance_score", 0.0) or 0.0),
    )
    if lexical_query_coverage(question, quote) <= 0:
        score -= 1.0
    if re.search(r"(?:出生|创始人|董事长|首席执行官|CEO|企业家|投资人|软件工程师|代表)", quote):
        score += 0.9
    if re.search(r"(?:元起|查看全部|服务中心|预约维修|退货|换货|包邮|售后|订单查询|ICP备|营业执照|违法和不良信息|知识产权侵权投诉)", quote):
        score -= 1.4
    if re.search(r"(?:Video|Image|快捷键说明|播放\s*/\s*暂停|退出全屏|快进|快退|按住此处可拖拽|详情)", quote):
        score -= 1.0
    if len(quote) > 450 and len(set(re.findall(r"[\u4e00-\u9fff]{2,}", quote))) > 80:
        score -= 0.35
    return score


def best_summary_source_ids(question: str, sources: list[dict[str, Any]], limit: int = 3) -> list[int]:
    scored: list[tuple[float, int]] = []
    for source in sources:
        source_id = int(source.get("source_id") or 0)
        if not source_id:
            continue
        score = web_evidence_quality_score(question, source)
        if score > 0:
            scored.append((score, source_id))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0] if scored else 0.0
    minimum_score = max(0.75, best_score - 0.5)
    return [source_id for score, source_id in scored if score >= minimum_score][:limit]


def research_web_source(question: str, source: dict[str, Any]) -> dict[str, Any]:
    researched = source.copy()
    try:
        page_text, resolved_url, content_type = fetch_public_webpage_text(str(source.get("url", "")))
        page_title, text = extract_webpage_text(page_text, content_type)
        chunks = webpage_chunks(text)
        ranked = sorted(
            ((webpage_chunk_score(question, chunk), chunk) for chunk in chunks),
            key=lambda item: item[0],
            reverse=True,
        )
        selected = [chunk for score, chunk in ranked[:2] if score > 0]
        if not selected and chunks:
            selected = chunks[:1]
        citation = human_text(
            " ".join(citation_for_text(chunk, question, max_len=520) for chunk in selected),
            max_len=1000,
            simplify=True,
        )
        if not citation:
            raise ValueError("Webpage did not contain readable text.")
    except Exception as exc:  # noqa: BLE001
        researched["page_fetch_status"] = "error"
        researched["page_fetch_error"] = human_text(str(exc), max_len=180)
        return researched
    researched.update(
        {
            "title": page_title or researched.get("title", ""),
            "resolved_url": resolved_url,
            "content_source": "webpage",
            "page_fetch_status": "ok",
            "page_chars": len(text),
            "page_chunk_score": round(ranked[0][0], 4) if ranked else 0.0,
            "snippet": citation,
            "excerpt": citation,
            "citation": citation,
        }
    )
    return researched


def enrich_web_search_result(question: str, result: dict[str, Any], limit: int = 5) -> dict[str, Any]:
    sources = [source.copy() for source in (result.get("sources") or [])]
    if not sources:
        result["web_research_used"] = False
        result["web_pages_fetched"] = 0
        result["web_pages_extracted"] = 0
        return result

    selected_sources = sources[: min(len(sources), limit, WEB_RESEARCH_SOURCE_LIMIT)]
    completed: queue.Queue[tuple[int, dict[str, Any]]] = queue.Queue()

    def invoke(index: int, source: dict[str, Any]) -> None:
        completed.put((index, research_web_source(question, source)))

    for index, source in enumerate(selected_sources):
        threading.Thread(target=invoke, args=(index, source), daemon=True).start()

    researched: dict[int, dict[str, Any]] = {}
    deadline = time.monotonic() + WEB_RESEARCH_TOTAL_TIMEOUT_SECONDS
    while len(researched) < len(selected_sources):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            index, source = completed.get(timeout=remaining)
        except queue.Empty:
            break
        researched[index] = source

    for index in range(len(selected_sources)):
        sources[index] = researched.get(
            index,
            {
                **sources[index],
                "page_fetch_status": "timeout",
                "page_fetch_error": "Webpage fetch timed out.",
            },
        )
    for index in range(len(selected_sources), len(sources)):
        sources[index]["page_fetch_status"] = "not_selected"

    for source_id, source in enumerate(sources, start=1):
        source["source_id"] = source_id
    extracted_ids = [source["source_id"] for source in sources if source.get("page_fetch_status") == "ok"]
    result["sources"] = sources
    result["citations"] = grounded_citations(result)
    result["summary_source_ids"] = best_summary_source_ids(question, sources) or extracted_ids[:3] or result.get("summary_source_ids", [])
    result["web_research_used"] = True
    result["web_pages_fetched"] = len(selected_sources)
    result["web_pages_extracted"] = len(extracted_ids)
    result["retrieval_mode"] = "web_search_page_rag"
    snippets = [source_quote(source) for source in sources[:3] if source_quote(source)]
    if snippets:
        result["answer"] = human_text(
            "本地知识库没有找到足够依据，联网研究到：" + "；".join(trim_terminal_punctuation(snippet) for snippet in snippets) + "。",
            simplify=True,
        )
    return result


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
            "web_research_used": web_result.get("web_research_used", False),
            "web_pages_fetched": web_result.get("web_pages_fetched", 0),
            "web_pages_extracted": web_result.get("web_pages_extracted", 0),
            "need_web_search": False,
            "retrieval_mode": "local_and_web_hybrid",
        }
    )
    return result


def augment_with_web(question: str, result: dict[str, Any], limit: int) -> dict[str, Any]:
    return merge_with_web_result(result, enrich_web_search_result(question, web_search(question, limit), limit), limit)


def attach_query_structure(question: str, result: dict[str, Any]) -> dict[str, Any]:
    if "query" not in result:
        result["query"] = {
            "original": question,
            "normalized": normalize_search_question(question),
            "canonical": canonical_question_for_cache(question),
            "web_query": web_query(question),
        }
    return result


def guidance_query_subject(question: str) -> str:
    normalized = normalize_search_question(normalized_retrieval_question(question)).strip()
    normalized = re.sub(r"[，。！？?；;：:\s]+", " ", normalized).strip()
    station = requested_station_name(normalized)
    if station and station_test_item_question(normalized):
        return station.upper() if re.fullmatch(r"[A-Za-z0-9_-]+", station) else station
    normalized = re.sub(r"^(?:请问|我想问|想问|帮我查一下|帮我查|查询一下|查询|查一下|查|关于)\s*", "", normalized).strip()
    normalized = re.sub(r"^(?:我的|我)\s*", "", normalized).strip()
    for _ in range(3):
        reduced = re.sub(
            r"(?:是什么|是啥|是多少|有哪些|有那些|有什么|怎么查|如何查|是|为|吗|呢)$",
            "",
            normalized,
        ).strip()
        if reduced == normalized:
            break
        normalized = reduced
    return normalized or question.strip()


def verified_local_suggestions(db_path: Path | None, question: str, subject: str) -> list[str]:
    if db_path is None or not db_path.exists():
        return []
    suggestions: list[str] = []
    abbreviation = abbreviation_query_term(subject)
    if abbreviation and structured_abbreviation_result(db_path, abbreviation) is not None:
        suggestions.append(f"{abbreviation}是什么")
    station = requested_station_name(question)
    if station:
        station_question = f"{station}有哪些测试项"
        if (
            structured_fft_station_result(db_path, station_question) is not None
            or structured_station_result(db_path, station_question) is not None
        ):
            suggestions.append(station_question)
    return list(dict.fromkeys(suggestions))[:3]


def local_candidate_has_answer(db_path: Path | None, question: str) -> bool:
    if db_path is None or not db_path.exists():
        return False
    abbreviation = abbreviation_query_term(question)
    if abbreviation and structured_abbreviation_result(db_path, abbreviation) is not None:
        return True
    station = requested_station_name(question)
    if station and station_test_item_question(question):
        return (
            structured_fft_station_result(db_path, question) is not None
            or structured_station_result(db_path, question) is not None
        )
    try:
        rows, _metadata = search(db_path, question, 5)
    except sqlite3.DatabaseError:
        return False
    return local_result_is_relevant(rows, question, db_path)


def suggested_local_questions(question: str, db_path: Path | None = None) -> list[str]:
    normalized = normalize_search_question(normalized_retrieval_question(question)).strip()
    subject = guidance_query_subject(question)
    verified = verified_local_suggestions(db_path, normalized, subject)
    if verified:
        return verified
    candidates: list[str] = []
    station = requested_station_name(normalized)
    if station:
        candidates.append(f"{station}有哪些测试项")
    wants_test_items = bool(re.search(r"测试|test|items?", normalized, re.IGNORECASE))
    for token in re.findall(r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_-]{1,30})(?![A-Za-z0-9_])", normalized):
        if token.lower() in ENGLISH_QUERY_STOP_WORDS:
            continue
        if token.lower() == "test":
            continue
        if token.upper() == token or any(ch.isdigit() for ch in token) or wants_test_items:
            display_token = token.upper() if len(token) <= 12 and re.fullmatch(r"[A-Za-z0-9_-]+", token) else token
            if wants_test_items or any(ch.isdigit() for ch in token):
                candidates.append(f"{display_token}有哪些测试项")
                candidates.append(f"{display_token}的测试方案是什么")
            else:
                candidates.append(f"{display_token}是什么")
            break
    if not candidates and subject:
        candidates.append(f"{subject}是什么")
    candidates = list(dict.fromkeys(candidates))[:3]
    if db_path is not None and db_path.exists():
        return [candidate for candidate in candidates if local_candidate_has_answer(db_path, candidate)][:3]
    return candidates


def guidance_subject(question: str, suggestions: list[str]) -> str:
    if suggestions:
        subject = re.split(r"有哪些测试项|的测试方案是什么|是什么", suggestions[0], maxsplit=1)[0].strip()
        if subject:
            return subject
    return guidance_query_subject(question)


def no_index_user_guidance(question: str, faq_missing: bool = True) -> dict[str, Any]:
    suggestions = suggested_local_questions(question)
    subject = guidance_subject(question, suggestions)
    target = suggestions[0] if suggestions else ""
    message = f"本地索引和 FAQ 都不可用，当前没有与「{subject}」相关的索引和 FAQ，无法从资料中判断这个问题。"
    if not faq_missing:
        message = f"本地索引不可用，FAQ 也没有命中与「{subject}」相关的足够依据。"
    if target:
        message += f" 你是否想问：{target}？请先建立索引后，再用这种更完整的问法查询。"
    else:
        message += " 请先建立索引后，再换成更完整、包含对象和问题类型的问法查询。"
    return {
        "kind": "no_index_no_faq" if faq_missing else "no_index_faq_no_match",
        "severity": "error",
        "message": message,
        "suggested_questions": suggestions,
    }


def local_no_match_user_guidance(question: str, db_path: Path | None = None) -> dict[str, Any]:
    suggestions = suggested_local_questions(question, db_path)
    subject = guidance_subject(question, suggestions)
    target = suggestions[0] if suggestions else ""
    message = f"本地索引存在，但当前问法没有命中与「{subject}」相关的足够依据。"
    if target:
        message += f" 你是否想问：{target}？请换成这种更完整的问法查询我们已有的索引答案。"
    else:
        message += " 请换成更完整、包含对象和问题类型的问法查询我们已有的索引答案。"
    return {
        "kind": "local_no_match",
        "severity": "error",
        "message": message,
        "suggested_questions": suggestions,
    }


def synthesize_with_api(question: str, result: dict[str, Any], config: dict[str, str]) -> dict[str, Any]:
    if result.get("source_type") not in {"knowledge_base", "web_search", "hybrid_web_knowledge", "faq_fallback"}:
        result["api_used"] = False
        return result

    citations = synthesis_citations(result)
    if not citations:
        result["api_used"] = False
        return result
    if not api_enabled(config):
        return chinese_grounded_fallback(result)

    base_url = config.get("LKA_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = config.get("LKA_API_MODEL", "gpt-5.2")
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
        validation = citation_validation(result)
        if not validation["ok"]:
            result["api_used"] = False
            result["api_validation_error"] = validation["issues"]
            return chinese_grounded_fallback(result)
        result["citation_validation"] = validation
    return result


def finalize_answer(question: str, result: dict[str, Any], config: dict[str, str], allow_api: bool) -> dict[str, Any]:
    attach_query_structure(question, result)
    if result.get("source_type") not in {"knowledge_base", "web_search", "hybrid_web_knowledge", "faq_fallback"}:
        return result
    if result.get("answer_mode") in {"api_grounded_summary", "api_agent_answer"}:
        return apply_citation_validation(prune_unreferenced_output_sources(polish_internal_answer(result)))
    if str(result.get("answer_mode", "")).startswith("local_structured_"):
        result["api_used"] = False
        return apply_citation_validation(polish_internal_answer(result))
    if allow_api:
        return apply_citation_validation(
            prune_unreferenced_output_sources(polish_internal_answer(synthesize_with_api(question, result, config)))
        )
    return apply_citation_validation(prune_unreferenced_output_sources(polish_internal_answer(chinese_grounded_fallback(result))))


def web_research_agent(question: str, config: dict[str, str], allow_api: bool, limit: int = 5) -> dict[str, Any]:
    result = enrich_web_search_result(question, web_search(question, limit), limit)
    if result.get("sources") and allow_api and api_enabled(config):
        result = synthesize_with_api(question, result, config)
        if result.get("api_used"):
            result["retrieval_mode"] = "web_research_grounded_api_summary"
        elif result.get("api_error"):
            result["retrieval_mode"] = "web_research_local_summary_after_api_error"
    else:
        result["api_used"] = False
    return result


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


def no_local_answer(
    use_web: bool,
    question: str,
    answer: str,
    metadata: dict[str, str] | None = None,
    config: dict[str, str] | None = None,
    allow_api: bool = True,
    limit: int = 5,
) -> dict[str, Any]:
    if use_web:
        return web_research_agent(question, config or {}, allow_api, limit)
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


def domain_retrievers() -> list[DomainRetriever]:
    return [
        abbreviation_retriever.create(structured_abbreviation_result),
        station_test_retriever.create(structured_fft_station_result),
        station_test_retriever.create(structured_station_result),
        excel_table_retriever.create(structured_n972_excel_result),
    ]


def run_domain_retrievers(
    db_path: Path,
    question: str,
    metadata: dict[str, str],
    refresh_scheduled: bool,
) -> dict[str, Any] | None:
    for retriever in domain_retrievers():
        result = retriever.retrieve(db_path, question, metadata, refresh_scheduled)
        if result is not None:
            result["retriever_plugin"] = retriever.name
            return result
    return None


AGENT_EVIDENCE_MAX_CHARS = 1800


def agent_retriever(db_path: Path, config: dict[str, str]) -> agentic_retrieval.Retrieve:
    def retrieve(query_text: str, limit: int) -> list[dict[str, Any]]:
        rows, _ = search(db_path, query_text, limit, config)
        rows = rows[:limit]
        for row in rows:
            # The whole section or table, as a local answer would cite it, so the model sees
            # complete steps and rows rather than a fragment.
            row["evidence"] = block_citation(full_section_text(db_path, row), query_text, AGENT_EVIDENCE_MAX_CHARS)
        return rows

    return retrieve


def agent_completion(config: dict[str, str]) -> agentic_retrieval.Complete:
    timeout = int(config.get("LKA_API_TIMEOUT_SECONDS", str(DEFAULT_API_TIMEOUT_SECONDS)) or DEFAULT_API_TIMEOUT_SECONDS)

    def complete(messages: list[dict[str, str]]) -> dict[str, Any]:
        return chat_completion_detailed(
            config.get("LKA_API_PROVIDER", "openai-compatible"),
            config.get("LKA_API_BASE_URL", "").rstrip("/"),
            config.get("LKA_API_KEY", ""),
            config.get("LKA_API_MODEL", ""),
            messages,
            temperature=0.0,
            timeout=timeout,
            json_output=True,
            max_tokens=agentic_retrieval.ANSWER_MAX_TOKENS,
            thinking=api_thinking(config),
        )

    return complete


def agent_trace(outcome: agentic_retrieval.AgentOutcome, config: dict[str, str]) -> dict[str, Any]:
    return {
        "version": agentic_retrieval.AGENT_VERSION,
        "decision": outcome.decision,
        "reason": outcome.reason,
        "rounds": outcome.rounds,
        "follow_up_queries": outcome.queries,
        "evidence_count": len(outcome.evidence),
        "usage": outcome.usage,
        "elapsed_ms": outcome.elapsed_ms,
        "thinking": api_thinking(config),
    }


def agent_answer_result(
    question: str,
    outcome: agentic_retrieval.AgentOutcome,
    metadata: dict[str, str],
    config: dict[str, str],
    refresh_scheduled: bool,
) -> dict[str, Any]:
    by_id = {item["id"]: item for item in outcome.evidence}
    sources: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    for source_id, evidence_id in enumerate(outcome.cited_ids, start=1):
        item = by_id[evidence_id]
        row = item["row"]
        source = {
            "source_id": source_id,
            "file": row.get("file_name"),
            "file_path": row.get("file_path"),
            "file_type": row.get("file_type"),
            "chunk_id": row.get("chunk_id"),
            "excerpt": item["text"],
            "citation": item["text"],
            "rerank_score": row.get("rerank_score"),
            "semantic_score": row.get("semantic_score"),
            "keyword_score": row.get("keyword_score"),
            "fts_score": row.get("fts_score"),
        }
        if row.get("context_chunk_ids"):
            source["context_chunk_ids"] = row["context_chunk_ids"]
        sources.append(source)
        citations.append({"source_id": source_id, "file": row.get("file_name"), "chunk_id": row.get("chunk_id"), "quote": item["text"]})
    return {
        "answer": agentic_retrieval.renumber_citations(outcome.answer, outcome.cited_ids),
        "answer_mode": "api_agent_answer",
        "source_type": "knowledge_base",
        "sources": sources,
        "citations": citations,
        "need_web_search": False,
        "web_search_used": False,
        "api_used": True,
        "api_provider": config.get("LKA_API_PROVIDER", "openai-compatible"),
        "api_model": config.get("LKA_API_MODEL", ""),
        "agent": agent_trace(outcome, config),
        "index_updated_at": metadata.get("indexed_at"),
        "embedding_model": metadata.get("embedding_model"),
        "vector_backend": metadata.get("vector_backend", "faiss"),
        "vector_embedding_model": metadata.get("vector_embedding_model"),
        "retrieval_mode": "agentic_llm_retrieval",
        "async_update_scheduled": refresh_scheduled,
        "index_status": "updating" if refresh_scheduled else "ready",
        "cache_hit": False,
    }


def query(db_path: Path, question: str, limit: int = 5, use_web: bool = False, allow_api: bool = True) -> dict[str, Any]:
    config = runtime_config()
    cache_variant = query_cache_variant(config, allow_api, use_web)
    if not db_path.exists():
        result = faq_fallback(question, limit)
        if result.get("sources"):
            result["degraded"] = True
            result["degrade_reason"] = "local_index_missing"
            if use_web:
                result = augment_with_web(question, result, limit)
            return finalize_answer(question, result, config, allow_api)
        faq_missing = not (PROCESSED_DIR / "faq.jsonl").exists()
        return finalize_answer(
            question,
            {
                **no_local_answer(
                    use_web,
                    question,
                    "本地知识库索引不存在，请先运行 ingest。",
                    config=config,
                    allow_api=allow_api,
                    limit=limit,
                ),
                "degraded": True,
                "degrade_reason": "local_index_missing",
                "user_guidance": no_index_user_guidance(question, faq_missing=faq_missing),
            }
            if not use_web
            else no_local_answer(
                use_web,
                question,
                "本地知识库索引不存在，请先运行 ingest。",
                config=config,
                allow_api=allow_api,
                limit=limit,
            ),
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
            apply_index_status(cached, refresh_scheduled)
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

    structured_result = run_domain_retrievers(db_path, question, metadata, refresh_scheduled)
    if structured_result is not None:
        result = finalize_answer(question, structured_result, config, allow_api)
        apply_index_status(result, refresh_scheduled)
        if cache_conn is not None:
            cache_put(cache_conn, question, limit, use_web, metadata.get("indexed_at", ""), result, cache_variant)
            cache_conn.close()
        return result

    verified_suggestions = verified_local_suggestions(db_path, question, guidance_query_subject(question))
    if station_test_item_question(question) and requested_station_name(question) and verified_suggestions:
        result = no_local_answer(
            use_web,
            question,
            "本地知识库没有找到足够依据。",
            metadata,
            config,
            allow_api,
            limit,
        )
        if not use_web:
            result["degraded"] = True
            result["degrade_reason"] = "local_no_match"
            result["user_guidance"] = local_no_match_user_guidance(question, db_path)
        apply_index_status(result, refresh_scheduled)
        result = finalize_answer(question, result, config, allow_api)
        if cache_conn is not None and (not result.get("web_search_used") or result.get("sources")) and not result.get("web_search_reason"):
            cache_put(cache_conn, question, limit, use_web, metadata.get("indexed_at", ""), result, cache_variant)
            cache_conn.close()
        return result

    agent_error = ""
    if allow_api and agent_enabled(config) and not unknown_names(question, db_path):
        try:
            outcome = agentic_retrieval.run_agent(
                question, agent_retriever(db_path, config), agent_completion(config), agent_max_rounds(config)
            )
        except (agentic_retrieval.AgentError, RuntimeError, OSError, ValueError, KeyError) as exc:
            # Unusable reply, timeout or network failure: answer locally instead.
            agent_error = human_text(str(exc), max_len=240)
        else:
            if outcome.decision == "answer":
                result = agent_answer_result(question, outcome, metadata, config, refresh_scheduled)
            else:
                reason = trim_terminal_punctuation(outcome.reason) if outcome.reason else ""
                result = no_local_answer(
                    use_web,
                    question,
                    f"本地知识库没有找到足够依据：{reason}。" if reason else "本地知识库没有找到足够依据。",
                    metadata,
                    config,
                    allow_api,
                    limit,
                )
                result["agent"] = agent_trace(outcome, config)
                if not use_web:
                    result["degraded"] = True
                    result["degrade_reason"] = "agent_no_evidence"
                    result["user_guidance"] = local_no_match_user_guidance(question, db_path)
            result = finalize_answer(question, result, config, allow_api)
            if result.get("source_type") == "knowledge_base":
                result["answer_quality"] = model_capabilities.evaluate_answer(
                    question, str(result.get("answer", "")), result.get("sources") or []
                )
            apply_index_status(result, refresh_scheduled)
            if cache_conn is not None and not result.get("web_search_reason"):
                cache_put(cache_conn, question, limit, use_web, metadata.get("indexed_at", ""), result, cache_variant)
                cache_conn.close()
            return result

    try:
        rows, metadata = search(db_path, question, limit, config)
    except sqlite3.DatabaseError:
        if cache_conn is not None:
            cache_conn.close()
        result = faq_fallback(question, limit)
        result["degraded"] = True
        result["degrade_reason"] = "local_index_unavailable"
        if use_web:
            result = augment_with_web(question, result, limit)
        return finalize_answer(question, result, config, allow_api)

    if not local_result_is_relevant(rows, question, db_path):
        missing = unknown_names(question, db_path)
        result = no_local_answer(
            use_web,
            question,
            f"本地知识库中没有出现「{'、'.join(missing)}」，没有找到足够依据。" if missing else "本地知识库没有找到足够依据。",
            metadata,
            config,
            allow_api,
            limit,
        )
        if not use_web:
            result["degraded"] = True
            result["degrade_reason"] = "local_no_match"
            result["user_guidance"] = local_no_match_user_guidance(question, db_path)
        apply_index_status(result, refresh_scheduled)
        result = finalize_answer(question, result, config, allow_api)
        if agent_error:
            result["agent_error"] = agent_error
        if (
            cache_conn is not None
            and not agent_error  # a fallback after a passing API failure must not be served from cache later
            and (not result.get("web_search_used") or result.get("sources"))
            and not result.get("web_search_reason")
        ):
            cache_put(cache_conn, question, limit, use_web, metadata.get("indexed_at", ""), result, cache_variant)
            cache_conn.close()
        return result
    if iterative_retrieval_question(question):
        min_source_score = max(LOCAL_SCORE_THRESHOLD * 0.8, rows[0]["rerank_score"] * 0.35)
        rows = [row for row in rows if row["rerank_score"] >= min_source_score][: max(limit, 8)]
    else:
        min_source_score = max(LOCAL_SCORE_THRESHOLD, rows[0]["rerank_score"] * 0.75)
        rows = [row for row in rows if row["rerank_score"] >= min_source_score]

    sources: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        citation = primary_citation(db_path, row, question) if index == 1 else citation_for_retrieval_row(row, question)
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
        if row.get("vector_score") is not None:
            source["vector_score"] = row["vector_score"]
        if row.get("semantic_candidate_mode"):
            source["semantic_candidate_mode"] = row["semantic_candidate_mode"]
        if row.get("llm_rerank_used"):
            source["llm_rerank_used"] = True
        if row.get("context_chunk_ids"):
            source["context_chunk_ids"] = row["context_chunk_ids"]
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
        "semantic_candidate_strategy": metadata.get("semantic_candidate_strategy"),
        "vector_backend": metadata.get("vector_backend", "faiss"),
        "vector_index_status": metadata.get("vector_index_status"),
        "vector_index_reason": metadata.get("vector_index_reason"),
        "vector_embedding_model": metadata.get("vector_embedding_model"),
        "vector_embedding_dimension": metadata.get("vector_embedding_dimension"),
        "query_route": json.loads(metadata.get("query_route", "{}") or "{}"),
        "retrieval_mode": "hybrid_faiss_fts_semantic_keyword"
        if metadata.get("vector_retrieval_used") == "true"
        else "hybrid_fts_semantic_keyword",
        "async_update_scheduled": refresh_scheduled,
        "index_status": "updating" if refresh_scheduled else "ready",
        "cache_hit": False,
    }
    structured_answer = structured_local_answer(db_path, question, sources)
    if structured_answer:
        result["answer"] = structured_answer
        result["answer_mode"] = "local_structured_table_summary"
    result = finalize_answer(question, result, config, allow_api)
    result["answer_quality"] = model_capabilities.evaluate_answer(question, str(result.get("answer", "")), result.get("sources") or [])
    if agent_error:
        result["agent_error"] = agent_error
    apply_index_status(result, refresh_scheduled)
    if cache_conn is not None and not agent_error and not result.get("web_search_reason"):
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
