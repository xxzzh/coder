#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Query the local SQLite FTS5 + embedding knowledge base, with optional web fallback.

This module is the main flow: answer cache, index status, the answer API and retrieval agent,
guidance when nothing is found, query() and the command line. The building blocks live in kb/:
text, terms (question analysis), config, retrieval, relevance (the answer gate), citations,
answer_text (answers without a model), structured (table answers) and web (web search).
Their names are re-exported here, so existing callers keep working.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import re
import sqlite3
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

import agentic_retrieval
from api_providers import chat_completion, chat_completion_detailed
from ingest_knowledge_base import ingest_in_progress, is_ignored_raw_file, raw_file_stat_fingerprint
import llm_budget
import model_capabilities
import project_runtime
from retrievers import abbreviation as abbreviation_retriever
from retrievers import excel_table as excel_table_retriever
from retrievers import station_test as station_test_retriever
from retrievers.base import DomainRetriever
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
    covering_row_index,
    knowledge_text,
    local_result_is_relevant,
    query_content_units,
    rows_about_named_subject,
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
from kb.web import (  # noqa: F401  (re-exported for callers of this module)
    DuckDuckGoParser,
    NoRedirectHandler,
    WEB_PAGE_CHUNK_CHARS,
    WEB_PAGE_FETCH_TIMEOUT_SECONDS,
    WEB_PAGE_MAX_BYTES,
    WEB_PAGE_MAX_TEXT_CHARS,
    WEB_RESEARCH_SOURCE_LIMIT,
    WEB_RESEARCH_TOTAL_TIMEOUT_SECONDS,
    WEB_SEARCH_MERGE_GRACE_SECONDS,
    WEB_SEARCH_TIMEOUT_SECONDS,
    WEB_SEARCH_TOTAL_TIMEOUT_SECONDS,
    WEB_TIMEOUT_SECONDS,
    WebPageTextParser,
    alternate_sources,
    best_summary_source_ids,
    bing_china_search,
    bing_rss_search,
    bing_rss_search_url,
    chinese_relevance_score,
    clean_bing_url,
    clean_web_source,
    decode_webpage,
    dedupe_sources,
    duckduckgo_search,
    english_relevance_score,
    enrich_web_search_result,
    extract_webpage_text,
    fetch_public_webpage_text,
    fetch_url_bytes,
    fetch_url_text,
    filter_web_sources,
    freshness_sensitive_question,
    html_visible_text,
    jina_baidu_search,
    jina_bing_search,
    merge_with_web_result,
    research_web_source,
    source_is_duplicate,
    source_quote,
    validate_public_web_url,
    web_evidence_quality_score,
    web_query,
    web_query_variants,
    web_search,
    web_search_error,
    web_search_result,
    web_search_variant,
    web_source_quality_score,
    webpage_chunk_score,
    webpage_chunks,
    wikipedia_search,
    wikipedia_summary,
)


project_runtime.apply_project_runtime_env()
HOT_QUERY_THRESHOLD = 2
CACHE_TTL_SECONDS = 3600
NEGATIVE_CACHE_TTL_SECONDS = 120
CACHE_VARIANT_VERSION = "retrieval-v57:mixed-documents"
# Answers built without a model (layout, gate, cited evidence). Raising it refreshes cached local
# answers only; cached API answers stay valid and are not paid for again.
LOCAL_ANSWER_VERSION = "local-v3"


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
        return f"{CACHE_VARIANT_VERSION}:no-api:{LOCAL_ANSWER_VERSION}"
    if not api_enabled(config):
        return f"{CACHE_VARIANT_VERSION}:api-disabled:{LOCAL_ANSWER_VERSION}"
    provider = config.get("LKA_API_PROVIDER", "openai-compatible")
    base_url = config.get("LKA_API_BASE_URL", "")
    model = config.get("LKA_API_MODEL", "")
    summary_model = llm_budget.model_for_task(config, "summary")
    if summary_model != model:
        model += f":summary-{summary_model}"
    agent = ""
    if agent_enabled(config):
        agent = f":{agentic_retrieval.AGENT_VERSION}:rounds-{agent_max_rounds(config)}:thinking-{api_thinking(config)}"
        agent_model = llm_budget.model_for_task(config, "agent")
        if agent_model != config.get("LKA_API_MODEL", ""):
            agent += f":model-{agent_model}"
        limits = (agent_evidence_max_chars(config), agent_evidence_max_items(config))
        if limits != (AGENT_EVIDENCE_MAX_CHARS, agentic_retrieval.EVIDENCE_MAX_ITEMS):
            agent += f":evidence-{limits[0]}x{limits[1]}"
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


SUMMARY_SYSTEM_PROMPT = (
    "You are a grounded knowledge base assistant. Answer only from the supplied citations. "
    "Always answer in Simplified Chinese, even when the question or citations are in another language. "
    "Translate relevant facts faithfully into Chinese. Do not invent facts or silently omit important details. "
    "Remove repeated statements from overlapping citations. Ignore citation content unrelated to the question. "
    "Integrate local knowledge and web evidence into one direct answer instead of listing search results. "
    "When citations conflict, state the conflict instead of choosing an unsupported claim. "
    "If the citations are insufficient, clearly say the available evidence is insufficient. "
    "Write a polished answer that is concise and easy to read, not copied fragments: open with one or two sentences "
    "that answer the question directly; when it covers several stages or categories, give each group a short line "
    "ending with '：' followed by '- ' bullets, with a blank line between groups; one point per bullet, in fluent "
    "sentences; merge overlapping points. Concise means tight wording, not fewer facts: keep every listed item, every "
    "step, and identifiers, numbers and units exactly as written. No Markdown tables, # headings or bold. "
    "Cite source ids like [1] at the end of each point."
)
# Longest citation sent to the summary model (LKA_API_CITATION_MAX_CHARS); long web pages and
# whole sections keep their beginning.
SUMMARY_CITATION_MAX_CHARS = 4000


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
    budget = llm_budget.task_settings({"LKA_API_MODEL": "gpt-5.2", **config}, "summary")
    model = budget["model"]
    timeout = int(config.get("LKA_API_TIMEOUT_SECONDS", str(DEFAULT_API_TIMEOUT_SECONDS)) or DEFAULT_API_TIMEOUT_SECONDS)
    max_chars = llm_budget.int_setting(config, "LKA_API_CITATION_MAX_CHARS", SUMMARY_CITATION_MAX_CHARS, 200)
    citations = [{**citation, "quote": llm_budget.clip(str(citation.get("quote", "")), max_chars)} for citation in citations]
    # Fixed instructions first, then the question and its citations, so the instructions are a cached prefix.
    messages = llm_budget.build_messages(
        SUMMARY_SYSTEM_PROMPT,
        llm_budget.stable_json({"question": question, "source_type": result.get("source_type"), "citations": citations}),
    )
    try:
        answer = chat_completion(
            config.get("LKA_API_PROVIDER", "openai-compatible"),
            base_url,
            config.get("LKA_API_KEY", ""),
            model,
            messages,
            temperature=0.1,
            timeout=timeout,
            max_tokens=budget["max_tokens"],
            thinking=budget["thinking"],
            task="summary",
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


# Evidence fed back to the agent per search hit (LKA_AGENT_EVIDENCE_MAX_CHARS) and in total
# (LKA_AGENT_EVIDENCE_MAX_ITEMS): every character is billed as input on each round.
AGENT_EVIDENCE_MAX_CHARS = 1800


def agent_evidence_max_chars(config: dict[str, str]) -> int:
    return llm_budget.int_setting(config, "LKA_AGENT_EVIDENCE_MAX_CHARS", AGENT_EVIDENCE_MAX_CHARS, 200, 20_000)


def agent_evidence_max_items(config: dict[str, str]) -> int:
    return llm_budget.int_setting(config, "LKA_AGENT_EVIDENCE_MAX_ITEMS", agentic_retrieval.EVIDENCE_MAX_ITEMS, 1, 60)


def agent_retriever(db_path: Path, config: dict[str, str]) -> agentic_retrieval.Retrieve:
    max_chars = agent_evidence_max_chars(config)

    def retrieve(query_text: str, limit: int) -> list[dict[str, Any]]:
        rows, _ = search(db_path, query_text, limit, config)
        rows = rows[:limit]
        for row in rows:
            # The whole section or table, as a local answer would cite it, so the model sees
            # complete steps and rows rather than a fragment.
            row["evidence"] = block_citation(full_section_text(db_path, row), query_text, max_chars)
        return rows

    return retrieve


def agent_completion(config: dict[str, str]) -> agentic_retrieval.Complete:
    timeout = int(config.get("LKA_API_TIMEOUT_SECONDS", str(DEFAULT_API_TIMEOUT_SECONDS)) or DEFAULT_API_TIMEOUT_SECONDS)
    budget = llm_budget.task_settings(config, "agent")

    def complete(messages: list[dict[str, str]]) -> dict[str, Any]:
        return chat_completion_detailed(
            config.get("LKA_API_PROVIDER", "openai-compatible"),
            config.get("LKA_API_BASE_URL", "").rstrip("/"),
            config.get("LKA_API_KEY", ""),
            budget["model"],
            messages,
            temperature=0.0,
            timeout=timeout,
            json_output=True,
            max_tokens=budget["max_tokens"],
            thinking=budget["thinking"],
            task="agent",
        )

    return complete


def agent_trace(outcome: agentic_retrieval.AgentOutcome, config: dict[str, str]) -> dict[str, Any]:
    metrics = llm_budget.usage_metrics(outcome.usage)
    return {
        "version": agentic_retrieval.AGENT_VERSION,
        "decision": outcome.decision,
        "reason": outcome.reason,
        "rounds": outcome.rounds,
        "follow_up_queries": outcome.queries,
        "evidence_count": len(outcome.evidence),
        "usage": outcome.usage,
        "calls": outcome.calls,
        "cache_hit_rate": metrics["cache_hit_rate"],
        "output_input_ratio": metrics["output_input_ratio"],
        "cost_usd": outcome.cost_usd,
        "model": llm_budget.model_for_task(config, "agent"),
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
        "api_model": llm_budget.model_for_task(config, "agent"),
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


_QUERY_FLIGHTS: dict[tuple[Any, ...], list[Any]] = {}
_QUERY_FLIGHTS_LOCK = threading.Lock()


@contextmanager
def single_flight(key: tuple[Any, ...]):
    """Requests for the same answer run one at a time: a repeated click or a second tab waits for
    the first request and then reads its answer from the answer cache instead of paying again."""
    with _QUERY_FLIGHTS_LOCK:
        entry = _QUERY_FLIGHTS.setdefault(key, [threading.Lock(), 0])
        entry[1] += 1
    try:
        with entry[0]:
            yield
    finally:
        with _QUERY_FLIGHTS_LOCK:
            entry[1] -= 1
            if not entry[1]:
                _QUERY_FLIGHTS.pop(key, None)


def query(db_path: Path, question: str, limit: int = 5, use_web: bool = False, allow_api: bool = True) -> dict[str, Any]:
    with single_flight((str(Path(db_path).resolve()), question.strip(), limit, use_web, allow_api)):
        return query_unshared(db_path, question, limit, use_web, allow_api)


def query_unshared(db_path: Path, question: str, limit: int = 5, use_web: bool = False, allow_api: bool = True) -> dict[str, Any]:
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
    # Only a request for a station's item list («SFT 有哪些测试项»): «X测试项目前是什么状态» asks
    # about one item's status, not about a station called X.
    if station_test_item_question(question) and enumeration_or_list_question(question) and requested_station_name(question) and verified_suggestions:
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
                question,
                agent_retriever(db_path, config),
                agent_completion(config),
                agent_max_rounds(config),
                agent_evidence_max_items(config),
            )
        except (agentic_retrieval.AgentError, RuntimeError, OSError, ValueError, KeyError, http.client.HTTPException) as exc:
            # Unusable reply, timeout or network failure (a connection dropped mid-reply raises
            # IncompleteRead, an HTTPException): answer locally instead.
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

    covering = covering_row_index(rows, question, db_path)
    if covering is not None and covering > 0 and rows[covering].get("doc_id") != rows[0].get("doc_id"):
        # The answer starts from the document that covers the question, not from another document
        # ranked above it (a fixture list above the limits the question asks for). Within one
        # document the ranking stays: its top chunk often answers in words the gate does not count.
        rows = [rows[covering], *rows[:covering], *rows[covering + 1 :]]
    if covering is None:
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
    rows = rows_about_named_subject(rows, question)

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
