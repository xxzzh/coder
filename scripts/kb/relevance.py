"""Relevance gate: does the retrieved evidence cover the question, and does the question name anything the index never mentions."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from kb.terms import CONCEPT_SYNONYMS, ENGLISH_QUERY_STOP_WORDS, QUESTION_FUNCTION_PHRASES, abbreviation_query_term, iterative_retrieval_question, normalized_retrieval_question


LOCAL_SCORE_THRESHOLD = 0.16

LOCAL_CONTENT_COVERAGE_THRESHOLD = 0.5

# A name found in at least this share of the chunks (the framework's own name, a station that
# every test sheet lists) does not pin a question to particular evidence: STPM matches a
# quarter of the index. When every name in a question is that broad, the evidence must also
# cover the question's Chinese part (what is asked about the name), at least half of it.
BROAD_NAME_SHARE = 0.10

# A two-character term found in at least half of the chunks (测试 in an index of test documents)
# says nothing about what is asked; checks of the asked part ignore it.
UBIQUITOUS_TERM_SHARE = 0.5

# 「X支持Y吗」 asks about Y: the evidence must mention Y, not just X.
SUPPORT_QUESTION_PATTERN = re.compile(r"支持(.+?)(?:吗|么|？|\?|$)")

# A name followed by a version: MES 3.0, USB 4.0, WIFI 6E, .NET 6, MES3.0. A version without a
# dot (6, 6E) must be separated by a space, so model codes such as AB123 are not split.
VERSIONED_NAME_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_]*?)(?:\s*(\d+\.\d+(?:\.\d+)*[A-Za-z]?)|\s+(\d+[A-Za-z]?))(?![A-Za-z0-9]|\.\d)"
)

KNOWLEDGE_TEXT_CACHE: dict[str, tuple[str, str]] = {}
CHUNK_TEXTS_CACHE: dict[str, tuple[str, list[str]]] = {}
NAME_SHARE_CACHE: dict[tuple[str, str, str], float] = {}
TERM_SHARE_CACHE: dict[tuple[str, str, str], float] = {}


def query_content_units(question: str) -> list[tuple[str, str, float]]:
    """Weighted subject units of a question as (kind, term, weight).

    Each ASCII term (identifier, abbreviation, English word) weighs 1. Each Chinese
    content span left after removing question-function words also weighs 1 in total,
    spread over its distinct bigrams, so 「添加一个新测试项」 still mostly matches the
    source wording 「添加新测试项」 instead of requiring the whole span verbatim.
    """
    normalized = normalized_retrieval_question(question)
    abbreviation = abbreviation_query_term(normalized)
    if abbreviation:
        return [("ascii", abbreviation.lower(), 1.0)]
    units: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for word in re.findall(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]*", normalized):
        lowered = word.lower()
        if len(lowered) < 2 or lowered in ENGLISH_QUERY_STOP_WORDS or lowered in seen:
            continue
        seen.add(lowered)
        units.append(("ascii", lowered, 1.0))
    chinese_text = normalized
    for phrase in QUESTION_FUNCTION_PHRASES:
        chinese_text = chinese_text.replace(phrase, " ")
    # A phrase with known synonyms is one unit, covered by any of its variants (目录 by 路径).
    for phrase in sorted(CONCEPT_SYNONYMS, key=len, reverse=True):
        if phrase in chinese_text:
            chinese_text = chinese_text.replace(phrase, " ")
            units.append(("concept", "|".join(CONCEPT_SYNONYMS[phrase]), 1.0))
    for span in re.findall(r"[\u4e00-\u9fff]{2,}", chinese_text):
        bigrams = [gram for gram in dict.fromkeys(span[index : index + 2] for index in range(len(span) - 1)) if gram not in seen]
        if not bigrams:
            continue
        seen.update(bigrams)
        units.extend(("cjk", gram, 1.0 / len(bigrams)) for gram in bigrams)
    return units


def content_coverage(units: list[tuple[str, str, float]], text: str) -> float:
    """Share of the question's subject weight found in text (0..1)."""
    total = sum(weight for _, _, weight in units)
    if not total:
        return 0.0
    lowered = text.lower()
    compact = re.sub(r"\s+", "", text)
    matched = 0.0
    for kind, term, weight in units:
        if kind == "ascii":
            if ascii_term_pattern(term).search(lowered):
                matched += weight
        elif kind == "concept":
            if any(concept_variant_found(variant, lowered, compact) for variant in term.split("|")):
                matched += weight
        elif term in compact:
            matched += weight
    return matched / total


def concept_variant_found(variant: str, lowered: str, compact: str) -> bool:
    if re.search(r"[\u4e00-\u9fff]", variant):
        return variant in compact
    if len(variant) >= 4:  # a long identifier may sit inside another: SampleColdBootFlag
        return variant.lower() in lowered or variant.lower().replace(" ", "") in compact.lower()
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(variant.lower())}(?![a-z0-9])", lowered))  # short: a whole word only


def ascii_term_pattern(term: str) -> re.Pattern[str]:
    # Short terms (PR, GW, AC) need a full word boundary; longer ones may prefix
    # a compound such as Camera -> CameraTest.
    return re.compile(rf"(?<![a-z0-9]){re.escape(term)}" + (r"(?![a-z0-9])" if len(term) < 3 else ""))


def knowledge_text(db_path: Path) -> str | None:
    """All chunk text of the index, lower-cased; reloaded whenever the index is rebuilt."""
    key = str(Path(db_path).resolve())
    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("SELECT value FROM metadata WHERE key = 'indexed_at'").fetchone()
            stamp = str(row[0]) if row else ""
            cached = KNOWLEDGE_TEXT_CACHE.get(key)
            if cached and cached[0] == stamp:
                return cached[1]
            text = "\n".join(str(value) for (value,) in conn.execute("SELECT text FROM chunks")).lower()
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return None
    KNOWLEDGE_TEXT_CACHE[key] = (stamp, text)
    return text


def index_stamp(db_path: Path) -> tuple[str, str]:
    key = str(Path(db_path).resolve())
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT value FROM metadata WHERE key = 'indexed_at'").fetchone()
    finally:
        conn.close()
    return key, str(row[0]) if row else ""


def chunk_texts(db_path: Path) -> list[str]:
    """Every chunk's text, lower-cased; reloaded whenever the index is rebuilt."""
    key, stamp = index_stamp(db_path)
    cached = CHUNK_TEXTS_CACHE.get(key)
    if cached and cached[0] == stamp:
        return cached[1]
    conn = sqlite3.connect(db_path)
    try:
        texts = [str(text).lower() for (text,) in conn.execute("SELECT text FROM chunks")]
    finally:
        conn.close()
    CHUNK_TEXTS_CACHE[key] = (stamp, texts)
    return texts


def name_share(term: str, db_path: Path) -> float:
    """Share of the index's chunks that mention a Latin-script name."""
    try:
        key, stamp = index_stamp(db_path)
        cache_key = (key, stamp, term)
        if cache_key not in NAME_SHARE_CACHE:
            texts = chunk_texts(db_path)
            pattern = ascii_term_pattern(term)
            NAME_SHARE_CACHE[cache_key] = sum(bool(pattern.search(text)) for text in texts) / len(texts) if texts else 0.0
        return NAME_SHARE_CACHE[cache_key]
    except sqlite3.DatabaseError:
        return 0.0


def term_share(term: str, db_path: Path) -> float:
    """Share of the index's chunks containing a Chinese term."""
    try:
        key, stamp = index_stamp(db_path)
        cache_key = (key, stamp, term)
        if cache_key not in TERM_SHARE_CACHE:
            texts = chunk_texts(db_path)
            TERM_SHARE_CACHE[cache_key] = sum(term in "".join(text.split()) for text in texts) / len(texts) if texts else 0.0
        return TERM_SHARE_CACHE[cache_key]
    except sqlite3.DatabaseError:
        return 0.0


def asked_part(units: list[tuple[str, str, float]], db_path: Path) -> list[tuple[str, str, float]]:
    """The units that say what is asked: not names, and not terms nearly every chunk contains."""
    asked = [unit for unit in units if unit[0] != "ascii"]
    informative = [unit for unit in asked if not (unit[0] == "cjk" and term_share(unit[1], db_path) >= UBIQUITOUS_TERM_SHARE)]
    return informative or asked


def required_parts(question: str, units: list[tuple[str, str, float]], db_path: Path | None) -> list[list[tuple[str, str, float]]]:
    """Parts of the question each piece of evidence must cover at least half of, besides the whole.

    - When every name in the question is broad (STPM, SFT), the asked part: STPM plus the
      common word 测试 is no evidence about STPM's 单元测试覆盖率.
    - In 「X支持Y吗」, Y: a chunk about X that never mentions Y does not answer whether X supports it.
    """
    if db_path is None:
        return []
    parts = []
    if broad_names_only(units, db_path):
        asked = asked_part(units, db_path)
        if asked:
            parts.append(asked)
    support = SUPPORT_QUESTION_PATTERN.search(normalized_retrieval_question(question))
    if support:
        supported = asked_part(query_content_units(support.group(1)), db_path)
        supported += [unit for unit in query_content_units(support.group(1)) if unit[0] == "ascii"]
        if supported:
            parts.append(supported)
    return parts


def broad_names_only(units: list[tuple[str, str, float]], db_path: Path | None) -> bool:
    """The question names something, but only names too common to point at particular evidence."""
    names = [term for kind, term, _ in units if kind == "ascii"]
    return bool(db_path is not None and names and all(name_share(name, db_path) >= BROAD_NAME_SHARE for name in names))


def unknown_names(question: str, db_path: Path) -> list[str]:
    """Names in a Chinese question that occur nowhere in the knowledge base.

    In a Chinese question, Latin-script terms are names and identifiers (Jenkins, FATAL,
    LogMsgType), and a word followed by a version (MES 3.0, Windows 7) names one release.
    If the index never mentions such a name it holds no evidence about it, however well
    the rest of the question matches: MES 2.0 chunks do not answer a question about MES 3.0.
    Chinese words are not checked this way: users' wording (记事本, 连不上) is often absent
    from the documents even when the documents answer the question.
    """
    if not re.search(r"[\u4e00-\u9fff]", question):
        return []
    corpus = knowledge_text(db_path)
    if corpus is None:
        return []
    unknown: list[str] = []
    normalized = normalized_retrieval_question(question)
    for match in VERSIONED_NAME_PATTERN.finditer(normalized):
        word, version = match.group(1), match.group(2) or match.group(3)
        dotted = "." if normalized[max(match.start(1) - 1, 0) : match.start(1)] == "." else ""  # .NET 6
        pattern = rf"(?<![a-z0-9]){re.escape(word.lower())}\s*{re.escape(version.lower())}(?![a-z0-9]|\.\d)"
        if not re.search(pattern, corpus):
            unknown.append(f"{dotted}{word} {version}")
    versioned = [name.lower().replace(" ", "") for name in unknown]
    for kind, term, _ in query_content_units(question):
        if kind != "ascii" or ascii_term_pattern(term).search(corpus) or any(term in name for name in versioned):
            continue
        # Report the name as the user wrote it (FATAL, not fatal).
        written = re.search(re.escape(term), question, re.IGNORECASE)
        unknown.append(written.group(0) if written else term)
    return unknown


def row_covers_question(
    row: dict[str, Any], units: list[tuple[str, str, float]], required: list[list[tuple[str, str, float]]] | None = None
) -> bool:
    if row.get("rerank_score", 0.0) < LOCAL_SCORE_THRESHOLD * 0.8:
        return False
    has_lexical_score = float(row.get("keyword_score", 0.0) or 0.0) > 0 or float(row.get("fts_score", 0.0) or 0.0) > 0
    text = str(row.get("text", ""))
    # Strictly more than half: a chunk that only shares the subject (a model code) but not the
    # asked attribute (售价) must not count as evidence.
    if not has_lexical_score or content_coverage(units, text) <= LOCAL_CONTENT_COVERAGE_THRESHOLD:
        return False
    return all(content_coverage(part, text) >= LOCAL_CONTENT_COVERAGE_THRESHOLD for part in required or [])


def gate_candidate_count(rows: list[dict[str, Any]], question: str) -> int:
    return min(len(rows), 8 if iterative_retrieval_question(question) else 3)


def local_result_is_relevant(rows: list[dict[str, Any]], question: str, db_path: Path | None = None) -> bool:
    if not rows or rows[0].get("rerank_score", 0.0) < LOCAL_SCORE_THRESHOLD:
        return False
    units = query_content_units(question)
    if not units:
        return False
    if db_path is not None and unknown_names(question, db_path):
        return False
    required = required_parts(question, units, db_path)
    return any(row_covers_question(row, units, required) for row in rows[: gate_candidate_count(rows, question)])

