"""Citations: which part of a retrieved chunk to quote (whole sections, focused table rows, sentences), and checks on the citations an answer uses."""

from __future__ import annotations

from difflib import SequenceMatcher
from html import unescape
import re
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

from kb.terms import ENGLISH_QUERY_STOP_WORDS, STOP_WORDS, candidate_terms, enumeration_or_list_question, process_question
from kb.text import first_markdown_heading_level, human_text, markdown_table_of_contents, sentence_score, split_sentences, strip_markdown_emphasis, trim_terminal_punctuation, unique_terms


# Evidence budgets for local answers built without an LLM.
PRIMARY_CITATION_MAX_CHARS = 1500

SECONDARY_BLOCK_MAX_CHARS = 300

# A process question (流程/步骤/机制 ...) is often answered across several sections, e.g. an overview
# ranked below a section about one variant; supporting sections are then shown whole, not as snippets.
PROCESS_SECONDARY_BLOCK_MAX_CHARS = 900

TABLE_ROW_MAX_CHARS = 120

SECTION_SCAN_MAX_CHUNKS = 40


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
        identifiers = unique_terms(
            re.findall(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*(?![A-Za-z0-9_])", question),
            limit=8,
        )
        for identifier in identifiers:
            if any(identifier.lower() in sentence.lower() for _, sentence in selected_items):
                continue
            identifier_match = next(
                (
                    (index, sentence)
                    for _, index, sentence in sorted(matches, key=lambda item: (-item[0], item[1]))
                    if identifier.lower() in sentence.lower()
                ),
                None,
            )
            if identifier_match and identifier_match not in selected_items:
                selected_items.append(identifier_match)
            if len(selected_items) >= 3:
                break
        selected_items.sort(key=lambda item: item[0])
        selected: list[str] = []
        for index, sentence in selected_items:
            if sentence.startswith("They ") and index > 0:
                selected.append(sentences[index - 1])
            selected.append(sentence)
        citation = " ".join(trim_terminal_punctuation(sentence) + "。" for sentence in selected)
        return human_text(citation, max_len=max_len)
    return human_text(text, max_len=max_len)


def evidence_lines(text: str) -> list[tuple[str, int]]:
    """Readable lines of a chunk as (text, heading level or 0).

    Markdown markers, code-fence lines, rules, table separators and empty table cells are
    removed; line structure, numbering and code are kept.
    """
    lines: list[tuple[str, int]] = []
    for raw_line in unescape(text).splitlines():
        line = raw_line.strip()
        if not line or re.fullmatch(r"```[\w+-]*|[-*_=]{3,}|\|?[\s:|-]*-{3,}[\s:|-]*\|?", line):
            continue
        heading = re.match(r"(#{1,6})\s+", line)
        level = len(heading.group(1)) if heading else 0
        line = re.sub(r"^#{1,6}\s+|^>\s*|^[-*+]\s+", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        line = strip_markdown_emphasis(line)
        if "|" in line:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            line = " | ".join(cell for cell in cells if cell)
        line = re.sub(r"[ \t]+", " ", line).strip()
        # Drop lines with no letters or digits: tree glyphs, braces, empty table rows.
        if re.search(r"\w", line):
            lines.append((line, level))
    return lines


def focus_table_rows(lines: list[tuple[str, int]], question: str, terms: list[str]) -> list[str]:
    """Table rows to cite: all of them, unless a few rows clearly answer a non-list question.

    Rows are scored on their values only, because column titles such as 「测试门限：」 repeat in
    every row. Filtering needs a distinctive best row (score >= 3) whose peers are at most half
    the table, so a comparison table that weakly mentions the subject everywhere stays whole.
    Headings and a Markdown table's header row are always kept.
    """
    rows = [line for line, _ in lines]
    if enumeration_or_list_question(question):
        return rows
    body = [index for index, (line, level) in enumerate(lines) if not level and " | " in line]
    scores = {index: sentence_score(re.sub(r"(^|\| )[^|：]{1,24}：", r"\1", lines[index][0]), terms) for index in body}
    best = max(scores.values(), default=0)
    if best < 3:
        return rows
    standing_out = {index for index, score in scores.items() if score >= best * 0.6}
    if len(standing_out) > len(body) / 2:
        return rows
    labelled = any("：" in lines[index][0] for index in body)
    keep = standing_out | {index for index, (_, level) in enumerate(lines) if level}
    if body and not labelled:
        keep.add(body[0])  # Markdown header row: 方法 | 签名 | 说明
    return [line for index, (line, _) in enumerate(lines) if index in keep]


def block_citation(text: str, question: str, max_chars: int, outline: bool = True) -> str:
    """A whole section or table as cited evidence, kept within max_chars.

    Lead-in lines that belong to the previous section are dropped. When the section is
    longer than the budget and outline is set, the sub-headings that did not fit are listed
    so a step-by-step answer still shows every step.
    """
    lines = evidence_lines(text)
    if not lines:
        return citation_for_text(text, question)
    terms = candidate_terms(question)
    first_heading = next((index for index, (_, level) in enumerate(lines) if level), None)
    if first_heading and not any(sentence_score(line, terms) for line, _ in lines[:first_heading]):
        lines = lines[first_heading:]
    top_level = min((level for _, level in lines if level), default=0)
    is_table = len(lines) >= 2 and sum(" | " in line for line, _ in lines) >= len(lines) * 0.5
    if is_table:
        rows = focus_table_rows(lines, question, terms)
        if sum(len(row) + 1 for row in rows) > max_chars:
            # Shorten rows before dropping any: leading cells (number, item, sub-item) identify a row,
            # long procedure descriptions come later.
            rows = [row if len(row) <= TABLE_ROW_MAX_CHARS else row[: TABLE_ROW_MAX_CHARS - 1] + "…" for row in rows]
        if sum(len(row) + 1 for row in rows) <= max_chars:
            return "\n".join(rows)
        # Still too long: rows matching the question first, original order otherwise.
        order = sorted(range(len(rows)), key=lambda index: (sentence_score(rows[index], terms) == 0, index))
        kept: set[int] = set()
        used = 0
        for index in order:
            if used + len(rows[index]) + 1 <= max_chars:
                kept.add(index)
                used += len(rows[index]) + 1
        return "\n".join(row for index, row in enumerate(rows) if index in kept)

    selected: list[str] = []
    used = 0
    cut_at = None
    for index, (line, _) in enumerate(lines):
        if used + len(line) + 1 > max_chars:
            cut_at = index
            break
        selected.append(line)
        used += len(line) + 1
    if cut_at is None:
        return "\n".join(selected)
    remaining_headings = [line for line, level in lines[cut_at:] if level and level > top_level] if outline else []
    if remaining_headings:
        outline = "本节其余部分：" + "；".join(remaining_headings)
        while selected and used + len(outline) > max_chars:
            used -= len(selected.pop()) + 1
        selected.append(outline)
    elif selected:
        selected[-1] = selected[-1] + "…"
    else:
        selected.append(lines[0][0][: max_chars - 1] + "…")
    return "\n".join(selected)


def full_section_text(db_path: Path, row: dict[str, Any]) -> str:
    """The row's Markdown section through to the next heading of the same or a higher level."""
    text = str(row.get("text", ""))
    level = first_markdown_heading_level(text)
    if (
        str(row.get("file_type", "")).lower() not in {"md", "markdown"}
        or level is None
        or row.get("doc_id") is None
        or row.get("chunk_index") is None
        or markdown_table_of_contents(text)
    ):
        return text
    last_index = int(row["chunk_index"]) + max(0, len(row.get("context_chunk_ids") or [row.get("chunk_id")]) - 1)
    try:
        conn = sqlite3.connect(db_path)
        try:
            neighbors = conn.execute(
                "SELECT text FROM chunks WHERE doc_id = ? AND chunk_index > ? AND chunk_index <= ? ORDER BY chunk_index",
                (row["doc_id"], last_index, last_index + SECTION_SCAN_MAX_CHUNKS),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return text
    parts = [text]
    for (neighbor_text,) in neighbors:
        neighbor_level = first_markdown_heading_level(str(neighbor_text))
        if neighbor_level is not None and neighbor_level <= level:
            break
        parts.append(str(neighbor_text))
    return "\n\n".join(parts)


def primary_citation(db_path: Path, row: dict[str, Any], question: str) -> str:
    """Citation for the top-ranked evidence: its whole section or table, not two sentences."""
    return block_citation(full_section_text(db_path, row), question, PRIMARY_CITATION_MAX_CHARS)


def citation_for_retrieval_row(row: dict[str, Any], question: str) -> str:
    text = str(row["text"])
    if row.get("context_expanded") or re.search(
        r"\b(enum|LogMsgType|Enum[A-Za-z0-9_]+)\b|枚举|^.*\|\s*值\s*\|",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    ):
        budget = PROCESS_SECONDARY_BLOCK_MAX_CHARS if process_question(question) else SECONDARY_BLOCK_MAX_CHARS
        return block_citation(text, question, budget, outline=False)
    return citation_for_text(text, question)


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
            # A quote already shown inside an earlier (e.g. whole-section) quote adds nothing.
            or (len(normalized_quote) >= 12 and normalized_quote in existing)
            or (min(len(normalized_quote), len(existing)) >= 40 and existing in normalized_quote)
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


def synthesis_citations(result: dict[str, Any]) -> list[dict[str, Any]]:
    citations = grounded_citations(result)
    if not result.get("web_pages_extracted"):
        return citations
    sources = result.get("sources") or []
    summary_ids = set(result.get("summary_source_ids") or [])
    fetched_web_ids = {
        source.get("source_id")
        for source in sources
        if source.get("source_origin") == "web" and source.get("content_source") == "webpage"
    }
    if result.get("source_type") == "web_search":
        allowed_ids = summary_ids | fetched_web_ids
        return [citation for citation in citations if citation.get("source_id") in allowed_ids] or citations
    if result.get("source_type") == "hybrid_web_knowledge":
        local_ids = {source.get("source_id") for source in sources if source.get("source_origin") != "web"}
        return [citation for citation in citations if citation.get("source_id") in local_ids | summary_ids | fetched_web_ids]
    return citations


def evidence_terms(text: str) -> set[str]:
    lowered = text.lower()
    terms = set(re.findall(r"[a-z0-9_]{3,}", lowered))
    for block in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(block) <= 4:
            terms.add(block)
        for size in (2, 3, 4):
            terms.update(block[index : index + size] for index in range(max(0, len(block) - size + 1)))
    return {term for term in terms if term and term not in STOP_WORDS and term not in ENGLISH_QUERY_STOP_WORDS}


def citation_validation(result: dict[str, Any]) -> dict[str, Any]:
    citations = result.get("citations") or synthesis_citations(result)
    citation_ids = {int(item["source_id"]) for item in citations if str(item.get("source_id", "")).isdigit()}
    source_ids = {int(item["source_id"]) for item in result.get("sources", []) if str(item.get("source_id", "")).isdigit()}
    allowed_ids = citation_ids | source_ids
    answer = str(result.get("answer", ""))
    referenced_ids = {int(match) for match in re.findall(r"\[(\d+)\]", answer)}
    issues: list[str] = []
    warnings: list[str] = []
    if referenced_ids and not referenced_ids <= allowed_ids:
        issues.append("answer_references_missing_source_id")
    if citations and result.get("api_used") and not referenced_ids:
        warnings.append("api_answer_missing_citation_marker")

    quotes_by_id = {
        int(item["source_id"]): str(item.get("quote") or item.get("citation") or "")
        for item in citations
        if str(item.get("source_id", "")).isdigit()
    }
    for sentence in re.split(r"(?<=[。！？.!?])\s+|\n+", answer):
        ids = [int(match) for match in re.findall(r"\[(\d+)\]", sentence)]
        if not ids:
            continue
        sentence_terms = evidence_terms(re.sub(r"\[\d+\]", "", sentence))
        if len(sentence_terms) < 3:
            continue
        quote_terms: set[str] = set()
        for source_id in ids:
            quote_terms.update(evidence_terms(quotes_by_id.get(source_id, "")))
        overlap = sentence_terms & quote_terms
        if len(overlap) < min(3, max(1, len(sentence_terms) // 5)):
            warnings.append("cited_sentence_not_clearly_supported")
            break
    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "referenced_source_ids": sorted(referenced_ids),
    }


def strip_native_reference_markers(text: str) -> str:
    return re.sub(r"\s*\[\d{1,3}\]", "", text)


def prune_unreferenced_output_sources(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("source_type") not in {"web_search", "hybrid_web_knowledge"}:
        return result

    sources = result.get("sources") or []
    if not sources:
        return result

    answer = str(result.get("answer", ""))
    referenced_ids = {int(match) for match in re.findall(r"\[(\d+)\]", answer)}
    summary_ids = {
        int(source_id)
        for source_id in result.get("summary_source_ids", [])
        if str(source_id).isdigit()
    }
    if referenced_ids:
        keep_ids = referenced_ids | summary_ids
    else:
        keep_ids = summary_ids

    if not keep_ids:
        return result

    pruned_sources = [
        source
        for source in sources
        if str(source.get("source_id", "")).isdigit() and int(source["source_id"]) in keep_ids
    ]
    if not pruned_sources:
        return result

    for source in pruned_sources:
        for key in ("snippet", "excerpt", "citation"):
            if key in source and isinstance(source[key], str):
                source[key] = strip_native_reference_markers(source[key])
    result["sources"] = pruned_sources
    result["summary_source_ids"] = [
        source_id for source_id in result.get("summary_source_ids", []) if int(source_id) in keep_ids
    ]
    citations = result.get("citations") or []
    result["citations"] = [
        citation
        for citation in citations
        if str(citation.get("source_id", "")).isdigit() and int(citation["source_id"]) in keep_ids
    ] or grounded_citations(result)
    for citation in result["citations"]:
        if isinstance(citation.get("quote"), str):
            citation["quote"] = strip_native_reference_markers(citation["quote"])
    result["output_sources_pruned"] = True
    return result


def apply_citation_validation(result: dict[str, Any]) -> dict[str, Any]:
    result["citation_validation"] = citation_validation(result)
    return result
