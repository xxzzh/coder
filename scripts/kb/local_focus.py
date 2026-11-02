"""Local answers without a model: the part of the retrieved evidence a question is about.

A retrieved chunk is often a whole short document (a policy with four sections, a handbook page)
that, shown whole, buries the asked detail. When the chunk holds several sections side by side
(一、年假 / 二、费用报销 / 三、加班), each line is compared with the question by meaning (the local
embedding model), read together with the title of its section, and the answer shows the section of
the best line. A section's own sub-sections are never split off: a chunk under one heading (a
procedure with its steps, an API section with its tables) is shown as before.

The same comparison lets a question in the user's own words find its section when the wording check
finds too few of its words (报销多久能到账 against 审批完成后 3 个工作日内付款). Such an answer is
labelled as the closest content found, and only given when one section clearly stands out: when
several sections are about equally close, the question is most likely not answered by any.
"""

from __future__ import annotations

from collections import OrderedDict
import re
import threading
from dataclasses import dataclass, field
from typing import Any

import model_capabilities

from kb.citations import PRIMARY_CITATION_MAX_CHARS, block_citation, evidence_lines
from kb.relevance import content_coverage
from kb.terms import enumeration_or_list_question, process_question
from kb.text import to_simplified


# Lines compared per question, and the characters of each read by the embedding model (the start
# of a long line says what it is about): on a CPU the model reads about 400 characters a second,
# so a question costs at most about 5 seconds the first time its evidence is read.
FOCUS_LINES_MAX = 32
FOCUS_LINE_CHARS = 64

# Line vectors kept between questions: the same documents are usually asked about again.
VECTOR_CACHE_SIZE = 4096

# Title lines in text without Markdown headings (PDF, TXT), from the outermost: 第一章, then 一、 /
# 第三条 / 【注意】, then 1. / 2.1 (only before a line that is not numbered the same way: otherwise
# they are the steps of a list). Markdown headings (levels 1-6) are outside all of them.
CHAPTER_TITLE_PATTERN = re.compile(r"^第[一二三四五六七八九十百\d]+[章部篇]")
SECTION_TITLE_PATTERN = re.compile(r"^(?:第?[一二三四五六七八九十百]+[、.．]\s*\S|第[一二三四五六七八九十百\d]+[节条]|【[^】]+】)")
NUMBERED_TITLE_PATTERN = re.compile(r"^\d{1,2}(?:\.\d{1,2})*(?:[.、]\s*|\s+)(?=\D)")
SECTION_TITLE_MAX_CHARS = 30

# A question in other words than the documents' gets the closest section when its best line is at
# least this close in meaning (cosine of the local embedding model) ...
RELATED_MIN_SCORE = 0.65

# ... and at least this much closer than the best line of any other section. Measured on made-up
# company documents and the developer's technical documents: the section that answers stands out by
# 0.07-0.20; for unanswerable questions several sections score within 0.06 of each other.
RELATED_MIN_MARGIN = 0.06

# Retrieved chunks read when looking for the closest section.
RELATED_ROWS = 3

# ... and only when one of them is at least this close to the question as a whole (the vector
# search score): otherwise nothing retrieved is about the question, and reading its lines would
# only cost time.
RELATED_MIN_CHUNK_SCORE = 0.5

# Sections whose share of the question's words is within this much of the best one's are told
# apart by meaning.
COVERAGE_TIE = 0.1

_VECTOR_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_VECTOR_CACHE_LOCK = threading.Lock()


@dataclass
class Section:
    row: int
    title: str
    title_level: int = 0
    lines: list[tuple[str, int]] = field(default_factory=list)  # (text, Markdown heading level or 0)
    scores: list[float] = field(default_factory=list)

    @property
    def best(self) -> float:
        return max(self.scores, default=0.0)

    @property
    def body(self) -> list[str]:
        return [line for line, _ in self.lines]

    def text(self) -> str:
        """The section as evidence text, Markdown heading markers restored."""
        parts = [("#" * self.title_level + " " if self.title_level else "") + self.title] if self.title else []
        parts += [("#" * level + " " if level else "") + line for line, level in self.lines]
        return "\n".join(parts)


def title_rank(line: str, level: int, following: str) -> int:
    """How far out a title line is (1-6 Markdown, 7-9 numbered titles), 0 for any other line."""
    if level:
        return level
    text = line.strip()
    if len(text) > SECTION_TITLE_MAX_CHARS or " | " in text or re.search(r"[。；;，,！？!?]$", text):
        return 0
    if CHAPTER_TITLE_PATTERN.match(text):
        return 7
    if SECTION_TITLE_PATTERN.match(text):
        return 8
    if NUMBERED_TITLE_PATTERN.match(text) and following and not NUMBERED_TITLE_PATTERN.match(following):
        return 9
    return 0


def split_sections(text: str, row: int, opening: bool = False) -> list[Section]:
    """The evidence's sections side by side: when the outermost kind of title in it occurs at
    least twice, each of those titles starts a section. A document's own title on the first line of
    its opening chunk (opening) is set apart as its name. Otherwise the evidence is one section:
    sub-sections under one heading stay together."""
    lines = evidence_lines(text)
    ranks = [title_rank(line, level, lines[index + 1][0] if index + 1 < len(lines) else "") for index, (line, level) in enumerate(lines)]
    titles = [rank for rank in ranks if rank]
    if opening and ranks and ranks[0] and titles.count(ranks[0]) == 1 and ranks[0] == min(titles):
        titles = titles[1:]  # the document's name
    outermost = min(titles, default=0)
    split_rank = outermost if outermost and titles.count(outermost) >= 2 else 0
    sections: list[Section] = []
    current = Section(row, "")
    for (line, level), rank in zip(lines, ranks):
        if split_rank and rank and rank <= split_rank:
            if current.lines or current.title:
                sections.append(current)
            current = Section(row, line, level)
        else:
            current.lines.append((line, level))
    if current.lines or current.title:
        sections.append(current)
    return sections


def line_vectors(texts: list[str], config: dict[str, str]) -> list[list[float]] | None:
    """Embedding vectors of texts, cached; None without a usable embedding model."""
    with _VECTOR_CACHE_LOCK:
        missing = list(dict.fromkeys(text for text in texts if text not in _VECTOR_CACHE))
    if missing:
        try:
            result = model_capabilities.embed_texts(missing, config)
        except Exception:  # noqa: BLE001  (no model: the answer keeps the whole evidence)
            return None
        vectors = result.get("vectors") or []
        if len(vectors) != len(missing):
            return None
        with _VECTOR_CACHE_LOCK:
            for text, vector in zip(missing, vectors):
                _VECTOR_CACHE[text] = vector
                _VECTOR_CACHE.move_to_end(text)
            while len(_VECTOR_CACHE) > VECTOR_CACHE_SIZE:
                _VECTOR_CACHE.popitem(last=False)
    with _VECTOR_CACHE_LOCK:
        found = [_VECTOR_CACHE.get(text) for text in texts]
    return None if any(vector is None for vector in found) else found  # type: ignore[return-value]


def score_sections(question: str, sections: list[Section], config: dict[str, str]) -> bool:
    """Set each line's closeness to the question; False without an embedding model. A line is read
    with its section title (保修：为 3 年…); a title without lines is read alone."""
    units: list[tuple[Section, str]] = []
    for section in sections:
        if not section.lines:
            units.append((section, section.title))
        units += [(section, f"{section.title}：{line}" if section.title else line) for line in section.body]
    units = units[:FOCUS_LINES_MAX]
    if not units:
        return False
    vectors = line_vectors([question, *(unit[:FOCUS_LINE_CHARS] for _, unit in units)], config)
    if vectors is None:
        return False
    query = vectors[0]
    scores: dict[int, list[float]] = {}
    for (section, _), vector in zip(units, vectors[1:]):
        scores.setdefault(id(section), []).append(sum(a * b for a, b in zip(query, vector)))
    for section in sections:
        section.scores = scores.get(id(section), [])
    return True


def names_only(section: Section) -> bool:
    """A title without lines, or a line without a title that only names the document (会议室预订说明
    above its sections)."""
    if not section.lines:
        return True
    if section.title or len(section.lines) != 1:
        return False
    line = section.lines[0][0]
    return len(line) <= SECTION_TITLE_MAX_CHARS and " | " not in line and not re.search(r"[。；;，,！？!?：:]", line)


def answerable_sections(sections: list[Section]) -> list[Section]:
    """Sections that can answer: a document's name above its sections only says what the
    document is, unless its evidence has nothing else."""
    rows_with_content = {section.row for section in sections if not names_only(section)}
    return [section for section in sections if not names_only(section) or section.row not in rows_with_content]


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", to_simplified(text)).lower()


def names_another_topic(chosen: Section, sections: list[Section], files: list[str], units: list[tuple[str, str, float]]) -> bool:
    """Whether the question names a topic that another document has a section for, while the
    chosen section never mentions it: «加班要谁审批» is not answered by a travel policy's «出差审批»
    when the handbook has a section «加班»."""
    text = compact_text(chosen.title + "\n" + "\n".join(chosen.body))
    titles = [compact_text(section.title) for section in sections if section.title and files[section.row] != files[chosen.row]]
    for kind, term, _ in units:
        if kind in {"cjk", "ascii"} and term not in text and any(term in title for title in titles):
            return True
    return False


def section_key(section: Section, files: list[str]) -> tuple[str, str]:
    """A section retrieved twice (two overlapping chunks of one file) is one section."""
    return files[section.row], section.title or (section.body[0] if section.body else "")


@dataclass
class Focus:
    row: int  # the retrieved row the shown section comes from
    section: Section | None  # None: the row is one section, shown as before
    related: bool  # the wording check failed: the closest content, for the user to check

    def citation(self, evidence: str, question: str) -> str:
        """The cited evidence: the chosen section, or the row's whole evidence within the
        primary budget."""
        return block_citation(self.section.text() if self.section else evidence, question, PRIMARY_CITATION_MAX_CHARS)


def opening(row: dict[str, Any]) -> bool:
    return int(row.get("chunk_index") or 0) <= 1


def focus_answer(
    question: str,
    rows: list[dict[str, Any]],
    first_text: str,
    config: dict[str, str],
    gate_passed: bool,
    units: list[tuple[str, str, float]],
) -> Focus | None:
    """What a local answer shows. When the wording check passed (on rows[0], whose evidence is
    first_text): the section of rows[0] the question is about, or None when rows[0] is one section
    (nothing to choose) or no embedding model can compare lines. When it failed: the closest section
    of the top rows if one clearly stands out, else None (the question is refused)."""
    if gate_passed:
        if enumeration_or_list_question(question) or process_question(question):
            return None  # a list or a procedure may span the sections: shown whole
        sections = answerable_sections(split_sections(first_text, 0, opening(rows[0])))
        if len(sections) < 2:
            return None
        # The section holding most of the question's words; among sections about equally close
        # in wording, the one closest in meaning.
        coverage = {id(section): content_coverage(units, "\n".join([section.title, *section.body])) for section in sections}
        best_coverage = max(coverage.values())
        tied = [section for section in sections if coverage[id(section)] >= best_coverage - COVERAGE_TIE]
        if len(tied) > 1:
            if not score_sections(question, tied, config):
                return None
            return Focus(0, max(tied, key=lambda section: section.best), related=False)
        return Focus(0, tied[0], related=False)
    top = rows[:RELATED_ROWS]
    if max(float(row.get("vector_score") or row.get("semantic_score") or 0.0) for row in top) < RELATED_MIN_CHUNK_SCORE:
        return None
    texts = [first_text, *(str(row.get("text", "")) for row in top[1:])]
    sections = answerable_sections(
        [section for index, text in enumerate(texts) for section in split_sections(text, index, opening(top[index]))]
    )
    if not sections or not score_sections(question, sections, config):
        return None
    sections = [section for section in sections if section.scores]
    if not sections:
        return None
    files = [str(row.get("file_name", "")) for row in top]
    chosen = max(sections, key=lambda section: section.best)
    others = [section.best for section in sections if section_key(section, files) != section_key(chosen, files)]
    if (
        chosen.best < RELATED_MIN_SCORE
        or chosen.best - max(others, default=0.0) < RELATED_MIN_MARGIN
        or names_another_topic(chosen, sections, files, units)
    ):
        return None
    split = sum(section.row == chosen.row for section in sections) > 1
    return Focus(chosen.row, chosen if split else None, related=True)


# Evidence from another document is shown next to the answer only when it holds at least this
# share of the question's words: a price list or a meeting-room rule under a question about
# a forklift is noise. Measured on the developer's documents: such pieces below a quarter
# almost never add an asked detail.
SUPPLEMENT_MIN_COVERAGE = 0.25


def supplements(rows: list[dict[str, Any]], units: list[tuple[str, str, float]]) -> list[dict[str, Any]]:
    """rows with the other documents' pieces that hardly mention the question left out; the first
    row and the pieces of its document stay."""
    if not rows:
        return rows
    first = rows[0].get("file_name")
    return [rows[0], *(row for row in rows[1:] if row.get("file_name") == first or content_coverage(units, str(row.get("text", ""))) >= SUPPLEMENT_MIN_COVERAGE)]
