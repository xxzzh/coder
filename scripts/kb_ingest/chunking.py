"""Text normalization, semantic chunking and the local hash embedding."""

from __future__ import annotations

import hashlib
import json
import math
import re


EMBEDDING_DIMS = 384

EMBEDDING_LSH_BUCKETS = 12

EMBEDDING_LSH_PAIR_BUCKETS = 4

CHUNK_MAX_CHARS = 1100

CHUNK_MIN_CHARS = 180

SEMANTIC_BREAK_SIMILARITY = 0.08


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int = CHUNK_MAX_CHARS, overlap: int = 0) -> list[str]:
    """Split by semantic boundaries instead of fixed windows.

    The splitter keeps headings with nearby content, starts new chunks when
    adjacent units drift semantically, and only falls back to character limits
    for unusually long units.
    """
    text = normalize_text(text)
    if not text:
        return []

    units = semantic_units(text)
    chunks: list[str] = []
    current_units: list[str] = []
    current_embedding: dict[int, float] = {}

    for unit in units:
        unit_chunks = _split_long_text(unit, size, overlap)
        if len(unit_chunks) > 1:
            if current_units:
                chunks.append("\n\n".join(current_units).strip())
                current_units = []
                current_embedding = {}
            chunks.extend(unit_chunks)
            continue

        projected = "\n\n".join([*current_units, unit]).strip()
        similarity = cosine_dicts(current_embedding, local_embedding_dict(unit)) if current_units else 1.0
        starts_new_section = bool(re.match(r"^\s{0,3}#{1,6}\s+", unit))
        should_break = (
            current_units
            and len(projected) > CHUNK_MIN_CHARS
            and (len(projected) > size or starts_new_section or similarity < SEMANTIC_BREAK_SIMILARITY)
        )
        if should_break:
            chunks.append("\n\n".join(current_units).strip())
            current_units = [unit]
        else:
            current_units.append(unit)
        current_embedding = local_embedding_dict("\n\n".join(current_units))

    if current_units:
        chunks.append("\n\n".join(current_units).strip())
    return [chunk for chunk in chunks if chunk]


def semantic_units(text: str) -> list[str]:
    blocks = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    units: list[str] = []
    pending_heading: str | None = None

    for block in blocks:
        if re.match(r"^\s{0,3}#{1,6}\s+", block):
            if pending_heading:
                units.append(pending_heading)
            pending_heading = block
            continue

        table_groups = split_table_block(block, pending_heading)
        if table_groups:
            units.extend(table_groups)
            pending_heading = None
            continue
        split_blocks = split_semantic_block(block)
        if pending_heading and split_blocks:
            split_blocks[0] = f"{pending_heading}\n{split_blocks[0]}"
            pending_heading = None
        units.extend(split_blocks)

    if pending_heading:
        units.append(pending_heading)
    return units


TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")


def split_table_block(block: str, heading: str | None = None) -> list[str]:
    """A Markdown table too long for one chunk, as groups of rows that each repeat the section
    heading and the table header: a row read on its own («| 项目甲 | … | 阻塞 |») otherwise loses
    what its columns mean and which overview it belongs to. Empty for anything else."""
    rows = [row.strip() for row in block.splitlines() if row.strip()]
    if len(block) <= CHUNK_MAX_CHARS or len(rows) < 3 or not TABLE_SEPARATOR.match(rows[1]) or not all(row.startswith("|") for row in rows):
        return []
    context = "\n".join(part for part in [heading, rows[0], rows[1]] if part)
    groups: list[str] = []
    current: list[str] = []
    for row in rows[2:]:
        if current and len(context) + sum(len(item) + 1 for item in current) + len(row) > CHUNK_MAX_CHARS:
            groups.append(context + "\n" + "\n".join(current))
            current = []
        current.append(row)
    if current:
        groups.append(context + "\n" + "\n".join(current))
    return groups


def split_semantic_block(block: str) -> list[str]:
    if len(block) <= CHUNK_MAX_CHARS:
        return [block]

    if "|" in block and "\n" in block:
        rows = [row.strip() for row in block.splitlines() if row.strip()]
        if rows:
            return rows

    sentences = re.split(r"(?<=[。！？；?!;])\s*|(?<=\.)\s+|\n+", block)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _split_long_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        breakpoint = max(text.rfind("\n", start, end), text.rfind("。", start, end), text.rfind(".", start, end))
        if breakpoint > start + size // 2:
            end = breakpoint + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def searchable_text(text: str) -> str:
    chinese_terms: list[str] = []
    for block in re.findall(r"[\u4e00-\u9fff]+", text):
        for size in (2, 3, 4):
            if len(block) < size:
                continue
            chinese_terms.extend(block[index : index + size] for index in range(len(block) - size + 1))
    return f"{text}\n{' '.join(chinese_terms)}"


def content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", "", text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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


def local_embedding(text: str, dims: int = EMBEDDING_DIMS) -> list[list[float]]:
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
        return []
    return [[index, round(value / norm, 6)] for index, value in sorted(vector.items()) if abs(value) > 1e-9]


def local_embedding_dict(text: str, dims: int = EMBEDDING_DIMS) -> dict[int, float]:
    return {int(index): float(value) for index, value in local_embedding(text, dims)}


def embedding_lsh_buckets(embedding: list[list[float]]) -> list[str]:
    top = sorted(
        ((int(index), float(value)) for index, value in embedding),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:EMBEDDING_LSH_BUCKETS]
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


def parse_embedding(raw: str) -> list[list[float]]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    embedding: list[list[float]] = []
    for item in payload:
        if isinstance(item, list) and len(item) == 2:
            try:
                embedding.append([int(item[0]), float(item[1])])
            except (TypeError, ValueError):
                continue
    return embedding


def cosine_dicts(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())
