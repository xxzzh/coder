#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the offline RAG evaluation scripts.

Nothing in this package calls an LLM. Collection talks to the system under
test; validation and scoring only use string/set rules so the results can be
audited line by line.
"""

from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
EVAL_DIR = ROOT / "eval"
DEFAULT_SET = EVAL_DIR / "eval_set.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
EXPERIMENT_LOG = EVAL_DIR / "experiment_log.md"
RAW_DIR = ROOT / "knowledge_base" / "raw"
DB_PATH = ROOT / "knowledge_base" / "index" / "knowledge.db"

CATEGORIES = {
    "A": "精确匹配",
    "B": "同义改写",
    "C": "表格/结构化",
    "D": "多片段综合",
    "E": "负样本",
}
POSITIVE_CATEGORIES = {"A", "B", "C", "D"}
NEGATIVE_CATEGORIES = {"E"}


def use_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def norm(text: Any) -> str:
    """Width-fold, drop all whitespace and lowercase, for robust substring checks."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text or ""))).lower()


def contains(haystack_norm: str, keyword: str) -> bool:
    needle = norm(keyword)
    return bool(needle) and needle in haystack_norm


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    """Load the evaluation set, which is kept local and not committed (it quotes internal documents)."""
    if not path.exists():
        raise SystemExit(
            f"评测集不存在：{path}\n"
            "eval_set.jsonl 摘录了内部资料原文，只保存在本地、不提交到 Git。"
            "请从本机备份复制过来，或按 eval/README.md 的格式新建。"
        )
    return load_jsonl(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def required_hits(keyword_count: int, ratio: float) -> int:
    """How many gold keywords one chunk must contain to count as a correct hit."""
    if keyword_count <= 0:
        return 0
    return min(keyword_count, max(1, math.ceil(keyword_count * ratio)))


def raw_files_by_name(raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    files: dict[str, Path] = {}
    if not raw_dir.exists():
        return files
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and not path.name.startswith("~$"):
            files.setdefault(path.name, path)
    return files


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[index]


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None
