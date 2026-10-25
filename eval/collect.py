#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run every evaluation question through the system and store the raw output.

This script only executes and records. It never decides whether an answer is
right; that is score.py's job. Two layers are captured per question:

- retrieval: the ranked chunks from the general hybrid search (search())
- end_to_end: the final answer from query(), including domain retrievers

The query cache and the background re-index trigger are disabled so every run
measures the current code, not an answer cached by an earlier version.
"""

from __future__ import annotations

import argparse
import os
import platform
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval_lib import DB_PATH, DEFAULT_SET, ROOT, SCRIPTS_DIR, load_eval_set, use_utf8_stdout, write_json, write_jsonl

sys.path.insert(0, str(SCRIPTS_DIR))

import query_knowledge_base as qkb  # noqa: E402

RETRIEVAL_KEEP = 10


def isolate_system(allow_api: bool) -> dict[str, str]:
    """Disable cache and async re-index; optionally strip API settings."""
    qkb.cache_get = lambda *args, **kwargs: None
    qkb.cache_put = lambda *args, **kwargs: None
    qkb.trigger_async_update = lambda *args, **kwargs: False
    base_config = qkb.runtime_config()
    if not allow_api:
        base_config = {**base_config, "LKA_USE_API": "false", "LKA_RERANK_ENABLED": "false"}
        frozen = dict(base_config)
        qkb.runtime_config = lambda: dict(frozen)
    return base_config


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {"commit": run("rev-parse", "--short", "HEAD"), "branch": run("branch", "--show-current"), "dirty": bool(run("status", "--porcelain"))}


def index_state(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    try:
        metadata = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
        documents = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        conn.close()
    return {
        "documents": documents,
        "chunks": chunks,
        "indexed_at": metadata.get("indexed_at"),
        "embedding_model": metadata.get("embedding_model"),
        "chunk_strategy": metadata.get("chunk_strategy"),
        "vector_index_status": metadata.get("vector_index_status"),
    }


def retrieval_rows(db_path: Path, question: str, k: int, config: dict[str, str]) -> list[dict[str, Any]]:
    rows, _ = qkb.search(db_path, question, k, config)
    return [
        {
            "rank": rank,
            "file_name": row.get("file_name"),
            "chunk_id": row.get("chunk_id"),
            "rerank_score": row.get("rerank_score"),
            "text": str(row.get("text", "")),
        }
        for rank, row in enumerate(rows[:RETRIEVAL_KEEP], start=1)
    ]


def end_to_end(db_path: Path, question: str, k: int, allow_api: bool) -> dict[str, Any]:
    result = qkb.query(db_path, question, k, use_web=False, allow_api=allow_api)
    return {
        "answer": str(result.get("answer", "")),
        "source_type": result.get("source_type"),
        "answer_mode": result.get("answer_mode"),
        "retrieval_mode": result.get("retrieval_mode"),
        "retriever_plugin": result.get("retriever_plugin"),
        "degrade_reason": result.get("degrade_reason"),
        "api_used": bool(result.get("api_used")),
        "sources": [
            {
                "source_id": source.get("source_id"),
                "file": source.get("file"),
                "chunk_id": source.get("chunk_id"),
                "excerpt": str(source.get("excerpt") or source.get("citation") or ""),
            }
            for source in result.get("sources") or []
        ],
    }


def collect(eval_set: list[dict[str, Any]], db_path: Path, out_dir: Path, k: int, allow_api: bool) -> Path:
    db_path = db_path.resolve()
    out_dir = out_dir.resolve()
    # The query module resolves knowledge_base/ paths relative to the project root.
    os.chdir(ROOT)
    config = isolate_system(allow_api)
    responses: list[dict[str, Any]] = []
    for position, item in enumerate(eval_set, start=1):
        question = str(item["question"])
        record: dict[str, Any] = {"id": item["id"], "question": question}
        started = time.perf_counter()
        try:
            record["retrieval"] = retrieval_rows(db_path, question, k, config)
        except Exception as exc:  # noqa: BLE001 - one broken question must not stop the run
            record["retrieval"] = []
            record["retrieval_error"] = f"{type(exc).__name__}: {exc}"
        record["retrieval_ms"] = round((time.perf_counter() - started) * 1000, 1)
        started = time.perf_counter()
        try:
            record["end_to_end"] = end_to_end(db_path, question, k, allow_api)
        except Exception as exc:  # noqa: BLE001
            record["end_to_end"] = {"answer": "", "source_type": "error", "sources": []}
            record["end_to_end_error"] = f"{type(exc).__name__}: {exc}"
        record["end_to_end_ms"] = round((time.perf_counter() - started) * 1000, 1)
        responses.append(record)
        print(f"[{position}/{len(eval_set)}] {item['id']} {record['end_to_end_ms']:.0f}ms {question}", flush=True)

    write_jsonl(out_dir / "raw_responses.jsonl", responses)
    write_json(
        out_dir / "run_meta.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": git_state(),
            "index": index_state(db_path),
            "k": k,
            "allow_api": allow_api,
            "web": False,
            "cache": "disabled",
            "python": platform.python_version(),
            "questions": len(eval_set),
        },
    )
    return out_dir


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser(description="Collect raw system outputs for the evaluation set.")
    parser.add_argument("out_dir")
    parser.add_argument("--set", default=str(DEFAULT_SET))
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--allow-api", action="store_true", help="Let the system call the configured LLM API.")
    parser.add_argument("--ids", nargs="*", help="Only run these item ids.")
    args = parser.parse_args()

    eval_set = load_eval_set(Path(args.set))
    if args.ids:
        wanted = set(args.ids)
        eval_set = [item for item in eval_set if item["id"] in wanted]
    collect(eval_set, Path(args.db), Path(args.out_dir), args.k, args.allow_api)


if __name__ == "__main__":
    main()
