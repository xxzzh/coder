#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure the API's prompt-cache hit rate and cost on real questions.

Answers each question with the API (the Web UI question suite's questions unless --questions
gives a file with one question per line). The answer cache is cleared for exactly these
questions first, so every answer is computed again and billed; the usage log then shows what
those calls cost. Pass 2 asks the same questions again: their prompts are byte-identical, so
nearly all input should be a cache hit.

    python scripts/measure_api_cache.py --passes 2
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import sqlite3
import sys
from pathlib import Path

import llm_budget
import query_knowledge_base as query_module


def default_questions() -> list[str]:
    import run_webui_question_suite as suite

    return list(dict.fromkeys(case["question"] for case in suite.QUERY_CASES))


def forget_answers(db_path: Path, questions: list[str], limit: int, variant: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        keys = [(query_module.cache_key(question, limit, False, variant),) for question in questions]
        removed = conn.executemany("DELETE FROM query_cache WHERE cache_key = ?", keys).rowcount
        conn.commit()
        return removed
    finally:
        conn.close()


def print_summary(label: str, records: list[dict]) -> None:
    total = llm_budget.summarize(records)["total"]
    print(
        f"{label}: {total['calls']} calls, cache hit rate {llm_budget.percent(total['cache_hit_rate'])}, "
        f"hit/miss/output tokens {total['cache_hit_tokens']}/{total['cache_miss_tokens']}/{total['output_tokens']}, "
        f"output/input {total['output_input_ratio']}, estimated cost ${total['cost_usd']:.6f} ({total['peak_calls']} at peak price)",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", help="file with one question per line")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--db", default=str(query_module.DB_PATH))
    args = parser.parse_args()
    if args.questions:
        args.questions = str(Path(args.questions).resolve())
    os.chdir(llm_budget.ROOT)  # the index and processed files are relative to the project folder

    config = query_module.runtime_config()
    if not query_module.api_enabled(config):
        print("API is not configured (LKA_USE_API, LKA_API_KEY, LKA_API_MODEL); nothing to measure.")
        return 1
    log = llm_budget.usage_log_path(llm_budget.settings())
    if log is None:
        print("Usage logging is off (LKA_API_USAGE_LOG=off); turn it on to measure.")
        return 1
    questions = (
        [line.strip() for line in Path(args.questions).read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.questions
        else default_questions()
    )
    if llm_budget.is_peak(datetime.now(timezone.utc)):
        print("Note: this is DeepSeek peak time (Beijing weekdays 9-12, 14-18); these calls cost double.")
    db_path = Path(args.db)
    variant = query_module.query_cache_variant(config, True, False)
    for number in range(1, args.passes + 1):
        started = datetime.now(llm_budget.BEIJING).isoformat(timespec="seconds")
        forget_answers(db_path, questions, args.limit, variant)
        for index, question in enumerate(questions, start=1):
            result = query_module.query(db_path, question, args.limit, use_web=False, allow_api=True)
            agent = result.get("agent") or {}
            print(
                f"  pass {number} [{index}/{len(questions)}] {question[:30]:30} mode={result.get('answer_mode')} "
                f"calls={agent.get('calls', 0)} hit={llm_budget.percent(agent.get('cache_hit_rate'))}",
                flush=True,
            )
        print_summary(f"pass {number} (since {started})", llm_budget.read_log(log, started))
    print(f"Full log: {log}\nReport any window: python scripts/llm_budget.py report --since <time>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
