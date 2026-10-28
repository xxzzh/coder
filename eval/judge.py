#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Judge each answer of a scored run by its meaning, next to the keyword score.

    python eval/judge.py eval/results/<run>

The keyword score (score.py) counts a correct answer as failed when it words a point differently
from the gold keywords ("use the configuration tool" against "do not edit it in a text editor"). Here
the configured chat model compares each answered positive with the gold answer and says whether
it is correct, partly correct or wrong; wording, order and format do not count. Refusals are wrong
without a call, and negatives keep the rule score (a refusal is correct).

Verdicts are cached in eval/results/judge_cache.jsonl by question, gold answer and answer, so a
rerun only pays for answers that changed. Calls are recorded in the usage log under "judge".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from eval_lib import DEFAULT_SET, RESULTS_DIR, SCRIPTS_DIR, load_eval_set, load_jsonl, use_utf8_stdout, write_json, write_jsonl

sys.path.insert(0, str(SCRIPTS_DIR))

import api_providers  # noqa: E402
import llm_budget  # noqa: E402
from kb.config import runtime_config  # noqa: E402

JUDGE_VERSION = "judge-v1"
CACHE_PATH = RESULTS_DIR / "judge_cache.jsonl"
VERDICTS = ("correct", "partial", "wrong")

SYSTEM_PROMPT = """你是问答评测员。给你一个问题、标准答案和系统答案，判断系统答案是否答对了这个问题。

判断规则：
- 只看事实：系统答案里的要点是否与标准答案一致、是否覆盖了标准答案的要点。
- 措辞、顺序、格式、详略不同都不扣分；说法不同但意思相同算一致（例如“必须用配置工具修改”与“不能用文本编辑器直接改”）。
- 多出的内容只要不与标准答案矛盾就不扣分。
- 标准答案给出的数值、名称、步骤，系统答案写错或漏掉就要扣分。

verdict 三选一：
- "correct"：覆盖了标准答案的全部要点，且没有矛盾。
- "partial"：覆盖了部分要点，且没有矛盾。
- "wrong"：答错、与标准答案矛盾，或基本没回答问题。

只输出一个 JSON 对象，例如：
{"verdict": "partial", "missing": "没有说明第 3 步的门限", "reason": "前两步与标准答案一致，缺少第 3 步"}"""


def cache_key(question: str, gold: str, answer: str) -> str:
    return hashlib.sha1(json.dumps([JUDGE_VERSION, question, gold, answer], ensure_ascii=False).encode("utf-8")).hexdigest()


def load_cache() -> dict[str, dict[str, Any]]:
    if not CACHE_PATH.exists():
        return {}
    return {row["key"]: row for row in load_jsonl(CACHE_PATH)}


def gold_answer(item: dict[str, Any]) -> str:
    keywords = "、".join(item.get("gold_chunk_keywords") or [])
    return f"{item.get('gold_answer_summary', '')}\n（要点关键词：{keywords}）" if keywords else str(item.get("gold_answer_summary", ""))


def ask_model(config: dict[str, str], question: str, gold: str, answer: str) -> dict[str, Any]:
    budget = llm_budget.task_settings(config, "judge")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"question": question, "gold_answer": gold, "system_answer": answer}, ensure_ascii=False)},
    ]
    reply = api_providers.chat_completion_detailed(
        config.get("LKA_API_PROVIDER", "openai-compatible"),
        config.get("LKA_API_BASE_URL", "").rstrip("/"),
        config.get("LKA_API_KEY", ""),
        budget["model"],
        messages,
        temperature=0.0,
        timeout=60,
        json_output=True,
        max_tokens=budget["max_tokens"],
        thinking=budget["thinking"],
        task="judge",
    )
    try:
        verdict = json.loads(reply.get("content") or "{}")
    except json.JSONDecodeError:
        verdict = {}
    if verdict.get("verdict") not in VERDICTS:
        return {"verdict": "unjudged", "missing": "", "reason": f"unusable reply: {str(reply.get('content'))[:120]}"}
    return {"verdict": verdict["verdict"], "missing": str(verdict.get("missing") or ""), "reason": str(verdict.get("reason") or "")}


def judge_run(run_dir: Path, set_path: Path = DEFAULT_SET) -> dict[str, Any]:
    config = runtime_config()
    items = {item["id"]: item for item in load_eval_set(set_path)}
    responses = {row["id"]: row for row in load_jsonl(run_dir / "raw_responses.jsonl")}
    scores = {row["id"]: row for row in load_jsonl(run_dir / "scores.jsonl")}
    cache = load_cache()
    new_cache: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for item_id, score in scores.items():
        item = items.get(item_id)
        if item is None or item["category"] == "E":
            continue
        answer = str(responses.get(item_id, {}).get("end_to_end", {}).get("answer", ""))
        if not score.get("answered"):
            verdict = {"verdict": "wrong", "missing": "", "reason": "refused"}
        else:
            key = cache_key(item["question"], gold_answer(item), answer)
            if key in cache:
                verdict = {name: cache[key][name] for name in ("verdict", "missing", "reason")}
            else:
                try:
                    verdict = ask_model(config, item["question"], gold_answer(item), answer)
                except Exception as exc:  # noqa: BLE001  (a dropped connection leaves this one unjudged)
                    verdict = {"verdict": "unjudged", "missing": "", "reason": f"API call failed: {str(exc)[:120]}"}
                if verdict["verdict"] in VERDICTS:
                    # Saved as soon as it is paid for, so an interrupted run does not pay twice.
                    new_cache.append({"key": key, **verdict})
                    cache[key] = new_cache[-1]
                    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with CACHE_PATH.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(new_cache[-1], ensure_ascii=False) + "\n")
        rows.append({"id": item_id, "category": item["category"], "question": item["question"], "keyword_pass": bool(score.get("pass")), **verdict})
    judged = [row for row in rows if row["verdict"] in VERDICTS]
    summary = {
        "judge_version": JUDGE_VERSION,
        "positives": len(rows),
        "correct": sum(row["verdict"] == "correct" for row in rows),
        "partial": sum(row["verdict"] == "partial" for row in rows),
        "wrong": sum(row["verdict"] == "wrong" for row in rows),
        "unjudged": len(rows) - len(judged),
        "keyword_pass": sum(row["keyword_pass"] for row in rows),
        "new_calls": len(new_cache),
    }
    summary["correct_rate"] = summary["correct"] / len(rows) if rows else 0.0
    summary["keyword_pass_rate"] = summary["keyword_pass"] / len(rows) if rows else 0.0
    write_jsonl(run_dir / "judge.jsonl", rows)
    write_json(run_dir / "judge_summary.json", summary)
    return {"summary": summary, "rows": rows}


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser(description="Judge a scored run's answers by meaning with the configured chat model.")
    parser.add_argument("run_dir")
    parser.add_argument("--set", default=str(DEFAULT_SET))
    args = parser.parse_args()
    result = judge_run(Path(args.run_dir), Path(args.set))
    summary = result["summary"]
    print(
        f"positives {summary['positives']}: correct {summary['correct']} ({summary['correct_rate']:.1%}), "
        f"partial {summary['partial']}, wrong {summary['wrong']}, unjudged {summary['unjudged']}; "
        f"keyword pass {summary['keyword_pass']} ({summary['keyword_pass_rate']:.1%}); new calls {summary['new_calls']}"
    )
    disagreements = [row for row in result["rows"] if row["keyword_pass"] != (row["verdict"] == "correct")]
    if disagreements:
        print("\n| id | 问题 | 关键词评分 | 按意思判 | 说明 |\n|---|---|---|---|---|")
        for row in disagreements:
            print(f"| {row['id']} | {row['question']} | {'通过' if row['keyword_pass'] else '未通过'} | {row['verdict']} | {row['missing'] or row['reason']} |")


if __name__ == "__main__":
    main()
