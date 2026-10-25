#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score raw system outputs against the evaluation set with plain rules.

No model is involved: every number below is a substring check or a set
operation that can be recomputed by hand from raw_responses.jsonl.

Retrieval layer (general hybrid search, top-k chunks):
  hit       a chunk from a gold doc containing >= ceil(hit_ratio * |keywords|) gold keywords
  Hit@1/3/k whether such a chunk is ranked within the first 1/3/k
  MRR@k     1 / rank of the first hit chunk (0 when none)
  KwCov@k   share of gold keywords found anywhere in the top-k gold-doc chunks

End-to-end layer (query(), including domain retrievers):
  answered       the system returned an answer instead of "no evidence"
  AnsCov         share of gold keywords that appear in the answer text
  pass           answered and AnsCov >= answer_pass_ratio
  negatives      correct when the system refuses (source_type "none")
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from eval_lib import (
    CATEGORIES,
    DEFAULT_SET,
    NEGATIVE_CATEGORIES,
    POSITIVE_CATEGORIES,
    contains,
    load_eval_set,
    load_jsonl,
    mean,
    norm,
    percentile,
    read_json,
    required_hits,
    use_utf8_stdout,
    write_json,
    write_jsonl,
)

REFUSAL_SOURCE_TYPES = {None, "", "none", "error"}


def corrupt_keywords(eval_set: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace every gold keyword with a string that cannot occur, for the self-check."""
    corrupted = []
    for item in eval_set:
        clone = dict(item)
        if item.get("gold_chunk_keywords"):
            clone["gold_chunk_keywords"] = [
                "zzcorrupt" + hashlib.sha1(f"{item['id']}:{keyword}".encode("utf-8")).hexdigest()[:12]
                for keyword in item["gold_chunk_keywords"]
            ]
        corrupted.append(clone)
    return corrupted


def score_positive(item: dict[str, Any], response: dict[str, Any], k: int, hit_ratio: float, answer_pass_ratio: float) -> dict[str, Any]:
    gold_docs = set(item["gold_doc_ids"])
    keywords = list(item["gold_chunk_keywords"])
    need = required_hits(len(keywords), hit_ratio)
    top = [row for row in response.get("retrieval", []) if int(row.get("rank", 0)) <= k]

    first_hit_rank = None
    covered: set[str] = set()
    for row in top:
        if row.get("file_name") not in gold_docs:
            continue
        text = norm(row.get("text", ""))
        found = {keyword for keyword in keywords if contains(text, keyword)}
        covered |= found
        if first_hit_rank is None and len(found) >= need:
            first_hit_rank = int(row["rank"])

    e2e = response.get("end_to_end", {})
    answered = e2e.get("source_type") not in REFUSAL_SOURCE_TYPES
    answer_text = norm(e2e.get("answer", ""))
    answer_found = [keyword for keyword in keywords if contains(answer_text, keyword)]
    answer_cov = len(answer_found) / len(keywords)
    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "required_hits": need,
        "first_hit_rank": first_hit_rank,
        "hit@1": first_hit_rank is not None and first_hit_rank <= 1,
        "hit@3": first_hit_rank is not None and first_hit_rank <= 3,
        "hit@k": first_hit_rank is not None,
        "rr": 1.0 / first_hit_rank if first_hit_rank else 0.0,
        "kw_cov@k": len(covered) / len(keywords),
        "answered": answered,
        "answer_cov": answer_cov,
        "answer_missing_keywords": [keyword for keyword in keywords if keyword not in answer_found],
        "pass": answered and answer_cov >= answer_pass_ratio,
        "answer_path": e2e.get("retriever_plugin") or e2e.get("answer_mode") or e2e.get("retrieval_mode") or e2e.get("source_type"),
        "retrieval_ms": response.get("retrieval_ms"),
        "end_to_end_ms": response.get("end_to_end_ms"),
        "errors": [value for key, value in response.items() if key.endswith("_error")],
    }


def score_negative(item: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    e2e = response.get("end_to_end", {})
    answered = e2e.get("source_type") not in REFUSAL_SOURCE_TYPES
    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "answered": answered,
        "correct_rejection": not answered,
        "answer_path": e2e.get("retriever_plugin") or e2e.get("answer_mode") or e2e.get("source_type"),
        "answer_preview": str(e2e.get("answer", ""))[:160],
        "end_to_end_ms": response.get("end_to_end_ms"),
        "errors": [value for key, value in response.items() if key.endswith("_error")],
    }


def aggregate_positive(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "hit@1": mean(float(row["hit@1"]) for row in rows),
        "hit@3": mean(float(row["hit@3"]) for row in rows),
        "hit@k": mean(float(row["hit@k"]) for row in rows),
        "mrr": mean(row["rr"] for row in rows),
        "kw_cov@k": mean(row["kw_cov@k"] for row in rows),
        "answered": mean(float(row["answered"]) for row in rows),
        "false_refusal": mean(float(not row["answered"]) for row in rows),
        "answer_cov": mean(row["answer_cov"] for row in rows),
        "pass": mean(float(row["pass"]) for row in rows),
    }


def score(
    eval_set: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    k: int = 5,
    hit_ratio: float = 0.5,
    answer_pass_ratio: float = 0.5,
    runnable_ids: set[str] | None = None,
) -> dict[str, Any]:
    by_id = {response["id"]: response for response in responses}
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    skipped: list[str] = []
    for item in eval_set:
        if runnable_ids is not None and item["id"] not in runnable_ids:
            skipped.append(item["id"])
            continue
        response = by_id.get(item["id"])
        if response is None:
            skipped.append(item["id"])
            continue
        if item["category"] in POSITIVE_CATEGORIES:
            positives.append(score_positive(item, response, k, hit_ratio, answer_pass_ratio))
        elif item["category"] in NEGATIVE_CATEGORIES:
            negatives.append(score_negative(item, response))

    by_category = {
        category: aggregate_positive([row for row in positives if row["category"] == category])
        for category in sorted(POSITIVE_CATEGORIES)
    }
    latencies = [float(row["end_to_end_ms"]) for row in positives + negatives if row.get("end_to_end_ms") is not None]
    summary = {
        "k": k,
        "hit_ratio": hit_ratio,
        "answer_pass_ratio": answer_pass_ratio,
        "positives": aggregate_positive(positives),
        "by_category": by_category,
        "negatives": {
            "n": len(negatives),
            "correct_rejection": mean(float(row["correct_rejection"]) for row in negatives),
        },
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95), "max": max(latencies) if latencies else None},
        "answer_paths": dict(Counter(str(row["answer_path"]) for row in positives).most_common()),
        "errors": sum(1 for row in positives + negatives if row["errors"]),
        "skipped": skipped,
    }
    return {"summary": summary, "positives": positives, "negatives": negatives}


def self_check(eval_set: list[dict[str, Any]], responses: list[dict[str, Any]], k: int, runnable_ids: set[str] | None) -> dict[str, Any]:
    """With corrupted gold keywords every positive metric must drop to zero."""
    corrupted = score(corrupt_keywords(eval_set), responses, k=k, runnable_ids=runnable_ids)
    positives = corrupted["summary"]["positives"]
    leaked = [
        row["id"]
        for row in corrupted["positives"]
        if row["hit@k"] or row["kw_cov@k"] > 0 or row["answer_cov"] > 0 or row["pass"]
    ]
    return {"ok": positives.get("n", 0) > 0 and not leaked, "leaked_ids": leaked, "corrupted_positive_summary": positives}


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def num(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def summary_markdown(result: dict[str, Any], meta: dict[str, Any] | None = None, check: dict[str, Any] | None = None) -> str:
    summary = result["summary"]
    k = summary["k"]
    lines: list[str] = ["# 评测结果", ""]
    if meta:
        git = meta.get("git", {})
        index = meta.get("index", {})
        lines += [
            f"- 运行时间：{meta.get('created_at')}",
            f"- 代码：{git.get('branch')}@{git.get('commit')}{'（有未提交改动）' if git.get('dirty') else ''}",
            f"- 索引：{index.get('documents')} 文档 / {index.get('chunks')} chunk，embedding={index.get('embedding_model')}",
            f"- API：{'允许' if meta.get('allow_api') else '禁用'}，联网：禁用，缓存：禁用",
            "",
        ]
    lines += [
        f"| 类别 | n | Hit@1 | Hit@3 | Hit@{k} | MRR@{k} | KwCov@{k} | 作答率 | 答案覆盖 | 通过率 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for category, stats in summary["by_category"].items():
        if not stats.get("n"):
            continue
        lines.append(
            f"| {category} {CATEGORIES[category]} | {stats['n']} | {pct(stats['hit@1'])} | {pct(stats['hit@3'])} | {pct(stats['hit@k'])} | "
            f"{num(stats['mrr'])} | {pct(stats['kw_cov@k'])} | {pct(stats['answered'])} | {pct(stats['answer_cov'])} | {pct(stats['pass'])} |"
        )
    total = summary["positives"]
    if total.get("n"):
        lines.append(
            f"| **A-D 合计** | {total['n']} | {pct(total['hit@1'])} | {pct(total['hit@3'])} | {pct(total['hit@k'])} | "
            f"{num(total['mrr'])} | {pct(total['kw_cov@k'])} | {pct(total['answered'])} | {pct(total['answer_cov'])} | {pct(total['pass'])} |"
        )
    negatives = summary["negatives"]
    latency = summary["latency_ms"]
    lines += [
        "",
        f"- 负样本正确拒答：{pct(negatives['correct_rejection'])}（n={negatives['n']}）",
        f"- 端到端延迟：P50 {num(latency['p50'], 0)} ms，P95 {num(latency['p95'], 0)} ms，最大 {num(latency['max'], 0)} ms",
        f"- 正样本答案来源：{summary['answer_paths']}",
        f"- 运行出错：{summary['errors']}，跳过：{len(summary['skipped'])}",
    ]
    if check is not None:
        lines.append(f"- 判分自检（破坏标准答案后应全部判 fail）：{'通过' if check['ok'] else '失败 ' + str(check['leaked_ids'])}")
    failed = [row for row in result["positives"] if not row["hit@k"] or not row["pass"]]
    if failed:
        lines += ["", "## 未通过的正样本", "", f"| id | 问题 | 首个命中排名 | KwCov@{k} | 作答 | 答案覆盖 | 答案缺少的关键词 |", "|---|---|---|---|---|---|---|"]
        for row in failed:
            missing = "、".join(row["answer_missing_keywords"][:4]) + ("…" if len(row["answer_missing_keywords"]) > 4 else "")
            lines.append(
                f"| {row['id']} | {row['question']} | {row['first_hit_rank'] or '未命中'} | {pct(row['kw_cov@k'])} | "
                f"{'是' if row['answered'] else '拒答'} | {pct(row['answer_cov'])} | {missing} |"
            )
    wrong_negatives = [row for row in result["negatives"] if not row["correct_rejection"]]
    if wrong_negatives:
        lines += ["", "## 负样本被错误作答", "", "| id | 问题 | 答案开头 |", "|---|---|---|"]
        for row in wrong_negatives:
            preview = row["answer_preview"].replace("|", "/").replace("\n", " ")
            lines.append(f"| {row['id']} | {row['question']} | {preview} |")
    return "\n".join(lines) + "\n"


def score_run(run_dir: Path, set_path: Path, k: int, hit_ratio: float, answer_pass_ratio: float) -> tuple[dict[str, Any], dict[str, Any]]:
    eval_set = load_eval_set(set_path)
    responses = load_jsonl(run_dir / "raw_responses.jsonl")
    validation_path = run_dir / "validation.json"
    runnable = set(read_json(validation_path)["runnable_ids"]) if validation_path.exists() else None
    meta = read_json(run_dir / "run_meta.json") if (run_dir / "run_meta.json").exists() else None
    result = score(eval_set, responses, k=k, hit_ratio=hit_ratio, answer_pass_ratio=answer_pass_ratio, runnable_ids=runnable)
    check = self_check(eval_set, responses, k, runnable)
    result["summary"]["self_check_ok"] = check["ok"]
    write_jsonl(run_dir / "scores.jsonl", result["positives"] + result["negatives"])
    write_json(run_dir / "summary.json", {**result["summary"], "self_check": check})
    (run_dir / "summary.md").write_text(summary_markdown(result, meta, check), encoding="utf-8")
    return result, check


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser(description="Score raw_responses.jsonl with rule-based metrics.")
    parser.add_argument("run_dir")
    parser.add_argument("--set", default=str(DEFAULT_SET))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--hit-ratio", type=float, default=0.5)
    parser.add_argument("--answer-pass-ratio", type=float, default=0.5)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    score_run(run_dir, Path(args.set), args.k, args.hit_ratio, args.answer_pass_ratio)
    print((run_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
