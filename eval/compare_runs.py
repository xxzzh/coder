#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-question diff between two scored evaluation runs.

    python eval/compare_runs.py eval/results/<before> eval/results/<after>

Lists every question whose retrieval hit, answered state or pass state changed,
so an average that moved can be traced back to concrete questions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_lib import load_jsonl, read_json, use_utf8_stdout
from score import pct


def flag(row: dict, key: str) -> str:
    return "Y" if row.get(key) else "-"


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser(description="Compare two evaluation runs question by question.")
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    before_dir, after_dir = Path(args.before), Path(args.after)
    before = {row["id"]: row for row in load_jsonl(before_dir / "scores.jsonl")}
    after = {row["id"]: row for row in load_jsonl(after_dir / "scores.jsonl")}

    print(f"before: {before_dir.name}\nafter:  {after_dir.name}\n")
    summary_before = read_json(before_dir / "summary.json")
    summary_after = read_json(after_dir / "summary.json")
    print("| 指标 | before | after |\n|---|---|---|")
    for key, label in (("hit@1", "Hit@1"), ("hit@k", "Hit@5"), ("mrr", "MRR"), ("kw_cov@k", "KwCov@5"), ("answered", "作答率"), ("pass", "通过率")):
        b = summary_before["positives"].get(key)
        a = summary_after["positives"].get(key)
        fmt = (lambda v: "-" if v is None else f"{v:.3f}") if key == "mrr" else pct
        print(f"| {label} | {fmt(b)} | {fmt(a)} |")
    print(f"| 负样本拒答 | {pct(summary_before['negatives'].get('correct_rejection'))} | {pct(summary_after['negatives'].get('correct_rejection'))} |")
    print(f"| P95 ms | {summary_before['latency_ms'].get('p95'):.0f} | {summary_after['latency_ms'].get('p95'):.0f} |\n")

    changed = []
    for item_id in sorted(set(before) & set(after)):
        b, a = before[item_id], after[item_id]
        keys = ("hit@k", "answered", "pass") if b["category"] != "E" else ("correct_rejection",)
        if any(bool(b.get(key)) != bool(a.get(key)) for key in keys) or b.get("first_hit_rank") != a.get("first_hit_rank"):
            changed.append((item_id, b, a))
    if not changed:
        print("没有题目发生变化。")
        return
    print("| id | 问题 | 命中排名 before→after | 作答 | 通过 |\n|---|---|---|---|---|")
    for item_id, b, a in changed:
        if b["category"] == "E":
            print(f"| {item_id} | {b['question']} | - | 拒答 {flag(b, 'correct_rejection')}→{flag(a, 'correct_rejection')} | - |")
            continue
        rank = f"{b.get('first_hit_rank') or '未中'}→{a.get('first_hit_rank') or '未中'}"
        print(f"| {item_id} | {b['question']} | {rank} | {flag(b, 'answered')}→{flag(a, 'answered')} | {flag(b, 'pass')}→{flag(a, 'pass')} |")


if __name__ == "__main__":
    main()
