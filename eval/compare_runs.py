#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-question diff between two scored evaluation runs.

    python eval/compare_runs.py eval/results/<before> eval/results/<after>

Lists every question whose retrieval hit, answered state or pass state changed,
so an average that moved can be traced back to concrete questions. When both runs were judged
(eval/judge.py), it also lists the questions whose verdict changed and whether their evidence
did: a verdict that moved on the same evidence is the API's wording varying, not the change.
An API run on a subset (eval/api_subset.py) is compared on the questions both runs have.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from api_subset import evidence
from eval_lib import load_jsonl, read_json, use_utf8_stdout
from score import pct


def flag(row: dict, key: str) -> str:
    return "Y" if row.get(key) else "-"


def print_summaries(before_dir: Path, after_dir: Path) -> None:
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


def print_verdicts(before_dir: Path, after_dir: Path, before: dict, after: dict) -> None:
    """Judged verdicts on the questions both runs have; refusals for negatives."""
    if not (before_dir / "judge.jsonl").exists() or not (after_dir / "judge.jsonl").exists():
        print("（有一次运行还没有按意思判分，跳过判分比较：先运行 eval/judge.py。）\n")
        return
    judged_before = {row["id"]: row.get("verdict") for row in load_jsonl(before_dir / "judge.jsonl")}
    judged_after = {row["id"]: row.get("verdict") for row in load_jsonl(after_dir / "judge.jsonl")}
    evidence_before, evidence_after = evidence(before_dir), evidence(after_dir)

    def outcome(row: dict, judged: dict) -> str:
        if row["category"] == "E":
            return "拒答" if row.get("correct_rejection") else "误答"
        return str(judged.get(row["id"]) or ("拒答" if not row.get("answered") else "未判分"))

    common = sorted(set(before) & set(after))
    positives = [item_id for item_id in common if before[item_id]["category"] != "E"]
    negatives = [item_id for item_id in common if before[item_id]["category"] == "E"]
    print("| 按意思判分 | before | after |\n|---|---|---|")
    print(f"| 可回答问题答对（共 {len(positives)} 道） | {sum(judged_before.get(i) == 'correct' for i in positives)} | {sum(judged_after.get(i) == 'correct' for i in positives)} |")
    print(f"| 没有答案的问题正确拒答（共 {len(negatives)} 道） | {sum(bool(before[i].get('correct_rejection')) for i in negatives)} | {sum(bool(after[i].get('correct_rejection')) for i in negatives)} |\n")
    moved = [(item_id, outcome(before[item_id], judged_before), outcome(after[item_id], judged_after)) for item_id in common]
    moved = [(item_id, b, a) for item_id, b, a in moved if b != a]
    if not moved:
        print("判分没有变化的题。\n")
        return
    print("| id | 问题 | 判分 before→after | 证据 |\n|---|---|---|---|")
    for item_id, b, a in moved:
        same = evidence_before.get(item_id) == evidence_after.get(item_id)
        print(f"| {item_id} | {before[item_id]['question']} | {b}→{a} | {'相同（措辞波动）' if same else '有变化'} |")
    print()


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
    if set(before) == set(after):
        print_summaries(before_dir, after_dir)
    else:
        common = len(set(before) & set(after))
        print(f"两次运行的题目不同（before {len(before)} 道，after {len(after)} 道）：只逐题比较共同的 {common} 道，汇总指标不能直接比较，不列出。\n")
    print_verdicts(before_dir, after_dir, before, after)

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
