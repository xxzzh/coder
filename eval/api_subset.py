#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The questions worth an API run after a change: those whose retrieved evidence changed.

    python eval/api_subset.py eval/results/<before-local> eval/results/<after-local> [--negatives 5]

API answers cost money and vary in wording from run to run, so a question whose evidence did not
change tells nothing about the change. Compare two local runs (before and after the change); the
questions whose top retrieved chunks differ, plus a few fixed negatives that catch new false
answers, are written as an eval set next to the after run (api_subset.jsonl), and the command that
runs them with the API is printed. When no evidence changed, no API run is needed.

The agent's first evidence is this same search for the question; the follow-up searches it writes
itself are not compared. A change to the API side (prompts, answer polishing, the agent loop)
changes no evidence at all: run the full set for those, and before every release.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_lib import DEFAULT_SET, ROOT, load_eval_set, load_jsonl, use_utf8_stdout

SUBSET_FILE = "api_subset.jsonl"


def evidence(run_dir: Path) -> dict[str, list[str]]:
    """Each question's retrieved chunks, by text (chunk ids shift when a document is read again)."""
    return {row["id"]: [str(hit.get("text", "")) for hit in row.get("retrieval", [])] for row in load_jsonl(run_dir / "raw_responses.jsonl")}


def fixed_negatives(items: list[dict], count: int) -> list[str]:
    """The same few negatives every time, spread over the set, so subset runs stay comparable."""
    negatives = sorted(item["id"] for item in items if item.get("category") == "E")
    if count <= 0 or not negatives:
        return []
    step = max(1, len(negatives) // count)
    return negatives[::step][:count]


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser(description="Write the questions whose evidence changed as an eval set for an API run.")
    parser.add_argument("before", help="local run before the change")
    parser.add_argument("after", help="local run after the change")
    parser.add_argument("--negatives", type=int, default=5, help="fixed negatives added to a non-empty subset (default 5)")
    parser.add_argument("--set", default=str(DEFAULT_SET))
    args = parser.parse_args()
    before_dir, after_dir = Path(args.before), Path(args.after)
    items = load_eval_set(Path(args.set))
    before, after = evidence(before_dir), evidence(after_dir)

    changed = [item["id"] for item in items if before.get(item["id"]) != after.get(item["id"])]
    print(f"before: {before_dir.name}\nafter:  {after_dir.name}\n")
    if not changed:
        print(f"{len(items)} 道题的证据都没有变化：这次改动不需要跑 API 评测。")
        return
    negatives = [item_id for item_id in fixed_negatives(items, args.negatives) if item_id not in changed]
    chosen = set(changed) | set(negatives)
    subset = [item for item in items if item["id"] in chosen]
    subset_path = after_dir / SUBSET_FILE
    subset_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in subset), encoding="utf-8")

    print(f"证据有变化：{len(changed)} 道（{'、'.join(changed)}）")
    if negatives:
        print(f"固定负样本：{len(negatives)} 道（{'、'.join(negatives)}）")
    shown = subset_path.resolve().relative_to(ROOT) if subset_path.resolve().is_relative_to(ROOT) else subset_path.resolve()
    label = after_dir.name.split("-", 2)[-1].removesuffix("-local")
    print(f"共 {len(subset)} / {len(items)} 道，已写入 {shown}\n")
    print("在项目目录下用 API 跑这些题（不写入 experiment_log，子集的平均数不能和全量比较）：")
    print(f'  .\\run-agent.bat eval --allow-api --set "{shown}" --label {label}-api-subset')
    print("跑完后按意思判分，再与上一次全量 API 评测逐题比较：")
    print("  .venv\\Scripts\\python.exe eval\\judge.py eval\\results\\<子集结果目录>")
    print("  .venv\\Scripts\\python.exe eval\\compare_runs.py eval\\results\\<上一次全量 API 结果> eval\\results\\<子集结果目录>")


if __name__ == "__main__":
    main()
