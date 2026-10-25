#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot evaluation: validate the set, collect raw outputs, score, self-check.

Examples:
    python eval/run_eval.py --label baseline
    python eval/run_eval.py --label gate-v2 --log "相关性门槛改为分词覆盖率"
"""

from __future__ import annotations

import argparse
import re
import warnings
from datetime import datetime
from pathlib import Path

from eval_lib import DB_PATH, DEFAULT_SET, EXPERIMENT_LOG, RAW_DIR, RESULTS_DIR, load_eval_set, use_utf8_stdout, write_json

import collect as collect_module
import score as score_module
import validate_set


LOG_HEADER = (
    "| 日期 | 运行 | 改动说明 | 代码 | Hit@1 | Hit@5 | MRR | KwCov@5 | 作答率 | 通过率 | 负样本拒答 | P95 ms |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def append_log(run_dir: Path, summary: dict, meta: dict, description: str) -> None:
    pos = summary["positives"]
    git = meta.get("git", {})
    code = f"{git.get('commit')}{'+dirty' if git.get('dirty') else ''}"
    row = (
        f"| {datetime.now().strftime('%Y-%m-%d')} | {run_dir.name} | {description} | {code} | "
        f"{score_module.pct(pos.get('hit@1'))} | {score_module.pct(pos.get('hit@k'))} | {score_module.num(pos.get('mrr'))} | "
        f"{score_module.pct(pos.get('kw_cov@k'))} | {score_module.pct(pos.get('answered'))} | {score_module.pct(pos.get('pass'))} | "
        f"{score_module.pct(summary['negatives'].get('correct_rejection'))} | {score_module.num(summary['latency_ms'].get('p95'), 0)} |\n"
    )
    if not EXPERIMENT_LOG.exists() or "| 日期 |" not in EXPERIMENT_LOG.read_text(encoding="utf-8"):
        EXPERIMENT_LOG.write_text("# 评测实验记录\n\n每次改动检索/问答逻辑后运行 `python eval/run_eval.py --log \"说明\"` 追加一行。\n\n" + LOG_HEADER, encoding="utf-8")
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(row)


def main() -> None:
    use_utf8_stdout()
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser(description="Validate, collect and score the evaluation set.")
    parser.add_argument("--label", default="run")
    parser.add_argument("--log", help="Append a row with this description to eval/experiment_log.md.")
    parser.add_argument("--set", default=str(DEFAULT_SET))
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--raw", default=str(RAW_DIR))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--allow-api", action="store_true")
    args = parser.parse_args()

    set_path = Path(args.set)
    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Index not found: {db_path}. Build it first: .\\run-agent.bat rebuild")
    label = re.sub(r"[^\w.-]+", "-", args.label).strip("-") or "run"
    run_dir = RESULTS_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{label}"

    eval_set = load_eval_set(set_path)
    validation = validate_set.validate(eval_set, Path(args.raw), db_path)
    write_json(run_dir / "validation.json", validation)
    for line in validation["errors"]:
        print("ERROR  ", line)
    for line in validation["warnings"]:
        print("WARN   ", line)
    if not validation["ok"]:
        raise SystemExit("Evaluation set is invalid; fix the errors above before scoring.")

    runnable = set(validation["runnable_ids"])
    collect_module.collect([item for item in eval_set if item["id"] in runnable], db_path, run_dir, args.k, args.allow_api)
    result, check = score_module.score_run(run_dir, set_path, args.k, 0.5, 0.5)
    print()
    print((run_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"Results: {run_dir}")
    if not check["ok"]:
        raise SystemExit("Scorer self-check FAILED: corrupted gold keywords still scored as hits.")
    if args.log:
        append_log(run_dir, result["summary"], score_module.read_json(run_dir / "run_meta.json"), args.log)
        print(f"Logged to {EXPERIMENT_LOG}")


if __name__ == "__main__":
    main()
