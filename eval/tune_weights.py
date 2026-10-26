#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grid-search the hybrid rerank weights (semantic / keyword / FTS) against the evaluation set.

Candidate generation (FAISS, FTS, keyword pools) does not depend on the weights; they only order
the merged pool. So each question's full candidate pool is captured once with its component
scores and re-scored offline for every weight combination, giving exactly what search() would
return with those weights.

Guarding against overfitting a small set:
- questions are split into two folds, stratified by category
- a setting qualifies only if it improves MRR on both folds and lowers neither Hit@1 nor Hit@5 on either fold,
  compared with the current weights
- the chosen setting maximises the smaller of the two fold gains
- the best setting of each fold is also reported on the other fold

Usage:
    python eval/tune_weights.py            # report only
    python eval/tune_weights.py --step 0.05
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from eval_lib import (
    DB_PATH,
    DEFAULT_SET,
    POSITIVE_CATEGORIES,
    RAW_DIR,
    RESULTS_DIR,
    ROOT,
    SCRIPTS_DIR,
    contains,
    load_eval_set,
    mean,
    norm,
    required_hits,
    use_utf8_stdout,
    write_json,
)

sys.path.insert(0, str(SCRIPTS_DIR))

import query_knowledge_base as qkb  # noqa: E402
from kb import retrieval as kb_retrieval  # noqa: E402

import collect  # noqa: E402
import validate_set  # noqa: E402

K = 5


def capture_pool(db_path: Path, question: str, config: dict[str, str]) -> list[dict[str, Any]]:
    """The whole candidate pool for a question, after search()'s own post-processing.

    rerank() is made to return every candidate instead of the top few, so search() applies its
    section/table context expansion to all of them, exactly as it would to the rows it returns.
    Hits are then judged on the same text the evaluation sees.
    """
    original = kb_retrieval.rerank

    def full_rerank(rows, question_text, limit, config_arg=None):  # noqa: ANN001
        return original(rows, question_text, 10**6, config_arg)

    kb_retrieval.rerank = full_rerank  # search() looks rerank up in kb.retrieval
    try:
        rows, _ = qkb.search(db_path, question, K, config)
    finally:
        kb_retrieval.rerank = original
    return rows


def build_pools(eval_set: list[dict[str, Any]], db_path: Path, config: dict[str, str]) -> list[dict[str, Any]]:
    pools = []
    for position, item in enumerate(eval_set, start=1):
        keywords = item["gold_chunk_keywords"]
        need = required_hits(len(keywords), 0.5)
        list_question = qkb.enumeration_or_list_question(qkb.retrieval_question_with_aliases(item["question"]))
        rows = []
        for row in capture_pool(db_path, item["question"], config):
            text = str(row.get("text", ""))
            rows.append(
                {
                    "semantic": float(row.get("semantic_score", 0.0) or 0.0),
                    "keyword": float(row.get("keyword_score", 0.0) or 0.0),
                    "fts": float(row.get("fts_score", 0.0) or 0.0),
                    "quality": float(row.get("information_quality_score", 1.0) or 0.0),
                    "boost": qkb.OVERVIEW_LIST_BOOST if list_question and qkb.TABLE_OVERVIEW_MARKER in text else 1.0,
                    "hit": row.get("file_name") in item["gold_doc_ids"]
                    and sum(contains(norm(text), keyword) for keyword in keywords) >= need,
                }
            )
        pools.append({"id": item["id"], "category": item["category"], "rows": rows})
        print(f"[{position}/{len(eval_set)}] {item['id']} pool={len(rows)} gold_in_pool={sum(r['hit'] for r in rows)}", flush=True)
    return pools


def first_hit_rank(pool: dict[str, Any], weights: tuple[float, float, float]) -> int | None:
    semantic, keyword, fts = weights
    scored = sorted(
        pool["rows"],
        key=lambda row: (semantic * row["semantic"] + keyword * row["keyword"] + fts * row["fts"]) * row["quality"] * row["boost"],
        reverse=True,
    )
    return next((rank for rank, row in enumerate(scored[:K], start=1) if row["hit"]), None)


def metrics(pools: list[dict[str, Any]], weights: tuple[float, float, float]) -> dict[str, float]:
    ranks = [first_hit_rank(pool, weights) for pool in pools]
    return {
        "hit@1": mean(float(rank == 1) for rank in ranks) or 0.0,
        "hit@5": mean(float(rank is not None) for rank in ranks) or 0.0,
        "mrr": mean(1.0 / rank if rank else 0.0 for rank in ranks) or 0.0,
    }


def stratified_folds(pools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_a, fold_b = [], []
    for category in sorted(POSITIVE_CATEGORIES):
        members = sorted((pool for pool in pools if pool["category"] == category), key=lambda pool: pool["id"])
        for index, pool in enumerate(members):
            (fold_a if index % 2 == 0 else fold_b).append(pool)
    return fold_a, fold_b


def grid(step: float) -> list[tuple[float, float, float]]:
    count = round(1 / step)
    combos = []
    for semantic_steps in range(count + 1):
        for keyword_steps in range(count + 1 - semantic_steps):
            semantic = round(semantic_steps * step, 4)
            keyword = round(keyword_steps * step, 4)
            combos.append((semantic, keyword, round(1 - semantic - keyword, 4)))
    return combos


def fmt(weights: tuple[float, float, float]) -> str:
    return f"sem {weights[0]:.2f} / kw {weights[1]:.2f} / fts {weights[2]:.2f}"


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser(description="Tune hybrid rerank weights with two-fold validation.")
    parser.add_argument("--set", default=str(DEFAULT_SET))
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    eval_set = load_eval_set(Path(args.set))
    validation = validate_set.validate(eval_set, RAW_DIR, db_path)
    runnable = set(validation["runnable_ids"])
    positives = [item for item in eval_set if item["id"] in runnable and item["category"] in POSITIVE_CATEGORIES]
    os.chdir(ROOT)
    config = collect.isolate_system(allow_api=False)
    pools = build_pools(positives, db_path, config)

    current = (qkb.RERANK_WEIGHTS["semantic"], qkb.RERANK_WEIGHTS["keyword"], qkb.RERANK_WEIGHTS["fts"])
    fold_a, fold_b = stratified_folds(pools)
    base = {"all": metrics(pools, current), "a": metrics(fold_a, current), "b": metrics(fold_b, current)}
    results = []
    for weights in grid(args.step):
        scores = {"all": metrics(pools, weights), "a": metrics(fold_a, weights), "b": metrics(fold_b, weights)}
        gain_a = scores["a"]["mrr"] - base["a"]["mrr"]
        gain_b = scores["b"]["mrr"] - base["b"]["mrr"]
        # Hit@1 is protected as well as Hit@5: local answers cite the rank-1 chunk as their main evidence.
        qualifies = (
            gain_a > 0
            and gain_b > 0
            and all(scores[fold][metric] >= base[fold][metric] for fold in ("a", "b") for metric in ("hit@1", "hit@5"))
        )
        results.append({"weights": weights, **scores, "gain_a": gain_a, "gain_b": gain_b, "qualifies": qualifies})

    def distance(weights: tuple[float, float, float]) -> float:
        return sum(abs(left - right) for left, right in zip(weights, current))

    qualifying = sorted(
        (row for row in results if row["qualifies"]),
        key=lambda row: (-min(row["gain_a"], row["gain_b"]), distance(row["weights"])),
    )
    best_a = max(results, key=lambda row: (row["a"]["mrr"], -distance(row["weights"])))
    best_b = max(results, key=lambda row: (row["b"]["mrr"], -distance(row["weights"])))

    print(f"\nfolds: A={len(fold_a)} B={len(fold_b)} questions, {len(results)} weight settings (step {args.step})")
    print(f"current  {fmt(current)} | all MRR {base['all']['mrr']:.3f} Hit@1 {base['all']['hit@1']:.3f} Hit@5 {base['all']['hit@5']:.3f} | A {base['a']['mrr']:.3f} B {base['b']['mrr']:.3f}")
    print(f"best on A {fmt(best_a['weights'])}: A {best_a['a']['mrr']:.3f} -> on B {best_a['b']['mrr']:.3f} (B baseline {base['b']['mrr']:.3f})")
    print(f"best on B {fmt(best_b['weights'])}: B {best_b['b']['mrr']:.3f} -> on A {best_b['a']['mrr']:.3f} (A baseline {base['a']['mrr']:.3f})")
    print(f"settings improving MRR on both folds without losing Hit@1 or Hit@5: {len(qualifying)}")
    for row in qualifying[:8]:
        print(
            f"  {fmt(row['weights'])} | all MRR {row['all']['mrr']:.3f} Hit@1 {row['all']['hit@1']:.3f} Hit@5 {row['all']['hit@5']:.3f}"
            f" | gain A {row['gain_a']:+.3f} B {row['gain_b']:+.3f}"
        )
    chosen = qualifying[0] if qualifying else None
    print("\nrecommended:", fmt(chosen["weights"]) if chosen else "keep current weights (nothing improves both folds)")

    out = RESULTS_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-weights"
    write_json(
        out / "report.json",
        {
            "current": current,
            "baseline": base,
            "folds": {"a": [pool["id"] for pool in fold_a], "b": [pool["id"] for pool in fold_b]},
            "best_on_a": best_a,
            "best_on_b": best_b,
            "qualifying": qualifying,
            "recommended": chosen["weights"] if chosen else None,
        },
    )
    print(f"report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
