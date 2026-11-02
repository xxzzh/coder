#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write the decisions of a filled-in review workbook (eval/review_sheet.py) into the eval set.

    python eval/apply_review.py eval/results/<review>.xlsx [--set eval/eval_set.jsonl] [--dry-run]

For each row of the 核对 sheet, by 你的结论:
- 正确: the item is marked review_status "verified".
- 修改: the new reference answer and/or keywords replace the old ones (keywords for a negative
  item are its absent terms), and the item is marked "verified".
- 删除: the item is removed.
- empty: the item is left as it is.

The eval set is copied to eval/results/ before it is written. The changed items are then checked
against the raw documents (eval/validate_set.py): every keyword must be found in the item's
documents, and no absent term of a negative item may appear in any document. A row whose new
keywords fail that check is reported and not applied, so a typo cannot slip into the standard.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_lib import DEFAULT_SET, RAW_DIR, RESULTS_DIR  # noqa: E402
from validate_set import validate  # noqa: E402

SHEET = "核对"
HEADERS = {
    "id": "编号",
    "decision": "你的结论",
    "summary": "修改后的标准答案",
    "keywords": "修改后的关键词（；分隔）",
    "note": "备注",
}


def cell_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def split_keywords(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[；;\n]", text) if part.strip()]


def read_decisions(path: Path) -> dict[str, dict[str, str]]:
    workbook = load_workbook(path, read_only=True)
    if SHEET not in workbook.sheetnames:
        raise SystemExit(f"{path} has no sheet named {SHEET!r}")
    rows = workbook[SHEET].iter_rows(values_only=True)
    header = [cell_text(value) for value in next(rows)]
    missing = [name for name in HEADERS.values() if name not in header]
    if missing:
        raise SystemExit(f"columns not found in {SHEET}: {missing}")
    position = {key: header.index(name) for key, name in HEADERS.items()}
    decisions: dict[str, dict[str, str]] = {}
    for row in rows:
        values = {key: cell_text(row[index]) if index < len(row) else "" for key, index in position.items()}
        if values["id"] and values["decision"]:
            decisions[values["id"]] = values
    return decisions


def apply(items: list[dict], decisions: dict[str, dict[str, str]], raw_dir: Path) -> tuple[list[dict], dict]:
    report: dict[str, list] = {"verified": [], "changed": [], "removed": [], "rejected": [], "unknown": [], "invalid_decision": []}
    by_id = {item["id"]: item for item in items}
    today = time.strftime("%Y-%m-%d")
    for item_id, decision in decisions.items():
        if item_id not in by_id:
            report["unknown"].append(item_id)
            continue
        if decision["decision"] not in ("正确", "修改", "删除"):
            report["invalid_decision"].append(f"{item_id}: {decision['decision']}")
    result: list[dict] = []
    for item in items:
        decision = decisions.get(item["id"])
        if not decision or decision["decision"] not in ("正确", "修改", "删除"):
            result.append(item)
            continue
        if decision["decision"] == "删除":
            report["removed"].append(item["id"])
            continue
        updated = dict(item)
        if decision["decision"] == "修改":
            if decision["summary"]:
                updated["gold_answer_summary"] = decision["summary"]
            if decision["keywords"]:
                field = "absent_terms" if item["category"] == "E" else "gold_chunk_keywords"
                updated[field] = split_keywords(decision["keywords"])
            # New keywords must hold against the documents before they become the standard.
            check = validate([updated], raw_dir)
            prefix = f"{item['id']}: "
            problems = list(
                dict.fromkeys(error.removeprefix(prefix) for error in check["errors"] if error.startswith(prefix))
            )
            if problems:
                report["rejected"].append({"id": item["id"], "problems": problems})
                result.append(item)
                continue
            report["changed"].append(item["id"])
        else:
            report["verified"].append(item["id"])
        updated["review_status"] = "verified"
        updated["reviewed_at"] = today
        if decision["note"]:
            updated["review_note"] = decision["note"]
        result.append(updated)
    return result, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a filled-in review workbook to the eval set.")
    parser.add_argument("workbook")
    parser.add_argument("--set", default=str(DEFAULT_SET))
    parser.add_argument("--raw", default=str(RAW_DIR))
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    args = parser.parse_args()

    set_path = Path(args.set)
    items = [json.loads(line) for line in set_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    decisions = read_decisions(Path(args.workbook))
    result, report = apply(items, decisions, Path(args.raw))
    summary = {
        "decisions_read": len(decisions),
        "verified": len(report["verified"]),
        "changed": report["changed"],
        "removed": report["removed"],
        "rejected": report["rejected"],
        "unknown_ids": report["unknown"],
        "invalid_decisions": report["invalid_decision"],
        "items_before": len(items),
        "items_after": len(result),
        "still_draft": sum(1 for item in result if item.get("review_status") != "verified"),
    }
    if not args.dry_run:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        backup = RESULTS_DIR / f"eval_set.before-review-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        shutil.copy2(set_path, backup)
        set_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in result), encoding="utf-8")
        summary["backup"] = str(backup)
    summary["written"] = not args.dry_run
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if report["rejected"]:
        print("Rejected rows keep their old reference answer; fix the keywords so each one appears in the document, then run again.")


if __name__ == "__main__":
    main()
