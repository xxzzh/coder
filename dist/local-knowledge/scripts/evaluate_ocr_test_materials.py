#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate extraction against generated OCR test materials."""

from __future__ import annotations

from difflib import SequenceMatcher
import json
import re
from pathlib import Path
from typing import Any

from ingest_knowledge_base import read_document_with_report


ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT / "test_materials" / "ocr" / "ground_truth.jsonl"
RESULTS = ROOT / "test_materials" / "ocr" / "ocr_eval_results.json"


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def phrase_hit(text: str, phrase: str) -> bool:
    text_norm = normalize(text)
    phrase_norm = normalize(phrase)
    if not phrase_norm:
        return True
    if phrase_norm in text_norm:
        return True
    ratio = SequenceMatcher(None, phrase_norm, text_norm).quick_ratio()
    return ratio > 0.88 and len(phrase_norm) >= 12


def evaluate_item(item: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / item["path"]
    text, report = read_document_with_report(path)
    phrases = item.get("required_phrases", [])
    hits = [phrase for phrase in phrases if phrase_hit(text, phrase)]
    return {
        "file": item["file"],
        "kind": item["kind"],
        "description": item["description"],
        "text_chars": len(text),
        "method": report.get("method"),
        "warnings": report.get("warnings"),
        "requires_review": report.get("requires_review"),
        "required_phrases": len(phrases),
        "matched_phrases": len(hits),
        "phrase_recall": round(len(hits) / len(phrases), 4) if phrases else 1.0,
        "missed_phrases": [phrase for phrase in phrases if phrase not in hits],
        "sample_text": text[:500],
    }


def main() -> None:
    if not GROUND_TRUTH.exists():
        raise SystemExit(f"Missing ground truth: {GROUND_TRUTH}")

    items = [json.loads(line) for line in GROUND_TRUTH.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = [evaluate_item(item) for item in items]
    total_required = sum(result["required_phrases"] for result in results)
    total_matched = sum(result["matched_phrases"] for result in results)
    summary = {
        "materials": len(results),
        "total_required_phrases": total_required,
        "total_matched_phrases": total_matched,
        "overall_phrase_recall": round(total_matched / total_required, 4) if total_required else 1.0,
        "requires_review": sum(1 for result in results if result["requires_review"]),
        "warning_files": sum(1 for result in results if result["warnings"]),
    }
    output = {"summary": summary, "results": results}
    RESULTS.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
