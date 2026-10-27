"""Extraction reports and the pending-review / approval workflow."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from kb_ingest.files import PROCESSED_DIR, raw_file_fingerprint


REVIEW_APPROVALS_FILE = "review_approvals.json"


def load_review_approvals(processed_dir: Path = PROCESSED_DIR) -> dict[str, dict[str, str]]:
    approvals_path = processed_dir / REVIEW_APPROVALS_FILE
    if not approvals_path.exists():
        return {}
    try:
        payload = json.loads(approvals_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    files = payload.get("files", {})
    return files if isinstance(files, dict) else {}


def write_review_approvals(approvals: dict[str, dict[str, str]], processed_dir: Path = PROCESSED_DIR) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    approvals_path = processed_dir / REVIEW_APPROVALS_FILE
    payload = {"files": approvals}
    approvals_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def review_approval_matches(approvals: dict[str, dict[str, str]], rel_path: str, path: Path) -> bool:
    approval = approvals.get(rel_path)
    return bool(approval and approval.get("fingerprint") == raw_file_fingerprint(path))


def extraction_reports(processed_dir: Path = PROCESSED_DIR) -> list[dict[str, Any]]:
    report_path = processed_dir / "extraction_report.jsonl"
    if not report_path.exists():
        return []
    reports: list[dict[str, Any]] = []
    with report_path.open("r", encoding="utf-8") as report_in:
        for line in report_in:
            if not line.strip():
                continue
            try:
                reports.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return reports


def pending_review_reports(raw_dir: Path, processed_dir: Path = PROCESSED_DIR) -> list[dict[str, Any]]:
    approvals = load_review_approvals(processed_dir)
    pending: list[dict[str, Any]] = []
    for report in extraction_reports(processed_dir):
        rel_path = str(report.get("file_path", ""))
        path = (raw_dir / rel_path).resolve()
        if not report.get("requires_review") or not path.exists():
            continue
        if review_approval_matches(approvals, rel_path, path):
            continue
        pending.append(report)
    return pending


def approve_review_file(raw_dir: Path, processed_dir: Path, rel_path: str) -> dict[str, str]:
    rel_path = rel_path.replace("\\", "/").strip("/")
    raw_root = raw_dir.resolve()
    path = (raw_root / rel_path).resolve()
    if raw_root not in path.parents or not path.is_file():
        raise ValueError("待复核文件不存在。")
    pending_paths = {str(report.get("file_path", "")) for report in pending_review_reports(raw_dir, processed_dir)}
    if rel_path not in pending_paths:
        raise ValueError("该文件当前不需要复核，或文件内容已经变化，请刷新列表。")
    approvals = load_review_approvals(processed_dir)
    approval = {
        "fingerprint": raw_file_fingerprint(path),
        "approved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    approvals[rel_path] = approval
    write_review_approvals(approvals, processed_dir)
    return {"file_path": rel_path, **approval}


def extraction_report(path: Path, file_type: str) -> dict[str, Any]:
    return {
        "file": path.name,
        "file_path": str(path),
        "file_type": file_type,
        "method": [],
        "text_chars": 0,
        "warnings": [],
        "requires_review": False,
    }


def write_extraction_report(
    processed_dir: Path,
    reports: list[dict[str, Any]],
    current_paths: set[str] | None = None,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_path = processed_dir / "extraction_report.jsonl"
    merged: dict[str, dict[str, Any]] = {}
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as report_in:
            for line in report_in:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("file_path"):
                    merged[str(item["file_path"])] = item
    for report in reports:
        if report.get("file_path"):
            merged[str(report["file_path"])] = report
    if current_paths is not None:
        merged = {file_path: report for file_path, report in merged.items() if file_path in current_paths}
    with report_path.open("w", encoding="utf-8") as report_out:
        for report in (merged[file_path] for file_path in sorted(merged)):
            report_out.write(json.dumps(report, ensure_ascii=False) + "\n")
