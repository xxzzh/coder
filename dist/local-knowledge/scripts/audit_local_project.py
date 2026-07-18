#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project-local audit gate for Local Knowledge Base Agent."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "knowledge_base" / "raw"
SUPPORTED_SUFFIXES = {".md", ".txt", ".doc", ".docx", ".pdf", ".xlsx"}
EXECUTABLE_SUFFIXES = {".exe", ".bat", ".cmd", ".ps1", ".vbs", ".js", ".msi", ".dll"}
REQUIRED_FILES = [
    "README.md",
    "run-agent.bat",
    "start.bat",
    "scripts/healthcheck_agent.py",
    "scripts/manage_knowledge_base.py",
    "scripts/query_knowledge_base.py",
    "scripts/test_retrieval_guardrails.py",
]


def run_step(command: list[str], timeout: int = 120) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    display_command = command[:]
    if display_command and Path(display_command[0]) == Path(sys.executable):
        display_command[0] = "python"
    return {
        "command": display_command,
        "exitCode": proc.returncode,
        "stdout": proc.stdout,
        "outputTail": proc.stdout.splitlines()[-20:],
        "errorTail": proc.stderr.splitlines()[-20:],
    }


def raw_files() -> list[Path]:
    if not RAW_DIR.exists():
        return []
    return [path for path in sorted(RAW_DIR.rglob("*")) if path.is_file()]


def check_required_files() -> dict[str, Any]:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    return {"ok": not missing, "missing": missing, "files": REQUIRED_FILES}


def check_raw_files() -> dict[str, Any]:
    files = raw_files()
    unsupported = [
        str(path.relative_to(RAW_DIR)).replace("\\", "/")
        for path in files
        if path.suffix.lower() not in SUPPORTED_SUFFIXES
    ]
    executables = [
        str(path.relative_to(RAW_DIR)).replace("\\", "/")
        for path in files
        if path.suffix.lower() in EXECUTABLE_SUFFIXES
    ]
    return {
        "ok": RAW_DIR.exists() and not unsupported and not executables,
        "raw_dir_exists": RAW_DIR.exists(),
        "supported_files": len(files) - len(unsupported),
        "unsupported_files": unsupported,
        "executables": executables,
    }


def parse_json_from_tail(step: dict[str, Any]) -> dict[str, Any] | None:
    text = str(step.get("stdout", ""))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def main() -> int:
    checks: list[dict[str, Any]] = []

    checks.append({"id": "required-files", **check_required_files()})
    checks.append({"id": "raw-files", **check_raw_files()})

    health = run_step([sys.executable, "scripts/healthcheck_agent.py"])
    health_payload = parse_json_from_tail(health)
    health.pop("stdout", None)
    checks.append(
        {
            "id": "healthcheck",
            "ok": health["exitCode"] == 0,
            "details": health,
            "summary": health_payload,
        }
    )

    guardrails = run_step([sys.executable, "scripts/test_retrieval_guardrails.py"])
    guardrails.pop("stdout", None)
    checks.append(
        {
            "id": "retrieval-guardrails",
            "ok": guardrails["exitCode"] == 0,
            "details": guardrails,
        }
    )

    sources = run_step([sys.executable, "scripts/manage_knowledge_base.py", "sources"])
    sources_payload = parse_json_from_tail(sources)
    sources.pop("stdout", None)
    duplicate_files = (sources_payload or {}).get("duplicate_files", [])
    checks.append(
        {
            "id": "sources",
            "ok": sources["exitCode"] == 0 and not duplicate_files,
            "details": sources,
            "summary": sources_payload,
        }
    )

    ok = all(check["ok"] for check in checks)
    print(
        json.dumps(
            {
                "target": "local-knowledge-agent",
                "targetRoot": str(ROOT),
                "operation": "local-audit",
                "ok": ok,
                "checks": checks,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
