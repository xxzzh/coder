#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install optional runtime dependencies into the project-local virtualenv."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import project_runtime


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=str(ROOT), check=True)


def main() -> int:
    env = project_runtime.apply_project_runtime_env()
    if not REQUIREMENTS.exists():
        print(f"Missing {REQUIREMENTS}", file=sys.stderr)
        return 2
    if not VENV_DIR.exists():
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    python_exe = VENV_DIR / "Scripts" / "python.exe"
    if not python_exe.exists():
        print(f"Virtualenv python not found: {python_exe}", file=sys.stderr)
        return 3
    run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(python_exe), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    print(
        json.dumps(
            {
                "ok": True,
                "venv": str(VENV_DIR),
                "python": str(python_exe),
                "pip_cache": env["PIP_CACHE_DIR"],
                "project_cache": env["LKA_PROJECT_CACHE_DIR"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

