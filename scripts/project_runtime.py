#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project-local runtime directories for downloads and caches."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
CACHE_DIR = TOOLS_DIR / "cache"
VENV_DIR = ROOT / ".venv"


def apply_project_runtime_env() -> dict[str, str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "LKA_PROJECT_ROOT": str(ROOT),
        "LKA_PROJECT_CACHE_DIR": str(CACHE_DIR),
        "PIP_CACHE_DIR": str(CACHE_DIR / "pip"),
        "XDG_CACHE_HOME": str(CACHE_DIR / "xdg-cache"),
        "XDG_DATA_HOME": str(CACHE_DIR / "xdg-data"),
        "HF_HOME": str(CACHE_DIR / "huggingface"),
        "HF_HUB_CACHE": str(CACHE_DIR / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(CACHE_DIR / "huggingface" / "transformers"),
        "TORCH_HOME": str(CACHE_DIR / "torch"),
        "PLAYWRIGHT_BROWSERS_PATH": str(CACHE_DIR / "playwright-browsers"),
        "NPM_CONFIG_CACHE": str(CACHE_DIR / "npm"),
        "ARGOS_PACKAGES_DIR": str(CACHE_DIR / "argos"),
        "ARGOS_PACKAGE_DIR": str(CACHE_DIR / "argos"),
    }
    for key, value in paths.items():
        os.environ.setdefault(key, value)
        Path(value).mkdir(parents=True, exist_ok=True) if key.endswith(("DIR", "HOME", "CACHE", "PATH")) else None
    return paths


PROJECT_ENV = apply_project_runtime_env()

