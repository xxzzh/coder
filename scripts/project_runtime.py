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
    # transformers otherwise starts a background download of a safetensors conversion whenever a
    # .bin-only model such as BAAI/bge-m3 loads (~2.3 GB), and the process cannot exit until it ends.
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")
    return paths


def reuse_local_stanza_resources() -> None:
    """Keep offline English-to-Chinese translation offline.

    Argos splits sentences with Stanza, whose pipelines re-download resources.json each time
    they start (DownloadMethod.DOWNLOAD_RESOURCES); without network that fails and nothing gets
    translated. Reuse the copy stored with the installed model instead: Stanza then only goes
    online for files that are missing, which the translation installer fetches once at install time.
    """
    try:
        import stanza  # type: ignore
        from stanza.pipeline.core import DownloadMethod  # type: ignore
    except ImportError:
        return
    original = stanza.Pipeline
    if getattr(original, "_lka_reuses_resources", False):
        return

    def pipeline(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("download_method", DownloadMethod.REUSE_RESOURCES)
        return original(*args, **kwargs)

    pipeline._lka_reuses_resources = True  # type: ignore[attr-defined]
    stanza.Pipeline = pipeline


PROJECT_ENV = apply_project_runtime_env()

