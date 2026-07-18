#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install the Argos offline English-to-Chinese translation model."""

from __future__ import annotations

import json
import sys

import project_runtime


def main() -> int:
    project_runtime.apply_project_runtime_env()
    try:
        from argostranslate import package, translate
    except ImportError:
        print("Missing argostranslate. Run: python -m pip install -r requirements.txt", file=sys.stderr)
        return 2

    installed = translate.get_installed_languages()
    english = next((language for language in installed if language.code == "en"), None)
    if english and any(translation.to_lang.code == "zh" for translation in english.translations_from):
        print(json.dumps({"installed": True, "from": "en", "to": "zh", "changed": False}, ensure_ascii=False, indent=2))
        return 0

    print("Downloading Argos offline translation model: en -> zh")
    package.update_package_index()
    available = package.get_available_packages()
    model = next((item for item in available if item.from_code == "en" and item.to_code == "zh"), None)
    if model is None:
        print("English-to-Chinese Argos model was not found in the package index.", file=sys.stderr)
        return 3

    path = model.download()
    package.install_from_path(path)
    print(
        json.dumps(
            {
                "installed": True,
                "from": "en",
                "to": "zh",
                "changed": True,
                "package_version": model.package_version,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
