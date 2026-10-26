#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install the Argos offline English-to-Chinese translation model."""

from __future__ import annotations

import json
import sys

import project_runtime

WARM_UP_SENTENCE = "Offline translation is ready."


def warm_up(translate) -> str:  # noqa: ANN001
    """Translate once while online, so Stanza fetches the sentence-splitting files the model does
    not ship. Later translations reuse them and work without network."""
    project_runtime.reuse_local_stanza_resources()
    return str(translate.translate(WARM_UP_SENTENCE, "en", "zh")).strip()


def main() -> int:
    project_runtime.apply_project_runtime_env()
    try:
        from argostranslate import package, translate
    except ImportError:
        print("Missing argostranslate. Run: python -m pip install -r requirements.txt", file=sys.stderr)
        return 2

    installed = translate.get_installed_languages()
    english = next((language for language in installed if language.code == "en"), None)
    result: dict[str, object] = {"installed": True, "from": "en", "to": "zh", "changed": False}
    if not (english and any(translation.to_lang.code == "zh" for translation in english.translations_from)):
        print("Downloading Argos offline translation model: en -> zh")
        package.update_package_index()
        available = package.get_available_packages()
        model = next((item for item in available if item.from_code == "en" and item.to_code == "zh"), None)
        if model is None:
            print("English-to-Chinese Argos model was not found in the package index.", file=sys.stderr)
            return 3
        package.install_from_path(model.download())
        result.update({"changed": True, "package_version": model.package_version})

    try:
        result["warm_up"] = {"en": WARM_UP_SENTENCE, "zh": warm_up(translate)}
    except Exception as exc:  # noqa: BLE001
        print(f"The model is installed, but a test translation failed: {exc}", file=sys.stderr)
        return 4
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
