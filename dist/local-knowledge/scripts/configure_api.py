#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive API setup with provider presets, model discovery, and validation."""

from __future__ import annotations

import getpass
import json

import project_runtime
from api_providers import (
    default_base_url,
    discover_models,
    provider_options,
    read_env,
    validate_chat_model,
    write_env,
)


def choose(prompt: str, options: list[str], default: int = 1) -> str:
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")
    raw = input(f"{prompt} [{default}]: ").strip()
    selected = int(raw or default)
    if selected < 1 or selected > len(options):
        raise ValueError("选择超出范围。")
    return options[selected - 1]


def main() -> int:
    project_runtime.apply_project_runtime_env()
    current = read_env()
    providers = provider_options()
    print("配置答案精炼 API")
    provider_name = choose("请选择 API 提供商", [item["name"] for item in providers])
    provider = next(item["id"] for item in providers if item["name"] == provider_name)
    default_url = default_base_url(provider)
    base_url = input(f"API base URL [{default_url}]: ").strip() or default_url
    api_key = getpass.getpass("API key（输入时不会显示）: ").strip()
    if not api_key:
        print("未输入 API key，配置未变更。")
        return 1

    discovery = discover_models(provider, base_url, api_key)
    if not discovery.get("ok"):
        print(json.dumps(discovery, ensure_ascii=False, indent=2))
        return 2
    if discovery.get("warning"):
        print(discovery["warning"])
    models = list(discovery["models"])
    model = choose("请选择用于引用去重和中文精炼的模型", models)
    validation = validate_chat_model(provider, base_url, api_key, model)
    if not validation.get("ok"):
        print("模型调用验证失败，配置未保存：")
        print(validation.get("error"))
        return 3

    values = {
        **current,
        "LKA_USE_API": "true",
        "LKA_API_PROVIDER": provider,
        "LKA_API_BASE_URL": base_url.rstrip("/"),
        "LKA_API_KEY": api_key,
        "LKA_API_MODEL": model,
        "LKA_API_TIMEOUT_SECONDS": current.get("LKA_API_TIMEOUT_SECONDS", "20"),
    }
    write_env(values)
    print(
        json.dumps(
            {
                "saved": True,
                "provider": provider,
                "base_url": base_url.rstrip("/"),
                "model": model,
                "validation": validation.get("answer"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
