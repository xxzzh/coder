#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configure the high-quality OpenAI-compatible embedding backend."""

from __future__ import annotations

import getpass
import json

import project_runtime
from api_providers import default_base_url, read_env, write_env
import model_capabilities


def main() -> int:
    project_runtime.apply_project_runtime_env()
    current = read_env()
    print("配置向量检索 Embedding API")
    print("默认推荐：OpenAI text-embedding-3-large，3072 维，FAISS 本地精确检索。")
    provider = input("Embedding provider [openai]: ").strip() or "openai"
    base_url = input(f"Embedding base URL [{default_base_url(provider)}]: ").strip() or default_base_url(provider)
    api_key = getpass.getpass("Embedding API key（输入时不会显示）: ").strip()
    if not api_key:
        print("未输入 API key，配置未变更。")
        return 1
    model = input(f"Embedding model [{model_capabilities.DEFAULT_EMBEDDING_MODEL}]: ").strip()
    model = model or model_capabilities.DEFAULT_EMBEDDING_MODEL
    dimensions = input(f"Embedding dimensions [{model_capabilities.DEFAULT_EMBEDDING_DIMENSIONS}]: ").strip()
    dimensions = dimensions or str(model_capabilities.DEFAULT_EMBEDDING_DIMENSIONS)
    values = {
        **current,
        "LKA_EMBEDDING_ENABLED": "true",
        "LKA_EMBEDDING_PROVIDER": provider,
        "LKA_EMBEDDING_BASE_URL": base_url.rstrip("/"),
        "LKA_EMBEDDING_API_KEY": api_key,
        "LKA_EMBEDDING_MODEL": model,
        "LKA_EMBEDDING_DIMENSIONS": dimensions,
        "LKA_EMBEDDING_BATCH_SIZE": current.get("LKA_EMBEDDING_BATCH_SIZE", "32"),
        "LKA_VECTOR_BACKEND": "faiss",
        "LKA_RERANK_ENABLED": current.get("LKA_RERANK_ENABLED", "false"),
        "LKA_LLM_CHUNKING_ENABLED": current.get("LKA_LLM_CHUNKING_ENABLED", "false"),
        "LKA_LLM_QUERY_ROUTING_ENABLED": current.get("LKA_LLM_QUERY_ROUTING_ENABLED", "false"),
    }
    validation = model_capabilities.embed_texts(["本地知识库 embedding 配置验证"], values, retries=0)
    if not validation.get("ok"):
        print("Embedding 调用验证失败，配置未保存：")
        print(validation.get("error"))
        return 2
    write_env(values)
    print(
        json.dumps(
            {
                "saved": True,
                "provider": provider,
                "base_url": base_url.rstrip("/"),
                "model": model,
                "requested_dimensions": int(dimensions),
                "actual_dimensions": validation.get("dimension"),
                "vector_backend": "faiss",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

