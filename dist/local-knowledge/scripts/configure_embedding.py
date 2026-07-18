#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configure the high-quality embedding backend."""

from __future__ import annotations

import json

import project_runtime
from api_providers import default_base_url, read_env, write_env
import model_capabilities


def main() -> int:
    project_runtime.apply_project_runtime_env()
    current = read_env()
    print("配置向量检索 Embedding")
    print("默认推荐：本地 BAAI/bge-m3，1024 维，FAISS 本地精确检索。")
    print("DeepSeek 当前不提供 embeddings；DeepSeek key 继续用于答案生成、rerank 和查询路由。")
    provider = input("Embedding provider [local-bge-m3]: ").strip() or "local-bge-m3"
    local_provider = model_capabilities.is_local_embedding_provider(provider)
    base_url = ""
    api_key = ""
    if not local_provider:
        base_url = input(f"Embedding base URL [{default_base_url(provider)}]: ").strip() or default_base_url(provider)
        api_key = input("Embedding API key: ").strip()
        if not api_key:
            print("未输入 API key，配置未变更。")
            return 1
    default_model = (
        model_capabilities.DEFAULT_LOCAL_EMBEDDING_MODEL
        if local_provider
        else model_capabilities.DEFAULT_EMBEDDING_MODEL
    )
    default_dimensions = (
        model_capabilities.DEFAULT_LOCAL_EMBEDDING_DIMENSIONS
        if local_provider
        else model_capabilities.DEFAULT_EMBEDDING_DIMENSIONS
    )
    model = input(f"Embedding model [{default_model}]: ").strip() or default_model
    dimensions = input(f"Embedding dimensions [{default_dimensions}]: ").strip() or str(default_dimensions)
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
        print("Embedding 验证失败，配置未保存：")
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
