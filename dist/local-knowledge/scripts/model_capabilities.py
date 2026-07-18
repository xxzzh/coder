#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional large-model capabilities for retrieval, embedding, and answer checks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from api_providers import api_request, chat_completion, default_base_url, infer_provider, token_plan_rejected


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
DEFAULT_EMBEDDING_DIMENSIONS = 1024


def read_env(path: Path = ENV_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def runtime_config(env_path: Path = ENV_PATH) -> dict[str, str]:
    config = read_env(env_path)
    for key in (
        "LKA_EMBEDDING_ENABLED",
        "LKA_EMBEDDING_PROVIDER",
        "LKA_EMBEDDING_BASE_URL",
        "LKA_EMBEDDING_API_KEY",
        "LKA_EMBEDDING_MODEL",
        "LKA_EMBEDDING_DIMENSIONS",
        "LKA_EMBEDDING_BATCH_SIZE",
        "LKA_VECTOR_BACKEND",
        "LKA_RERANK_ENABLED",
        "LKA_LLM_CHUNKING_ENABLED",
        "LKA_LLM_QUERY_ROUTING_ENABLED",
        "LKA_API_PROVIDER",
        "LKA_API_BASE_URL",
        "LKA_API_KEY",
        "LKA_API_MODEL",
        "LKA_API_TIMEOUT_SECONDS",
    ):
        if os.environ.get(key):
            config[key] = os.environ[key]
    return config


def enabled(config: dict[str, str], key: str, default: bool = False) -> bool:
    raw = config.get(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def embedding_config(config: dict[str, str] | None = None) -> dict[str, Any]:
    config = config or runtime_config()
    provider = infer_provider(
        config.get("LKA_EMBEDDING_BASE_URL", ""),
        config.get("LKA_EMBEDDING_API_KEY", ""),
        config.get("LKA_EMBEDDING_PROVIDER", "openai-compatible"),
    )
    base_url = config.get("LKA_EMBEDDING_BASE_URL", "") or default_base_url(provider)
    api_key = config.get("LKA_EMBEDDING_API_KEY", "")
    model = config.get("LKA_EMBEDDING_MODEL", "")
    try:
        dimensions = int(config.get("LKA_EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS) or DEFAULT_EMBEDDING_DIMENSIONS)
    except ValueError:
        dimensions = DEFAULT_EMBEDDING_DIMENSIONS
    try:
        batch_size = max(1, min(int(config.get("LKA_EMBEDDING_BATCH_SIZE", "32") or 32), 128))
    except ValueError:
        batch_size = 32
    usable = (
        enabled(config, "LKA_EMBEDDING_ENABLED")
        and bool(base_url)
        and bool(api_key)
        and bool(model)
        and not token_plan_rejected(base_url, api_key)
    )
    return {
        "enabled": enabled(config, "LKA_EMBEDDING_ENABLED"),
        "usable": usable,
        "provider": provider,
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "model": model,
        "requested_dimensions": dimensions,
        "batch_size": batch_size,
        "backend": config.get("LKA_VECTOR_BACKEND", "faiss").strip().lower() or "faiss",
    }


def embedding_status(config: dict[str, str] | None = None) -> dict[str, Any]:
    cfg = embedding_config(config)
    public = {key: value for key, value in cfg.items() if key != "api_key"}
    if not cfg["enabled"]:
        return {**public, "ready": False, "reason": "disabled"}
    if not cfg["usable"]:
        return {**public, "ready": False, "reason": "missing_or_invalid_embedding_config"}
    return {**public, "ready": True, "reason": "ready"}


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


def normalize_dense_vector(values: list[float]) -> list[float]:
    total = sum(value * value for value in values) ** 0.5
    if not total:
        return values
    return [float(value) / total for value in values]


def embed_texts(
    texts: list[str],
    config: dict[str, str] | None = None,
    retries: int = 2,
    sleep_seconds: float = 0.4,
) -> dict[str, Any]:
    cfg = embedding_config(config)
    if not cfg["usable"]:
        return {"ok": False, "vectors": [], "error": "embedding_not_configured", "config": embedding_status(config)}

    vectors: list[list[float]] = []
    base_url = str(cfg["base_url"]).rstrip("/")
    timeout = int((config or {}).get("LKA_API_TIMEOUT_SECONDS", "20") or 20)
    for start in range(0, len(texts), int(cfg["batch_size"])):
        batch = texts[start : start + int(cfg["batch_size"])]
        payload: dict[str, Any] = {"model": cfg["model"], "input": batch}
        if cfg.get("requested_dimensions"):
            payload["dimensions"] = cfg["requested_dimensions"]
        last_error = ""
        for attempt in range(retries + 1):
            try:
                data = api_request(
                    "POST",
                    f"{base_url}/embeddings",
                    str(cfg["provider"]),
                    str(cfg["api_key"]),
                    payload=payload,
                    timeout=timeout,
                )
                items = sorted(data.get("data", []), key=lambda item: int(item.get("index", 0)))
                batch_vectors = [normalize_dense_vector([float(value) for value in item.get("embedding", [])]) for item in items]
                if len(batch_vectors) != len(batch):
                    raise RuntimeError(f"embedding count mismatch: expected {len(batch)}, got {len(batch_vectors)}")
                vectors.extend(batch_vectors)
                last_error = ""
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < retries:
                    time.sleep(sleep_seconds * (attempt + 1))
        if last_error:
            return {"ok": False, "vectors": [], "error": last_error, "config": embedding_status(config)}

    dimensions = len(vectors[0]) if vectors else 0
    return {
        "ok": True,
        "vectors": vectors,
        "dimension": dimensions,
        "model": cfg["model"],
        "provider": cfg["provider"],
        "backend": cfg["backend"],
    }


def classify_query(question: str, config: dict[str, str] | None = None) -> dict[str, Any]:
    text = question.strip()
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9&./_-]{1,12}", text):
        return {"type": "abbreviation", "confidence": 0.9, "rewrites": [text]}
    if re.search(r"站位|测试项|测试子项|门限|判定|治具|设备|N972|FFT", text, flags=re.IGNORECASE):
        return {"type": "excel_table", "confidence": 0.85, "rewrites": [text]}
    if re.search(r"代码|函数|枚举|LogMsgType|Enum|框架|流程|机制", text, flags=re.IGNORECASE):
        return {"type": "code_or_document", "confidence": 0.7, "rewrites": [text]}
    if enabled(config or runtime_config(), "LKA_LLM_QUERY_ROUTING_ENABLED"):
        # Keep this first implementation deterministic; model routing can replace this shape later.
        return {"type": "general_rag", "confidence": 0.55, "rewrites": [text]}
    return {"type": "general_rag", "confidence": 0.5, "rewrites": [text]}


def rerank_chunks(question: str, chunks: list[dict[str, Any]], config: dict[str, str] | None = None) -> dict[str, Any]:
    config = config or runtime_config()
    if not enabled(config, "LKA_RERANK_ENABLED") or not chunks:
        return {"ok": False, "reason": "disabled_or_empty", "ordered_chunk_ids": []}
    if not (
        config.get("LKA_USE_API", "false").lower() == "true"
        and config.get("LKA_API_KEY")
        and config.get("LKA_API_MODEL")
    ):
        return {"ok": False, "reason": "chat_model_not_configured", "ordered_chunk_ids": []}

    candidates = []
    for index, chunk in enumerate(chunks[:12], start=1):
        candidates.append(
            {
                "rank": index,
                "chunk_id": chunk.get("chunk_id"),
                "text": str(chunk.get("text", ""))[:900],
            }
        )
    prompt = (
        "请按与问题的相关性从高到低重排候选证据。只返回 JSON 数组，元素为 chunk_id 字符串。\n"
        f"问题：{question}\n候选：{json.dumps(candidates, ensure_ascii=False)}"
    )
    try:
        answer = chat_completion(
            config.get("LKA_API_PROVIDER", "openai-compatible"),
            config.get("LKA_API_BASE_URL", ""),
            config.get("LKA_API_KEY", ""),
            config.get("LKA_API_MODEL", ""),
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=int(config.get("LKA_API_TIMEOUT_SECONDS", "20") or 20),
        )
        match = re.search(r"\[[\s\S]*\]", answer)
        ordered = json.loads(match.group(0) if match else answer)
        ordered_ids = [str(item) for item in ordered if isinstance(item, str)]
        return {"ok": bool(ordered_ids), "ordered_chunk_ids": ordered_ids, "raw": answer}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc), "ordered_chunk_ids": []}


def evaluate_answer(question: str, answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    if sources and not re.search(r"\[\d+\]", answer):
        issues.append("answer_missing_citation_marker")
    if not sources and "没有找到足够依据" not in answer and "没有足够依据" not in answer:
        issues.append("missing_no_evidence_guidance")
    if question.strip() and len(answer.strip()) < 4:
        issues.append("answer_too_short")
    return {"ok": not issues, "issues": issues}

