#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional large-model capabilities for retrieval, embedding, and answer checks."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from api_providers import api_request, chat_completion, default_base_url, infer_provider, token_plan_rejected
import llm_budget
import project_runtime


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 3072
DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_LOCAL_EMBEDDING_DIMENSIONS = 1024
LOCAL_EMBEDDING_PROVIDERS = {"local", "local-bge-m3", "sentence-transformers", "huggingface"}
LOCAL_EMBEDDING_MODEL: Any = None


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
        "LKA_API_THINKING",
        "LKA_API_PRO_TASKS",
        "LKA_API_PRO_MODEL",
        "LKA_MAX_TOKENS_RERANK",
        "LKA_API_USAGE_LOG",
    ):
        if os.environ.get(key):
            config[key] = os.environ[key]
    return config


def enabled(config: dict[str, str], key: str, default: bool = False) -> bool:
    raw = config.get(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def is_local_embedding_provider(provider: str) -> bool:
    return provider.strip().lower() in LOCAL_EMBEDDING_PROVIDERS


def local_embedding_dependency_status() -> dict[str, Any]:
    try:
        import sentence_transformers  # type: ignore  # noqa: F401
        import torch  # type: ignore  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}
    return {"available": True, "error": None}


def embedding_config(config: dict[str, str] | None = None) -> dict[str, Any]:
    config = config or runtime_config()
    raw_provider = config.get("LKA_EMBEDDING_PROVIDER", "openai").strip() or "openai"
    if is_local_embedding_provider(raw_provider):
        provider = raw_provider.lower()
        base_url = ""
        api_key = ""
        model = config.get("LKA_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL) or DEFAULT_LOCAL_EMBEDDING_MODEL
        default_dimensions = DEFAULT_LOCAL_EMBEDDING_DIMENSIONS
    else:
        provider = infer_provider(
            config.get("LKA_EMBEDDING_BASE_URL", ""),
            config.get("LKA_EMBEDDING_API_KEY", ""),
            raw_provider,
        )
        base_url = config.get("LKA_EMBEDDING_BASE_URL", "") or default_base_url(provider)
        api_key = config.get("LKA_EMBEDDING_API_KEY", "")
        model = config.get("LKA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        default_dimensions = DEFAULT_EMBEDDING_DIMENSIONS
    try:
        dimensions = int(config.get("LKA_EMBEDDING_DIMENSIONS", default_dimensions) or default_dimensions)
    except ValueError:
        dimensions = default_dimensions
    try:
        batch_size = max(1, min(int(config.get("LKA_EMBEDDING_BATCH_SIZE", "32") or 32), 128))
    except ValueError:
        batch_size = 32
    if is_local_embedding_provider(provider):
        dependency = local_embedding_dependency_status()
        usable = enabled(config, "LKA_EMBEDDING_ENABLED") and bool(model) and bool(dependency["available"])
    else:
        dependency = None
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
        "local_dependency": dependency,
    }


def embedding_status(config: dict[str, str] | None = None) -> dict[str, Any]:
    cfg = embedding_config(config)
    public = {key: value for key, value in cfg.items() if key != "api_key"}
    if not cfg["enabled"]:
        return {**public, "ready": False, "reason": "disabled"}
    if not cfg["usable"]:
        reason = "missing_or_invalid_embedding_config"
        if is_local_embedding_provider(str(cfg.get("provider", ""))) and cfg.get("local_dependency"):
            reason = "local_embedding_dependency_missing"
        return {**public, "ready": False, "reason": reason}
    return {**public, "ready": True, "reason": "ready"}


def local_model_revision(model_name: str, cache_folder: Path) -> str:
    """Snapshot commit of a cached Hub model ("local-path" for a directory, "unknown" if not cached)."""
    if Path(model_name).is_dir():
        return "local-path"
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore

        cached = try_to_load_from_cache(model_name, "config.json", cache_dir=str(cache_folder))
    except Exception:  # noqa: BLE001
        return "unknown"
    return Path(cached).parent.name if isinstance(cached, str) else "unknown"


def embedding_model_key(cfg: dict[str, Any]) -> str:
    """Identifies which model produced a vector, so cached vectors are never mixed across models."""
    provider = str(cfg.get("provider", ""))
    model = str(cfg.get("model", ""))
    if is_local_embedding_provider(provider):
        revision = local_model_revision(model, project_runtime.CACHE_DIR / "huggingface")
        return f"local|{model}|{revision}|normalized"
    return f"api|{provider}|{cfg.get('base_url', '')}|{model}|{cfg.get('requested_dimensions', '')}"


QUERY_EMBEDDING_CACHE: "OrderedDict[tuple[str, str], list[float]]" = OrderedDict()
QUERY_EMBEDDING_CACHE_SIZE = 256
QUERY_EMBEDDING_LOCK = threading.Lock()


def embed_query(text: str, config: dict[str, str] | None = None) -> dict[str, Any]:
    """Embed one question, reusing the vector when the same question is embedded again in this process.

    A refused question is searched twice (answer attempt, then suggestion check); each local
    BGE-M3 encode costs ~0.1 s on CPU.
    """
    cfg = embedding_config(config)
    key = (embedding_model_key(cfg), text)
    with QUERY_EMBEDDING_LOCK:
        if key in QUERY_EMBEDDING_CACHE:
            QUERY_EMBEDDING_CACHE.move_to_end(key)
            return {"ok": True, "vectors": [QUERY_EMBEDDING_CACHE[key]], "cached": True}
    result = embed_texts([text], config)
    if result.get("ok") and result.get("vectors"):
        with QUERY_EMBEDDING_LOCK:
            QUERY_EMBEDDING_CACHE[key] = result["vectors"][0]
            while len(QUERY_EMBEDDING_CACHE) > QUERY_EMBEDDING_CACHE_SIZE:
                QUERY_EMBEDDING_CACHE.popitem(last=False)
    return result


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
    if is_local_embedding_provider(str(cfg["provider"])):
        return embed_texts_local(texts, cfg)

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


def local_model_is_cached(model_name: str, cache_folder: Path) -> bool:
    """True when the model's files are already in the project cache (or model_name is a local path)."""
    if Path(model_name).is_dir():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore

        return isinstance(try_to_load_from_cache(model_name, "config.json", cache_dir=str(cache_folder)), str)
    except Exception:  # noqa: BLE001
        return False


def embed_texts_local(texts: list[str], cfg: dict[str, Any]) -> dict[str, Any]:
    global LOCAL_EMBEDDING_MODEL
    project_runtime.apply_project_runtime_env()
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model_name = str(cfg["model"])
        if LOCAL_EMBEDDING_MODEL is None or getattr(LOCAL_EMBEDDING_MODEL, "_lka_model_name", "") != model_name:
            cache_folder = project_runtime.CACHE_DIR / "huggingface"
            # Once cached, never touch the Hub: offline machines would stall here, and transformers
            # otherwise starts a background download of a safetensors conversion (~2.3 GB for bge-m3)
            # that also keeps the process from exiting.
            LOCAL_EMBEDDING_MODEL = SentenceTransformer(
                model_name,
                cache_folder=str(cache_folder),
                local_files_only=local_model_is_cached(model_name, cache_folder),
            )
            setattr(LOCAL_EMBEDDING_MODEL, "_lka_model_name", model_name)
        vectors_raw = LOCAL_EMBEDDING_MODEL.encode(
            texts,
            batch_size=int(cfg["batch_size"]),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = [[float(value) for value in row] for row in vectors_raw.tolist()]
        dimension = len(vectors[0]) if vectors else 0
        return {
            "ok": True,
            "vectors": vectors,
            "dimension": dimension,
            "model": model_name,
            "provider": cfg["provider"],
            "backend": cfg["backend"],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "vectors": [], "error": str(exc), "config": embedding_status()}


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


RERANK_PROMPT = (
    "请按与问题的相关性从高到低重排候选证据。问题和候选以 JSON 给出（question、candidates）。"
    "只返回 JSON 数组，元素为 chunk_id 字符串，不要解释。"
)


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
    # Fixed instruction as the system prompt (a cached prefix), the question and candidates last.
    messages = llm_budget.build_messages(RERANK_PROMPT, llm_budget.stable_json({"question": question, "candidates": candidates}))
    budget = llm_budget.task_settings(config, "rerank")
    try:
        answer = chat_completion(
            config.get("LKA_API_PROVIDER", "openai-compatible"),
            config.get("LKA_API_BASE_URL", ""),
            config.get("LKA_API_KEY", ""),
            budget["model"],
            messages,
            temperature=0.0,
            timeout=int(config.get("LKA_API_TIMEOUT_SECONDS", "20") or 20),
            max_tokens=budget["max_tokens"],
            thinking=budget["thinking"],
            task="rerank",
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
