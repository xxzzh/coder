"""Paths and configuration: .env loading, runtime overrides, answer-API and agent switches."""

from __future__ import annotations

import os
from pathlib import Path

from api_providers import token_plan_rejected
import llm_budget
import model_capabilities


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/kb/config.py -> project root

# From the install folder, not the current folder: started from anywhere else, the tool would
# read or rebuild another folder's knowledge_base.
DB_PATH = PROJECT_ROOT / "knowledge_base" / "index" / "knowledge.db"

RAW_DIR = PROJECT_ROOT / "knowledge_base" / "raw"

PROCESSED_DIR = PROJECT_ROOT / "knowledge_base" / "processed"

ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_API_TIMEOUT_SECONDS = 20


def load_env_file(path: Path = ENV_PATH) -> dict[str, str]:
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


def runtime_config() -> dict[str, str]:
    config = load_env_file()
    for key in (
        "LKA_USE_API",
        "LKA_API_PROVIDER",
        "LKA_API_BASE_URL",
        "LKA_API_KEY",
        "LKA_API_MODEL",
        "LKA_API_TIMEOUT_SECONDS",
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
        "LKA_AGENT_ENABLED",
        "LKA_AGENT_MAX_ROUNDS",
        "LKA_API_THINKING",
        "LKA_API_PRO_TASKS",
        "LKA_API_PRO_MODEL",
        "LKA_MAX_TOKENS_AGENT",
        "LKA_MAX_TOKENS_SUMMARY",
        "LKA_MAX_TOKENS_RERANK",
        "LKA_AGENT_EVIDENCE_MAX_CHARS",
        "LKA_AGENT_EVIDENCE_MAX_ITEMS",
        "LKA_API_CITATION_MAX_CHARS",
        "LKA_API_USAGE_LOG",
    ):
        if os.environ.get(key):
            config[key] = os.environ[key]
    return config


def api_enabled(config: dict[str, str]) -> bool:
    return (
        config.get("LKA_USE_API", "false").lower() == "true"
        and bool(config.get("LKA_API_KEY"))
        and bool(config.get("LKA_API_MODEL"))
        and api_backend_allowed(config)
    )


def api_backend_allowed(config: dict[str, str]) -> bool:
    base_url = config.get("LKA_API_BASE_URL", "").lower()
    api_key = config.get("LKA_API_KEY", "").lower()
    return not token_plan_rejected(base_url, api_key)


def agent_enabled(config: dict[str, str]) -> bool:
    """The retrieval agent runs whenever the answer API is usable, unless LKA_AGENT_ENABLED=false."""
    return api_enabled(config) and model_capabilities.enabled(config, "LKA_AGENT_ENABLED", default=True)


def agent_max_rounds(config: dict[str, str]) -> int:
    try:
        return max(1, min(3, int(config.get("LKA_AGENT_MAX_ROUNDS", "2") or 2)))
    except ValueError:
        return 2


def api_thinking(config: dict[str, str]) -> str:
    return llm_budget.configured_thinking(config)
