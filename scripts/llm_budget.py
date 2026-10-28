#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cost control for chat-model calls: output budget and model tier per task, a cache-friendly
message layout, and a usage log with cache hit rate and estimated cost.

DeepSeek bills a call as cache-hit input x a very low price + cache-miss input x the normal price
+ output x the highest price, and its context cache only matches an exact prefix starting at the
first token. So the fixed part of every prompt goes first and stays byte-identical, everything that
changes per request goes last, and the usage of every call is logged so the hit rate can be checked:

    python scripts/llm_budget.py report [--since 2026-10-27T12:00] [--log logs/api_usage.jsonl]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
DEFAULT_USAGE_LOG = ROOT / "logs" / "api_usage.jsonl"

# Output budget and thinking per task. Tasks that only classify or pick (rerank, the connection
# test) get a small budget and never think; tasks that write the answer get room for a complete
# one. LKA_MAX_TOKENS_<TASK> (e.g. LKA_MAX_TOKENS_SUMMARY) overrides the budget.
# thinking "config" follows LKA_API_THINKING (default disabled: thinking tokens are billed as output).
TASKS: dict[str, dict[str, Any]] = {
    "agent": {"max_tokens": 3000, "thinking": "config", "pro_allowed": True},  # judge evidence, search again or write the cited answer
    "summary": {"max_tokens": 2000, "thinking": "config", "pro_allowed": True},  # grounded answer from citations or web pages
    "rerank": {"max_tokens": 400, "thinking": "disabled", "pro_allowed": False},  # a JSON array of chunk ids
    "connection_test": {"max_tokens": 16, "thinking": "disabled", "pro_allowed": False},
    "judge": {"max_tokens": 300, "thinking": "disabled", "pro_allowed": False},  # eval/judge.py: a JSON verdict per answer
    "chat": {"max_tokens": 2000, "thinking": "config", "pro_allowed": False},  # any other caller
}

# Model tier: every task uses LKA_API_MODEL (Flash) unless LKA_API_PRO_TASKS names it, e.g.
# LKA_API_PRO_TASKS=agent. The Pro model is LKA_API_PRO_MODEL, or the provider's default below.
DEFAULT_PRO_MODELS = {"deepseek": "deepseek-v4-pro"}

# USD per million tokens, off-peak. LKA_PRICE_<MODEL> = "cache_hit,cache_miss,output" adds or
# overrides a model, e.g. LKA_PRICE_DEEPSEEK_V4_PRO=0.01,0.5,2.0 (the model name upper-cased,
# every other character as "_").
PRICES: dict[str, dict[str, float]] = {
    "deepseek-flash": {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.60},
}
# Peak hours, Beijing time on weekdays, cost double. Public holidays are not modelled.
BEIJING = timezone(timedelta(hours=8))
PEAK_HOURS = ((9, 12), (14, 18))
PEAK_MULTIPLIER = 2.0

_LOG_LOCK = threading.Lock()


def settings(env_path: Path = ENV_PATH) -> dict[str, str]:
    """.env values, with LKA_* environment variables taking precedence."""
    values: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({key: value for key, value in os.environ.items() if key.startswith("LKA_") and value})
    return values


def int_setting(config: dict[str, str], key: str, default: int, low: int = 1, high: int = 1_000_000) -> int:
    try:
        return max(low, min(high, int(config.get(key, "") or default)))
    except ValueError:
        return default


def configured_thinking(config: dict[str, str]) -> str:
    # DeepSeek thinks by default, which multiplies latency and output tokens; the answers here work without it.
    value = str(config.get("LKA_API_THINKING", "disabled")).strip().lower()
    return value if value in {"enabled", "disabled"} else "disabled"


def pro_tasks(config: dict[str, str]) -> list[str]:
    names = [item.strip().lower() for item in str(config.get("LKA_API_PRO_TASKS", "")).split(",")]
    return [name for name in names if name in TASKS and TASKS[name]["pro_allowed"]]


def model_for_task(config: dict[str, str], task: str) -> str:
    base = config.get("LKA_API_MODEL", "")
    if task not in pro_tasks(config):
        return base
    provider = str(config.get("LKA_API_PROVIDER", "")).strip().lower()
    return config.get("LKA_API_PRO_MODEL") or DEFAULT_PRO_MODELS.get(provider) or base


def task_settings(config: dict[str, str], task: str) -> dict[str, Any]:
    profile = TASKS.get(task, TASKS["chat"])
    thinking = configured_thinking(config) if profile["thinking"] == "config" else profile["thinking"]
    return {
        "model": model_for_task(config, task),
        "max_tokens": int_setting(config, f"LKA_MAX_TOKENS_{task.upper()}", profile["max_tokens"], 1, 32_000),
        "thinking": thinking,
    }


def stable_json(value: Any) -> str:
    """The same value always serializes to the same bytes: sorted keys, fixed separators."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_messages(
    system: str,
    dynamic: str,
    few_shot: list[dict[str, str]] | None = None,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Fixed prefix first (system prompt, then few-shot examples), then the conversation so far
    exactly as sent before, then this turn's dynamic content (question, evidence, time) last.

    A follow-up turn passes the previous turn's messages (after the system prompt) plus the model's
    reply as history, so every earlier byte is a cache hit and only the new turn is billed in full.
    """
    messages = [{"role": "system", "content": system}]
    messages.extend({"role": item["role"], "content": item["content"]} for item in few_shot or [])
    messages.extend({"role": item["role"], "content": item["content"]} for item in history or [])
    messages.append({"role": "user", "content": dynamic})
    return messages


def clip(text: str, max_chars: int) -> str:
    """Shorten text fed back to the model; long sections and pages keep their beginning."""
    text = str(text)
    return text if len(text) <= max_chars else text[: max(0, max_chars - 1)].rstrip() + "…"


def usage_metrics(usage: dict[str, Any]) -> dict[str, Any]:
    prompt = int(usage.get("prompt_tokens") or 0)
    hit = int(usage.get("prompt_cache_hit_tokens") or 0)
    raw_miss = usage.get("prompt_cache_miss_tokens")
    miss = int(raw_miss) if isinstance(raw_miss, int) else max(0, prompt - hit)
    output = int(usage.get("completion_tokens") or 0)
    details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
    input_tokens = hit + miss
    return {
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "output_tokens": output,
        "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
        "cache_hit_rate": round(hit / input_tokens, 4) if input_tokens else None,
        "output_input_ratio": round(output / input_tokens, 4) if input_tokens else None,
    }


def is_peak(moment: datetime) -> bool:
    local = moment.astimezone(BEIJING)
    return local.weekday() < 5 and any(start <= local.hour < end for start, end in PEAK_HOURS)


def model_prices(model: str, config: dict[str, str]) -> dict[str, float] | None:
    override = config.get("LKA_PRICE_" + re.sub(r"[^A-Z0-9]", "_", model.upper()))
    if override:
        try:
            hit, miss, output = (float(part) for part in override.split(","))
            return {"cache_hit": hit, "cache_miss": miss, "output": output}
        except ValueError:
            pass
    return PRICES.get(model)


def estimate_cost_usd(model: str, metrics: dict[str, Any], moment: datetime, config: dict[str, str]) -> float | None:
    prices = model_prices(model, config)
    if prices is None:
        return None
    base = (
        metrics["cache_hit_tokens"] * prices["cache_hit"]
        + metrics["cache_miss_tokens"] * prices["cache_miss"]
        + metrics["output_tokens"] * prices["output"]
    ) / 1_000_000
    return round(base * (PEAK_MULTIPLIER if is_peak(moment) else 1.0), 8)


def usage_log_path(config: dict[str, str]) -> Path | None:
    value = str(config.get("LKA_API_USAGE_LOG", "")).strip()
    if value.lower() in {"off", "false", "0", "none"}:
        return None
    if not value:
        return DEFAULT_USAGE_LOG
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def record_call(
    task: str,
    provider: str,
    model: str,
    usage: dict[str, Any],
    elapsed_ms: int,
    moment: datetime | None = None,
    config: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Log one call's token usage, cache hit rate and estimated cost; returns the record."""
    moment = moment or datetime.now(timezone.utc)
    config = settings() if config is None else config
    metrics = usage_metrics(usage)
    record = {
        "time": moment.astimezone(BEIJING).isoformat(timespec="seconds"),
        "task": task,
        "provider": provider,
        "model": model,
        **metrics,
        "peak": is_peak(moment),
        "cost_usd": estimate_cost_usd(model, metrics, moment, config),
        "elapsed_ms": elapsed_ms,
    }
    path = usage_log_path(config)
    if path is not None:
        try:
            with _LOG_LOCK:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(stable_json(record) + "\n")
        except OSError:
            pass  # cost logging must never break an answer
    return record


def read_log(path: Path, since: str = "") -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and str(record.get("time", "")) >= since:
            records.append(record)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def totals(items: list[dict[str, Any]]) -> dict[str, Any]:
        hit = sum(int(item.get("cache_hit_tokens") or 0) for item in items)
        miss = sum(int(item.get("cache_miss_tokens") or 0) for item in items)
        output = sum(int(item.get("output_tokens") or 0) for item in items)
        costs = [item["cost_usd"] for item in items if isinstance(item.get("cost_usd"), (int, float))]
        return {
            "calls": len(items),
            "cache_hit_tokens": hit,
            "cache_miss_tokens": miss,
            "output_tokens": output,
            "cache_hit_rate": round(hit / (hit + miss), 4) if hit + miss else None,
            "output_input_ratio": round(output / (hit + miss), 4) if hit + miss else None,
            "cost_usd": round(sum(costs), 6),
            "unpriced_calls": len(items) - len(costs),
            "peak_calls": sum(1 for item in items if item.get("peak")),
        }

    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_task.setdefault(f"{record.get('task')} / {record.get('model')}", []).append(record)
    return {"total": totals(records), "by_task": {name: totals(items) for name, items in sorted(by_task.items())}}


def percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat-model cost report from the usage log.")
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report", help="cache hit rate, output/input ratio and estimated cost")
    report.add_argument("--since", default="", help="only calls at or after this Beijing time, e.g. 2026-10-27T12:00")
    report.add_argument("--log", default="", help="usage log (default: LKA_API_USAGE_LOG or logs/api_usage.jsonl)")
    report.add_argument("--json", action="store_true")
    args = parser.parse_args()

    path = Path(args.log) if args.log else usage_log_path(settings())
    if path is None:
        print("usage logging is off (LKA_API_USAGE_LOG=off)")
        return 1
    summary = summarize(read_log(path, args.since))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    rows = [("TOTAL", summary["total"]), *summary["by_task"].items()]
    print(f"{path}  since {args.since or 'the beginning'}")
    print(f"{'task / model':34} {'calls':>5} {'hit':>9} {'miss':>8} {'output':>8} {'hit rate':>8} {'out/in':>7} {'cost USD':>10} {'peak':>5}")
    for name, item in rows:
        print(
            f"{name[:34]:34} {item['calls']:>5} {item['cache_hit_tokens']:>9} {item['cache_miss_tokens']:>8} {item['output_tokens']:>8} "
            f"{percent(item['cache_hit_rate']):>8} {item['output_input_ratio'] if item['output_input_ratio'] is not None else '-':>7} "
            f"{item['cost_usd']:>10.6f} {item['peak_calls']:>5}"
        )
    if summary["total"]["unpriced_calls"]:
        print(f"{summary['total']['unpriced_calls']} calls have no price (set LKA_PRICE_<MODEL>=cache_hit,cache_miss,output)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
