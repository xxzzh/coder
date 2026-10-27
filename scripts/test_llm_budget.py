#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline tests for chat-model cost control: cache-friendly prompts, output budgets, the model
tier switch, request de-duplication and the usage log."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
import time

os.environ["LKA_API_USAGE_LOG"] = "off"  # test calls must not land in the real usage log

import agentic_retrieval as agent
import api_providers
import llm_budget
import model_capabilities
import query_knowledge_base as query_module

FEW_SHOT = [
    {"role": "user", "content": llm_budget.stable_json({"question": "开机流程", "evidence": [{"id": 1, "text": "先自检"}]})},
    {"role": "assistant", "content": llm_budget.stable_json({"decision": "answer", "answer": "先自检 [#1]"})},
]


def request(question: str, timestamp: str) -> list[dict[str, str]]:
    dynamic = llm_budget.stable_json({"question": question, "now": timestamp, "evidence": [{"id": 1, "file": "a.md", "text": "甲"}]})
    return llm_budget.build_messages(agent.SYSTEM_PROMPT, dynamic, few_shot=FEW_SHOT)


def prefix_bytes(messages: list[dict[str, str]], count: int) -> bytes:
    return llm_budget.stable_json(messages[:count]).encode("utf-8")


def check_prefix_is_byte_identical() -> None:
    """Two requests that differ only in the question and the time share their prefix byte for byte."""
    first = request("开机流程是什么", "2026-10-27T10:00:00+08:00")
    second = request("BootUp 调用了哪些函数", "2026-10-27T15:30:12+08:00")
    fixed = 1 + len(FEW_SHOT)  # system prompt + few-shot examples
    assert prefix_bytes(first, fixed) == prefix_bytes(second, fixed)
    assert first[-1]["role"] == "user" and first[-1] != second[-1]  # only the last message differs
    assert len(first) == len(second) == fixed + 1
    # Neither the question nor the time leaks into the prefix.
    assert "开机流程是什么" not in first[0]["content"] and "2026-10-27" not in prefix_bytes(first, fixed).decode("utf-8")
    # The same input always serializes to the same bytes, whatever the key order it was built in.
    assert llm_budget.stable_json({"b": 1, "a": {"y": 2, "x": 3}}) == llm_budget.stable_json({"a": {"x": 3, "y": 2}, "b": 1})
    # The real callers put only fixed text in the system prompt.
    for system in (agent.SYSTEM_PROMPT, query_module.SUMMARY_SYSTEM_PROMPT, model_capabilities.RERANK_PROMPT):
        a = llm_budget.build_messages(system, llm_budget.stable_json({"question": "甲", "now": "t1"}))
        b = llm_budget.build_messages(system, llm_budget.stable_json({"question": "乙", "now": "t2"}))
        assert prefix_bytes(a, 1) == prefix_bytes(b, 1)


def corpus_retrieve(corpus: dict[str, list[dict]]):
    return lambda query_text, limit: [dict(row) for row in corpus.get(query_text, [])]


def check_agent_rounds_only_append() -> None:
    """Round 2 resends round 1 unchanged (a cached prefix) and adds only the new evidence."""
    retrieve = corpus_retrieve(
        {
            "开机流程": [{"chunk_id": "a", "file_name": "guide.md", "evidence": "开机流程：先自检，再加载配置"}],
            "BootUp": [{"chunk_id": "b", "file_name": "guide.md", "evidence": "BootUp() 依次调用 HardwareCheck() 和 LoadSettings()"}],
        }
    )
    sent: list[list[dict[str, str]]] = []
    replies = iter(
        [
            '{"decision": "search", "queries": ["BootUp"], "reason": "缺少函数名"}',
            '{"decision": "answer", "answer": "开机时调用 BootUp()。\\n- 先自检 [#1]\\n- 依次调用两个函数 [#2]"}',
        ]
    )

    def complete(messages: list[dict[str, str]]) -> dict:
        sent.append([dict(message) for message in messages])
        return {"content": next(replies), "usage": {"prompt_tokens": 100, "prompt_cache_hit_tokens": 60, "prompt_cache_miss_tokens": 40, "completion_tokens": 20}}

    outcome = agent.run_agent("开机流程", retrieve, complete)
    assert outcome.decision == "answer" and outcome.cited_ids == [1, 2] and outcome.calls == 2
    round_one, round_two = sent
    assert round_two[: len(round_one)] == round_one  # byte-identical prefix
    assert round_two[len(round_one)] == {"role": "assistant", "content": '{"decision": "search", "queries": ["BootUp"], "reason": "缺少函数名"}'}
    new_payload = json.loads(round_two[-1]["content"])
    assert [item["id"] for item in new_payload["evidence"]] == [2] and new_payload["final_round"] is True
    assert outcome.usage["prompt_cache_hit_tokens"] == 120


def check_repeated_searches_stop_early() -> None:
    """A follow-up search that only repeats an earlier one, or finds nothing new, ends the loop without another call."""
    retrieve = corpus_retrieve({"开机流程": [{"chunk_id": "a", "file_name": "guide.md", "evidence": "开机流程：先自检"}]})
    for proposed in (["开机 流程", "开机流程！"], ["nothing-like-this"]):
        calls: list[int] = []

        def complete(messages: list[dict[str, str]], proposed=proposed) -> dict:
            calls.append(1)
            return {"content": json.dumps({"decision": "search", "queries": proposed, "reason": "证据不足"}, ensure_ascii=False)}

        outcome = agent.run_agent("开机流程", retrieve, complete, max_rounds=3)
        assert outcome.decision == "refuse" and len(calls) == 1 and outcome.reason == "证据不足", (proposed, outcome)
    assert agent.new_queries(["BootUp 流程", "bootup流程", "  ", "开机流程"], ["开机流程"]) == ["BootUp 流程"]


def check_cost_metrics() -> None:
    usage = {"prompt_tokens": 1000, "prompt_cache_hit_tokens": 900, "prompt_cache_miss_tokens": 100, "completion_tokens": 200}
    metrics = llm_budget.usage_metrics(usage)
    assert metrics["cache_hit_rate"] == 0.9 and metrics["output_input_ratio"] == 0.2
    off_peak = datetime(2026, 10, 27, 4, 30, tzinfo=timezone.utc)  # Tuesday 12:30 Beijing time
    peak = datetime(2026, 10, 27, 2, 0, tzinfo=timezone.utc)  # Tuesday 10:00 Beijing time
    saturday = datetime(2026, 10, 31, 2, 0, tzinfo=timezone.utc)
    assert not llm_budget.is_peak(off_peak) and llm_budget.is_peak(peak) and not llm_budget.is_peak(saturday)
    expected = (900 * 0.003 + 100 * 0.15 + 200 * 0.60) / 1_000_000
    assert abs(llm_budget.estimate_cost_usd("deepseek-flash", metrics, off_peak, {}) - expected) < 1e-12
    assert abs(llm_budget.estimate_cost_usd("deepseek-flash", metrics, peak, {}) - 2 * expected) < 1e-12
    assert llm_budget.estimate_cost_usd("unknown-model", metrics, peak, {}) is None
    priced = {"LKA_PRICE_DEEPSEEK_V4_PRO": "0.01,0.5,2"}
    assert llm_budget.estimate_cost_usd("deepseek-v4-pro", metrics, off_peak, priced) == round((9 + 50 + 400) / 1_000_000, 8)
    # Providers without cache fields: everything counts as a miss.
    assert llm_budget.usage_metrics({"prompt_tokens": 50, "completion_tokens": 5})["cache_miss_tokens"] == 50
    with tempfile.TemporaryDirectory() as temp_dir:
        log = Path(temp_dir) / "usage.jsonl"
        config = {"LKA_API_USAGE_LOG": str(log)}
        llm_budget.record_call("agent", "deepseek", "deepseek-flash", usage, 10, off_peak, config)
        llm_budget.record_call("rerank", "deepseek", "deepseek-flash", {"prompt_tokens": 100, "prompt_cache_miss_tokens": 100, "completion_tokens": 10}, 5, peak, config)
        records = llm_budget.read_log(log)
        assert [record["task"] for record in records] == ["agent", "rerank"] and records[0]["time"] == "2026-10-27T12:30:00+08:00"
        total = llm_budget.summarize(records)["total"]
        assert total["calls"] == 2 and total["cache_hit_rate"] == round(900 / 1100, 4) and total["peak_calls"] == 1
        assert llm_budget.read_log(log, since="2026-10-27T12:00") == records[:1]
        assert llm_budget.usage_log_path({"LKA_API_USAGE_LOG": "off"}) is None


def check_task_budgets_and_model_tier() -> None:
    config = {"LKA_API_PROVIDER": "deepseek", "LKA_API_MODEL": "deepseek-flash"}
    for task in ("agent", "summary", "rerank", "connection_test"):
        assert llm_budget.task_settings(config, task)["model"] == "deepseek-flash"  # Flash unless switched
    assert llm_budget.task_settings(config, "rerank")["max_tokens"] < llm_budget.task_settings(config, "summary")["max_tokens"]
    assert llm_budget.task_settings({**config, "LKA_API_THINKING": "enabled"}, "rerank")["thinking"] == "disabled"
    assert llm_budget.task_settings({**config, "LKA_API_THINKING": "enabled"}, "agent")["thinking"] == "enabled"
    assert llm_budget.task_settings({**config, "LKA_MAX_TOKENS_SUMMARY": "900"}, "summary")["max_tokens"] == 900
    pro = {**config, "LKA_API_PRO_TASKS": "agent, rerank"}
    assert llm_budget.task_settings(pro, "agent")["model"] == "deepseek-v4-pro"
    assert llm_budget.task_settings(pro, "summary")["model"] == "deepseek-flash"
    assert llm_budget.task_settings(pro, "rerank")["model"] == "deepseek-flash"  # simple tasks never switch
    assert llm_budget.task_settings({**pro, "LKA_API_PRO_MODEL": "my-pro"}, "agent")["model"] == "my-pro"
    # The answer cache keeps its key unless the switch changes which model answers.
    api_config = {"LKA_USE_API": "true", "LKA_API_KEY": "k", **config}
    default_variant = query_module.api_cache_variant(api_config, True)
    assert "deepseek-v4-pro" not in default_variant
    assert "model-deepseek-v4-pro" in query_module.api_cache_variant({**api_config, "LKA_API_PRO_TASKS": "agent"}, True)


def check_summary_request() -> None:
    """The summary call sends its budget, thinking switch and task name, and clips long citations."""
    captured: dict = {}
    original = query_module.chat_completion

    def fake(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs, messages=args[4], model=args[3])
        return "依据 [1]，答案如下。"

    result = {
        "source_type": "web_search",
        "answer": "x",
        "sources": [{"source_id": 1, "file": "page", "citation": "甲" * 5000}],
        "citations": [],
    }
    config = {"LKA_USE_API": "true", "LKA_API_KEY": "k", "LKA_API_MODEL": "deepseek-flash", "LKA_API_PROVIDER": "deepseek", "LKA_API_CITATION_MAX_CHARS": "300"}
    query_module.chat_completion = fake
    try:
        query_module.synthesize_with_api("问题", result, config)
    finally:
        query_module.chat_completion = original
    assert captured["task"] == "summary" and captured["thinking"] == "disabled" and captured["max_tokens"] == 2000
    assert captured["messages"][0]["content"] == query_module.SUMMARY_SYSTEM_PROMPT
    quote = json.loads(captured["messages"][-1]["content"])["citations"][0]["quote"]
    assert len(quote) == 300 and quote.endswith("…")


def check_identical_requests_are_sent_once() -> None:
    """Concurrent identical calls share one request and one bill; answering the same question twice at once runs once."""
    sent: list[dict] = []
    original = api_providers.api_request

    def slow_request(method, url, provider, api_key, payload=None, timeout=20):  # noqa: ANN001
        sent.append(payload)
        time.sleep(0.3)
        return {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 1}}

    api_providers.api_request = slow_request
    results: list[dict] = []
    messages = [{"role": "user", "content": "同一个请求"}]
    try:
        threads = [
            threading.Thread(
                target=lambda: results.append(
                    api_providers.chat_completion_detailed("deepseek", "https://api.deepseek.com", "k", "deepseek-flash", messages)
                )
            )
            for _ in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        api_providers.api_request = original
    assert len(sent) == 1 and [result["content"] for result in results] == ["ok"] * 3
    assert sum(1 for result in results if result["record"]) == 1  # logged (billed) once

    running = {"now": 0, "max": 0}
    lock = threading.Lock()
    original_query = query_module.query_unshared

    def counted(*args, **kwargs):  # noqa: ANN002, ANN003
        with lock:
            running["now"] += 1
            running["max"] = max(running["max"], running["now"])
        time.sleep(0.2)
        with lock:
            running["now"] -= 1
        return {}

    query_module.query_unshared = counted
    try:
        threads = [threading.Thread(target=query_module.query, args=(Path("x.db"), "同一个问题")) for _ in range(3)]
        threads.append(threading.Thread(target=query_module.query, args=(Path("x.db"), "另一个问题")))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        query_module.query_unshared = original_query
    assert running["max"] == 2  # the three identical questions ran one at a time, the other alongside


def main() -> None:
    check_prefix_is_byte_identical()
    check_agent_rounds_only_append()
    check_repeated_searches_stop_early()
    check_cost_metrics()
    check_task_budgets_and_model_tier()
    check_summary_request()
    check_identical_requests_are_sent_once()
    print("llm budget: ok")


if __name__ == "__main__":
    main()
