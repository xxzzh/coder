#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agentic retrieval: an LLM reads the retrieved evidence, then answers, searches again, or refuses.

The lexical relevance gate cannot tell a paraphrase (开机 for BootUp) from a question whose
subject the documents never cover, and the rank order decides which chunk a local answer cites.
Here the model sees the whole candidate pool, judges whether it answers the question, and may
ask for up to three follow-up searches worded the way the documents are written.

Retrieval and the model are passed in as functions, so this module holds only the loop, the
prompt and the checks on the model's reply.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

AGENT_VERSION = "agent-v1"
FIRST_ROUND_RESULTS = 10
FOLLOW_UP_QUERIES = 3
FOLLOW_UP_RESULTS = 5
EVIDENCE_MAX_ITEMS = 18
ANSWER_MAX_TOKENS = 3000
# DeepSeek's JSON mode documents that it occasionally returns empty content; one retry covers it.
REPLY_ATTEMPTS = 2

# query, limit -> rows carrying "evidence" (the text shown to the model), "file_name", "chunk_id"
Retrieve = Callable[[str, int], list[dict[str, Any]]]
# messages -> {"content": reply text, "usage": token counts}
Complete = Callable[[list[dict[str, str]]], dict[str, Any]]

SYSTEM_PROMPT = """你是本地知识库的检索问答助手。用户问题和从知识库检索到的证据片段以 JSON 给出。你只能依据这些证据回答，不能用自己的知识补充。

先判断证据能否回答问题：问题问的对象和它要的那项信息（属性、步骤、数值、原因、位置等）都要在证据里有直接依据。证据只讲了同一对象的其他方面，不算能回答。用户的说法和文档不同没有关系，例如“开机”对应 BootUp、“设置文件”对应“配置文件”，只要意思一致就算有依据。

然后三选一：
1. decision 为 "answer"：证据足以回答。用简体中文写完整答案：
   - 第一句直接回答问题，接着用“- ”开头的列表逐条列出要点；不要用 Markdown 表格、标题或加粗。
   - 列举类问题列出证据中的全部项；流程、步骤类问题按原顺序完整列出。
   - 参数名、标识符、文件路径、数值和单位照证据原文写，不要翻译或改写。
   - 每条要点末尾标注支持它的证据编号，写成 [#3]，多个来源写成 [#1][#3]；只引用真正支持该要点的编号。方括号里的数值区间等原文内容照抄即可，不要写成 [#…] 的形式。
   - 只回答问到的内容；证据之间有冲突时分别列出并注明编号，不要擅自取舍。
2. decision 为 "search"：现有证据不足，但答案可能在知识库其他位置。在 queries 里给出 1 到 3 个新的检索语句，尽量用文档里可能出现的写法（英文标识符、参数名、章节标题式的说法），不要重复 searched_queries 里已有的。
3. decision 为 "refuse"：知识库里没有这项信息。证据没有提到的事，不要推断成“不会”“没有”“不支持”来作答，这种情况也选 "refuse"。在 reason 里用一句话说明缺少什么。

final_round 为 true 时不能选 "search"，只能选 "answer" 或 "refuse"。

只输出一个 JSON 对象，例如：
{"decision": "answer", "answer": "设备启动时先自检，再加载配置。\\n- 先执行硬件自检 HardwareCheck() [#2]\\n- 再读取 config.ini [#3]", "evidence_ids": [2, 3], "queries": [], "reason": "证据 2、3 说明了启动顺序"}"""


class AgentError(RuntimeError):
    """The model's reply could not be used; the caller falls back to the local answer path."""


@dataclass
class AgentOutcome:
    decision: str  # "answer" or "refuse"
    answer: str
    reason: str
    evidence: list[dict[str, Any]]
    cited_ids: list[int]
    rounds: int
    queries: list[str]
    usage: dict[str, int] = field(default_factory=dict)
    elapsed_ms: int = 0


def evidence_key(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())


def add_evidence(evidence: list[dict[str, Any]], rows: list[dict[str, Any]]) -> int:
    """Append rows not already shown (or contained in shown evidence); returns how many were added."""
    added = 0
    keys = [evidence_key(item["text"]) for item in evidence]
    for row in rows:
        if len(evidence) >= EVIDENCE_MAX_ITEMS:
            break
        text = str(row.get("evidence") or row.get("text") or "").strip()
        key = evidence_key(text)
        if not key or any(key in existing for existing in keys):
            continue
        evidence.append({"id": len(evidence) + 1, "file": str(row.get("file_name", "")), "text": text, "row": row})
        keys.append(key)
        added += 1
    return added


def parse_reply(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        reply = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentError(f"model reply is not JSON: {text[:120]!r}") from exc
    if not isinstance(reply, dict) or reply.get("decision") not in {"answer", "search", "refuse"}:
        raise AgentError(f"model reply has no valid decision: {text[:120]!r}")
    return reply


# The model cites evidence as [#3]. Plain brackets are left alone: the documents are full of
# bracketed data such as thresholds [10,20] or [-3,-7], which must never be read as citations.
CITATION_PATTERN = re.compile(r"\[#\s*(\d+(?:\s*[,，、]\s*#?\s*\d+)*)\s*\]")


def citation_group_ids(group: str) -> list[int]:
    return [int(value) for value in re.findall(r"\d+", group)]


def cited_ids(answer: str) -> list[int]:
    """Evidence ids in order of first citation; [#1][#3] and [#1, #3] both count."""
    ids: list[int] = []
    for group in CITATION_PATTERN.findall(answer):
        for value in citation_group_ids(group):
            if value not in ids:
                ids.append(value)
    return ids


def ask(
    complete: Complete,
    question: str,
    evidence: list[dict[str, Any]],
    searched: list[str],
    final_round: bool,
    usage: dict[str, int],
) -> dict[str, Any]:
    payload = {
        "question": question,
        "final_round": final_round,
        "searched_queries": searched,
        "evidence": [{"id": item["id"], "file": item["file"], "text": item["text"]} for item in evidence],
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    last_error: Exception | None = None
    for _ in range(REPLY_ATTEMPTS):
        try:
            response = complete(messages)
        except RuntimeError as exc:
            # Rate limits and server errors are worth one more try; anything else (bad key,
            # bad request) will not improve. Timeouts are not retried, to keep the wait bounded.
            if not re.match(r"HTTP (429|5\d\d)", str(exc)):
                raise
            last_error = exc
            continue
        for key, value in (response.get("usage") or {}).items():
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + value
        try:
            return parse_reply(str(response.get("content", "")))
        except AgentError as exc:
            last_error = exc
    raise AgentError(f"no usable reply after {REPLY_ATTEMPTS} attempts: {last_error}")


def run_agent(question: str, retrieve: Retrieve, complete: Complete, max_rounds: int = 2) -> AgentOutcome:
    started = time.perf_counter()
    evidence: list[dict[str, Any]] = []
    add_evidence(evidence, retrieve(question, FIRST_ROUND_RESULTS))
    searched = [question]
    usage: dict[str, int] = {}
    rounds = 0
    reply: dict[str, Any] = {}
    max_rounds = max(1, max_rounds)
    while rounds < max_rounds:
        rounds += 1
        final_round = rounds == max_rounds
        reply = ask(complete, question, evidence, searched, final_round, usage)
        if reply["decision"] != "search":
            break
        if final_round:
            reply = {**reply, "decision": "refuse", "reason": str(reply.get("reason") or "检索后仍没有足够依据")}
            break
        queries = [str(item).strip() for item in reply.get("queries") or [] if str(item).strip()]
        queries = [item for item in dict.fromkeys(queries) if item not in searched][:FOLLOW_UP_QUERIES]
        for query in queries:
            add_evidence(evidence, retrieve(query, FOLLOW_UP_RESULTS))
            searched.append(query)

    decision = reply["decision"]
    answer = str(reply.get("answer") or "").strip()
    reason = str(reply.get("reason") or "").strip()
    ids: list[int] = []
    if decision == "answer":
        ids = cited_ids(answer)
        known = {item["id"] for item in evidence}
        if not answer or not ids:
            raise AgentError("answer cites no evidence")
        if not set(ids) <= known:
            raise AgentError(f"answer cites unknown evidence ids {sorted(set(ids) - known)}")
    return AgentOutcome(
        decision=decision,
        answer=answer,
        reason=reason,
        evidence=evidence,
        cited_ids=ids,
        rounds=rounds,
        queries=searched[1:],
        usage=usage,
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )


def renumber_citations(answer: str, ids: list[int]) -> str:
    """Rewrite [#evidence id] as [source number] in citation order; [#3, #5] becomes [1][2]."""
    mapping = {old: new for new, old in enumerate(ids, start=1)}

    def replace(match: re.Match[str]) -> str:
        return "".join(f"[{mapping[value]}]" for value in citation_group_ids(match.group(1)) if value in mapping)

    return CITATION_PATTERN.sub(replace, answer)
