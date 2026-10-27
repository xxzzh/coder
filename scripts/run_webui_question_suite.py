#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automated Web UI question verification against a running Web UI.

Transports:
- http (default): asks the questions through the Web UI's own /api/query endpoint and checks the
  page from its HTML and /api/health, rendered in the same wording as the page. Nothing to install.
- playwright: drives a real browser through playwright-cli (needs Node.js/npx and a Playwright browser).

Usage:
    .venv\\Scripts\\python.exe scripts\\run_webui_question_suite.py
    .venv\\Scripts\\python.exe scripts\\run_webui_question_suite.py --allow-api
    .venv\\Scripts\\python.exe scripts\\run_webui_question_suite.py --transport playwright

--allow-api asks as the Web UI itself does when an answer API is configured: the retrieval agent
then answers most questions, so the local table/dictionary mode expectations do not apply to them.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
NPX = shutil.which("npx.cmd") or shutil.which("npx") or "npx"
SESSION = "lka-ui-test"
WEBUI_URL = "http://127.0.0.1:8765/"


def parse_playwright_stdout(stdout: str) -> Any:
    outer = json.loads(stdout)
    value: Any = outer.get("result")
    for _ in range(5):
        if not isinstance(value, str):
            return value
        try:
            value = json.loads(value)
        except Exception:
            return value
    return value


def pw_eval(js: str, timeout: int = 120, label: str = "") -> Any:
    cmd = [
        NPX,
        "--yes",
        "--package",
        "@playwright/cli",
        "playwright-cli",
        f"-s={SESSION}",
        "eval",
        js,
        "--json",
    ]
    last: subprocess.CompletedProcess[str] | None = None
    for attempt in range(3):
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            encoding="utf-8",
        )
        last = result
        if result.returncode == 0 and result.stdout.strip():
            try:
                return parse_playwright_stdout(result.stdout)
            except Exception:
                pass
        if attempt == 0:
            subprocess.run(
                [
                    NPX,
                    "--yes",
                    "--package",
                    "@playwright/cli",
                    "playwright-cli",
                    f"-s={SESSION}",
                    "open",
                    WEBUI_URL,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=120,
                encoding="utf-8",
            )
        time.sleep(1)
    stdout = last.stdout[:300] if last else ""
    stderr = last.stderr[:300] if last else ""
    raise RuntimeError(f"playwright eval failed {label}: stdout={stdout!r} stderr={stderr!r}")


def status_from_failures(failures: list[str]) -> str:
    return "FAIL" if failures else "PASS"


def normalize_assertion_text(text: str) -> str:
    text = (text or "").lower()
    return re.sub(r"[\s，,。；;：:/\\()（）【】\[\]<>《》\"'`_\-—–.]+", "", text)


def contains_expected(text: str, expected: str) -> bool:
    if expected in text:
        return True
    return normalize_assertion_text(expected) in normalize_assertion_text(text)


QUERY_CASES: list[dict[str, Any]] = [
    {
        "id": "Q01",
        "area": "FFT站位",
        "question": "FFT站位测试",
        "expected_mode": "local_structured_table",
        "must_contain": [
            "自动化测试 / 寸动线自动化测试 / 清洁屏幕",
            "人工功能测试 MES：FFT / IR CameraTest / IR Camera Test",
            "人工功能测试 MES：FFT / 手写笔功能测试 / 充电功能测试",
            "人工功能测试 MES：FFT / FFT 过站 / FFT扫描过站",
        ],
        "must_not_contain": ["NFCReadTest NFC功能测试；Auto-Camera Test"],
        "group": "fft_station",
    },
    {
        "id": "Q02",
        "area": "FFT站位",
        "question": "FFT站位测试项",
        "expected_mode": "local_structured_table",
        "must_contain": [
            "自动化功能测试 MES:ATS / 摄像头测试",
            "人工功能测试 MES：FFT / RSSI / 天线信号强度测试",
        ],
        "must_not_contain": ["NFCReadTest NFC功能测试；Auto-Camera Test"],
        "group": "fft_station",
    },
    {
        "id": "Q03",
        "area": "人工功能/FFT",
        "question": "FFT测试",
        "expected_answer_mode": "local_structured_table_summary",
        # Every item and sub-item name, in any layout (a polished API answer rewrites the table rows).
        "must_contain": [
            "IR Camera Test",
            "Sensor Test",
            "Lid Hall Sensor Test",
            "手写笔功能测试",
            "充电功能测试",
            "FFT扫描过站",  # the sub-item; a grouped answer may title its group just "过站"
        ],
        "group": "fft_manual",
    },
    {
        "id": "Q04",
        "area": "人工功能/FFT",
        "question": "FFT测试项",
        "expected_answer_mode": "local_structured_table_summary",
        "must_contain": ["IR Camera Test", "WIFI吞吐量测试", "T-put测试"],
        "group": "fft_manual",
    },
    {
        "id": "Q05",
        "area": "人工功能/FFT",
        "question": "人工功能测试",
        "expected_answer_mode": "local_structured_table_summary",
        "must_contain": ["IR Camera Test", "异音测试", "开合转轴异响"],
        "group": "fft_manual",
    },
    {
        "id": "Q06",
        "area": "缩写词典",
        "question": "PR",
        "expected_mode": "local_structured_dictionary",
        "must_contain": [
            "Pilot Run：试运行",
            "Public Relations：公共关系",
            "Performance Rating：个人绩效考评等级",
        ],
        "must_not_contain": ["本地知识库没有找到足够依据", "none"],
    },
    {
        "id": "Q07",
        "area": "缩写词典",
        "question": "COB",
        "expected_mode": "local_structured_dictionary",
        "must_contain": ["Chip on Board：板上芯片", "Close of Business：停业(下班)"],
    },
    {
        "id": "Q08",
        "area": "缩写词典",
        "question": "GW",
        "expected_mode": "local_structured_dictionary",
        "must_contain": ["Gross Weight：毛重"],
        "must_not_contain": ["本地知识库没有找到足够依据"],
    },
    {
        "id": "Q09",
        "area": "缩写词典",
        "question": "CPK",
        "expected_mode": "local_structured_dictionary",
        "must_contain": ["Process Capability Index：工序能力指数"],
        "must_not_contain": ["HDCP Key"],
    },
    {
        "id": "Q10",
        "area": "缩写全称",
        "question": "Pilot Run",
        "must_contain": ["试运行"],
        "must_not_contain": ["本地知识库没有找到足够依据"],
    },
    {
        "id": "Q11",
        "area": "工单流程",
        "question": "工单数据获取流程",
        "must_contain": ["mWoNoteItemList", "GetWorkOrderData()", "GetWorkOrderData_V20()", "WONOTELIST.txt"],
        "must_not_contain": ["本地知识库没有找到足够依据"],
    },
    {
        "id": "Q12",
        "area": "枚举/类型",
        "question": "客户类型有哪些",
        "must_contain_any": ["EnumCustomerType", "客户类型", "枚举"],
        "must_not_contain": ["本地知识库没有找到足够依据"],
    },
    {
        "id": "Q13",
        "area": "枚举/日志",
        "question": "LogMsgType有哪些值",
        "must_contain": ["LogMsgType"],
        "must_not_contain": ["本地知识库没有找到足够依据"],
    },
    {
        "id": "Q14",
        "area": "恢复机制",
        "question": "恢复机制是什么",
        "must_contain": ["EnableReadLocalStatus", "LocalData", "状态"],
        "must_not_contain": ["现有证据不足", "本地知识库没有找到足够依据"],
    },
    {
        "id": "Q15",
        "area": "本地缓存",
        "question": "本地缓存逻辑",
        "must_contain_any": ["缓存", "cache", "query_cache"],
        "must_not_contain": ["本地知识库没有找到足够依据"],
        "group": "local_cache",
    },
    {
        "id": "Q16",
        "area": "本地缓存",
        "question": "本地缓存",
        "must_contain_any": ["缓存", "cache", "query_cache"],
        "must_not_contain": ["本地知识库没有找到足够依据"],
        "group": "local_cache",
    },
    {
        "id": "Q17",
        "area": "负例",
        "question": "火星咖啡机保修政策是什么",
        "expected_source_type": "none",
        "must_contain": ["本地知识库没有找到足够依据"],
    },
]

UI_CHECKS: list[dict[str, Any]] = [
    {
        "id": "UI01",
        "area": "页面加载",
        "question": "标题",
        "selector": "h1",
        "must_contain": ["Local Knowledge Base Agent"],
    },
    {
        "id": "UI02",
        "area": "页面控件",
        "question": "本地知识库/online",
        "selector": "body",
        "must_contain": ["本地知识库", "online"],
    },
    {
        "id": "UI03",
        "area": "状态模块",
        "question": "索引时间/OCR",
        "selector": "body",
        "must_contain": ["索引时间：", "OCR：可用（离线/online）"],
    },
]


def http_get(path: str, timeout: int = 180) -> tuple[int, Any]:
    with urlopen(WEBUI_URL.rstrip("/") + path, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if path.startswith("/api/") else body


def page_texts_http() -> dict[str, str]:
    """What the page shows, read without a browser: the static HTML with scripts removed, plus the
    status lines the page renders from /api/health, worded as in the page."""
    _, html = http_get("/")
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.S)
    visible = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S | re.I)
    visible = re.sub(r"<[^>]+>", " ", visible)
    _, health = http_get("/api/health")
    status_lines = [
        f"索引时间：{(health.get('index') or {}).get('indexed_at') or '-'}",
        f"OCR：{'可用（离线/online）' if (health.get('optional_extractors') or {}).get('ocr_ready') else '不可用'}",
    ]
    return {"h1": re.sub(r"<[^>]+>", "", h1.group(1)).strip() if h1 else "", "body": visible + "\n" + "\n".join(status_lines)}


def page_text_playwright(selector: str, label: str) -> str:
    data = pw_eval(
        "async () => JSON.stringify({text:(document.querySelector(%s)?.innerText || '')})" % json.dumps(selector),
        label=label,
    )
    return data.get("text", "")


def query_http(question: str, allow_api: bool) -> dict[str, Any]:
    params = urlencode({"question": question, "limit": "5", "use_web": "false", "allow_api": str(allow_api).lower()})
    started = time.perf_counter()
    try:
        status, data = http_get(f"/api/query?{params}")
    except Exception as exc:  # noqa: BLE001
        status, data = 0, {"answer": "", "error": str(exc)}
    return {"ok": status == 200, "status": status, "elapsedMs": round((time.perf_counter() - started) * 1000), "data": data}


def query_playwright(question: str, allow_api: bool, label: str) -> dict[str, Any]:
    question_json = json.dumps(question, ensure_ascii=True)
    js = (
        "async () => {"
        "const started=performance.now();"
        f"const params=new URLSearchParams({{question:{question_json},limit:'5',use_web:'false',allow_api:'{str(allow_api).lower()}'}});"
        "const resp=await fetch('/api/query?'+params.toString());"
        "const data=await resp.json();"
        "return JSON.stringify({ok:resp.ok,status:resp.status,elapsedMs:Math.round(performance.now()-started),data});"
        "}"
    )
    return pw_eval(js, timeout=180, label=label)


def cited_sources(data: dict[str, Any]) -> set[str]:
    return {
        str(source.get("chunk_id") or f"{source.get('file')}|{str(source.get('excerpt') or '')[:60]}")
        for source in data.get("sources") or []
    }


def run_suite(transport: str = "http", allow_api: bool = False) -> dict[str, Any]:
    if transport == "playwright":
        subprocess.run(
            [NPX, "--yes", "--package", "@playwright/cli", "playwright-cli", f"-s={SESSION}", "open", WEBUI_URL],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=120,
            encoding="utf-8",
        )
        page_texts: dict[str, str] = {}
    else:
        page_texts = page_texts_http()

    rows: list[dict[str, Any]] = []

    for check in UI_CHECKS:
        if transport == "playwright":
            text = page_text_playwright(check["selector"], check["id"])
        else:
            text = page_texts.get(check["selector"], "")
        failures = [f"缺少: {item}" for item in check["must_contain"] if not contains_expected(text, item)]
        rows.append(
            {
                "id": check["id"],
                "area": check["area"],
                "question": check["question"],
                "status": status_from_failures(failures),
                "failures": failures,
                "mode": "dom" if transport == "playwright" else "html+health",
                "answerMode": "",
                "sourceType": "",
                "sourceCount": "",
                "cacheHit": "",
                "elapsedMs": "",
                "answerPreview": text.replace("\n", " ")[:160],
                "fullAnswer": "",
                "group": "",
            }
        )

    for case in QUERY_CASES:
        if transport == "playwright":
            result = query_playwright(case["question"], allow_api, case["id"])
        else:
            result = query_http(case["question"], allow_api)
        data = result.get("data", {})
        answer = data.get("answer", "") or ""
        failures: list[str] = []

        for item in case.get("must_contain", []):
            if not contains_expected(answer, item):
                failures.append(f"缺少: {item}")
        if case.get("must_contain_any") and not any(contains_expected(answer, item) for item in case["must_contain_any"]):
            failures.append("未命中任一: " + " / ".join(case["must_contain_any"]))
        for item in case.get("must_not_contain", []):
            if contains_expected(answer, item):
                failures.append(f"不应包含: {item}")
        # The table/dictionary mode expectations describe local answers. When an answer API is
        # configured the retrieval agent may answer instead; the content checks above still apply.
        agent_answer = data.get("answer_mode") == "api_agent_answer"
        if case.get("expected_mode") and not agent_answer and data.get("retrieval_mode") != case["expected_mode"]:
            failures.append(f"retrieval_mode={data.get('retrieval_mode')}，期望 {case['expected_mode']}")
        if case.get("expected_answer_mode") and not agent_answer and data.get("answer_mode") != case["expected_answer_mode"]:
            failures.append(f"answer_mode={data.get('answer_mode')}，期望 {case['expected_answer_mode']}")
        if case.get("expected_source_type") and data.get("source_type") != case["expected_source_type"]:
            failures.append(f"source_type={data.get('source_type')}，期望 {case['expected_source_type']}")
        if not result.get("ok", False):
            failures.append(f"HTTP {result.get('status')} {data.get('error', '')}".strip())

        rows.append(
            {
                "id": case["id"],
                "area": case["area"],
                "question": case["question"],
                "status": status_from_failures(failures),
                "failures": failures,
                "mode": data.get("retrieval_mode", ""),
                "answerMode": data.get("answer_mode", ""),
                "sourceType": data.get("source_type", ""),
                "sourceCount": len(data.get("sources") or []),
                "cacheHit": bool(data.get("cache_hit")),
                "elapsedMs": result.get("elapsedMs"),
                "answerPreview": answer.replace("\n", " ")[:260],
                "fullAnswer": answer,
                "citedSources": sorted(cited_sources(data)),
                "group": case.get("group", ""),
            }
        )

    # Different wordings of one question must rest on the same evidence. Their wording may differ
    # (an LLM never repeats itself word for word), so each answer has to share at least one cited
    # source with the first answer of its group.
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("group"):
            groups.setdefault(row["group"], []).append(row)
    for group, items in groups.items():
        base = set(items[0]["citedSources"])
        mismatches = [
            f"{item['id']}:{item['question']} 与 {items[0]['id']} 没有共同引用来源"
            for item in items[1:]
            if not base & set(item["citedSources"])
        ]
        rows.append(
            {
                "id": "G-" + group,
                "area": "一致性",
                "question": group,
                "status": status_from_failures(mismatches),
                "failures": mismatches,
                "mode": "shared evidence",
                "answerMode": "",
                "sourceType": "",
                "sourceCount": "",
                "cacheHit": "",
                "elapsedMs": "",
                "answerPreview": ", ".join(item["id"] for item in items),
                "fullAnswer": "",
                "group": "",
            }
        )

    summary = {
        "transport": transport,
        "allow_api": allow_api,
        "total": len(rows),
        "pass": sum(1 for row in rows if row["status"] == "PASS"),
        "fail": sum(1 for row in rows if row["status"] == "FAIL"),
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Check the running Web UI with the question suite.")
    parser.add_argument("--transport", choices=("http", "playwright"), default="http")
    parser.add_argument("--allow-api", action="store_true", help="Ask as the Web UI does when an answer API is configured.")
    args = parser.parse_args()
    print(json.dumps(run_suite(args.transport, args.allow_api), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
