#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Randomized evidence-based QA checks for indexed Excel test-plan content.

The suite intentionally builds questions from the indexed N972 Excel source
instead of from prior user-reported regressions. Each generated case carries a
source-cell substring that must appear in the WebUI answer.
"""

from __future__ import annotations

import json
import base64
import random
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "knowledge_base" / "index" / "knowledge.db"
WEBUI_URL = "http://127.0.0.1:8765"
PLAYWRIGHT = Path("D:/nodejs/npx.cmd")
SESSION = "lka-ui-test"
SEED = 20260606
TARGET_CASES = 40
CHUNK_SIZE = 9


@dataclass
class Case:
    case_id: str
    area: str
    question: str
    expected: list[str]
    source_line: str


@dataclass
class Result:
    case_id: str
    area: str
    question: str
    expected: list[str]
    status: str
    reason: str
    source_line: str
    answer: str


def normalize(text: str) -> str:
    text = re.sub(r"\s+", "", text or "").lower()
    return re.sub(r"[，,。；;：:、.。/\\()（）【】\[\]<>《》\"'`_\-—–]+", "", text)


def compact(text: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.split("|")]


def nonempty(values: Iterable[str]) -> list[str]:
    return [v.strip() for v in values if v and v.strip()]


def load_n972_text() -> str:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"knowledge DB not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            """
            SELECT text
            FROM documents
            WHERE file_name = 'N972-ACER_Test plan_V3.1.xlsx'
            ORDER BY indexed_at DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        raise RuntimeError("N972-ACER_Test plan_V3.1.xlsx is not indexed in documents.text")
    return row[0]


def valid_expected(text: str) -> bool:
    if not text:
        return False
    text = text.strip()
    if len(text) < 2 or len(text) > 120:
        return False
    weak_values = {
        "NA",
        "N/A",
        "无",
        "なし",
        "TBD",
        "------",
        "-",
        "--",
        "OK",
        "Y",
        "N",
        "程序",
        "MES",
        "程序&MES",
        "程序/MES",
        "待导入",
        "已导入",
    }
    if text in weak_values:
        return False
    return True


def tool_like(text: str) -> bool:
    return bool(re.search(r"\.(?:exe|bat|cmd|dll|ps1)$", structured := text.strip(), re.I) or "factory tool" in structured.lower())


def preferred_anchor(item: str, subitem: str) -> str:
    if valid_expected(subitem) and not tool_like(subitem):
        return subitem
    return item


def short_expected(text: str, min_len: int = 4, max_len: int = 36) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_len:
        return text
    # Prefer a compact leading phrase, preserving numbers and key nouns.
    parts = re.split(r"[，,。；;、/]+", text)
    for part in parts:
        part = part.strip()
        if min_len <= len(part) <= max_len:
            return part
    return text[:max_len]


def build_cases(text: str) -> list[Case]:
    cases: list[Case] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    worksheet_re = re.compile("\u5de5\u4f5c\u8868 ([^\n]+)")
    matches = list(worksheet_re.finditer(text))
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.group(1).strip(), text[start:end]))

    def add(area: str, question: str, expected: Iterable[str], source_line: str) -> None:
        exp = [short_expected(e) for e in expected if valid_expected(short_expected(e))]
        if not exp:
            return
        key = (question, tuple(exp))
        if key in seen:
            return
        seen.add(key)
        cases.append(
            Case(
                case_id=f"T{len(cases) + 1:03d}",
                area=area,
                question=question,
                expected=exp,
                source_line=compact(source_line, 220),
            )
        )

    # Product/specification style rows. These are not based on user-reported
    # questions; they are sampled from common Excel specification fields.
    spec_field_re = re.compile(
        r"^(Main battery type|Quick Charge support|Adaptor Type|Connector Type|"
        r"WLAN|Bluetooth|HDMI|Type-C|Headphone|Power button|Battery indicator|"
        r"Camera indicator|Pixel- Front|Pixel- Rear|Codec|Microphone|Speakers|"
        r"Sensor|Keyboard|TouchPad)\b",
        re.I,
    )
    feature_sections = [
        (name, body)
        for name, body in sections
        if "feature list" in name.lower() or "\u4ea7\u54c1\u89c4\u683c" in name
    ]
    for _, body in feature_sections:
        for line in [line.strip() for line in body.splitlines() if "|" in line]:
            cs = nonempty(cells(line))
            if len(cs) < 2:
                continue
            field = cs[0]
            if spec_field_re.search(field):
                value = next((v for v in cs[1:] if valid_expected(v)), "")
                if value:
                    add("Excel规格字段", f"N972 的 {field} 是什么？", [value], line)

    # Structured test-plan rows. The common columns are approximately:
    # no, category, item, sub-item, tool, version, source, threshold, scheme,
    # judgement, fixture. We validate several independent cell types.
    noisy = re.compile(
        r"^(#|编号|序列|站位|工位|Category|sub-item|测试项|测试项目|测试子项|"
        r"Test Item|Item|工具名称|工具信息|版本|Rev|Remark|备注|当前状态)$",
        re.I,
    )
    excluded_section = re.compile(
        "\u4fee\u8ba2|\u6e05\u5355|check list|Test Flow|\u5e94\u7b54|\u95e8\u9650\u8bc4\u5ba1",
        re.I,
    )
    test_section = re.compile(
        r"SMT|DLTEST|FAT|FRT|FFT|SWDL|LCD|OOBE|Run In|"
        "\u529f\u80fd\u6d4b\u8bd5|\u8001\u5316|\u552e\u540e\u5907\u4ef6|\u670d\u52a1\u5907\u4ef6|"
        "\u95e8\u9650\u8868|\u6d4b\u8bd5\u95e8\u9650",
        re.I,
    )
    test_sections = [
        (name, body)
        for name, body in sections
        if test_section.search(name) and not excluded_section.search(name)
    ]

    for section_name, body in test_sections:
        for line in [line.strip() for line in body.splitlines() if "|" in line]:
            raw = cells(line)
            cs = raw + [""] * max(0, 12 - len(raw))
            if cs[0].strip().isdigit() or (not cs[0].strip() and not cs[1].strip() and cs[2].strip()):
                item = cs[2].strip() or cs[1].strip()
                subitem = cs[3].strip()
                tool = cs[4].strip()
                threshold = cs[7].strip()
                scheme = cs[8].strip()
                judgement = cs[9].strip()
                fixture = cs[10].strip()
            else:
                item = cs[1].strip()
                subitem = cs[2].strip()
                tool = cs[3].strip()
                threshold = cs[4].strip()
                scheme = cs[5].strip()
                judgement = cs[6].strip()
                fixture = cs[8].strip()
            if not item and not subitem:
                continue
            item = item or subitem

            if not item or noisy.search(item) or len(item) > 60:
                continue
            if noisy.search(subitem or ""):
                subitem = ""
            source_line = f"{section_name}: {line}"

            if valid_expected(subitem) and not noisy.search(subitem) and not tool_like(subitem):
                add("Excel测试子项", f"N972 中 {item} 的测试子项是什么？", [subitem], source_line)
            if valid_expected(tool) and tool.lower() not in {"tool", "test tool"}:
                anchor = preferred_anchor(item, subitem)
                add("Excel测试工具", f"N972 中 {anchor} 使用什么测试工具？", [tool], source_line)
            if valid_expected(threshold):
                anchor = preferred_anchor(item, subitem)
                add("Excel门限", f"N972 中 {anchor} 的门限或标准是什么？", [threshold], source_line)
            if valid_expected(scheme):
                anchor = preferred_anchor(item, subitem)
                add("Excel测试方案", f"N972 中 {anchor} 的测试方案是什么？", [scheme], source_line)
            if valid_expected(judgement):
                anchor = preferred_anchor(item, subitem)
                add("Excel判定依据", f"N972 中 {anchor} 的判定方式是什么？", [judgement], source_line)
            if valid_expected(fixture):
                anchor = preferred_anchor(item, subitem)
                add("Excel治具设备", f"N972 中 {anchor} 需要什么治具或设备？", [fixture], source_line)

    return cases


def choose_diverse_cases(cases: list[Case]) -> list[Case]:
    rng = random.Random(SEED)
    buckets: dict[str, list[Case]] = {}
    for case in cases:
        buckets.setdefault(case.area, []).append(case)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    per_area_limit = {
        "Excel规格字段": 8,
        "Excel测试子项": 6,
        "Excel测试工具": 6,
        "Excel门限": 6,
        "Excel测试方案": 6,
        "Excel判定依据": 4,
        "Excel治具设备": 4,
        "Excel站位覆盖": 2,
    }

    selected: list[Case] = []
    for area in sorted(buckets):
        selected.extend(buckets[area][: per_area_limit.get(area, 3)])

    rng.shuffle(selected)
    return selected[:TARGET_CASES]


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


def open_webui_session() -> None:
    subprocess.run(
        [
            str(PLAYWRIGHT),
            "--yes",
            "--package",
            "@playwright/cli",
            "playwright-cli",
            f"-s={SESSION}",
            "open",
            WEBUI_URL + "/",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        encoding="utf-8",
    )


def pw_eval(js: str, timeout: int = 180, label: str = "") -> Any:
    cmd = [
        str(PLAYWRIGHT),
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
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            encoding="utf-8",
        )
        last = proc
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                return parse_playwright_stdout(proc.stdout)
            except Exception:
                pass
        if attempt == 0:
            open_webui_session()
        time.sleep(1)
    stdout = last.stdout[:300] if last else ""
    stderr = last.stderr[:300] if last else ""
    raise RuntimeError(f"playwright eval failed {label}: stdout={stdout!r} stderr={stderr!r}")


def run_playwright_chunk(questions: list[str]) -> list[str]:
    payload = json.dumps(questions, ensure_ascii=True, separators=(",", ":"))
    payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    js = (
        "async () => {"
        f"const qs=JSON.parse(atob('{payload_b64}'));"
        "const out=[];"
        "for (const q of qs) {"
        "const params=new URLSearchParams({question:q,limit:'5',use_web:'false',allow_api:'false'});"
        "const resp=await fetch('/api/query?'+params.toString());"
        "let data={};"
        "try { data=await resp.json(); } catch (e) { data={answer:await resp.text()}; }"
        "out.push(data.answer || data.response || data.message || JSON.stringify(data));"
        "}"
        "return JSON.stringify(out);"
        "}"
    )
    result = pw_eval(js, timeout=180, label=f"chunk-{len(questions)}")
    if isinstance(result, list):
        return [str(item) for item in result]
    raise RuntimeError(f"unexpected playwright result: {result!r}")


def evaluate(cases: list[Case]) -> list[Result]:
    results: list[Result] = []
    open_webui_session()
    for start in range(0, len(cases), CHUNK_SIZE):
        chunk = cases[start : start + CHUNK_SIZE]
        answers = run_playwright_chunk([c.question for c in chunk])
        for case, answer in zip(chunk, answers):
            ans_norm = normalize(answer)
            missing = [e for e in case.expected if normalize(e) not in ans_norm]
            if missing:
                status = "FAIL"
                reason = "missing: " + " / ".join(missing)
            else:
                status = "PASS"
                reason = "matched expected source substring"
            results.append(
                Result(
                    case_id=case.case_id,
                    area=case.area,
                    question=case.question,
                    expected=case.expected,
                    status=status,
                    reason=reason,
                    source_line=case.source_line,
                    answer=compact(answer, 320),
                )
            )
    return results


def main() -> int:
    text = load_n972_text()
    all_cases = build_cases(text)
    cases = choose_diverse_cases(all_cases)
    if len(cases) < TARGET_CASES:
        print(f"warning: generated only {len(cases)} cases from {len(all_cases)} candidates", file=sys.stderr)
    results = evaluate(cases)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = len(results) - passed

    print(json.dumps(
        {
            "file": "N972-ACER_Test plan_V3.1.xlsx",
            "seed": SEED,
            "candidate_cases": len(all_cases),
            "sampled_cases": len(cases),
            "pass": passed,
            "fail": failed,
            "results": [asdict(r) for r in results],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
