#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a workbook for checking the eval set's reference answers by hand.

One row per question, the most doubtful first: questions the API answer did not fully match,
then those the local answer missed, then the rest. Each row shows the reference answer and
keywords, where in which document they come from, and the system's latest answers with their
verdicts, so a reviewer can tell a wrong reference from a wrong answer. The reviewer fills in the
yellow columns; eval/apply_review.py writes the decisions back into eval/eval_set.jsonl.

    python eval/review_sheet.py --api-run eval/results/<API run> --local-run eval/results/<local run> \
        [--suggestions eval/results/<suggestions>.json] [--out eval/results/<name>.xlsx]

The workbook and any suggestions file describe the indexed documents, so they belong in
eval/results/, which stays on this machine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_lib import CATEGORIES, DEFAULT_SET, RESULTS_DIR  # noqa: E402

FONT = "Arial"
ANSWER_CHARS = 900
DECISIONS = ("正确", "修改", "删除")
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
HEADER_FILL = PatternFill("solid", fgColor="1769AA")
PRIORITY_FILLS = {1: PatternFill("solid", fgColor="FCE4E4"), 2: PatternFill("solid", fgColor="FFF4E5")}
THIN = Side(style="thin", color="D8DEE6")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

# (header, width, input column?)
COLUMNS = [
    ("优先级", 7, False),
    ("编号", 7, False),
    ("类别", 11, False),
    ("问题", 28, False),
    ("标准答案（现在）", 40, False),
    ("关键词（现在，；分隔）", 26, False),
    ("出处（文档 / 章节）", 26, False),
    ("系统答案（开 API）", 55, False),
    ("判分（开 API）", 22, False),
    ("系统答案（本机）", 45, False),
    ("判分（本机）", 22, False),
    ("建议", 36, False),
    ("你的结论", 10, True),
    ("修改后的标准答案", 40, True),
    ("修改后的关键词（；分隔）", 26, True),
    ("备注", 24, True),
]
DECISION_COLUMN = "M"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_outputs(run_dir: Path | None) -> tuple[dict[str, dict], dict[str, dict]]:
    """Answers and judge verdicts of a run, by question id (empty when no run is given)."""
    if not run_dir:
        return {}, {}
    answers = {row["id"]: row.get("end_to_end") or {} for row in read_jsonl(run_dir / "raw_responses.jsonl")}
    judge_path = run_dir / "judge.jsonl"
    verdicts = {row["id"]: row for row in read_jsonl(judge_path)} if judge_path.exists() else {}
    return answers, verdicts


def refused(answer: dict) -> bool:
    return answer.get("source_type") in (None, "", "none")


def verdict_text(item: dict, answer: dict, verdicts: dict[str, dict]) -> tuple[str, bool]:
    """A short verdict and whether it marks the question as doubtful."""
    if not answer:
        return "", False
    if item["category"] == "E":
        return ("正确拒答", False) if refused(answer) else ("给了答案（应拒答）", True)
    verdict = verdicts.get(item["id"])
    if not verdict:
        return ("拒答" if refused(answer) else "作答（未判分）"), refused(answer)
    names = {"correct": "答对", "partial": "不完整", "wrong": "答错", "unjudged": "未判分"}
    label = names.get(verdict.get("verdict"), str(verdict.get("verdict")))
    reason = str(verdict.get("reason") or "")
    if verdict.get("verdict") == "correct":
        return label, False
    return f"{label}：{reason[:160]}", True


def build(eval_set: Path, api_run: Path | None, local_run: Path | None, suggestions: dict[str, str], out: Path) -> dict:
    items = read_jsonl(eval_set)
    api_answers, api_verdicts = run_outputs(api_run)
    local_answers, local_verdicts = run_outputs(local_run)

    rows = []
    for item in items:
        api_answer, local_answer = api_answers.get(item["id"], {}), local_answers.get(item["id"], {})
        api_text, api_doubt = verdict_text(item, api_answer, api_verdicts)
        local_text, local_doubt = verdict_text(item, local_answer, local_verdicts)
        priority = 1 if api_doubt or item["id"] in suggestions else 2 if local_doubt else 3
        keywords = item.get("gold_chunk_keywords") if item["category"] != "E" else item.get("absent_terms")
        source = "；".join(item.get("gold_doc_ids") or []) + (f" / {item['source_ref']}" if item.get("source_ref") else "")
        rows.append(
            [
                priority,
                item["id"],
                f"{item['category']} {CATEGORIES.get(item['category'], '')}",
                item["question"],
                item.get("gold_answer_summary", ""),
                "；".join(keywords or []),
                source or ("（负样本：资料里应当没有）" if item["category"] == "E" else ""),
                str(api_answer.get("answer") or "")[:ANSWER_CHARS],
                api_text,
                str(local_answer.get("answer") or "")[:ANSWER_CHARS],
                local_text,
                suggestions.get(item["id"], ""),
            ]
        )
    order = {category: index for index, category in enumerate("ABCDE")}
    rows.sort(key=lambda row: (row[0], order.get(row[1][0], 9), row[1]))

    workbook = Workbook()
    guide = workbook.active
    guide.title = "说明"
    sheet = workbook.create_sheet("核对")
    summary = workbook.create_sheet("汇总")

    # Review sheet
    for index, (header, width, is_input) in enumerate(COLUMNS, start=1):
        cell = sheet.cell(row=1, column=index, value=header)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF")
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        cell.border = BORDER
        sheet.column_dimensions[cell.column_letter].width = width
    for row_index, values in enumerate(rows, start=2):
        for column_index, value in enumerate(values, start=1):
            cell = sheet.cell(row=row_index, column=column_index, value=value)
            cell.font = Font(name=FONT, size=10)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = BORDER
            if values[0] in PRIORITY_FILLS and column_index <= 2:
                cell.fill = PRIORITY_FILLS[values[0]]
        for column_index in range(len(values) + 1, len(COLUMNS) + 1):
            cell = sheet.cell(row=row_index, column=column_index)
            cell.fill = INPUT_FILL
            cell.font = Font(name=FONT, size=10)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = BORDER
        sheet.row_dimensions[row_index].height = 120
    last_row = len(rows) + 1
    sheet.freeze_panes = "E2"
    sheet.auto_filter.ref = f"A1:{sheet.cell(row=1, column=len(COLUMNS)).column_letter}{last_row}"
    choices = DataValidation(type="list", formula1='"' + ",".join(DECISIONS) + '"', allow_blank=True)
    choices.error = "请选择：正确 / 修改 / 删除"
    choices.prompt = "正确：标准答案没问题；修改：在右边两列写新的；删除：这道题不要了"
    sheet.add_data_validation(choices)
    choices.add(f"{DECISION_COLUMN}2:{DECISION_COLUMN}{last_row}")
    sheet[f"{DECISION_COLUMN}1"].comment = Comment("下拉选择：正确 / 修改 / 删除；留空表示还没核对", "review_sheet")

    # Summary sheet: counts update as decisions are filled in.
    decisions_ref = f"'核对'!${DECISION_COLUMN}$2:${DECISION_COLUMN}${last_row}"
    priority_ref = f"'核对'!$A$2:$A${last_row}"
    summary_rows = [
        ("题目总数", f"=COUNTA('核对'!$B$2:$B${last_row})"),
        ("已核对：正确", f'=COUNTIF({decisions_ref},"正确")'),
        ("已核对：修改", f'=COUNTIF({decisions_ref},"修改")'),
        ("已核对：删除", f'=COUNTIF({decisions_ref},"删除")'),
        ("还没核对", "=B2-B3-B4-B5"),
        ("优先级 1 的题（建议先看）", f"=COUNTIF({priority_ref},1)"),
        ("优先级 1 里还没核对的", f'=COUNTIFS({priority_ref},1,{decisions_ref},"")'),
    ]
    summary["A1"], summary["B1"] = "项目", "数量"
    for cell in (summary["A1"], summary["B1"]):
        cell.font = Font(name=FONT, bold=True, color="FFFFFF")
        cell.fill = HEADER_FILL
    for index, (label, formula) in enumerate(summary_rows, start=2):
        summary.cell(row=index, column=1, value=label).font = Font(name=FONT)
        summary.cell(row=index, column=2, value=formula).font = Font(name=FONT)
    summary.column_dimensions["A"].width = 30
    summary.column_dimensions["B"].width = 10

    # Guide sheet
    run_note = f"系统答案来自：开 API {api_run.name if api_run else '（无）'}；本机 {local_run.name if local_run else '（无）'}。"
    guide_lines = [
        ("标准答案核对表", True),
        (f"生成时间 {time.strftime('%Y-%m-%d %H:%M')}，共 {len(rows)} 道题。{run_note}", False),
        ("", False),
        ("怎么核对一道题", True),
        ("1. 看“问题”和“标准答案（现在）”：标准答案是否正好回答了这个问题——不多（问题没问的内容不该算必答）、不少、没有错。", False),
        ("2. 打开“出处”里的文档和章节，确认标准答案和关键词与原文一致（数值、名称照原文）。", False),
        ("3. 负样本（类别 E）：确认资料里确实没有这项信息；如果资料其实写了，就把它改成正样本（在备注里写清楚），或删除。", False),
        ("4. 参考右边的系统答案和判分：系统被判“不完整/答错”时，先判断是系统错了，还是标准答案要求过多。“建议”列是我的看法，仅供参考。", False),
        ("5. 在黄色列里填写结论（只改黄色列，其他列不要动）：", False),
        ("   · 你的结论：下拉选择“正确”（标准答案没问题）、“修改”（在右边写新的标准答案和/或关键词）、“删除”（这道题不要了）；留空表示还没核对。", False),
        ("   · 修改后的标准答案：一两句话写出应答到的要点；只改关键词时可以留空。", False),
        ("   · 修改后的关键词：答案里必须出现的原文词语，用“；”分隔，每个都要能在原文里找到；负样本填资料里不应出现的词。", False),
        ("建议先看“优先级 1”（红色编号）的题，再看“优先级 2”（橙色），其余的抽查即可。“汇总”页会随填写自动统计进度。", False),
        ("", False),
        ("示例（格式参考，不在核对页里）", True),
        ("编号 B99｜问题：设备预热要多久｜标准答案（现在）：预热 30 秒后目视检查；拍照测试用二维码图片比对", False),
        ("你的结论：修改｜修改后的标准答案：预热 30 秒后再目视检查影像｜修改后的关键词：30秒；目视｜备注：问题只问预热时长，拍照比对不算必答", False),
        ("", False),
        ("填完以后", True),
        ("保存并关闭这个文件，在项目目录运行：", False),
        (r".venv\Scripts\python.exe eval\apply_review.py <这个文件的路径>", False),
        ("它会先备份评测集，再把结论写回 eval\\eval_set.jsonl（正确/修改的题标为 verified，删除的题移除），并检查新的关键词是否都能在原文里找到。", False),
        ("之后重新跑评测，准确率就是按核对过的标准计算的。可以分几次核对，每次填完都可以运行一次。", False),
    ]
    for index, (text, bold) in enumerate(guide_lines, start=1):
        cell = guide.cell(row=index, column=1, value=text)
        cell.font = Font(name=FONT, bold=bold, size=12 if bold else 10)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    guide.column_dimensions["A"].width = 120
    workbook.active = 1
    workbook.calculation.fullCalcOnLoad = True
    out.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(out)
    return {"rows": len(rows), "priority": {p: sum(1 for row in rows if row[0] == p) for p in (1, 2, 3)}, "out": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a workbook for checking the eval set's reference answers.")
    parser.add_argument("--set", default=str(DEFAULT_SET))
    parser.add_argument("--api-run", help="An API-mode eval run (judged), e.g. eval/results/<time>-<label>.")
    parser.add_argument("--local-run", help="A local eval run (judged).")
    parser.add_argument("--suggestions", help="JSON {question id: suggestion} shown in the 建议 column.")
    parser.add_argument("--out", default=str(RESULTS_DIR / f"answer_review_{time.strftime('%Y%m%d')}.xlsx"))
    args = parser.parse_args()
    suggestions = json.loads(Path(args.suggestions).read_text(encoding="utf-8")) if args.suggestions else {}
    result = build(
        Path(args.set),
        Path(args.api_run) if args.api_run else None,
        Path(args.local_run) if args.local_run else None,
        suggestions,
        Path(args.out),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
