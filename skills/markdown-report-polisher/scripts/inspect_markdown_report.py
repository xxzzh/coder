#!/usr/bin/env python3
"""检查 Markdown 报告中常见的结构缺口。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REQUIRED_SECTIONS = {
    "摘要": ("summary", "摘要"),
    "进展": ("progress", "进展"),
    "风险": ("risks", "risk", "风险"),
    "决策": ("decisions", "decision", "决策"),
    "行动项": ("action items", "actions", "行动项", "待办事项"),
}
OWNER_PATTERN = re.compile(r"\b(owner|owners|assignee|担当|负责人)\b", re.IGNORECASE)
DATE_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b(today|tomorrow|friday|monday|eod)\b|今天|明天|周五|周一|下周",
    re.IGNORECASE,
)
ACTION_PATTERN = re.compile(r"^\s*[-*]\s+(?:\[[ xX]\]\s+)?(.+)", re.MULTILINE)


def headings(markdown: str) -> list[str]:
    return [match.group(1).strip().lower() for match in re.finditer(r"^#{1,3}\s+(.+)$", markdown, re.MULTILINE)]


def inspect(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    found_headings = headings(text)
    issues: list[str] = []

    for label, aliases in REQUIRED_SECTIONS.items():
        if not any(any(alias in heading for alias in aliases) for heading in found_headings):
            issues.append(f"缺少章节：{label}")

    actions = ACTION_PATTERN.findall(text)
    unchecked_actions = [line for line in actions if not line.startswith("[x]")]
    if not actions:
        issues.append("缺少项目符号行动项")
    if unchecked_actions and not OWNER_PATTERN.search(text):
        issues.append("行动项可能缺少负责人")
    if unchecked_actions and not DATE_PATTERN.search(text):
        issues.append("行动项可能缺少截止日期")

    long_lines = [index for index, line in enumerate(text.splitlines(), start=1) if len(line) > 140]
    if long_lines:
        preview = ", ".join(str(line) for line in long_lines[:5])
        issues.append(f"超过 140 个字符的长行：{preview}")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 Markdown 报告。")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    issues = inspect(args.path)
    if not issues:
        print("通过：报告具备预期的高层结构。")
        return

    print("报告问题：")
    for issue in issues:
        print(f"- {issue}")


if __name__ == "__main__":
    main()
