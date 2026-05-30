#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一个最小但完整的客服工单 agent 示例。"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


Ticket = dict[str, Any]
ToolResult = dict[str, Any]


@dataclass(frozen=True)
class Step:
    name: str
    reason: str
    tool: str | None = None


class SupportTools:
    """本地工具，用来模拟真实 agent 会调用的 API 或系统。"""

    def __init__(self) -> None:
        self.articles = {
            "refund": "未使用套餐可在 30 天内申请退款。",
            "login": "请用户先重置密码，并检查 MFA 或验证码设置。",
            "billing": "需要收集发票编号、套餐名称和付款方式后四位。",
        }

    def search_kb(self, topic: str) -> ToolResult:
        return {"topic": topic, "article": self.articles.get(topic, "没有匹配到知识库文章。")}

    def create_refund_case(self, ticket: Ticket) -> ToolResult:
        return {
            "case_id": f"RF-{abs(hash(ticket.get('id', 'unknown'))) % 100000:05d}",
            "status": "已创建草稿",
        }

    def escalate(self, ticket: Ticket, reason: str) -> ToolResult:
        return {
            "queue": "高级客服队列",
            "reason": reason,
            "ticket_id": ticket.get("id", "unknown"),
        }


class CustomerSupportAgent:
    def __init__(self, tools: SupportTools) -> None:
        self.tools = tools
        self.tool_registry: dict[str, Callable[..., ToolResult]] = {
            "search_kb": self.tools.search_kb,
            "create_refund_case": self.tools.create_refund_case,
            "escalate": self.tools.escalate,
        }

    def run(self, ticket: Ticket) -> dict[str, Any]:
        observation = self.observe(ticket)
        plan = self.plan(observation)
        tool_results = self.act(ticket, observation, plan)
        return self.respond(ticket, observation, plan, tool_results)

    def observe(self, ticket: Ticket) -> dict[str, Any]:
        text = f"{ticket.get('subject', '')} {ticket.get('body', '')}".lower()
        topic = self._classify_topic(text)
        priority = "high" if re.search(r"\burgent\b|\bchargeback\b|\blawyer\b|紧急|拒付|律师|投诉", text) else "normal"
        return {"topic": topic, "priority": priority, "text": text}

    def plan(self, observation: dict[str, Any]) -> list[Step]:
        steps = [Step("查询政策", "先用知识库内容约束回复，避免凭空判断。", "search_kb")]
        if observation["topic"] == "refund":
            steps.append(Step("准备退款工单", "退款诉求需要创建 case 记录。", "create_refund_case"))
        if observation["priority"] == "high":
            steps.append(Step("升级处理", "高风险表达需要高级客服复核。", "escalate"))
        return steps

    def act(self, ticket: Ticket, observation: dict[str, Any], plan: list[Step]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for step in plan:
            if step.tool == "search_kb":
                results.append({"step": step.name, **self.tool_registry[step.tool](observation["topic"])})
            elif step.tool == "create_refund_case":
                results.append({"step": step.name, **self.tool_registry[step.tool](ticket)})
            elif step.tool == "escalate":
                results.append({"step": step.name, **self.tool_registry[step.tool](ticket, step.reason)})
        return results

    def respond(
        self,
        ticket: Ticket,
        observation: dict[str, Any],
        plan: list[Step],
        tool_results: list[ToolResult],
    ) -> dict[str, Any]:
        action_items = ["基于匹配到的知识库政策回复用户。"]
        if observation["topic"] == "refund":
            action_items.append("确认退款资格，并提交已创建的退款工单草稿。")
        if observation["priority"] == "high":
            action_items.append("等待高级客服复核后再发送最终回复。")

        return {
            "ticket_id": ticket.get("id", "unknown"),
            "classification": {"topic": observation["topic"], "priority": observation["priority"]},
            "plan": [{"name": step.name, "reason": step.reason, "tool": step.tool} for step in plan],
            "tool_results": tool_results,
            "next_actions": action_items,
        }

    @staticmethod
    def _classify_topic(text: str) -> str:
        if "refund" in text or "cancel" in text or "退款" in text or "取消" in text:
            return "refund"
        if "login" in text or "password" in text or "mfa" in text or "登录" in text or "密码" in text or "验证码" in text:
            return "login"
        if "invoice" in text or "billing" in text or "charged" in text or "发票" in text or "账单" in text or "扣费" in text:
            return "billing"
        return "general"


def read_skill_name(skill_file: Path) -> str:
    text = skill_file.read_text(encoding="utf-8-sig")
    match = re.search(r"^name:\s*(.+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip().strip("\"'")
    return skill_file.parent.name


def list_skills(extra_roots: list[Path]) -> list[dict[str, str]]:
    workspace_root = Path(__file__).resolve().parents[2]
    default_roots = [
        workspace_root / "skills",
        Path.home() / ".codex" / "skills",
    ]
    skills: list[dict[str, str]] = []
    seen: set[Path] = set()

    for root in [*default_roots, *extra_roots]:
        if not root.exists():
            continue
        for skill_file in root.rglob("SKILL.md"):
            skill_dir = skill_file.parent.resolve()
            if skill_dir in seen:
                continue
            seen.add(skill_dir)
            skills.append({"name": read_skill_name(skill_file), "path": str(skill_dir)})

    return sorted(skills, key=lambda item: item["name"])


def load_ticket(path: Path | None, ticket_id: str | None, subject: str | None, body: str | None) -> Ticket:
    if path is None:
        ticket = {
            "id": "T-1001",
            "subject": "紧急：误续费后需要退款",
            "body": "我今天被扣费了，需要退款，否则我会申请拒付。",
        }
    else:
        ticket = json.loads(path.read_text(encoding="utf-8-sig"))

    if ticket_id is not None:
        ticket["id"] = ticket_id
    if subject is not None:
        ticket["subject"] = subject
    if body is not None:
        ticket["body"] = body
    return ticket


def print_skills(skills: list[dict[str, str]]) -> None:
    if not skills:
        print("没有找到可见的 skills。")
        return
    for skill in skills:
        print(f"- {skill['name']}: {skill['path']}")


def result_only(full_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_results": full_result["tool_results"],
        "next_actions": full_result["next_actions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行客服工单 agent 示例。")
    parser.add_argument("--ticket", type=Path, help="JSON 工单文件路径。不提供时使用内置演示工单。")
    parser.add_argument("--id", dest="ticket_id", help="工单编号。")
    parser.add_argument("--subject", help="直接输入工单标题。")
    parser.add_argument("--body", help="直接输入工单正文。")
    parser.add_argument("--list-skills", action="store_true", help="列出当前 agent 可看到的 skills。")
    parser.add_argument("--skills-root", action="append", type=Path, default=[], help="额外的 skills 根目录，可重复传入。")
    parser.add_argument("--verbose", action="store_true", help="输出完整分类、计划和工具调用细节。")
    args = parser.parse_args()

    if args.list_skills:
        print_skills(list_skills(args.skills_root))
        return

    full_result = CustomerSupportAgent(SupportTools()).run(load_ticket(args.ticket, args.ticket_id, args.subject, args.body))
    output = full_result if args.verbose else result_only(full_result)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
