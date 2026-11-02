#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chinese text for start.bat: its menu, prompts and messages.

cmd misreads some multi-byte lines of a UTF-8 batch file (they run as commands), so start.bat stays
ASCII and has this script print the text; Python writes UTF-8 to the console reliably.

    python scripts/start_menu.py menu          the menu, ending in the prompt for a number
    python scripts/start_menu.py <message>     one of MESSAGES
"""

from __future__ import annotations

import sys

MENU = """================================================
  本地知识库问答助手
================================================

  首次安装（按顺序各做一次）
    1. 安装依赖
    2. 初始化配置
    3. 安装本地向量模型（约 2.2 GB，首次需要联网）
    4. 配置大模型 API（可选，也可以在网页的设置页配置）

  日常使用
    5. 启动 Web UI（在网页里上传资料、提问）
    6. 检查安装和资料

  不用网页上传资料时
    7. 打开资料文件夹
    8. 处理新增和修改的资料
    9. 全部重新处理

  其他
   10. 命令行提问
   11. 列出已处理的资料
   12. 安装离线英译中模型
   13. OCR 测试材料评测
   14. 生成 OCR 测试材料

    0. 退出

请输入编号："""

MESSAGES = {
    "invalid": "没有这个编号，请重新输入。",
    "web": "正在启动 Web UI：http://127.0.0.1:8765/\n用完关闭这个窗口，或按 Ctrl+C 停止。",
    "open_raw": "把 PDF、Word（.docx / .doc）、Excel（.xlsx）、Markdown（.md）、TXT 文件放进打开的文件夹，再选 8 处理新增和修改的资料。",
    "rebuild_strict": "结果里 requires_review 大于 0 时，到 Web UI 的“资料”页查看需要确认的资料。",
    "verify": "看最后的 summary 一行：显示“安装正常”或“可以使用”就可以用了。",
    "ask_prompt": "请输入问题：",
    "ask_empty": "问题不能为空。",
    "ocr_eval_missing": "找不到 OCR 评测脚本（scripts\\evaluate_ocr_test_materials.py）。",
    "ocr_gen_missing": "找不到 OCR 测试材料生成脚本（scripts\\generate_ocr_test_materials.py）。",
}

# Printed without a line break: set /p reads the answer right after them.
PROMPTS = {"menu", "ask_prompt"}


def main() -> int:
    key = sys.argv[1] if len(sys.argv) > 1 else "menu"
    text = MENU if key == "menu" else MESSAGES.get(key)
    if text is None:
        return 2
    sys.stdout.reconfigure(encoding="utf-8")
    print(text, end="" if key in PROMPTS else "\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
