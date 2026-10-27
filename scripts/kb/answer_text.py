"""Answer text without a model: polishing, offline English-to-Chinese translation and the grounded local answer."""

from __future__ import annotations

from html import unescape
import re
from typing import Any

import project_runtime

from kb.citations import strip_native_reference_markers, synthesis_citations
from kb.text import human_text, strip_markdown_emphasis, to_simplified, trim_terminal_punctuation


def build_answer(question: str, sources: list[dict[str, Any]]) -> str:
    citations = [source["citation"] for source in sources[:3] if source.get("citation")]
    if not citations:
        return "本地知识库没有找到足够依据。"
    if len(citations) == 1:
        return f"根据本地知识库，{trim_terminal_punctuation(citations[0])}。"
    return "根据本地知识库，" + "；".join(trim_terminal_punctuation(item) for item in citations) + "。"


FENCED_CODE_PATTERN = re.compile(r"(?m)(^```[^\n]*\n[\s\S]*?\n```[ \t]*$)")


def polish_internal_answer_text(text: str) -> str:
    """Tidy an answer's text; fenced code blocks are kept byte for byte (pipes, semicolons and
    commas in code are not punctuation)."""
    text = str(text or "")
    if not text:
        return text
    parts = FENCED_CODE_PATTERN.split(text)
    if len(parts) == 1:
        return polish_plain_answer_text(text)
    polished = ""
    for index, part in enumerate(parts):
        if index % 2:
            polished += ("\n" if polished else "") + part.strip()
            continue
        plain = polish_plain_answer_text(part)
        if plain:
            separator = "\n\n" if re.match(r"[ \t]*\n[ \t]*\n", part) else "\n"
            polished += (separator if polished else "") + plain
    return polished.strip()


def polish_plain_answer_text(text: str) -> str:
    text = unescape(text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\((?:https?|file|#)[^)]+\)", r"\1", text)
    # Keep code inside fences; only the fence markers are noise in a plain-text answer.
    text = re.sub(r"(?m)^[ \t]*```[\w+-]*[ \t]*$\n?", "", text)
    text = strip_markdown_emphasis(text)
    text = re.sub(r"[┌┬┐├┼┤└┴┘│┃╭╮╰╯╔╦╗╠╬╣╚╩╝]+", " ", text)
    text = re.sub(r"[─━—]{2,}", " ", text)
    # Markdown table separator rows such as |----|:---:| carry no content.
    text = re.sub(r"(?:\|\s*:?-{3,}:?\s*)+\|?", "|", text)
    text = re.sub(r"(?<![\w-])-{3,}(?![\w-])", " ", text)
    # Table cell pipes become commas; slashes and backslashes belong to paths and identifiers.
    text = re.sub(r"\s*\|+\s*", "，", text)
    text = re.sub(r"[<>《》]{2,}", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]+", "", text)
    # Only commas next to Chinese text are prose; [1.25,3.75] or f(a,b) must stay intact.
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s*,\s*|\s*,\s*(?=[\u4e00-\u9fff])", "，", text)
    text = re.sub(r"，{2,}", "，", text)
    text = re.sub(r"[ \t]+([，。；：、！？])", r"\1", text)
    text = re.sub(r"([，；：、])[ \t]+", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"(?m)^[ \t]*[，;；:：]+[ \t]*", "", text)
    text = re.sub(r"(?m)[ \t]*[，;；]+[ \t]*$", "", text)
    text = re.sub(r"([：:])(?=-\s)", r"\1\n", text)
    text = re.sub(r"([。！？])，", r"\1", text)
    text = re.sub(r"，([。！？])", r"\1", text)
    return text.strip()


def polish_internal_answer(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("source_type") == "knowledge_base" and result.get("answer"):
        result = result.copy()
        result["answer"] = polish_internal_answer_text(str(result.get("answer", "")))
    return result


# The offline English-to-Chinese model was trained on Wikipedia and sometimes appends this
# citation boilerplate to garbled output; such a translation is not used.
MACHINE_TRANSLATION_ARTIFACT = "原始内容存档"


def looks_like_code(text: str) -> bool:
    """Source code reads the same in every language; machine translation only garbles it."""
    tokens = text.split()
    code_tokens = [
        token
        for token in tokens
        if re.search(r"[(){}\[\];=<>]|::|->|\w\.\w|[a-z][A-Z]|(?:^|\W)_\w", token)
    ]
    return len(code_tokens) >= max(2, len(tokens) * 0.4)


def local_translate_to_chinese(text: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", str(text or "")):
        # Chinese evidence needs no translation: keep its line structure and full length
        # (citations are already budgeted) instead of flattening it into 520 characters.
        lines = (re.sub(r"[ \t]+", " ", line).strip() for line in to_simplified(unescape(str(text))).splitlines())
        return "\n".join(strip_markdown_emphasis(line) for line in lines if line)
    text = human_text(text, max_len=520, simplify=True)
    if not text:
        return ""
    french_markers = (
        " à ",
        " au ",
        " aux ",
        " avec ",
        " dans ",
        " des ",
        " du ",
        " est ",
        " et ",
        " les ",
        " mon ",
        " nous ",
        " pour ",
        " que ",
        " qui ",
        " une ",
        " vous ",
    )
    lowered = f" {text.lower()} "
    if re.search(r"[àâçéèêëîïôùûüÿœæ]", lowered) or sum(marker in lowered for marker in french_markers) >= 2:
        return ""
    if looks_like_code(text):
        return ""
    try:
        from argostranslate import translate  # type: ignore

        project_runtime.reuse_local_stanza_resources()
        translated = human_text(translate.translate(text, "en", "zh"), max_len=520, simplify=True)
        if MACHINE_TRANSLATION_ARTIFACT in translated:
            return ""
        replacements = {
            "Ohm的定律": "欧姆定律",
            "Ohm定律": "欧姆定律",
            "反应API": "Responses API",
            "客户SDK": "客户端 SDK",
        }
        for original, replacement in replacements.items():
            translated = translated.replace(original, replacement)
        translated = re.sub(r"(?<=[\u4e00-\u9fff]),", "，", translated)
        translated = re.sub(r"(?<=[\u4e00-\u9fff])\.", "。", translated)
        return translated
    except Exception:
        return ""


OFFLINE_TRANSLATION_SETUP_HINT = (
    "需要联网安装一次离线英译中模型：\n"
    "python -m pip install -r requirements.txt\n"
    ".\\run-agent.bat translation-setup\n"
    ".\\run-agent.bat verify"
)


# Without a model the answer is assembled from the evidence itself. These rules turn the
# evidence's own structure into a readable answer instead of one bullet per raw line: section
# titles and step names become group names, table rows and Excel records become one bullet per
# row or field, and code stays a code block.
CODE_LINE_PATTERN = re.compile(
    r"^\s*(?://|/\*|\*/|#include\b|using\s+[\w.]+|namespace\s"
    r"|(?:public|private|protected|internal|static|override|virtual|class|void|return|var|const)\b"
    r"|(?:if|for|foreach|while|switch|catch)\s*\(|else\b|do\s*\{|try\b|get\s*\{|set\s*\{"
    r"|\[[A-Za-z_][\w.]*(?:\(.*\))?(?:,\s*.+)?\]\s*$|[{}]\s*;?\s*$)"
    r"|[;{]\s*(?://.*)?$"
)
HEADING_PATTERN = re.compile(r"^(?:第?[一二三四五六七八九十百]+[、.．]\s*\S|步骤\s*\d+|阶段\s*\d+|【[^】]+】|#{1,6}\s+\S)")
# A tree diagram (├── 阶段1 / │ └── 子步骤) keeps its shape in a monospaced block; flattened
# into bullets it would lose which step belongs to which stage.
TREE_LINE_PATTERN = re.compile(r"^[\s│┃|]*[├└┣┗]\s*[─━]+|^[\s]*[│┃](?:\s|$)")
EMPTY_CELL_VALUES = {"", "-", "—", "–", "/", "\\", "na", "n/a", "无", "null", "none"}


def is_code_line(line: str) -> bool:
    text = line.strip().rstrip("…").rstrip()  # a line cut off by the citation budget is still code
    if "→" in text:
        return False
    return bool(CODE_LINE_PATTERN.search(text)) or bool(re.fullmatch(r"<[!?/\w][^<>]*>.*", text) and text.endswith(">"))


def numbered_steps(value: str) -> list[str]:
    """ "1.先… 2.再… 3.最后…" -> one step each; a single number (2.5V, 1.0 版) is not a list."""
    starts = [match.start() for match in re.finditer(r"(?<![\d.\w])(?:\d{1,2}[.、](?!\d)|[①②③④⑤⑥⑦⑧⑨])", value)]
    if len(starts) < 2:
        return []
    head = value[: starts[0]].strip()
    steps = [value[start:end].strip() for start, end in zip(starts, [*starts[1:], len(value)])]
    return ([head] if head else []) + [step for step in steps if step]


def record_lines(fields: list[tuple[str, str]]) -> list[str]:
    """One bullet per field; a field holding numbered steps becomes a group of its own after the
    plain fields, so the fields that follow it are not read as its steps."""
    plain: list[str] = []
    grouped: list[str] = []
    for name, value in fields:
        steps = numbered_steps(value)
        if steps and name:
            grouped += ["", name, *[f"- {step}" for step in steps]]
        else:
            plain.append(f"- {name}：{value}" if name else f"- {value}")
    return plain + grouped


def is_heading_line(line: str) -> bool:
    text = line.strip()
    return len(text) <= 40 and " | " not in text and (bool(HEADING_PATTERN.match(text)) or text.endswith(("：", ":")))


def table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in re.split(r"\s*\|\s*", line.strip().strip("|"))]


def evidence_record(line: str) -> tuple[str, list[tuple[str, str]]] | None:
    """An Excel record, "表名 | 字段：值 | 字段：值 …" -> (表名, fields). The row number and empty
    fields are dropped; a field named twice is merged."""
    cells = [cell for cell in table_cells(line) if cell]
    named = [cell for cell in cells[1:] if "：" in cell]
    if len(cells) < 3 or len(named) * 3 < (len(cells) - 1) * 2:
        return None
    title = cells[0] if "：" not in cells[0] else ""
    fields: dict[str, list[str]] = {}
    for cell in cells[1:] if title else cells:
        name, _, value = cell.partition("：")
        name, value = name.strip(), value.strip()
        if not value:
            name, value = "", name
        if name == "#" or value.lower() in EMPTY_CELL_VALUES:
            continue
        fields.setdefault(name, [])
        if value not in fields[name]:
            fields[name].append(value)
    return title, [(name, "，".join(values)) for name, values in fields.items()]


def table_row_bullet(header: list[str], row: list[str]) -> str:
    if len(header) == 2 and len(row) == 2:
        return f"- {row[0]}：{row[1]}" if row[1].lower() not in EMPTY_CELL_VALUES else f"- {row[0]}"
    details = [f"{name}：{value}" if name else value for name, value in zip(header[1:], row[1:]) if value.lower() not in EMPTY_CELL_VALUES]
    # The first column labels the row (an enum value may well be NONE or NULL): only a dash is empty.
    first = row[0] if row[0].strip() not in {"", "-", "—", "–"} else ""
    if first and details:
        return f"- {first} — {'；'.join(details)}"
    return f"- {first or '；'.join(details)}"


def evidence_layout(text: str) -> tuple[str, str, list[str]]:
    """(kind, title, lines) for one piece of evidence. kind is "record" (an Excel row), "prose"
    (a single sentence or paragraph) or "section"; lines are answer lines: bullets, group names,
    and fenced code blocks."""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return "prose", "", []
    if len(lines) == 1:
        record = evidence_record(lines[0])
        if record:
            title, fields = record
            return "record", title, record_lines(fields)
        return "prose", "", [lines[0]]
    title = ""
    if len(lines[0]) <= 40 and " | " not in lines[0] and not is_code_line(lines[0]) and not lines[0].endswith(("。", "；")):
        title, lines = lines[0].rstrip("：:"), lines[1:]
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        end = index
        while end < len(lines) and TREE_LINE_PATTERN.match(lines[end]):
            end += 1
        if end - index >= 3:
            out += ["```", *lines[index:end], "```"]
            index = end
            continue
        end = index
        while end < len(lines) and is_code_line(lines[end]):
            end += 1
        if end - index >= 2:
            out += ["```", *lines[index:end], "```"]
            index = end
            continue
        if " | " in line and not evidence_record(line):
            header = table_cells(line)
            end = index + 1
            while end < len(lines) and " | " in lines[end] and len(table_cells(lines[end])) == len(header):
                end += 1
            if end - index >= 2:
                out += [table_row_bullet(header, table_cells(row)) for row in lines[index + 1 : end]]
                index = end
                continue
        record = evidence_record(line)
        if record:
            record_title, fields = record
            if record_title:
                out += ["", record_title]
            out += record_lines(fields)
        elif is_heading_line(line) and index + 1 < len(lines):
            out += ["", line.rstrip("：:")]
        else:
            out.append(f"- {line}")
        index += 1
    return "section", title, out


def cite_block(lines: list[str], source_id: Any) -> list[str]:
    """Put the citation on the block's last text line, never inside a code block."""
    inside = False
    last_text = -1
    for position, line in enumerate(lines):
        if line == "```":
            inside = not inside
        elif not inside and line:
            last_text = position
    lines = list(lines)
    if last_text >= 0:
        lines[last_text] = f"{lines[last_text]} [{source_id}]"
    return lines


def has_text_line(lines: list[str]) -> bool:
    inside = False
    for line in lines:
        if line == "```":
            inside = not inside
        elif not inside and line:
            return True
    return False


def clean_title(title: str) -> str:
    # "工作表 参数表 / 站点 / 项目。 参数表" names a sheet by its path: keep the sheet.
    title = title.rsplit("。", 1)[-1].strip() or title
    return re.sub(r"^(?:第?[一二三四五六七八九十百]+[、.．]|\d+(?:\.\d+)*[、.．]?|#{1,6})\s*", "", title).strip() or title


def local_structured_answer(pieces: list[tuple[str, Any]], source_type: str | None) -> str:
    """The local answer: an opening line that names what was found, the best evidence laid out
    by its structure, then the other evidence in groups of its own."""
    where = {"web_search": "联网搜索资料", "hybrid_web_knowledge": "本地知识库和联网资料"}.get(str(source_type), "本地知识库")
    kind, title, lines = evidence_layout(pieces[0][0])

    if kind == "prose":
        opening, body = f"{lines[0]} [{pieces[0][1]}]", []
    else:
        if kind == "record":
            opening = f"在{where}的「{clean_title(title)}」中找到这条记录：" if title else f"在{where}中找到这条记录："
        else:
            opening = f"以下是{where}中「{clean_title(title)}」的内容：" if title else f"{where}中的相关内容："
        if not has_text_line(lines):  # only code: the opening line carries the citation
            opening = opening.rstrip("：") + f" [{pieces[0][1]}]："
        body = cite_block(lines, pieces[0][1])
    blocks = [opening, "\n".join(body).strip()]
    loose: list[str] = []
    for text, source_id in pieces[1:]:
        kind, title, lines = evidence_layout(text)
        if kind == "prose":
            loose.append(f"- {lines[0]} [{source_id}]")
            continue
        name = clean_title(title) if title else ("另见" if kind == "section" else "相关记录")
        if not has_text_line(lines):  # only code: the group name carries the citation
            name += f" [{source_id}]"
        blocks.append(name + "：\n" + "\n".join(cite_block(lines, source_id)).strip())
    if loose:
        blocks.append("补充：\n" + "\n".join(loose))
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(block for block in blocks if block))


def chinese_grounded_fallback(result: dict[str, Any]) -> dict[str, Any]:
    if not result.get("sources"):
        result["api_used"] = False
        return result
    source_type = result.get("source_type")
    citations = synthesis_citations(result)
    summary_source_ids = set(result.get("summary_source_ids") or [])
    if summary_source_ids:
        citations = [citation for citation in citations if citation.get("source_id") in summary_source_ids]
    citation_limit = 2 if source_type == "web_search" else 3
    pieces: list[tuple[str, Any]] = []
    for citation in citations[:citation_limit]:
        translated = local_translate_to_chinese(str(citation.get("quote", "")))
        if translated:
            translated = strip_native_reference_markers(translated)
            pieces.append((trim_terminal_punctuation(translated), citation["source_id"]))
    if pieces:
        result["answer"] = local_structured_answer(pieces, source_type)
        result["answer_mode"] = "local_translation_grounded_summary"
    elif source_type == "web_search":
        result["answer"] = (
            "已找到相关联网资料，但当前缺少可用的本地翻译模型。请查看下方引用来源；为避免误译，原始引用保持不变。\n\n"
            + OFFLINE_TRANSLATION_SETUP_HINT
        )
        result["answer_mode"] = "chinese_grounded_fallback"
    else:
        result["answer"] = (
            "已找到相关本地资料，但当前缺少可用的本地翻译模型。请查看下方引用来源；为避免误译，原始引用保持不变。\n\n"
            + OFFLINE_TRANSLATION_SETUP_HINT
        )
        result["answer_mode"] = "chinese_grounded_fallback"
    result["api_used"] = False
    return result
