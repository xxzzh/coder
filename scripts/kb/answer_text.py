"""Answer text without a model: polishing, offline English-to-Chinese translation and the grounded local answer."""

from __future__ import annotations

from html import unescape
import re
from typing import Any

import project_runtime

from kb.citations import SECONDARY_BLOCK_MAX_CHARS, strip_native_reference_markers, synthesis_citations
from kb.text import human_text, strip_markdown_emphasis, to_simplified, trim_terminal_punctuation


def build_answer(question: str, sources: list[dict[str, Any]]) -> str:
    citations = [source["citation"] for source in sources[:3] if source.get("citation")]
    if not citations:
        return "本地知识库没有找到足够依据。"
    if len(citations) == 1:
        return f"根据本地知识库，{trim_terminal_punctuation(citations[0])}。"
    return "根据本地知识库，" + "；".join(trim_terminal_punctuation(item) for item in citations) + "。"


def polish_internal_answer_text(text: str) -> str:
    text = str(text or "")
    if not text:
        return text
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
    translated_citations = [f"{' '.join(text.splitlines())} [{source_id}]" for text, source_id in pieces]
    if translated_citations:
        if source_type == "web_search":
            prefix = "根据联网搜索资料，"
        elif source_type == "hybrid_web_knowledge":
            prefix = "根据本地知识库和联网资料，"
        else:
            prefix = "根据本地知识库，"
        lead_text, lead_id = pieces[0]
        if "\n" in lead_text:
            # A whole section or table: one bullet per line so the Web UI renders a list.
            answer = prefix.rstrip("，") + "：\n\n" + "\n".join(f"- {line}" for line in lead_text.splitlines()) + f" [{lead_id}]"
            # A supporting section longer than a snippet was cited whole (process questions):
            # it keeps its own list instead of being flattened into the one-line 补充.
            whole_sections = [(text, source_id) for text, source_id in pieces[1:] if "\n" in text and len(text) > SECONDARY_BLOCK_MAX_CHARS]
            snippets = [f"{' '.join(text.splitlines())} [{source_id}]" for text, source_id in pieces[1:] if (text, source_id) not in whole_sections]
            for text, source_id in whole_sections:
                answer += "\n\n另见：\n\n" + "\n".join(f"- {line}" for line in text.splitlines()) + f" [{source_id}]"
            if snippets:
                answer += "\n\n补充：" + "；".join(snippets) + "。"
            result["answer"] = answer
        else:
            result["answer"] = prefix + "；".join(translated_citations) + "。"
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
