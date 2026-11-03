"""Follow-up questions in a conversation: a question that leans on earlier turns («它最多重试几次»,
«那 SFT 呢», «门限呢») is rewritten into one that stands alone, so retrieval, the answer cache and
the retrieval agent all see a complete question.

A question that already says what it is about is used exactly as typed, so single questions behave
as before. With the answer API on, the model fills in what a follow-up leaves out; without it, or
when that call fails, a few rules do: a pronoun takes the previous question's subject, «那 X 呢»
asks the previous question about X, «X呢» asks about the previous subject's X. When the documents
write X where the previous question has another word (排气温度 and 排气压力, 火车乘坐… and
飞机乘坐…), «X呢» swaps that word instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from api_providers import chat_completion_detailed
import llm_budget

from kb import relevance
from kb.config import DEFAULT_API_TIMEOUT_SECONDS, api_enabled
from kb.local_focus import title_rank


# Recent turns sent along with a follow-up, and how much of each is kept.
HISTORY_TURNS = 3
HISTORY_QUESTION_CHARS = 200
HISTORY_ANSWER_CHARS = 300
QUESTION_MAX_CHARS = 300

# Words that point back at something said before. «该» only as a pronoun («该测试»), not in «应该».
PRONOUNS = re.compile(
    r"它们|它|这个|那个|这些|那些|这项|那项|这种|那种|这一步|这一项|这里|那里|上述|"
    r"上面(?:提到)?的|前面(?:提到)?的|刚才(?:说|提到)?的|(?<!应)该(?=[一-鿿A-Za-z])"
)
# Openers that continue the conversation: «那…», «还有…».
LEADING = re.compile(r"^(?:那么|那|还有|另外|此外|然后|再问一下|顺便问一下)[，,、\s]*")
# A question that starts with what it asks, leaving out what it is about: «怎么判定», «具体步骤是什么».
PREDICATE_START = re.compile(
    r"^(?:怎么|怎样|如何|为什么|为何|是否|有没有|能不能|能否|可不可以|可以|需不需要|需要|要不要|应该|应当|"
    r"多少|多久|几|哪些|哪个|什么|具体|详细|一般|通常|默认|最多|最少|至少)"
)
# Adjectives that read as «X的具体步骤», not «X具体步骤».
NOUN_MODIFIER_START = re.compile(r"^(?:具体|详细|一般|通常|默认)")
QUESTION_WORD = re.compile(r"怎么|怎样|如何|为什么|为何|是否|有没有|能不能|能否|可以|需要|多少|多久|几|哪些|哪个|什么|是|有|会|要")
POLITE_PREFIX = re.compile(r"^(?:请问|我想问一下|我想问|想问一下|想问|帮我查一下|查一下|关于)[，,\s]*")
END_PUNCTUATION = "？?。！!；;，, "


def clean_history(history: Any) -> list[dict[str, str]]:
    """The last few turns from the client, each {question, answer}, trimmed; anything else dropped."""
    turns: list[dict[str, str]] = []
    if not isinstance(history, list):
        return turns
    for item in history[-HISTORY_TURNS:]:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()[:HISTORY_QUESTION_CHARS]
        answer = str(item.get("answer") or "").strip()[:HISTORY_ANSWER_CHARS]
        if question:
            turns.append({"question": question, "answer": answer})
    return turns


def needs_context(question: str) -> bool:
    """Whether the question leans on earlier turns rather than saying what it is about."""
    text = question.strip().rstrip(END_PUNCTUATION)
    core = LEADING.sub("", POLITE_PREFIX.sub("", text))
    return bool(
        PRONOUNS.search(text)
        or core != POLITE_PREFIX.sub("", text)
        or core.endswith("呢")
        or PREDICATE_START.match(core)
        or len(core) <= 4
    )


def subject_of(question: str) -> str:
    """What a question is about: the part before its first «的» («X100 Fan测试的完整流程是什么»),
    else the part before its first question word («G-sensor怎么测试»)."""
    text = POLITE_PREFIX.sub("", question.strip().rstrip(END_PUNCTUATION))
    if "的" in text:
        head = text.split("的", 1)[0].strip()
        if len(head) >= 2 and not PREDICATE_START.match(head):
            return head
    match = QUESTION_WORD.search(text)
    if match and match.start() >= 2:
        return text[: match.start()].strip()
    # «怎么订会议室» is about what follows its question word.
    lead = PREDICATE_START.match(text)
    rest = text[lead.end() :].strip() if lead else ""
    if len(rest) >= 2 and not QUESTION_WORD.match(rest):
        tail = QUESTION_WORD.search(rest)
        return (rest[: tail.start()] if tail and tail.start() >= 2 else rest).strip()
    return ""


# Occurrences of a word read from the documents when looking for its siblings.
CONTEXT_SCAN_LIMIT = 200


def word_contexts(word: str, text: str) -> set[str]:
    """The two characters before («<排气») and after («乘坐>») each place text uses word."""
    contexts: set[str] = set()
    for count, match in enumerate(re.finditer(re.escape(word.lower()), text)):
        if count >= CONTEXT_SCAN_LIMIT:
            break
        left = text[max(0, match.start() - 2) : match.start()]
        right = text[match.end() : match.end() + 2]
        if re.fullmatch(r"[一-鿿a-z0-9]{2}", left):
            contexts.add("<" + left)
        if re.fullmatch(r"[一-鿿a-z0-9]{2}", right):
            contexts.add(right + ">")
    return contexts


def listed_together(word: str, other: str, knowledge: str) -> bool:
    """Whether one line of the documents uses both words in the same frame: «火车乘坐高铁二等座；
    飞机乘坐经济舱»."""
    for count, match in enumerate(re.finditer(re.escape(other.lower()), knowledge)):
        if count >= CONTEXT_SCAN_LIMIT:
            break
        line = line_at(knowledge, match.start())
        if word.lower() in line and word_contexts(word, line) & word_contexts(other, line):
            return True
    return False


def line_at(text: str, position: int) -> str:
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    return text[start : end if end >= 0 else len(text)]


def titles_a_section(word: str, knowledge: str) -> bool:
    """Whether the documents have a section titled with word (二、费用报销, ## 加班): then «那X呢»
    asks about that topic, not about X of the previous question's subject."""
    for count, match in enumerate(re.finditer(re.escape(word.lower()), knowledge)):
        if count >= CONTEXT_SCAN_LIMIT:
            break
        line = line_at(knowledge, match.start()).strip()
        marks = len(line) - len(line.lstrip("#"))
        title = line.lstrip("#").strip()
        # The title names the topic itself (费用报销, 加班), not something of it (价格表).
        if title_rank(title, min(marks, 6), "-") and title.rstrip("：:").endswith(word.lower()):
            return True
    return False


def sibling_word(other: str, previous: str, knowledge: str) -> str:
    """The word of the previous question that other takes the place of, as the documents show it:
    they write other in the same frame as it does in the previous question (排气温度 in
    «空压机排气温度标准», 排气压力 in the documents: 温度 for 压力), or list both in one line in
    the same frame (火车乘坐… ；飞机乘坐…). Empty when there is no such word."""
    wanted = word_contexts(other, knowledge)
    if not wanted:
        return ""
    content = QUESTION_WORD.sub(" ", previous)
    lowered = previous.lower()
    candidates = dict.fromkeys(
        span[start : start + size]
        for span in re.findall(r"[一-鿿]{2,}", content.replace("的", " "))
        for size in range(2, min(4, len(other) + 1) + 1)
        for start in range(len(span) - size + 1)
    )
    for word in sorted(candidates, key=len, reverse=True):
        if set(word) & set(other):
            continue
        shared = word_contexts(word, knowledge) & wanted
        if not shared:
            continue
        place = lowered.find(word.lower())
        frame = {"<" + lowered[place - 2 : place], lowered[place + len(word) : place + len(word) + 2] + ">"}
        if shared & frame or listed_together(word, other, knowledge):
            return word
    return ""


def rewrite_by_rules(question: str, previous: str, knowledge: str = "") -> str:
    """knowledge: the documents' text, lower-cased, used to tell which word «X呢» swaps."""
    subject = subject_of(previous)
    if not subject:
        return question
    text = question.strip()
    led = LEADING.match(POLITE_PREFIX.sub("", text)) is not None
    core = LEADING.sub("", POLITE_PREFIX.sub("", text)).rstrip(END_PUNCTUATION)
    if not core:
        return question
    elided = re.fullmatch(r"(.+?)呢", core)
    if elided and not PRONOUNS.search(elided.group(1)):
        other = elided.group(1).strip()
        if re.search(r"[A-Za-z0-9]", other):
            # Another model number of the same series swaps just that number: «E3呢» after «出现E1怎么办».
            prefix = re.match(r"[A-Za-z]+", other)
            for token in re.findall(r"[A-Za-z]+[A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*", previous):
                if prefix and token != other and token.lower().startswith(prefix.group().lower()) and re.search(r"\d", other):
                    return previous.replace(token, other, 1)
            # A name or model number is another subject: «那 SFT 呢» asks the previous question about SFT.
            if subject in previous:
                return previous.replace(subject, other, 1)
        else:
            plain = POLITE_PREFIX.sub("", previous.strip())
            sibling = sibling_word(other, plain, knowledge) if knowledge and "的" not in previous else ""
            if sibling:
                return plain.replace(sibling, other, 1)
            # «那加班呢» after «报销多久能到账»: a topic the documents have a section for is asked
            # about on its own; the previous question's wording (多久能到账) may not fit it.
            if led and knowledge and titles_a_section(other, knowledge):
                return other
            # «那报销呢» after «年假怎么算» asks the same about 报销.
            if led and "的" not in previous and subject in plain:
                return plain.replace(subject, other, 1)
        return f"{subject}的{other}是什么"
    pronoun = PRONOUNS.search(core)
    if pronoun:
        after = core[pronoun.end() :]
        # «这个测试的门限» about «X100 Fan测试» -> «X100 Fan测试的门限», not «…测试测试的门限».
        for size in range(min(len(after), len(subject)), 1, -1):
            if subject.endswith(after[:size]):
                after = after[size:]
                break
        rewritten = core[: pronoun.start()] + subject + after
        # «它的判定标准呢» -> «…的判定标准是什么».
        return re.sub(r"呢$", "是什么", rewritten)
    if PREDICATE_START.match(core):
        joiner = "的" if NOUN_MODIFIER_START.match(core) else ""
        return f"{subject}{joiner}{core}"
    return question


FOLLOWUP_SYSTEM_PROMPT = (
    "你把对话中最新的问题改写成一个能单独理解的完整问题，用于检索资料。\n"
    "规则：\n"
    "1. 只补全最新问题省略或指代的对象（如“它”“这个”“那X呢”“门限呢”），对象从之前的对话中取。\n"
    "2. 保留用户的原话、名称、型号、数字和语言，不添加对话中没有的信息，不回答问题。\n"
    "3. 最新问题本身已经完整时，原样返回。\n"
    '只输出 JSON：{"question": "改写后的问题"}'
)


def rewrite_with_api(question: str, history: list[dict[str, str]], config: dict[str, str]) -> str:
    lines = ["之前的对话："]
    for turn in history:
        lines.append(f"问：{turn['question']}")
        if turn["answer"]:
            lines.append(f"答：{turn['answer']}")
    lines.extend(["", f"最新问题：{question}"])
    budget = llm_budget.task_settings(config, "followup")
    timeout = int(config.get("LKA_API_TIMEOUT_SECONDS", str(DEFAULT_API_TIMEOUT_SECONDS)) or DEFAULT_API_TIMEOUT_SECONDS)
    reply = chat_completion_detailed(
        config.get("LKA_API_PROVIDER", "openai-compatible"),
        config.get("LKA_API_BASE_URL", "").rstrip("/"),
        config.get("LKA_API_KEY", ""),
        budget["model"],
        llm_budget.build_messages(FOLLOWUP_SYSTEM_PROMPT, "\n".join(lines)),
        temperature=0.0,
        timeout=min(timeout, 30),
        json_output=True,
        max_tokens=budget["max_tokens"],
        thinking=budget["thinking"],
        task="followup",
    )
    text = str(reply.get("content") or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    rewritten = str(json.loads(text).get("question") or "").strip()
    if not rewritten or len(rewritten) > QUESTION_MAX_CHARS:
        raise ValueError("unusable follow-up rewrite")
    return rewritten


def standalone_question(
    question: str,
    history: Any,
    config: dict[str, str],
    allow_api: bool = True,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """{"question": what to search for, "rewritten": bool, "method": "none"|"api"|"rules"}."""
    question = question.strip()
    turns = clean_history(history)
    if not turns or not needs_context(question):
        return {"question": question, "rewritten": False, "method": "none"}
    if allow_api and api_enabled(config):
        try:
            rewritten = rewrite_with_api(question, turns, config)
        except Exception:  # noqa: BLE001  (a failed rewrite falls back to the rules)
            pass
        else:
            return {"question": rewritten, "rewritten": rewritten != question, "method": "api"}
    knowledge = (relevance.knowledge_text(db_path) or "") if db_path is not None else ""
    rewritten = rewrite_by_rules(question, turns[-1]["question"], knowledge)
    return {"question": rewritten, "rewritten": rewritten != question, "method": "rules"}
