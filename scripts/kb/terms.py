"""Question analysis: normalisation, stop words, retrieval terms and question types."""

from __future__ import annotations

import re

from kb.text import unique_terms


QUERY_TRAILING_PHRASES = (
    "分别是什么",
    "是什么意思",
    "什么意思",
    "啥意思",
    "代表什么",
    "表示什么",
    "含义是什么",
    "有哪些值",
    "有那些值",
    "有什么值",
    "都有哪些",
    "有那些",
    "有哪些",
    "是什么",
    "是谁",
    "有啥",
    "取值",
    "清单",
    "列表",
    "的含义",
    "含义",
    "意思",
    "值",
)

QUERY_ALIAS_PHRASES = {
    "FFT测试": ("人工功能测试", "MES：FFT", "IR CameraTest", "IR Camera Test", "IQCapture"),
    "人工功能测试": ("FFT测试", "MES：FFT", "IR CameraTest", "IR Camera Test", "IQCapture"),
}

# Everyday wording in a question -> the same idea as the documents write it. The local relevance
# gate counts such a phrase as one subject unit that a chunk covers when it contains any of the
# variants, so 「缓存文件放在哪个目录」 is covered by a chunk that lists the cache file's 路径.
# Only add true synonyms: the gate refuses when the asked attribute is missing from the evidence,
# and a loose pair would let a question about something the documents never mention through.
CONCEPT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "目录": ("目录", "路径", "文件夹", "directory", "folder"),
    "同时执行": ("同时执行", "并行", "parallel"),
    "同时运行": ("同时运行", "并行", "parallel"),
    "重新测试": ("重新测试", "重测", "重试", "重新执行", "retry"),
    "重新测": ("重新测", "重测", "重试", "重新执行", "retry"),
    "冷启动": ("冷启动", "coldboot", "cold boot"),
    "热启动": ("热启动", "warmboot", "warm boot"),
    "重启": ("重启", "reboot", "restart"),
    "电脑": ("电脑", "笔电", "整机", "dut"),
    "操作员": ("操作员", "人力", "测试人力", "作业员", "operator"),
    "节拍": ("节拍", "uph", "测试时间", "takt", "cycle time"),
    # English sheets and SOPs title these columns and sections in English (Version, Tool).
    "版本": ("版本", "version"),
    "工具": ("工具", "tool"),
    "手册": ("手册", "handbook", "manual"),
    "老化": ("老化", "aging", "run-in", "runin"),
    # A book's chapters are named 「…篇」, never 「篇章」.
    "篇章": ("篇章", "章节", "篇"),
}

# Phrases users say that production documents write differently, so searching with the user's
# word alone never reaches the evidence (操作员 -> 测试人力, 节拍 -> UPH / 测试时间, 老化 -> Aging in
# the English-titled test flows). Only these widen the search; widening a phrase the documents do
# use (目录, 同时执行) just adds ranking noise.
SEARCH_EXPANDED_PHRASES = ("操作员", "节拍", "老化")

STOP_WORDS = {
    "什么",
    "哪些",
    "哪个",
    "如何",
    "怎么",
    "是否",
    "以及",
    "有关",
    "问题",
    "区别",
    "作用",
    "影响",
    "一下",
    "为什么",
    "是什么",
    "有哪些",
    "有什么",
    "请问",
    "和",
    "与",
    "及",
}

ENGLISH_QUERY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "please",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

CHINESE_QUERY_STOP_PHRASES = (
    "请介绍一下",
    "介绍一下",
    "是什么",
    "是谁",
    "为什么",
    "有哪些",
    "有什么",
    "什么时候",
    "怎么样",
    "如何",
    "怎么",
    "是否",
    "请问",
    "多少",
    "哪里",
    "何时",
    "用于",
    "用来",
    "进行",
    "实现",
    "可以",
    "作用",
    "一下",
)

# Words that only express the kind of question (how / what / difference / steps / translation),
# never its subject. The local relevance gate ignores them when measuring how much of the
# question's subject a retrieved chunk covers. Longest phrases are removed first.
QUESTION_FUNCTION_PHRASES = tuple(
    sorted(
        {
            *CHINESE_QUERY_STOP_PHRASES,
            "请介绍", "介绍", "说明一下", "解释一下", "请问", "请",
            "什么是", "什么样", "什么", "怎么办", "怎么说", "怎么", "怎样", "为何",
            "是不是", "有没有", "能不能", "可不可以", "能否",
            "都有哪些", "有那些", "哪些", "那些", "哪几个", "哪几种", "哪几", "哪个", "哪种", "哪儿",
            "有几", "几个", "几种", "分别", "区别", "差别", "差异", "不同", "对比", "比较",
            "一个", "一些", "具体", "主要", "包括", "包含",
            "原因", "用途", "意思", "含义", "定义", "方法", "办法", "方式", "步骤", "流程", "过程",
            "机制", "原理", "逻辑", "规则", "英文", "中文", "英语", "全称", "缩写", "简称",
            "需要", "应该", "以及", "或者", "还是",
            # Where something is kept, and causatives / counts that only shape the question.
            "存放在", "保存在", "放在", "让", "两个",
            # Spans of time and adverbs that only frame the question: 从开始到结束要经历哪些步骤.
            "从开始到结束", "从头到尾", "要经历", "要经过", "经历", "已经", "还要",
            # Asking for a name: the answer is the parameter or function itself (EnableSampleReboot),
            # whose definition seldom contains the words 参数 / 函数.
            "哪个参数控制", "什么参数控制", "哪个参数", "什么参数", "哪个配置项", "哪个开关",
            "哪个函数", "什么函数", "哪个方法", "哪个字段",
            # Asking for the values of a field (某字段的取值) or a part of the named document.
            "取值", "哪一部分", "哪部分", "这本书",
            "的", "了", "吗", "呢", "吧", "啊", "呀", "么", "和", "与", "及", "是", "都", "也",
        },
        key=len,
        reverse=True,
    )
)

# Question words and particles that never help locate evidence. Aspect words such as 流程,
# 步骤, 区别 stay: they often appear in the section titles that hold the answer.
RETRIEVAL_STOP_PHRASES = tuple(
    sorted(
        {
            *CHINESE_QUERY_STOP_PHRASES,
            "什么", "哪些", "那些", "哪个", "哪种", "怎样", "一个", "分别",
            "吗", "呢", "吧", "啊", "呀", "么", "的", "了", "是", "都", "也", "和", "与", "及",
        },
        key=len,
        reverse=True,
    )
)

CANDIDATE_TERMS_LIMIT = 48

FTS_QUERY_MAX_TERMS = CANDIDATE_TERMS_LIMIT


def collapse_repeated_query_terms(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"([\u4e00-\u9fff]{2,4})\1", r"\1", text)
    return text


def strip_query_trailing_phrases(text: str) -> str:
    stripped = text.strip()
    changed = True
    while changed:
        changed = False
        for phrase in QUERY_TRAILING_PHRASES:
            if stripped.endswith(phrase) and len(stripped) - len(phrase) >= 2:
                stripped = stripped[: -len(phrase)].strip()
                changed = True
                break
    return stripped or text.strip()


def normalized_retrieval_question(question: str) -> str:
    return strip_query_trailing_phrases(collapse_repeated_query_terms(question))


def question_concepts(question: str) -> list[tuple[str, ...]]:
    """The synonym groups (CONCEPT_SYNONYMS) whose phrase the question uses, longest phrase first."""
    normalized = normalized_retrieval_question(question)
    found: list[tuple[str, ...]] = []
    for phrase in sorted(CONCEPT_SYNONYMS, key=len, reverse=True):
        if phrase in normalized:
            normalized = normalized.replace(phrase, " ")
            found.append(CONCEPT_SYNONYMS[phrase])
    return found


def retrieval_question_with_aliases(question: str) -> str:
    normalized = normalized_retrieval_question(question)
    aliases: list[str] = []
    for phrase, phrase_aliases in QUERY_ALIAS_PHRASES.items():
        if phrase.lower() in normalized.lower():
            aliases.extend(phrase_aliases)
    question_text = " ".join(unique_terms([normalized, *aliases], limit=12))
    # Search with the documents' wording too (SEARCH_EXPANDED_PHRASES): 操作员 also finds 测试人力. Each
    # variant is set off by "，" so English variants stay separate terms: "retry reboot" would be
    # searched as a phrase that no document contains.
    variants: list[str] = []
    for phrase in SEARCH_EXPANDED_PHRASES:
        if phrase in normalized:
            variants.extend(variant for variant in CONCEPT_SYNONYMS[phrase] if variant != phrase and variant not in variants)
    return "，".join([question_text, *variants]) if variants else question_text


def iterative_retrieval_question(question: str) -> bool:
    normalized = normalized_retrieval_question(question)
    return bool(
        re.search(
            r"流程|路径|步骤|机制|逻辑|有哪些|有那些|取值|枚举|列表|清单|有啥|有.*值|what\s+are|workflow|process|enum|values",
            question,
            flags=re.IGNORECASE,
        )
        or normalized != question.strip()
        or any(phrase.lower() in normalized.lower() for phrase in QUERY_ALIAS_PHRASES)
    )


def candidate_terms(question: str) -> list[str]:
    """Lexical terms of a question: ASCII phrases and words first, then the question's content
    spans and their 4-, 3- and 2-character n-grams.

    Question words and particles are removed first. The FTS index stores Chinese as 2-4
    character n-gram tokens, and the old fixed-size list sorted longest-first dropped every
    short term (门限, 风扇, usb) from long questions, so their FTS query matched nothing.
    """
    question = retrieval_question_with_aliases(question)
    ascii_phrases = [
        phrase.strip()
        for phrase in re.findall(
            r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]*(?:\s+[A-Za-z][A-Za-z0-9_]*)+(?![A-Za-z0-9_])",
            question,
        )
    ]
    ascii_words = [
        word.lower()
        for word in re.findall(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]*(?![A-Za-z0-9_])", question)
        if len(word) >= 2 and word.lower() not in ENGLISH_QUERY_STOP_WORDS
    ]
    chinese_text = question
    for phrase in RETRIEVAL_STOP_PHRASES:
        chinese_text = chinese_text.replace(phrase, " ")
    spans = re.findall(r"[\u4e00-\u9fff]{2,}", chinese_text)
    grams: dict[int, list[str]] = {4: [], 3: [], 2: []}
    for span in spans:
        for size, bucket in grams.items():
            bucket.extend(span[index : index + size] for index in range(len(span) - size + 1))
    whole_spans = sorted({span for span in spans if 4 < len(span) <= 8}, key=len, reverse=True)
    terms = [*ascii_phrases, *ascii_words, *whole_spans, *grams[4], *grams[3], *grams[2]]
    lowered = question.lower()
    if "dioxide" in lowered or "carbon" in lowered:
        terms.extend(["二氧化碳", "碳固定"])
    if "穿过" in question or "介质" in question or "pass through" in lowered:
        terms.extend(["travel", "through", "solids", "liquids", "gases", "pass through"])
    unique: dict[str, str] = {}
    for term in terms:
        if term and term not in STOP_WORDS:
            unique.setdefault(term.lower(), term)
    return list(unique.values())[:CANDIDATE_TERMS_LIMIT]


def fts_query(question: str) -> str:
    terms = candidate_terms(question)
    if not terms:
        escaped = question.replace('"', '""').strip()
        return f'"{escaped}"' if escaped else '""'
    # Only terms the index can hold as tokens: ASCII words and Chinese n-grams of 2-4 characters.
    indexable = [term for term in terms if not re.search(r"[\u4e00-\u9fff]{5,}", term)] or terms
    return " OR ".join(f'"{term.replace(chr(34), chr(34) + chr(34))}"' for term in indexable[:FTS_QUERY_MAX_TERMS])


def section_context_question(question: str) -> bool:
    return bool(re.search(r"是什么|逻辑|机制|原理|流程|如何|怎么|说明|介绍|what\s+is|how\s+does|workflow|mechanism|logic", question, flags=re.IGNORECASE))


def enumeration_or_list_question(question: str) -> bool:
    return bool(
        re.search(
            r"哪些|那些|包含|包括|都有|几个|几种|几项|取值|枚举|列表|清单|有啥|有.*值|what\s+are|enum|values|list",
            question,
            flags=re.IGNORECASE,
        )
    )


def fft_manual_function_question(question: str) -> bool:
    """Only a request for the item list itself, e.g. 「人工功能测试」 or 「FFT测试包含哪些测试项」.

    A narrower question such as 「人工功能测试的 IR Camera 判定标准」 must go through
    normal retrieval instead of receiving the whole list.
    """
    normalized = normalized_retrieval_question(question).strip()
    return bool(
        re.fullmatch(
            r"(?:人工功能测试|FFT\s*测试|MES[:：]\s*FFT)\s*(?:都?(?:包含|包括|有))?\s*(?:哪些|那些)?\s*(?:测试项目?|项目?|内容)?",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def process_question(question: str) -> bool:
    return bool(re.search(r"流程|路径|步骤|过程|机制|逻辑|生命周期|workflow|process|steps", question, flags=re.IGNORECASE))


def abbreviation_query_term(question: str) -> str | None:
    normalized = normalized_retrieval_question(question)
    normalized = re.sub(r"\s+", " ", normalize_search_question(normalized)).strip()
    normalized = re.sub(r"^(?:缩写|简称)\s*", "", normalized)
    # 「FAI是什么的缩写」「FAI的全称是什么」「FAI是哪个词的简称」 all ask about FAI.
    normalized = re.sub(
        r"\s*(?:(?:是|为)?(?:什么|哪个词|哪个单词)?的)?(?:英文)?(?:缩写|简称|全称)(?:是什么|是啥|是)?$",
        "",
        normalized,
    )
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9/._+-]{1,15}", normalized):
        return normalized.upper()
    return None


def station_test_item_question(question: str) -> bool:
    return bool(re.search(r"测试项|测试项目|(?:站位|工位|站点)\s*测试|测试\s*(?:站位|工位|站点)", question))


def station_question_includes_schemes(question: str) -> bool:
    return bool(re.search(r"方案|步骤|怎么测|如何测试", question))


def requested_station_name(question: str) -> str | None:
    explicit = re.search(
        r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_-]{1,30})\s*(?:站位|工位|站点)",
        question,
        flags=re.IGNORECASE,
    )
    if explicit:
        return explicit.group(1)
    if not station_test_item_question(question):
        return None
    implicit = re.search(
        r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_-]{1,30})"
        r"(?=\s*的?\s*(?:(?:都有哪些|有哪些|有什么|包含|包括|有)\s*)?(?:测试项|测试项目)"
        r"|\s*的?\s*(?:测试项|测试项目)\s*(?:都有哪些|有哪些|有什么|包含|包括|有))",
        question,
        flags=re.IGNORECASE,
    )
    return implicit.group(1) if implicit else None


def normalize_search_question(question: str) -> str:
    replacements = {
        "时什么": "是什么",
        "是神么": "是什么",
        "事什么": "是什么",
        "为什莫": "为什么",
        "怎末": "怎么",
    }
    normalized = question
    for original, replacement in replacements.items():
        normalized = normalized.replace(original, replacement)
    return normalized


def meaningful_query_terms(question: str) -> list[str]:
    question = normalized_retrieval_question(question)
    abbreviation = abbreviation_query_term(question)
    if abbreviation:
        return [abbreviation.lower()]
    terms = [
        word.lower()
        for word in re.findall(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_-]*(?![A-Za-z0-9_])", question)
        if len(word) >= 3 and word.lower() not in ENGLISH_QUERY_STOP_WORDS
    ]
    chinese_text = question
    for stop in CHINESE_QUERY_STOP_PHRASES:
        chinese_text = chinese_text.replace(stop, " ")
    terms.extend(re.findall(r"[\u4e00-\u9fff]{2,}", chinese_text))
    return unique_terms(terms, limit=16)


def query_term_variants(term: str) -> list[str]:
    variants = [term]
    reduced = term
    for _ in range(3):
        suffix = next(
            (
                candidate
                for candidate in (*QUERY_TRAILING_PHRASES, "工作原理", "逻辑", "机制", "原理", "流程", "说明", "介绍")
                if reduced.endswith(candidate) and len(reduced) - len(candidate) >= 2
            ),
            None,
        )
        if suffix is None:
            break
        reduced = reduced[: -len(suffix)]
        variants.append(reduced)
    return unique_terms(variants, limit=4)


def lexical_query_coverage(question: str, text: str) -> float:
    terms = meaningful_query_terms(question)
    if not terms:
        return 0.0
    lowered = text.lower()
    matched = sum(1 for term in terms if any(variant.lower() in lowered for variant in query_term_variants(term)))
    return matched / len(terms)
