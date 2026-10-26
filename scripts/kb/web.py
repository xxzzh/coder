"""Web search and research: search providers, public page fetching and extraction, relevance filtering and de-duplication of web sources."""

from __future__ import annotations

import base64
from difflib import SequenceMatcher
from html import unescape
from html.parser import HTMLParser
from ipaddress import ip_address
import json
import queue
import re
import socket
import threading
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote, quote_plus, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
from xml.etree import ElementTree

from kb.citations import citation_for_text, grounded_citations
from kb.retrieval import cosine, local_embedding
from kb.terms import CHINESE_QUERY_STOP_PHRASES, ENGLISH_QUERY_STOP_WORDS, candidate_terms, lexical_query_coverage, normalize_search_question
from kb.text import human_text, sentence_score, split_sentences, trim_terminal_punctuation


WEB_TIMEOUT_SECONDS = 5


WEB_SEARCH_TIMEOUT_SECONDS = 14


WEB_SEARCH_TOTAL_TIMEOUT_SECONDS = 20


WEB_SEARCH_MERGE_GRACE_SECONDS = 1.2


WEB_RESEARCH_SOURCE_LIMIT = 3


WEB_RESEARCH_TOTAL_TIMEOUT_SECONDS = 12


WEB_PAGE_FETCH_TIMEOUT_SECONDS = 7


WEB_PAGE_MAX_BYTES = 750_000


WEB_PAGE_MAX_TEXT_CHARS = 60_000


WEB_PAGE_CHUNK_CHARS = 900


def fetch_url_bytes(request: Request, timeout: int = WEB_TIMEOUT_SECONDS) -> bytes:
    try:
        import requests  # type: ignore

        try:
            response = requests.get(
                request.full_url,
                headers=dict(request.header_items()),
                timeout=(3, timeout),
            )
            response.raise_for_status()
            return response.content
        except Exception:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
    except ImportError:
        with urlopen(request, timeout=timeout) as response:
            return response.read()


def fetch_url_text(request: Request, timeout: int = WEB_TIMEOUT_SECONDS) -> str:
    return fetch_url_bytes(request, timeout).decode("utf-8", errors="ignore")


class DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._active: str | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        classes = set(attrs_dict.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self._flush()
            self._current = {"title": "", "url": self._clean_url(attrs_dict.get("href", "")), "snippet": ""}
            self._active = "title"
            self._buffer = []
        elif "result__snippet" in classes and self._current is not None:
            self._flush()
            self._active = "snippet"
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._active:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._active == "title" and tag == "a":
            self._flush()
        elif self._active == "snippet" and tag in {"a", "div"}:
            self._flush()

    def close(self) -> None:
        self._flush()
        super().close()

    def _flush(self) -> None:
        if self._active and self._current is not None:
            value = human_text("".join(self._buffer), simplify=True)
            if value:
                self._current[self._active] = value
            if self._active == "snippet" and self._current.get("title") and self._current.get("url"):
                self.results.append(clean_web_source(self._current))
                self._current = None
            self._active = None
            self._buffer = []

    @staticmethod
    def _clean_url(url: str) -> str:
        if url.startswith("//"):
            url = "https:" + url
        parsed = urlparse(url)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
            target = parse_qs(parsed.query).get("uddg", [""])[0]
            if target:
                return target
        return url


def web_query(question: str) -> str:
    question = normalize_search_question(question)
    if not re.search(r"[\u4e00-\u9fff]", question):
        return question

    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", " ", question)
    for stop in CHINESE_QUERY_STOP_PHRASES:
        text = text.replace(stop, " ")
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    if len(chinese_terms) >= 2:
        return " ".join(chinese_terms[:4])
    if chinese_terms:
        return chinese_terms[0]
    return text.strip() or question


def web_query_variants(question: str) -> list[str]:
    normalized = normalize_search_question(question)
    variants = [web_query(normalized), normalized]
    if re.search(r"[\u4e00-\u9fff]", normalized):
        compact = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", " ", normalized).strip()
        variants.append(compact)
    return list(dict.fromkeys(variant for variant in variants if variant.strip()))[:3]


def chinese_relevance_score(question: str, text: str) -> float | None:
    chinese_text = question
    for stop in CHINESE_QUERY_STOP_PHRASES:
        chinese_text = chinese_text.replace(stop, " ")
    blocks = re.findall(r"[\u4e00-\u9fff]{2,}", chinese_text)
    if not blocks:
        return None

    scores: list[float] = []
    for block in blocks:
        if block in text:
            scores.append(1.0)
            continue
        grams = {block[index : index + 2] for index in range(len(block) - 1)}
        scores.append(sum(1 for gram in grams if gram in text) / len(grams) if grams else 0.0)
    return sum(scores) / len(scores)


def english_relevance_score(question: str, text: str) -> float | None:
    terms = [
        word.lower()
        for word in re.findall(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_-]*(?![A-Za-z0-9_])", question)
        if len(word) >= 3 and word.lower() not in ENGLISH_QUERY_STOP_WORDS
    ]
    if not terms:
        return None
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered) / len(set(terms))


def filter_web_sources(
    question: str,
    sources: list[dict[str, str]],
    limit: int,
    allow_relaxed: bool = False,
) -> list[dict[str, str]]:
    normalized_question = normalize_search_question(question)
    ranked: list[tuple[float, dict[str, str]]] = []
    for source in sources:
        cleaned = clean_web_source(source)
        text = f"{cleaned.get('title', '')} {cleaned.get('snippet', '')}"
        chinese_score = chinese_relevance_score(normalized_question, text)
        english_score = english_relevance_score(normalized_question, text)
        scores = [
            score
            for score in (chinese_score, english_score)
            if score is not None
        ]
        relevance = max(scores, default=0.0)
        if chinese_score is not None:
            minimum_relevance = 0.75 if not allow_relaxed else 0.3
        else:
            minimum_relevance = 0.67
        if relevance >= minimum_relevance:
            cleaned["relevance_score"] = round(relevance, 4)
            ranked.append((relevance, cleaned))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [source for _, source in ranked[:limit]]


def source_quote(source: dict[str, Any]) -> str:
    return str(source.get("citation") or source.get("excerpt") or source.get("snippet") or source.get("title") or "")


def source_is_duplicate(source: dict[str, Any], existing: list[dict[str, Any]]) -> bool:
    url = str(source.get("url", "")).strip().lower()
    file_path = str(source.get("file_path", "")).strip().lower()
    chunk_id = str(source.get("chunk_id", "")).strip().lower()
    normalized_quote = re.sub(r"\W+", "", source_quote(source).lower())
    for item in existing:
        if url and url == str(item.get("url", "")).strip().lower():
            return True
        if file_path and chunk_id and file_path == str(item.get("file_path", "")).strip().lower() and chunk_id == str(item.get("chunk_id", "")).strip().lower():
            return True
        existing_quote = re.sub(r"\W+", "", source_quote(item).lower())
        if normalized_quote and existing_quote and (
            normalized_quote == existing_quote
            or (
                min(len(normalized_quote), len(existing_quote)) >= 40
                and (normalized_quote in existing_quote or existing_quote in normalized_quote)
            )
            or SequenceMatcher(None, normalized_quote, existing_quote).ratio() >= 0.9
        ):
            return True
    return False


def dedupe_sources(sources: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in sources:
        if source_is_duplicate(source, result):
            continue
        result.append(source.copy())
        if len(result) >= limit:
            break
    return result


def alternate_sources(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for index in range(max(len(left), len(right))):
        if index < len(left):
            merged.append(left[index])
        if index < len(right):
            merged.append(right[index])
    return merged


def web_source_quality_score(question: str, source: dict[str, Any]) -> float:
    text = f"{source.get('title', '')} {source.get('snippet', '')}"
    score = float(source.get("relevance_score", 0.0) or 0.0)
    normalized_question = normalize_search_question(question)
    if "是什么" in normalized_question or "是谁" in normalized_question:
        if re.search(r"(?:是|为|指|属于|出生|创始人|董事长|首席执行官|CEO).{0,30}(?:平台|网站|品牌|工具|服务|产品|概念|方法|公司|企业家|投资人|软件工程师|代表)", text):
            score += 0.8
        elif re.search(r"(?:是|为|指|属于|出生|创始人|董事长|首席执行官|CEO)", text):
            score += 0.35
    if re.search(r"版权所有|ICP备|隐私政策|用户协议|联系我们", text):
        score -= 0.8
    if re.search(r"为什么|怎么样|课程|模板|优惠|讨论", str(source.get("title", ""))):
        score -= 0.15
    return score


def duckduckgo_search(question: str, limit: int = 5) -> list[dict[str, str]]:
    request = Request(
        f"https://duckduckgo.com/html/?q={quote_plus(question)}",
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; local-kb-agent/1.0)",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    try:
        html = fetch_url_text(request)
    except Exception as exc:
        raise RuntimeError(f"DuckDuckGo request failed: {human_text(str(exc), max_len=160)}") from exc

    parser = DuckDuckGoParser()
    parser.feed(html)
    parser.close()
    return parser.results[:limit]


def bing_rss_search(question: str, limit: int = 5) -> list[dict[str, str]]:
    return bing_rss_search_url(
        f"https://www.bing.com/search?format=rss&q={quote_plus(question)}",
        limit,
        provider_name="Bing RSS",
    )


def bing_china_search(question: str, limit: int = 5) -> list[dict[str, str]]:
    return bing_rss_search_url(
        f"https://cn.bing.com/search?format=rss&q={quote_plus(question)}",
        limit,
        provider_name="Bing China RSS",
    )


def bing_rss_search_url(url: str, limit: int, provider_name: str) -> list[dict[str, str]]:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; local-kb-agent/1.0)",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    try:
        root = ElementTree.fromstring(fetch_url_bytes(request))
    except Exception as exc:
        raise RuntimeError(f"{provider_name} request failed: {human_text(str(exc), max_len=160)}") from exc

    results: list[dict[str, str]] = []
    for item in root.findall("./channel/item"):
        title = item.findtext("title", default="")
        url = item.findtext("link", default="")
        snippet = item.findtext("description", default="")
        if title and url:
            results.append(clean_web_source({"title": title, "url": url, "snippet": snippet}))
        if len(results) >= limit:
            break
    return results


def jina_baidu_search(question: str, limit: int = 5) -> list[dict[str, str]]:
    request = Request(
        f"https://r.jina.ai/http://www.baidu.com/s?wd={quote_plus(question)}",
        headers={"User-Agent": "Mozilla/5.0 (compatible; local-kb-agent/1.0)"},
    )
    try:
        markdown = fetch_url_text(request)
    except Exception as exc:
        raise RuntimeError(f"Baidu request failed: {human_text(str(exc), max_len=160)}") from exc

    results: list[dict[str, str]] = []
    pattern = re.compile(r"^###\s+\[(.*?)\]\((.*?)\)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(markdown))
    for index, match in enumerate(matches):
        title = match.group(1)
        url = unescape(match.group(2)).strip()
        snippet_start = match.end()
        snippet_end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        snippet = markdown[snippet_start:snippet_end]
        if title.strip().lower() in {"app", "课程", "竞赛"}:
            continue
        if title and url:
            results.append(clean_web_source({"title": title, "url": url, "snippet": snippet}, max_snippet=300))
        if len(results) >= limit:
            break
    return results


def web_search_variant(
    question: str,
    search_question: str,
    limit: int = 5,
    timeout_seconds: float = WEB_SEARCH_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    normalized_question = normalize_search_question(question)
    providers = (
        ("baidu", jina_baidu_search),
        ("bing_china", bing_china_search),
        ("duckduckgo", duckduckgo_search),
        ("wikipedia", wikipedia_search),
        ("jina_bing", jina_bing_search),
        ("bing_rss", bing_rss_search),
    )
    results: queue.Queue[tuple[str, list[dict[str, str]], str | None]] = queue.Queue()

    def invoke(provider_name: str, provider: Any) -> None:
        try:
            results.put((provider_name, provider(search_question, limit), None))
        except Exception as exc:  # noqa: BLE001
            results.put((provider_name, [], human_text(str(exc), max_len=180)))

    for provider_name, provider in providers:
        threading.Thread(target=invoke, args=(provider_name, provider), daemon=True).start()

    aggregated_sources: list[dict[str, str]] = []
    selected_providers: list[str] = []
    diagnostics: dict[str, Any] = {"query": search_question, "providers": {}}
    deadline = time.monotonic() + timeout_seconds
    merge_deadline: float | None = None
    completed = 0
    while completed < len(providers):
        remaining = deadline - time.monotonic()
        if merge_deadline is not None:
            remaining = min(remaining, merge_deadline - time.monotonic())
        if remaining <= 0:
            break
        try:
            provider_name, sources, error = results.get(timeout=remaining)
        except queue.Empty:
            break
        completed += 1
        relevant_sources = filter_web_sources(normalized_question, sources, limit)
        relaxed_filter_used = False
        if not relevant_sources and sources:
            relevant_sources = filter_web_sources(normalized_question, sources, limit, allow_relaxed=True)
            relaxed_filter_used = bool(relevant_sources)
        diagnostics["providers"][provider_name] = {
            "raw_sources": len(sources),
            "relevant_sources": len(relevant_sources),
            "relaxed_filter_used": relaxed_filter_used,
            "error": error,
        }
        if relevant_sources:
            if provider_name not in selected_providers:
                selected_providers.append(provider_name)
            for source in relevant_sources:
                source["search_provider"] = provider_name
            aggregated_sources.extend(relevant_sources)
            if merge_deadline is None:
                merge_deadline = time.monotonic() + WEB_SEARCH_MERGE_GRACE_SECONDS

    if aggregated_sources:
        for source in aggregated_sources:
            source["quality_score"] = round(web_source_quality_score(normalized_question, source), 4)
        ranked_sources = sorted(
            aggregated_sources,
            key=lambda source: float(source.get("quality_score", 0.0) or 0.0),
            reverse=True,
        )
        sources = dedupe_sources(ranked_sources, limit)
        return web_search_result(sources, providers=selected_providers), diagnostics
    diagnostics["timed_out_providers"] = len(providers) - completed
    return None, diagnostics


def freshness_sensitive_question(question: str) -> bool:
    lowered = question.lower()
    markers = (
        "最新",
        "当前",
        "目前",
        "今天",
        "即将",
        "停止使用",
        "弃用",
        "下线",
        "过期",
        "截止",
        "何时",
        "什么时候",
        "latest",
        "current",
        "today",
        "deprecated",
        "deprecation",
        "sunset",
        "end of life",
        "eol",
    )
    return any(marker in lowered for marker in markers)


def html_visible_text(html: str) -> str:
    text = re.sub(r"(?is)<script[^>]*>.*?</script>|<style[^>]*>.*?</style>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(text).replace("\x00", "")).strip()


class WebPageTextParser(HTMLParser):
    BLOCK_TAGS = {
        "article",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "p",
        "pre",
        "section",
        "td",
        "th",
        "tr",
    }
    SKIP_TAGS = {"aside", "canvas", "footer", "form", "header", "nav", "noscript", "script", "style", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.blocks: list[str] = []
        self._block_parts: list[str] = []
        self._title_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if self._skip_depth:
            self._skip_depth += 1
            return
        if tag in self.SKIP_TAGS:
            self._flush_block()
            self._skip_depth = 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in self.BLOCK_TAGS:
            self._flush_block()

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self.title = human_text(" ".join(self._title_parts), max_len=180, simplify=True)
            self._title_parts = []
            self._in_title = False
            return
        if tag in self.BLOCK_TAGS:
            self._flush_block()

    def handle_data(self, data: str) -> None:
        value = re.sub(r"\s+", " ", unescape(data).replace("\x00", " ")).strip()
        if not value or self._skip_depth:
            return
        if self._in_title:
            self._title_parts.append(value)
        else:
            self._block_parts.append(value)

    def close(self) -> None:
        self._flush_block()
        super().close()

    def text(self) -> str:
        self._flush_block()
        blocks: list[str] = []
        for block in self.blocks:
            if block and (not blocks or block != blocks[-1]):
                blocks.append(block)
        return "\n".join(blocks)

    def _flush_block(self) -> None:
        block = human_text(" ".join(self._block_parts), simplify=True)
        if block:
            self.blocks.append(block)
        self._block_parts = []


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def validate_public_web_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Only public http/https webpage URLs are allowed.")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
        raise ValueError("Local webpage URLs are not allowed.")
    try:
        addresses = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError(f"Cannot resolve webpage host: {hostname}") from exc
    if not addresses:
        raise ValueError(f"Cannot resolve webpage host: {hostname}")
    for address in addresses:
        value = ip_address(address[4][0].split("%", 1)[0])
        if not value.is_global:
            raise ValueError("Non-public webpage addresses are not allowed.")
    return url


def decode_webpage(data: bytes, charset: str | None = None) -> str:
    candidates = [charset]
    head = data[:4096].decode("ascii", errors="ignore")
    match = re.search(r"(?i)charset\s*=\s*[\"']?\s*([a-z0-9._-]+)", head)
    if match:
        candidates.append(match.group(1))
    candidates.extend(["utf-8", "gb18030"])
    best = ""
    best_replacements = WEB_PAGE_MAX_BYTES
    for encoding in candidates:
        if not encoding:
            continue
        try:
            decoded = data.decode(encoding, errors="replace")
        except LookupError:
            continue
        replacements = decoded.count("\ufffd")
        if replacements < best_replacements:
            best = decoded
            best_replacements = replacements
        if replacements == 0:
            break
    return best


def fetch_public_webpage_text(url: str, timeout: int = WEB_PAGE_FETCH_TIMEOUT_SECONDS) -> tuple[str, str, str]:
    opener = build_opener(NoRedirectHandler())
    current_url = url
    for _ in range(4):
        validate_public_web_url(current_url)
        request = Request(
            current_url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; local-kb-agent/1.0)",
                "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )
        try:
            response = opener.open(request, timeout=timeout)
        except HTTPError as exc:
            location = exc.headers.get("Location", "")
            if exc.code in {301, 302, 303, 307, 308} and location:
                current_url = urljoin(current_url, location)
                continue
            raise
        with response:
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
                raise ValueError(f"Unsupported webpage content type: {content_type}")
            data = response.read(WEB_PAGE_MAX_BYTES + 1)
            if len(data) > WEB_PAGE_MAX_BYTES:
                data = data[:WEB_PAGE_MAX_BYTES]
            return decode_webpage(data, response.headers.get_content_charset()), current_url, content_type
    raise ValueError("Too many webpage redirects.")


def extract_webpage_text(page_text: str, content_type: str = "text/html") -> tuple[str, str]:
    if content_type == "text/plain":
        return "", human_text(page_text, max_len=WEB_PAGE_MAX_TEXT_CHARS)
    parser = WebPageTextParser()
    parser.feed(page_text)
    parser.close()
    text = parser.text()
    if not text:
        text = html_visible_text(page_text)
    return parser.title, text[:WEB_PAGE_MAX_TEXT_CHARS]


def webpage_chunks(text: str, max_chars: int = WEB_PAGE_CHUNK_CHARS) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0
    paragraphs = [human_text(part) for part in re.split(r"\n+", text) if human_text(part)]
    for paragraph in paragraphs:
        parts = split_sentences(paragraph) if len(paragraph) > max_chars else [paragraph]
        for part in parts:
            if not part:
                continue
            if current and current_chars + len(part) + 1 > max_chars:
                chunks.append(" ".join(current))
                current = []
                current_chars = 0
            current.append(part)
            current_chars += len(part) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def webpage_chunk_score(question: str, chunk: str) -> float:
    lexical = sentence_score(chunk, candidate_terms(question))
    semantic = max(0.0, cosine(local_embedding(question), local_embedding(chunk)))
    relevance = max(
        score or 0.0
        for score in (
            chinese_relevance_score(normalize_search_question(question), chunk),
            english_relevance_score(question, chunk),
        )
    )
    return lexical + semantic * 4.0 + relevance * 3.0


def web_evidence_quality_score(question: str, source: dict[str, Any]) -> float:
    quote = source_quote(source)
    score = max(
        float(source.get("quality_score", 0.0) or 0.0),
        float(source.get("relevance_score", 0.0) or 0.0),
    )
    if lexical_query_coverage(question, quote) <= 0:
        score -= 1.0
    if re.search(r"(?:出生|创始人|董事长|首席执行官|CEO|企业家|投资人|软件工程师|代表)", quote):
        score += 0.9
    if re.search(r"(?:元起|查看全部|服务中心|预约维修|退货|换货|包邮|售后|订单查询|ICP备|营业执照|违法和不良信息|知识产权侵权投诉)", quote):
        score -= 1.4
    if re.search(r"(?:Video|Image|快捷键说明|播放\s*/\s*暂停|退出全屏|快进|快退|按住此处可拖拽|详情)", quote):
        score -= 1.0
    if len(quote) > 450 and len(set(re.findall(r"[\u4e00-\u9fff]{2,}", quote))) > 80:
        score -= 0.35
    return score


def best_summary_source_ids(question: str, sources: list[dict[str, Any]], limit: int = 3) -> list[int]:
    scored: list[tuple[float, int]] = []
    for source in sources:
        source_id = int(source.get("source_id") or 0)
        if not source_id:
            continue
        score = web_evidence_quality_score(question, source)
        if score > 0:
            scored.append((score, source_id))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0] if scored else 0.0
    minimum_score = max(0.75, best_score - 0.5)
    return [source_id for score, source_id in scored if score >= minimum_score][:limit]


def research_web_source(question: str, source: dict[str, Any]) -> dict[str, Any]:
    researched = source.copy()
    try:
        page_text, resolved_url, content_type = fetch_public_webpage_text(str(source.get("url", "")))
        page_title, text = extract_webpage_text(page_text, content_type)
        chunks = webpage_chunks(text)
        ranked = sorted(
            ((webpage_chunk_score(question, chunk), chunk) for chunk in chunks),
            key=lambda item: item[0],
            reverse=True,
        )
        selected = [chunk for score, chunk in ranked[:2] if score > 0]
        if not selected and chunks:
            selected = chunks[:1]
        citation = human_text(
            " ".join(citation_for_text(chunk, question, max_len=520) for chunk in selected),
            max_len=1000,
            simplify=True,
        )
        if not citation:
            raise ValueError("Webpage did not contain readable text.")
    except Exception as exc:  # noqa: BLE001
        researched["page_fetch_status"] = "error"
        researched["page_fetch_error"] = human_text(str(exc), max_len=180)
        return researched
    researched.update(
        {
            "title": page_title or researched.get("title", ""),
            "resolved_url": resolved_url,
            "content_source": "webpage",
            "page_fetch_status": "ok",
            "page_chars": len(text),
            "page_chunk_score": round(ranked[0][0], 4) if ranked else 0.0,
            "snippet": citation,
            "excerpt": citation,
            "citation": citation,
        }
    )
    return researched


def enrich_web_search_result(question: str, result: dict[str, Any], limit: int = 5) -> dict[str, Any]:
    sources = [source.copy() for source in (result.get("sources") or [])]
    if not sources:
        result["web_research_used"] = False
        result["web_pages_fetched"] = 0
        result["web_pages_extracted"] = 0
        return result

    selected_sources = sources[: min(len(sources), limit, WEB_RESEARCH_SOURCE_LIMIT)]
    completed: queue.Queue[tuple[int, dict[str, Any]]] = queue.Queue()

    def invoke(index: int, source: dict[str, Any]) -> None:
        completed.put((index, research_web_source(question, source)))

    for index, source in enumerate(selected_sources):
        threading.Thread(target=invoke, args=(index, source), daemon=True).start()

    researched: dict[int, dict[str, Any]] = {}
    deadline = time.monotonic() + WEB_RESEARCH_TOTAL_TIMEOUT_SECONDS
    while len(researched) < len(selected_sources):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            index, source = completed.get(timeout=remaining)
        except queue.Empty:
            break
        researched[index] = source

    for index in range(len(selected_sources)):
        sources[index] = researched.get(
            index,
            {
                **sources[index],
                "page_fetch_status": "timeout",
                "page_fetch_error": "Webpage fetch timed out.",
            },
        )
    for index in range(len(selected_sources), len(sources)):
        sources[index]["page_fetch_status"] = "not_selected"

    for source_id, source in enumerate(sources, start=1):
        source["source_id"] = source_id
    extracted_ids = [source["source_id"] for source in sources if source.get("page_fetch_status") == "ok"]
    result["sources"] = sources
    result["citations"] = grounded_citations(result)
    result["summary_source_ids"] = best_summary_source_ids(question, sources) or extracted_ids[:3] or result.get("summary_source_ids", [])
    result["web_research_used"] = True
    result["web_pages_fetched"] = len(selected_sources)
    result["web_pages_extracted"] = len(extracted_ids)
    result["retrieval_mode"] = "web_search_page_rag"
    snippets = [source_quote(source) for source in sources[:3] if source_quote(source)]
    if snippets:
        result["answer"] = human_text(
            "本地知识库没有找到足够依据，联网研究到：" + "；".join(trim_terminal_punctuation(snippet) for snippet in snippets) + "。",
            simplify=True,
        )
    return result


def web_search(question: str, limit: int = 5) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    deadline = time.monotonic() + WEB_SEARCH_TOTAL_TIMEOUT_SECONDS
    for variant in web_query_variants(question):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        result, diagnostics = web_search_variant(question, variant, limit, min(WEB_SEARCH_TIMEOUT_SECONDS, remaining))
        attempts.append(diagnostics)
        if result is not None:
            result["web_search_attempts"] = attempts
            return result

    providers = [provider for attempt in attempts for provider in attempt.get("providers", {}).values()]
    raw_sources = sum(int(provider.get("raw_sources", 0) or 0) for provider in providers)
    successful_providers = sum(1 for provider in providers if not provider.get("error"))
    if raw_sources:
        return web_search_error(
            "联网搜索已完成，但没有找到与问题足够相关的可信结果。请补充关键词或换一种更具体的问法。",
            reason="no_relevant_results",
            attempts=attempts,
        )
    if successful_providers:
        return web_search_error(
            "联网搜索已完成，但搜索源没有返回可用结果。请补充关键词或稍后重试。",
            reason="no_search_results",
            attempts=attempts,
        )
    return web_search_error(
        "联网搜索源暂时无法访问。请检查网络连接或稍后重试。",
        reason="providers_unavailable",
        attempts=attempts,
    )


def wikipedia_search(question: str, limit: int = 5) -> list[dict[str, str]]:
    languages = ["zh", "en"] if re.search(r"[\u4e00-\u9fff]", question) else ["en", "zh"]
    request_succeeded = False
    last_error: Exception | None = None
    for language in languages:
        url = (
            f"https://{language}.wikipedia.org/w/api.php?"
            f"action=opensearch&search={quote_plus(question)}&limit={limit}&namespace=0"
            f"&format=json&variant=zh-cn"
        )
        try:
            request = Request(url, headers={"User-Agent": "local-kb-agent/1.0", "Accept-Language": "zh-CN,zh;q=0.9"})
            data = json.loads(fetch_url_text(request))
            request_succeeded = True
        except Exception as exc:
            last_error = exc
            continue
        if len(data) < 4 or not data[1]:
            continue

        titles: list[str] = data[1]
        descriptions: list[str] = data[2]
        urls: list[str] = data[3]
        results: list[dict[str, str]] = []
        for index, title in enumerate(titles[:limit]):
            snippet = descriptions[index] if index < len(descriptions) else ""
            if not snippet:
                snippet = wikipedia_summary(language, title)
            results.append(
                clean_web_source(
                    {
                        "title": title,
                        "url": urls[index] if index < len(urls) else f"https://{language}.wikipedia.org/wiki/{quote(title)}",
                        "snippet": snippet,
                    }
                )
            )
        if results:
            return results
    if not request_succeeded and last_error is not None:
        raise RuntimeError(f"Wikipedia request failed: {human_text(str(last_error), max_len=160)}") from last_error
    return []


def wikipedia_summary(language: str, title: str) -> str:
    url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    try:
        request = Request(url, headers={"User-Agent": "local-kb-agent/1.0", "Accept-Language": "zh-CN,zh;q=0.9"})
        data = json.loads(fetch_url_text(request))
    except Exception:
        return ""
    return str(data.get("extract", ""))[:300]


def jina_bing_search(question: str, limit: int = 5) -> list[dict[str, str]]:
    request = Request(
        f"https://r.jina.ai/https://www.bing.com/search?q={quote_plus(question)}",
        headers={"User-Agent": "Mozilla/5.0 (compatible; local-kb-agent/1.0)"},
    )
    try:
        markdown = fetch_url_text(request)
    except Exception as exc:
        raise RuntimeError(f"Jina Bing request failed: {human_text(str(exc), max_len=160)}") from exc

    results: list[dict[str, str]] = []
    pattern = re.compile(r"^\d+\.\s+##\s+\[(.*?)\]\((.*?)\)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(markdown))
    for index, match in enumerate(matches):
        title = match.group(1)
        url = clean_bing_url(unescape(match.group(2)).strip())
        snippet_start = match.end()
        snippet_end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        snippet = markdown[snippet_start:snippet_end]
        if title and url:
            results.append(clean_web_source({"title": title, "url": url, "snippet": snippet}, max_snippet=300))
        if len(results) >= limit:
            break
    return results


def clean_bing_url(url: str) -> str:
    parsed = urlparse(url)
    encoded = parse_qs(parsed.query).get("u", [""])[0]
    if parsed.netloc.endswith("bing.com") and encoded.startswith("a1"):
        payload = encoded[2:]
        padding = "=" * (-len(payload) % 4)
        try:
            return base64.urlsafe_b64decode(payload + padding).decode("utf-8")
        except Exception:
            return url
    return url


def clean_web_source(source: dict[str, str], max_snippet: int = 320) -> dict[str, str]:
    return {
        "title": human_text(source.get("title", ""), max_len=160, simplify=True),
        "url": source.get("url", "").strip(),
        "snippet": human_text(source.get("snippet", ""), max_len=max_snippet, simplify=True),
    }


def web_search_result(
    sources: list[dict[str, str]],
    provider: str | None = None,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    selected_providers = providers or ([provider] if provider else [])
    cleaned_sources: list[dict[str, Any]] = []
    for source in dedupe_sources(sources, len(sources)):
        cleaned = clean_web_source(source)
        cleaned["source_origin"] = "web"
        if source.get("search_provider"):
            cleaned["search_provider"] = source["search_provider"]
        if source.get("relevance_score") is not None:
            cleaned["relevance_score"] = source["relevance_score"]
        if source.get("quality_score") is not None:
            cleaned["quality_score"] = source["quality_score"]
        cleaned_sources.append(cleaned)
    for source_id, source in enumerate(cleaned_sources, start=1):
        source["source_id"] = source_id
    snippets = [source["snippet"] or source["title"] for source in cleaned_sources[:3] if source["snippet"] or source["title"]]
    quality_scores = [float(source.get("quality_score", 0.0) or 0.0) for source in cleaned_sources]
    best_quality = max(quality_scores, default=0.0)
    summary_source_ids = [
        source["source_id"]
        for source in cleaned_sources
        if float(source.get("quality_score", 0.0) or 0.0) >= best_quality - 0.45
    ][:3]
    answer = "本地知识库没有找到足够依据，联网搜索到：" + "；".join(
        trim_terminal_punctuation(snippet) for snippet in snippets
    ) + "。"
    return {
        "answer": human_text(answer, simplify=True),
        "source_type": "web_search",
        "sources": cleaned_sources,
        "summary_source_ids": summary_source_ids,
        "web_search_provider": "+".join(selected_providers) or None,
        "web_search_providers": selected_providers,
        "need_web_search": False,
        "web_search_used": True,
    }


def web_search_error(answer: str, reason: str | None = None, attempts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "answer": human_text(answer, simplify=True),
        "source_type": "web_search",
        "sources": [],
        "need_web_search": False,
        "web_search_used": True,
        "web_search_reason": reason,
        "web_search_attempts": attempts or [],
    }


def merge_with_web_result(local_result: dict[str, Any], web_result: dict[str, Any], limit: int) -> dict[str, Any]:
    local_sources = [source.copy() for source in (local_result.get("sources") or [])]
    web_sources = [source.copy() for source in (web_result.get("sources") or [])]
    if not local_sources:
        return web_result
    if not web_sources:
        result = local_result.copy()
        result["web_search_used"] = True
        result["web_search_reason"] = web_result.get("web_search_reason")
        result["web_search_attempts"] = web_result.get("web_search_attempts", [])
        return result

    for source in local_sources:
        source["source_origin"] = source.get("source_origin") or "local"
    for source in web_sources:
        source["source_origin"] = "web"

    sources = dedupe_sources(alternate_sources(local_sources, web_sources), max(6, limit * 2))
    for source_id, source in enumerate(sources, start=1):
        source["source_id"] = source_id

    result = local_result.copy()
    result.update(
        {
            "source_type": "hybrid_web_knowledge",
            "sources": sources,
            "citations": grounded_citations({"sources": sources}),
            "web_search_provider": web_result.get("web_search_provider"),
            "web_search_providers": web_result.get("web_search_providers", []),
            "web_search_attempts": web_result.get("web_search_attempts", []),
            "web_search_used": True,
            "web_research_used": web_result.get("web_research_used", False),
            "web_pages_fetched": web_result.get("web_pages_fetched", 0),
            "web_pages_extracted": web_result.get("web_pages_extracted", 0),
            "need_web_search": False,
            "retrieval_mode": "local_and_web_hybrid",
        }
    )
    return result
