#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web UI of the Local Knowledge Base Agent: a chat page for questions (with follow-ups), a documents
page (upload, processing progress, review, delete) and a settings page (API, maintenance).

The page itself is scripts/web_page.html. Who may use which part is decided in web_access.py:
by default the UI listens on this computer only; opened to the network it needs a password, and
documents and settings can still only be managed from this computer.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from api_providers import (  # noqa: E402
    default_base_url,
    discover_models,
    infer_provider,
    provider_options,
    read_env as read_api_env,
    set_env_value,
    validate_chat_model,
    write_env as write_api_env,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
RAW_DIR = ROOT / "knowledge_base" / "raw"
DB_PATH = ROOT / "knowledge_base" / "index" / "knowledge.db"
PROCESSED_DIR = ROOT / "knowledge_base" / "processed"
TRASH_DIR = ROOT / "knowledge_base" / "trash"
PAGE_FILE = SCRIPTS_DIR / "web_page.html"
DEFAULT_APP_TITLE = "知识库问答助手"
JSON_BODY_MAX_BYTES = 1024 * 1024
# A rejected upload this small is read and dropped, so the browser gets the reason instead of a
# reset connection; a larger one is not read.
REJECTED_UPLOAD_DRAIN_BYTES = 16 * 1024 * 1024
OCR_EVAL_SCRIPT = SCRIPTS_DIR / "evaluate_ocr_test_materials.py"
WEB_UI_STDERR_LOG = ROOT / "logs" / "web-ui.stderr.log"
LOG_ROTATE_BYTES = 5 * 1024 * 1024

os.chdir(ROOT)
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from healthcheck_agent import (  # noqa: E402
    SUPPORTED,
    api_backend_allowed,
    db_stats,
    extraction_report_stats,
    load_env,
    offline_translation_status,
    optional_extractors,
    sqlite_has_fts5,
    unsupported_raw_files,
)
from manage_knowledge_base import sources as kb_sources  # noqa: E402
from ingest_knowledge_base import (  # noqa: E402
    approve_review_file,
    ingest_in_progress,
    is_ignored_raw_file,
    pending_review_reports,
)
from kb import followup  # noqa: E402
from kb.config import api_enabled, runtime_config  # noqa: E402
import model_capabilities  # noqa: E402
import project_runtime  # noqa: E402
from query_knowledge_base import query as kb_query  # noqa: E402
import vector_store  # noqa: E402
from web_access import (  # noqa: E402
    LOOPBACK_HOST_NAMES,
    Access,
    cleared_cookie,
    cross_site_reason,
    host_name,
    is_loopback_address,
    is_loopback_bind,
    session_cookie,
)
from web_stats import DEFAULT_MINUTES_SAVED_PER_ANSWER, QuestionLog  # noqa: E402
from web_documents import (  # noqa: E402
    IndexJob,
    check_upload,
    documents_payload,
    remove_document,
    review_reason,
    save_upload,
    upload_max_bytes,
)

project_runtime.apply_project_runtime_env()


def raw_file_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    if not RAW_DIR.exists():
        return counts
    for path in RAW_DIR.rglob("*"):
        if path.is_file() and not is_ignored_raw_file(path) and path.suffix.lower() in SUPPORTED:
            counts[path.suffix.lower()] = counts.get(path.suffix.lower(), 0) + 1
    return dict(sorted(counts.items()))


def health_payload() -> dict[str, Any]:
    env = load_env(ROOT / ".env")
    stats = db_stats()
    extraction = extraction_report_stats()
    extractors = optional_extractors()
    api_configured = (
        env.get("LKA_USE_API", "false").lower() == "true"
        and bool(env.get("LKA_API_KEY"))
        and bool(env.get("LKA_API_MODEL"))
    )
    api_usable = api_backend_allowed(env)
    raw_counts = raw_file_counts()
    unsupported_files = unsupported_raw_files()
    chunks = int(stats.get("chunks", 0) or 0) if isinstance(stats, dict) else 0
    embedding = model_capabilities.embedding_status(env)
    vector_index = vector_store.vector_status(sqlite_chunk_count=chunks)
    ok = (
        sqlite_has_fts5()
        and RAW_DIR.exists()
        and stats.get("exists") is True
        and not stats.get("error")
        and int(stats.get("documents", 0) or 0) > 0
        and chunks > 0
        and int(extraction.get("requires_review", 0) or 0) == 0
        and not unsupported_files
    )
    return {
        "ok": ok,
        "python_version": sys.version.split()[0],
        "sqlite_fts5": sqlite_has_fts5(),
        "raw_dir": str(RAW_DIR),
        "raw_supported_files": sum(raw_counts.values()),
        "raw_counts": raw_counts,
        "raw_unsupported_files": unsupported_files,
        "index": stats,
        "optional_extractors": extractors,
        "offline_translation": offline_translation_status(),
        "online_translation": {
            "en_to_zh_ready": api_configured and api_usable,
            "provider": env.get("LKA_API_PROVIDER", "none"),
        },
        "extraction_report": extraction,
        "embedding": embedding,
        "vector_index": vector_index,
        "llm_capabilities": {
            "chunking_enabled": model_capabilities.enabled(env, "LKA_LLM_CHUNKING_ENABLED"),
            "query_routing_enabled": model_capabilities.enabled(env, "LKA_LLM_QUERY_ROUTING_ENABLED"),
            "rerank_enabled": model_capabilities.enabled(env, "LKA_RERANK_ENABLED"),
            "degraded": not bool(vector_index.get("ready")) or not bool(embedding.get("ready")),
            "degrade_reason": vector_index.get("reason") if not vector_index.get("ready") else embedding.get("reason"),
        },
        "api_configured": api_configured,
        "api_backend_usable": api_usable,
        "api_provider": env.get("LKA_API_PROVIDER", "none"),
        "scale_warning": chunks >= 50000,
        "pending_reviews": len(pending_review_reports(RAW_DIR, PROCESSED_DIR)),
        "job": JOBS.snapshot(),
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# An answer API call that failed falls back to answering on this computer. Users see why, in
# words they can act on, instead of silently getting the less accurate local answer.
API_FAILURE_KINDS = (
    (r"HTTP 402|Insufficient Balance|insufficient_balance", "balance", "大模型 API 余额不足，需要到服务商平台充值"),
    (r"HTTP 40[13]|Authentication|Unauthorized|invalid.{0,12}key", "key", "大模型 API Key 无效或没有权限，需要在设置页重新配置"),
    (r"HTTP 429|rate.?limit|Too Many Requests", "rate", "大模型 API 调用太频繁被限流，稍后会自动恢复"),
    (r"timed? ?out|URLError|Connection|getaddrinfo|Remote end closed|IncompleteRead", "network", "连不上大模型服务（网络或服务暂时不可用）"),
)
# How long a failure keeps showing in the status bar when no later API answer succeeded.
API_FAILURE_SHOWN_SECONDS = 30 * 60
_api_failure_lock = threading.Lock()
_last_api_failure: dict[str, Any] | None = None


def api_failure(result: dict[str, Any]) -> dict[str, str] | None:
    text = str(result.get("agent_error") or result.get("api_error") or "")
    if not text:
        return None
    for pattern, kind, message in API_FAILURE_KINDS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return {"kind": kind, "message": message}
    return {"kind": "other", "message": "大模型 API 调用失败"}


def note_api_outcome(result: dict[str, Any]) -> dict[str, str] | None:
    """Remember the latest API failure for the status bar; an answer the API gave clears it."""
    global _last_api_failure
    failure = api_failure(result)
    with _api_failure_lock:
        if failure:
            _last_api_failure = {**failure, "time": time.time()}
        elif result.get("api_used"):
            _last_api_failure = None
    return failure


def recent_api_failure() -> dict[str, Any] | None:
    with _api_failure_lock:
        failure = dict(_last_api_failure) if _last_api_failure else None
    if not failure or time.time() - float(failure["time"]) > API_FAILURE_SHOWN_SECONDS:
        return None
    if not api_enabled(runtime_config()):
        return None
    return failure


def indexed_document_count(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(db_path)
        try:
            return int(conn.execute("SELECT COUNT(*) FROM documents WHERE duplicate_of IS NULL").fetchone()[0] or 0)
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def status_payload() -> dict[str, Any]:
    """What every user's page needs: whether there is anything to ask about, and whether it is changing."""
    return {
        "ok": True,
        "documents": indexed_document_count(DB_PATH),
        "needs_review": len(pending_review_reports(RAW_DIR, PROCESSED_DIR)),
        "processing": JOBS.running() or ingest_in_progress(DB_PATH),
        "api": api_enabled(runtime_config()),
        "api_failure": recent_api_failure(),
    }


def api_config_payload() -> dict[str, Any]:
    env = read_api_env()
    provider = infer_provider(
        env.get("LKA_API_BASE_URL", ""),
        env.get("LKA_API_KEY", ""),
        env.get("LKA_API_PROVIDER", ""),
    )
    configured = (
        env.get("LKA_USE_API", "false").lower() == "true"
        and bool(env.get("LKA_API_KEY"))
        and bool(env.get("LKA_API_MODEL"))
    )
    return {
        "configured": configured,
        "usable": configured and api_backend_allowed(env),
        "enabled": env.get("LKA_USE_API", "false").lower() == "true",
        "has_key": bool(env.get("LKA_API_KEY")) and bool(env.get("LKA_API_MODEL")),
        "provider": provider,
        "base_url": env.get("LKA_API_BASE_URL", ""),
        "model": env.get("LKA_API_MODEL", ""),
        "providers": provider_options(),
    }


def redact_secrets(text: str) -> str:
    return re.sub(r"(sk-[A-Za-z0-9_-]{8,}|tp-[A-Za-z0-9_-]{8,})", "***", str(text))


def rotate_web_ui_logs() -> None:
    try:
        if not WEB_UI_STDERR_LOG.exists() or WEB_UI_STDERR_LOG.stat().st_size < LOG_ROTATE_BYTES:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = WEB_UI_STDERR_LOG.with_name(f"web-ui.stderr.{stamp}.log")
        WEB_UI_STDERR_LOG.rename(target)
    except OSError:
        return


def reviews_payload() -> dict[str, Any]:
    reviews = pending_review_reports(RAW_DIR, PROCESSED_DIR)
    return {
        "pending": len(reviews),
        "reviews": [
            {
                "file_path": item.get("file_path"),
                "file_type": item.get("file_type"),
                "warnings": item.get("warnings", []),
                "reasons": [review_reason(warning) for warning in item.get("warnings", [])],
                "text_chars": item.get("text_chars", 0),
            }
            for item in reviews
        ],
    }


def api_key_from_payload(payload: dict[str, Any], base_url: str) -> str:
    """The key typed in, or else the saved one, but only for the saved address: a changed address
    gets no key it was not given, so the saved key never goes to a server it was not meant for."""
    api_key = str(payload.get("api_key", "")).strip()
    if api_key:
        return api_key
    env = read_api_env()
    if base_url.rstrip("/") == env.get("LKA_API_BASE_URL", "").rstrip("/"):
        return env.get("LKA_API_KEY", "")
    return ""


def save_api_config(payload: dict[str, Any]) -> dict[str, Any]:
    provider = str(payload.get("provider", "openai-compatible")).strip()
    base_url = str(payload.get("base_url", "")).strip() or default_base_url(provider)
    api_key = api_key_from_payload(payload, base_url)
    model = str(payload.get("model", "")).strip()
    if not api_key or not model:
        raise ValueError("API key 和模型不能为空。")
    validation = validate_chat_model(provider, base_url, api_key, model)
    if not validation.get("ok"):
        raise ValueError(f"模型调用验证失败：{validation.get('error')}")
    env = read_api_env()
    env.update(
        {
            "LKA_USE_API": "true",
            "LKA_API_PROVIDER": provider,
            "LKA_API_BASE_URL": base_url.rstrip("/"),
            "LKA_API_KEY": api_key,
            "LKA_API_MODEL": model,
            "LKA_API_TIMEOUT_SECONDS": env.get("LKA_API_TIMEOUT_SECONDS", "20"),
        }
    )
    write_api_env(env)
    return {"ok": True, "provider": provider, "base_url": base_url.rstrip("/"), "model": model}


def set_api_mode(payload: dict[str, Any]) -> dict[str, Any]:
    """Switch between answering with the model API and answering on this computer only; the saved
    key and model stay, so switching back needs no new setup."""
    enabled = bool(payload.get("enabled"))
    env = read_api_env()
    if enabled and not (env.get("LKA_API_KEY") and env.get("LKA_API_MODEL")):
        raise ValueError("还没有配置 API：请先在下方填好 API Key、读取并选择模型，点“保存并验证”。")
    set_env_value("LKA_USE_API", "true" if enabled else "false")
    return {"ok": True, **api_config_payload()}


def verify_embedding_api() -> dict[str, Any]:
    config = model_capabilities.runtime_config()
    result = model_capabilities.embed_texts(["本地知识库 embedding 连接测试"], config, retries=0)
    public = {
        key: value
        for key, value in result.items()
        if key not in {"vectors", "config"}
    }
    if result.get("config"):
        public["config"] = result["config"]
    if public.get("error"):
        public["error"] = redact_secrets(str(public["error"]))
    return {"ok": bool(result.get("ok")), **public}


def verify_vector_index() -> dict[str, Any]:
    stats = db_stats()
    chunks = int(stats.get("chunks", 0) or 0) if isinstance(stats, dict) else 0
    status = vector_store.vector_status(sqlite_chunk_count=chunks)
    return {"ok": bool(status.get("ready")), "status": status}


def retrieval_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question", "")).strip() or "本地知识库状态"
    config = model_capabilities.runtime_config()
    stats = db_stats()
    chunks = int(stats.get("chunks", 0) or 0) if isinstance(stats, dict) else 0
    return {
        "ok": True,
        "question": question,
        "query_route": model_capabilities.classify_query(question, config),
        "embedding": model_capabilities.embedding_status(config),
        "vector_index": vector_store.vector_status(sqlite_chunk_count=chunks),
        "rerank_enabled": model_capabilities.enabled(config, "LKA_RERANK_ENABLED"),
    }


def run_ocr_eval() -> dict[str, Any]:
    if not OCR_EVAL_SCRIPT.exists():
        return {"ok": False, "error": f"Missing script: {OCR_EVAL_SCRIPT}"}
    completed = subprocess.run(
        [sys.executable, str(OCR_EVAL_SCRIPT)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError:
        data = {"stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}
    data["ok"] = completed.returncode == 0
    data["returncode"] = completed.returncode
    return data




JOBS = IndexJob(RAW_DIR, DB_PATH, PROCESSED_DIR, redact=redact_secrets)
# Set in main(): the access password, and whether the UI listens on this computer only.
ACCESS = Access("")
LOCAL_ONLY = True


def web_setting(name: str, default: str = "") -> str:
    return os.environ.get(name) or load_env(ROOT / ".env").get(name, "") or default


def app_title() -> str:
    return web_setting("LKA_APP_TITLE", DEFAULT_APP_TITLE)[:60]


QUESTIONS = QuestionLog(
    ROOT / "logs" / "questions.jsonl",
    ROOT / "logs" / "answer_feedback.jsonl",
    enabled=web_setting("LKA_QUESTION_LOG", "on").lower() != "off",
)


def minutes_saved_per_answer() -> float:
    try:
        return max(0.0, float(web_setting("LKA_MINUTES_SAVED_PER_ANSWER", str(DEFAULT_MINUTES_SAVED_PER_ANSWER))))
    except ValueError:
        return float(DEFAULT_MINUTES_SAVED_PER_ANSWER)


def page_html(admin: bool) -> str:
    title = app_title()
    config = {
        "admin": admin,
        "title": title,
        "loginRequired": ACCESS.required,
        "uploadMaxMb": upload_max_bytes() // 1024 // 1024,
    }
    # "</" escaped: a title can never end the script element.
    blob = json.dumps(config, ensure_ascii=False).replace("</", "<\\/")
    page = PAGE_FILE.read_text(encoding="utf-8")
    page = re.sub(r"/\*LKA_CONFIG\*/\{[^\n]*\};", lambda _: f"{blob};", page, count=1)
    page = page.replace("<title>知识库问答助手</title>", f"<title>{html.escape(title)}</title>", 1)
    return page.replace('<h1 id="appTitle">知识库问答助手</h1>', f'<h1 id="appTitle">{html.escape(title)}</h1>', 1)


def login_html() -> str:
    title = html.escape(app_title())
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ margin:0; min-height:100vh; display:grid; place-items:center; background:#f5f6f8;
         font:14px/1.6 "Segoe UI","Microsoft YaHei",Arial,sans-serif; color:#1d2329; }}
  form {{ background:#fff; border:1px solid #dde3ea; border-radius:12px; padding:28px; width:min(360px, 90vw); }}
  h1 {{ font-size:18px; margin:0 0 6px; }}
  p {{ color:#66717f; margin:0 0 16px; }}
  input {{ width:100%; box-sizing:border-box; padding:9px 10px; border:1px solid #dde3ea; border-radius:6px; font:inherit; }}
  button {{ margin-top:12px; width:100%; padding:9px; border:0; border-radius:6px; background:#1769aa; color:#fff; font:inherit; cursor:pointer; }}
  .error {{ color:#b42318; min-height:1.6em; margin-top:8px; }}
</style></head>
<body><form id="form"><h1>{title}</h1><p>请输入访问密码</p>
<input id="password" type="password" autocomplete="current-password" autofocus>
<button type="submit">进入</button><div id="error" class="error"></div></form>
<script>
document.getElementById('form').onsubmit = async (event) => {{
  event.preventDefault();
  const response = await fetch('/api/login', {{method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{password: document.getElementById('password').value}})}});
  const data = await response.json().catch(() => ({{}}));
  if (response.ok) location.reload(); else document.getElementById('error').textContent = data.error || '登录失败';
}};
</script></body></html>"""


# Everyone signed in may ask; managing documents and settings is limited to this computer.
SHARED_GET = {"/api/status", "/api/query"}
SHARED_POST = {"/api/query-stream", "/api/query", "/api/logout", "/api/feedback"}
LEGACY_JOB_PATHS = {"/api/update": "update", "/api/rebuild-strict": "rebuild", "/api/vector-rebuild": "vector-rebuild"}


class RequestRejected(Exception):
    def __init__(self, message: str, status: HTTPStatus) -> None:
        super().__init__(message)
        self.status = status


class AgentHandler(BaseHTTPRequestHandler):
    server_version = "LocalKnowledgeAgent/1.2"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), redact_secrets(fmt % args)))

    # ---- who may do what ----

    def client_is_local(self) -> bool:
        return is_loopback_address(self.client_address[0])

    def check_request(self, path: str) -> None:
        if LOCAL_ONLY and host_name(str(self.headers.get("Host") or "")) not in LOOPBACK_HOST_NAMES:
            raise RequestRejected("请用 http://127.0.0.1 打开本页面。", HTTPStatus.FORBIDDEN)
        if path.startswith("/api/") or self.command == "POST":
            if cross_site_reason(self.headers, LOCAL_ONLY):
                raise RequestRejected("请求被拒绝：只接受本页面发出的请求。", HTTPStatus.FORBIDDEN)
        if self.command == "POST":
            content_type = str(self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            expected = "application/octet-stream" if path == "/api/upload" else "application/json"
            if content_type != expected:
                raise RequestRejected("请求格式不正确。", HTTPStatus.UNSUPPORTED_MEDIA_TYPE)

    def check_signed_in(self) -> None:
        if not ACCESS.signed_in(self.headers.get("Cookie")):
            raise RequestRejected("请先登录。", HTTPStatus.UNAUTHORIZED)

    def check_manager(self) -> None:
        if not self.client_is_local():
            raise RequestRejected("资料和设置只能在运行本服务的电脑上管理。", HTTPStatus.FORBIDDEN)

    # ---- routes ----

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            self.check_request(path)
            if path in {"/", "/index.html"}:
                if ACCESS.signed_in(self.headers.get("Cookie")):
                    self.send_html(page_html(admin=self.client_is_local()))
                else:
                    self.send_html(login_html())
                return
            if path == "/favicon.ico":
                self.send_empty(HTTPStatus.NO_CONTENT)
                return
            self.check_signed_in()
            if path not in SHARED_GET:
                self.check_manager()
            if path == "/api/status":
                self.send_json(status_payload())
            elif path == "/api/query":
                params = parse_qs(parsed.query)
                question = params.get("question", [""])[0].strip()
                if not question:
                    raise RequestRejected("Missing question", HTTPStatus.BAD_REQUEST)
                self.send_json(
                    kb_query(
                        DB_PATH,
                        question,
                        limit=int(params.get("limit", ["5"])[0] or 5),
                        use_web=params.get("use_web", ["false"])[0].lower() == "true",
                        allow_api=params.get("allow_api", ["true"])[0].lower() != "false",
                    )
                )
            elif path == "/api/health":
                self.send_json(health_payload())
            elif path == "/api/sources":
                self.send_json(kb_sources(DB_PATH))
            elif path == "/api/documents":
                self.send_json(documents_payload(RAW_DIR, DB_PATH, PROCESSED_DIR, JOBS.running()))
            elif path == "/api/reviews":
                self.send_json(reviews_payload())
            elif path == "/api/api-config":
                self.send_json(api_config_payload())
            elif path == "/api/jobs":
                self.send_json(JOBS.snapshot())
            elif path == "/api/stats":
                days_text = parse_qs(parsed.query).get("days", ["30"])[0]
                days = int(days_text) if days_text.isdigit() and int(days_text) > 0 else None
                self.send_json(QUESTIONS.stats(days, minutes_saved_per_answer()))
            else:
                raise RequestRejected("Not found", HTTPStatus.NOT_FOUND)
        except RequestRejected as exc:
            self.send_error_json(str(exc), exc.status)
        except Exception as exc:  # noqa: BLE001
            self.send_error_json(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        self.body_read = False
        try:
            self.check_request(path)
            if path == "/api/login":
                self.login()
                return
            self.check_signed_in()
            if path not in SHARED_POST:
                self.check_manager()
            if path == "/api/query-stream":
                self.stream_query(self.read_json())
            elif path == "/api/query":
                payload = self.read_json()
                question = str(payload.get("question", "")).strip()
                if not question:
                    raise RequestRejected("Missing question", HTTPStatus.BAD_REQUEST)
                self.send_json(
                    kb_query(
                        DB_PATH,
                        question,
                        limit=int(payload.get("limit", 5) or 5),
                        use_web=bool(payload.get("use_web", False)),
                        allow_api=bool(payload.get("allow_api", True)),
                    )
                )
            elif path == "/api/feedback":
                payload = self.read_json()
                QUESTIONS.feedback(str(payload.get("id", "")), bool(payload.get("helpful")))
                self.send_json({"ok": True})
            elif path == "/api/logout":
                ACCESS.sign_out(self.headers.get("Cookie"))
                self.send_json({"ok": True}, headers={"Set-Cookie": cleared_cookie()})
            elif path == "/api/upload":
                self.upload()
            elif path == "/api/documents/delete":
                removed = remove_document(str(self.read_json().get("file_path", "")), RAW_DIR, TRASH_DIR)
                self.send_json({"ok": True, "removed": removed, "job": JOBS.start("update")})
            elif path == "/api/jobs":
                self.send_json(JOBS.start(str(self.read_json().get("kind", "update"))))
            elif path in LEGACY_JOB_PATHS:
                self.send_json(JOBS.start(LEGACY_JOB_PATHS[path]))
            elif path == "/api/reviews/approve":
                approval = approve_review_file(RAW_DIR, PROCESSED_DIR, str(self.read_json().get("file_path", "")))
                self.send_json({"ok": True, "approval": approval, "job": JOBS.start("update")})
            elif path == "/api/embedding-verify":
                self.send_json(verify_embedding_api())
            elif path == "/api/vector-verify":
                self.send_json(verify_vector_index())
            elif path == "/api/retrieval-diagnostics":
                self.send_json(retrieval_diagnostics(self.read_json()))
            elif path == "/api/ocr-eval":
                self.send_json(run_ocr_eval())
            elif path == "/api/open-raw-folder":
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                os.startfile(RAW_DIR)  # type: ignore[attr-defined]
                self.send_json({"ok": True})
            elif path == "/api/api-models":
                payload = self.read_json()
                provider = str(payload.get("provider", "openai-compatible")).strip()
                base_url = str(payload.get("base_url", "")).strip() or default_base_url(provider)
                api_key = api_key_from_payload(payload, base_url)
                if not api_key:
                    raise ValueError("请输入 API Key（换了 API 地址时需要重新输入）。")
                result = discover_models(provider, base_url, api_key)
                if not result.get("ok"):
                    raise ValueError(str(result.get("error", "没有找到可用模型。")))
                self.send_json(result)
            elif path == "/api/api-save":
                self.send_json(save_api_config(self.read_json()))
            elif path == "/api/api-mode":
                self.send_json(set_api_mode(self.read_json()))
            else:
                raise RequestRejected("Not found", HTTPStatus.NOT_FOUND)
        except RequestRejected as exc:
            self.discard_unread_body()
            self.send_error_json(str(exc), exc.status)
        except ValueError as exc:
            self.discard_unread_body()
            self.send_error_json(str(exc), HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            self.discard_unread_body()
            self.send_error_json(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def discard_unread_body(self) -> None:
        """A request refused before its body was read: read and drop a small body first. Windows
        resets a connection closed with unread data, and the browser would then see a network error
        instead of the reason; a large body is not read and the connection is closed."""
        if getattr(self, "body_read", True):
            return
        self.body_read = True
        length = int(self.headers.get("Content-Length", "0") or 0)
        if 0 < length <= REJECTED_UPLOAD_DRAIN_BYTES:
            self.rfile.read(length)
        elif length > 0:
            self.close_connection = True

    def login(self) -> None:
        if not ACCESS.required:
            self.send_json({"ok": True})
            return
        client = self.client_address[0]
        if ACCESS.locked_out(client):
            raise RequestRejected("密码错误次数太多，请 10 分钟后再试。", HTTPStatus.TOO_MANY_REQUESTS)
        token = ACCESS.sign_in(client, str(self.read_json().get("password", "")))
        if not token:
            raise RequestRejected("密码不正确。", HTTPStatus.UNAUTHORIZED)
        self.send_json({"ok": True}, headers={"Set-Cookie": session_cookie(token)})

    def upload(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        file_name = unquote(str(self.headers.get("X-File-Name") or ""))
        # Name and size are checked before reading, so a refused file's body is still unread and
        # discard_unread_body() can drop it; once reading starts, the body counts as read.
        check_upload(file_name, length)
        self.body_read = True
        saved = save_upload(self.rfile, length, file_name, RAW_DIR)
        self.send_json({"ok": True, **saved})

    # ---- responses ----

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        if length > JSON_BODY_MAX_BYTES:
            raise RequestRejected("请求内容太大。", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        self.body_read = True
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        data = json.loads(raw) if raw.strip() else {}
        return data if isinstance(data, dict) else {}

    def send_common_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")

    def send_html(self, content: str) -> None:
        body = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_common_headers()
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK, headers: dict[str, str] | None = None) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_common_headers()
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status: HTTPStatus) -> None:
        self.send_json({"ok": False, "error": redact_secrets(message)}, status)

    def send_empty(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def send_stream_event(self, event: str, data: dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def stream_query(self, payload: dict[str, Any]) -> None:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise RequestRejected("请输入问题。", HTTPStatus.BAD_REQUEST)

        limit = int(payload.get("limit", 5) or 5)
        use_web = bool(payload.get("use_web", False))
        allow_api = bool(payload.get("allow_api", True))
        history = payload.get("history")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()

        result_box: dict[str, Any] = {}
        error_box: dict[str, Exception] = {}

        def run_query() -> None:
            try:
                # A follow-up («它最多重试几次») is searched as the complete question it stands for.
                understood = followup.standalone_question(question, history, runtime_config(), allow_api)
                result_box["understood"] = understood
                result_box["result"] = kb_query(DB_PATH, understood["question"], limit=limit, use_web=use_web, allow_api=allow_api)
            except Exception as exc:  # noqa: BLE001
                error_box["error"] = exc

        started_at = time.monotonic()
        self.send_stream_event("status", {"message": "正在查找资料，资料不足时会联网搜索…" if use_web else "正在查找资料…"})
        worker = threading.Thread(target=run_query, daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(timeout=1.0)
            if worker.is_alive():
                elapsed = max(1, int(time.monotonic() - started_at))
                self.send_stream_event("status", {"message": "正在整理答案…", "elapsed_seconds": elapsed})

        if error_box:
            self.send_stream_event("error", {"error": redact_secrets(str(error_box["error"]))})
            return

        result = result_box["result"]
        understood = result_box["understood"]
        sources = result.get("sources", [])
        failure = note_api_outcome(result)
        sys.stderr.write(
            "query "
            + json.dumps(
                {
                    "question": question,
                    "understood_as": understood["question"] if understood["rewritten"] else None,
                    "followup_method": understood["method"],
                    "source_type": result.get("source_type"),
                    "web_search_provider": result.get("web_search_provider"),
                    "web_search_reason": result.get("web_search_reason"),
                    "sources": len(sources),
                    "api_failure": failure["kind"] if failure else None,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        sys.stderr.flush()
        if understood["rewritten"]:
            self.send_stream_event("understood", {"question": understood["question"], "method": understood["method"]})
        self.send_stream_event(
            "status",
            {
                "message": f"找到 {len(sources)} 处相关资料，正在生成答案…" if sources else "正在生成答案…",
                "elapsed_seconds": max(0, int(time.monotonic() - started_at)),
            },
        )
        self.send_stream_event(
            "meta",
            {
                "source_type": result.get("source_type"),
                "web_search_provider": result.get("web_search_provider"),
                "web_search_reason": result.get("web_search_reason"),
                "cache_hit": result.get("cache_hit", False),
                "api_used": result.get("api_used", False),
                "agent_rounds": (result.get("agent") or {}).get("rounds"),
                "user_guidance": result.get("user_guidance"),
                "async_update_scheduled": result.get("async_update_scheduled", False),
                "index_status": result.get("index_status", "ready"),
                "index_message": result.get("index_message"),
                "api_failure": failure,
            },
        )
        record_id = QUESTIONS.record(
            question,
            understood["question"] if understood["rewritten"] else None,
            result,
            time.monotonic() - started_at,
            remote=not self.client_is_local(),
            api_failure=failure["kind"] if failure else None,
        )
        self.send_stream_event("sources", {"sources": sources})
        self.send_stream_event("answer-start", {})
        answer = str(result.get("answer") or "没有返回答案")
        for start in range(0, len(answer), 18):
            self.send_stream_event("answer-delta", {"text": answer[start : start + 18]})
            time.sleep(0.025)
        self.send_stream_event(
            "done",
            {
                "source_type": result.get("source_type"),
                "sources_count": len(sources),
                "elapsed_seconds": round(time.monotonic() - started_at, 2),
                "record_id": record_id,
            },
        )


def network_addresses() -> list[str]:
    try:
        names = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        return []
    return sorted({name for name in names if not name.startswith("127.")})


def main() -> None:
    global ACCESS, LOCAL_ONLY
    rotate_web_ui_logs()
    parser = argparse.ArgumentParser(description="Run the Web UI of the Local Knowledge Base Agent.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. The default keeps the UI on this computer; 0.0.0.0 opens it to the network and needs LKA_WEB_PASSWORD.",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the Web UI in the default browser.")
    args = parser.parse_args()

    LOCAL_ONLY = is_loopback_bind(args.host)
    password = web_setting("LKA_WEB_PASSWORD")
    if not LOCAL_ONLY and not password:
        print(f"对外开放（--host {args.host}）时必须设置访问密码：在 .env 中加一行 LKA_WEB_PASSWORD=你的密码，然后重新启动。")
        sys.exit(2)
    ACCESS = Access(password)

    server = ThreadingHTTPServer((args.host, args.port), AgentHandler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Local Knowledge Base Agent Web UI: {url}")
    if not LOCAL_ONLY:
        for address in network_addresses():
            print(f"同一网络的电脑可以打开：http://{address}:{args.port}/（需要访问密码；只能提问，资料和设置只能在本机管理）")
    print("Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Web UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
