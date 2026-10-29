#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web UI checks: follow-up questions, access rules, uploads, the document list and the
background processing job, and the HTTP server end to end (on a temporary knowledge base).

Nothing here touches the real knowledge base or calls an API: the index job runs a stand-in
for the ingest, and the server under test is pointed at temporary folders.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

os.environ["LKA_API_USAGE_LOG"] = "off"  # test calls must not land in the real usage log

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from kb import followup  # noqa: E402
from kb_ingest import progress  # noqa: E402
import web_access  # noqa: E402
import web_documents  # noqa: E402


def test_followup_rules() -> None:
    previous = "X100 Fan测试的完整流程是什么"
    cases = {
        "它最多重试几轮？": "X100 Fan测试最多重试几轮",
        "这个测试的门限是多少": "X100 Fan测试的门限是多少",
        "那Speaker测试呢": "Speaker测试的完整流程是什么",
        "门限呢？": "X100 Fan测试的门限是什么",
        "怎么判定通过": "X100 Fan测试怎么判定通过",
        "具体步骤是什么": "X100 Fan测试的具体步骤是什么",
        "那么它的判定标准呢": "X100 Fan测试的判定标准是什么",
    }
    for question, expected in cases.items():
        assert followup.needs_context(question), question
        assert followup.rewrite_by_rules(question, previous) == expected, (question, followup.rewrite_by_rules(question, previous))
    # Questions that say what they are about are used as typed.
    for question in ("G-sensor怎么测试", "前置摄像头测试前要预热多久", "SFT有哪些测试项", "AB平台的架构分为哪几层"):
        assert not followup.needs_context(question), question
    assert followup.rewrite_by_rules("那SFT呢", "FFT有哪些测试项") == "SFT有哪些测试项"
    assert followup.subject_of("请问AB平台的架构分为哪几层") == "AB平台"
    assert followup.subject_of("G-sensor怎么测试") == "G-sensor"

    # No history, or a complete question: unchanged and no model call (API disabled here anyway).
    config = {"LKA_USE_API": "false"}
    alone = followup.standalone_question("它最多重试几轮", [], config)
    assert alone == {"question": "它最多重试几轮", "rewritten": False, "method": "none"}
    history = [{"question": previous, "answer": "按顺序执行…"}]
    complete = followup.standalone_question("SFT有哪些测试项", history, config)
    assert complete["rewritten"] is False and complete["method"] == "none"
    follow = followup.standalone_question("它最多重试几轮", history, config)
    assert follow == {"question": "X100 Fan测试最多重试几轮", "rewritten": True, "method": "rules"}
    # Malformed history from a client is ignored, never an error.
    assert followup.clean_history("x") == [] and followup.clean_history([1, {"answer": "a"}]) == []
    assert len(followup.clean_history([{"question": str(i)} for i in range(9)])) == followup.HISTORY_TURNS


def test_followup_api_failure_falls_back_to_rules() -> None:
    original = followup.rewrite_with_api
    try:
        def broken(*args: object, **kwargs: object) -> str:
            raise RuntimeError("timeout")

        followup.rewrite_with_api = broken
        config = {"LKA_USE_API": "true", "LKA_API_KEY": "k", "LKA_API_MODEL": "m", "LKA_API_BASE_URL": "https://api.deepseek.com"}
        history = [{"question": "X100 Fan测试的完整流程是什么", "answer": ""}]
        result = followup.standalone_question("它最多重试几轮", history, config)
        assert result["method"] == "rules" and result["question"] == "X100 Fan测试最多重试几轮"

        followup.rewrite_with_api = lambda question, turns, cfg: "X100 Fan测试最多重试几轮？"
        assert followup.standalone_question("它最多重试几轮", history, config)["method"] == "api"
    finally:
        followup.rewrite_with_api = original


def test_access_rules() -> None:
    assert web_access.host_name("127.0.0.1:8765") == "127.0.0.1"
    assert web_access.host_name("[::1]:8765") == "::1"
    assert web_access.host_name("localhost") == "localhost"
    assert web_access.is_loopback_address("127.0.0.1") and web_access.is_loopback_address("::1")
    assert web_access.is_loopback_address("::ffff:127.0.0.1")
    assert not web_access.is_loopback_address("192.168.1.20") and not web_access.is_loopback_address("bad")
    assert web_access.is_loopback_bind("127.0.0.1") and not web_access.is_loopback_bind("0.0.0.0")

    page = {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765", "Sec-Fetch-Site": "same-origin"}
    assert web_access.cross_site_reason(page, local_only=True) == ""
    assert web_access.cross_site_reason({"Host": "127.0.0.1:8765"}, local_only=True) == ""  # a script, no browser
    assert web_access.cross_site_reason({**page, "Origin": "https://evil.example"}, local_only=True)
    assert web_access.cross_site_reason({**page, "Sec-Fetch-Site": "cross-site"}, local_only=True)
    assert web_access.cross_site_reason({**page, "Origin": "null"}, local_only=True)
    # DNS rebinding: the page's own origin, but under another host name.
    rebound = {"Host": "evil.example:8765", "Origin": "http://evil.example:8765", "Sec-Fetch-Site": "same-origin"}
    assert web_access.cross_site_reason(rebound, local_only=True)
    lan = {"Host": "192.168.1.5:8765", "Origin": "http://192.168.1.5:8765", "Sec-Fetch-Site": "same-origin"}
    assert web_access.cross_site_reason(lan, local_only=False) == ""

    open_access = web_access.Access("")
    assert not open_access.required and open_access.signed_in(None)
    access = web_access.Access("secret")
    assert access.required and not access.signed_in(None)
    original_sleep = web_access.time.sleep
    web_access.time.sleep = lambda seconds: None
    try:
        assert access.sign_in("10.0.0.2", "wrong") is None
        token = access.sign_in("10.0.0.2", "secret")
        assert token and access.signed_in(f"{web_access.SESSION_COOKIE}={token}")
        assert not access.signed_in(f"{web_access.SESSION_COOKIE}=forged")
        access.sign_out(f"{web_access.SESSION_COOKIE}={token}")
        assert not access.signed_in(f"{web_access.SESSION_COOKIE}={token}")
        for _ in range(web_access.LOGIN_FAILURES_ALLOWED):
            access.sign_in("10.0.0.9", "wrong")
        assert access.locked_out("10.0.0.9") and access.sign_in("10.0.0.9", "secret") is None
        assert not access.locked_out("10.0.0.2")
    finally:
        web_access.time.sleep = original_sleep


def test_upload_names_and_storage() -> None:
    assert web_documents.safe_upload_name("报告.PDF") == "报告.PDF"
    assert web_documents.safe_upload_name("..\\..\\windows\\evil.docx") == "evil.docx"
    assert web_documents.safe_upload_name("a/b/c.md") == "c.md"
    assert web_documents.safe_upload_name('a<b>:"c".xlsx') == "a_b___c_.xlsx"
    assert web_documents.safe_upload_name("CON.txt") == "_CON.txt"
    assert len(web_documents.safe_upload_name("长" * 300 + ".pdf")) == web_documents.FILE_NAME_MAX_CHARS
    for bad in ("tool.exe", "noext", "", ".hidden.md", "~$lock.docx", ".pdf"):
        try:
            web_documents.safe_upload_name(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")

    with tempfile.TemporaryDirectory() as temp:
        raw = Path(temp) / "raw"
        trash = Path(temp) / "trash"
        first = web_documents.save_upload(io.BytesIO(b"# A\nalpha"), 9, "说明.md", raw)
        assert first == {"file_path": "markdown/说明.md", "file_name": "说明.md", "size": 9, "replaced": False}
        again = web_documents.save_upload(io.BytesIO(b"# A\nbeta!"), 9, "说明.md", raw)
        assert again["replaced"] and (raw / "markdown" / "说明.md").read_bytes() == b"# A\nbeta!"
        assert web_documents.save_upload(io.BytesIO(b"%PDF"), 4, "x.pdf", raw)["file_path"] == "pdf/x.pdf"
        # A short body (the browser went away) leaves no partial file behind.
        try:
            web_documents.save_upload(io.BytesIO(b"abc"), 10, "cut.docx", raw)
        except ValueError:
            pass
        else:
            raise AssertionError("short upload accepted")
        assert not any(path.name.startswith("~$") or path.name == "cut.docx" for path in raw.rglob("*"))
        try:
            web_documents.save_upload(io.BytesIO(b""), 0, "empty.md", raw)
        except ValueError:
            pass
        else:
            raise AssertionError("empty upload accepted")

        removed = web_documents.remove_document("markdown/说明.md", raw, trash)
        assert not (raw / "markdown" / "说明.md").exists() and Path(removed["moved_to"]).exists()
        web_documents.save_upload(io.BytesIO(b"# B\nagain"), 9, "说明.md", raw)
        removed_again = web_documents.remove_document("markdown/说明.md", raw, trash)
        assert Path(removed_again["moved_to"]).exists() and Path(removed["moved_to"]).read_bytes() == b"# A\nbeta!"
        for bad in ("../outside.md", "markdown/missing.md", ""):
            try:
                web_documents.remove_document(bad, raw, trash)
            except ValueError:
                continue
            raise AssertionError(f"removed {bad!r}")

        listing = web_documents.documents_payload(raw, Path(temp) / "none.db", Path(temp) / "processed", processing=True)
        assert [(item["file_path"], item["status"]) for item in listing["documents"]] == [("pdf/x.pdf", "processing")]
        listing = web_documents.documents_payload(raw, Path(temp) / "none.db", Path(temp) / "processed", processing=False)
        assert listing["documents"][0]["status"] == "failed"


def test_review_reasons() -> None:
    assert "扫描件" in web_documents.review_reason("Extracted text is very short; source may be image-only, encrypted, or layout-heavy.")
    assert web_documents.review_reason("DOCX contains 3 embedded image(s), but OCR returned no image text.") == "Word 文档里有 3 张图片，没能识别出其中的文字。"
    xlsx = web_documents.review_reason("XLSX content not extracted: 2 embedded object(s) not extracted (attached file on sheet A), embedded image(s) returned no OCR text.")
    assert xlsx == "Excel 中有部分内容没能读取：2 个嵌入对象（attached file on sheet A）；图片中的文字。", xlsx
    assert web_documents.review_reason("something new") == "something new"


def test_question_log_and_stats() -> None:
    import web_stats

    with tempfile.TemporaryDirectory() as temp:
        log = web_stats.QuestionLog(Path(temp) / "q.jsonl", Path(temp) / "f.jsonl")
        answered = {"source_type": "knowledge_base", "api_used": True, "answer": "整机保修 2 年", "sources": [{"file": "保修.md"}, {"file": "保修.md"}]}
        refused = {"source_type": "none", "answer": "本地知识库没有找到足够依据。", "sources": []}
        first = log.record("整机保修多久？", None, answered, 2.0, remote=False)
        log.record("整机保修多久", None, answered, 1.0, remote=True)
        log.record("它电池呢", "电池保修多久", refused, 3.0, remote=False)
        log.record("电池保修多久？", None, refused, 2.5, remote=False)
        log.feedback(first, False)
        log.feedback(first, True)  # the latest feedback counts
        try:
            log.feedback("../../etc", True)
        except ValueError:
            pass
        else:
            raise AssertionError("bad feedback id accepted")
        stats = log.stats(days=7, minutes_per_answer=6)
        totals = stats["totals"]
        assert totals["questions"] == 4 and totals["answered"] == 2 and totals["refused"] == 2, totals
        assert totals["answer_rate"] == 0.5 and totals["followups"] == 1 and totals["remote_questions"] == 1
        assert totals["hours_saved"] == 0.2 and totals["helpful_rate"] == 1.0 and totals["rated"] == 1
        assert stats["top_questions"][0] == {**stats["top_questions"][0], "question": "整机保修多久", "count": 2}
        assert [row["question"] for row in stats["unanswered"]] == ["电池保修多久？"] and stats["unanswered"][0]["count"] == 2
        assert stats["top_documents"] == [{"file": "保修.md", "count": 2}]
        assert len(stats["per_day"]) == 7 and stats["per_day"][-1]["answered"] == 2 and stats["per_day"][-1]["refused"] == 2
        assert stats["unhelpful"] == []
        assert log.stats(days=None)["totals"]["questions"] == 4

        quiet = web_stats.QuestionLog(Path(temp) / "off.jsonl", Path(temp) / "off-f.jsonl", enabled=False)
        quiet.record("x", None, answered, 1.0, remote=False)
        assert not (Path(temp) / "off.jsonl").exists() and quiet.stats()["recording"] is False


def test_api_failures() -> None:
    import web_agent

    balance = 'HTTP 402: {"error":{"message":"Insufficient Balance","type":"insufficient_balance_error"}}'
    assert web_agent.api_failure({"agent_error": balance})["kind"] == "balance"
    assert web_agent.api_failure({"agent_error": "HTTP 401: Authentication Fails"})["kind"] == "key"
    assert web_agent.api_failure({"api_error": "The read operation timed out"})["kind"] == "network"
    assert web_agent.api_failure({"agent_error": "HTTP 429: Too Many Requests"})["kind"] == "rate"
    assert web_agent.api_failure({"agent_error": "something odd"})["kind"] == "other"
    assert web_agent.api_failure({"answer": "fine"}) is None

    original_enabled = web_agent.api_enabled
    web_agent.api_enabled = lambda config: True
    try:
        web_agent.note_api_outcome({"agent_error": balance})
        assert web_agent.recent_api_failure()["kind"] == "balance"
        web_agent.note_api_outcome({"source_type": "none"})  # a local refusal says nothing about the API
        assert web_agent.recent_api_failure()["kind"] == "balance"
        web_agent.note_api_outcome({"api_used": True, "source_type": "knowledge_base"})  # the API answered again
        assert web_agent.recent_api_failure() is None
        web_agent.note_api_outcome({"agent_error": balance})
        web_agent.api_enabled = lambda config: False  # switched to local mode: nothing to warn about
        assert web_agent.recent_api_failure() is None
    finally:
        web_agent.api_enabled = original_enabled
        web_agent.note_api_outcome({"api_used": True})


def test_set_env_value() -> None:
    import api_providers

    with tempfile.TemporaryDirectory() as temp:
        env = Path(temp) / ".env"
        env.write_bytes("# comment kept\nLKA_USE_API=true\nLKA_API_KEY=sk-x\n".encode("utf-8"))
        api_providers.set_env_value("LKA_USE_API", "false", env)
        api_providers.set_env_value("LKA_APP_TITLE", "客户助手", env)
        assert env.read_bytes().decode("utf-8") == "# comment kept\nLKA_USE_API=false\nLKA_API_KEY=sk-x\nLKA_APP_TITLE=客户助手\n"
        crlf = Path(temp) / "crlf.env"
        crlf.write_bytes(b"A=1\r\nLKA_USE_API=true\r\n")
        api_providers.set_env_value("LKA_USE_API", "false", crlf)
        assert crlf.read_bytes() == b"A=1\r\nLKA_USE_API=false\r\n"  # the file's own line endings stay


def wait_for(job: web_documents.IndexJob, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while job.running():
        if time.time() > deadline:
            raise AssertionError("job did not finish")
        time.sleep(0.02)
    return job.snapshot()


def test_index_job() -> None:
    gate = threading.Event()
    seen: list[str] = []
    calls: list[str] = []

    def fake_update(raw_dir: Path, db_path: Path, processed_dir: Path, strict_extraction: bool = False) -> dict:
        calls.append("update")
        assert strict_extraction
        progress.report("reading", done=0, total=2, file="a.pdf")
        gate.wait(5)
        progress.report("embedding", done=64, total=128)
        return {"documents": 2, "indexed_documents": 2, "chunks": 10, "requires_review": 0, "skipped": 0}

    def failing_rebuild(*args: object, **kwargs: object) -> dict:
        raise RuntimeError("disk full sk-abcdefghijkl")

    original_update, original_rebuild = web_documents.kb_update, web_documents.kb_rebuild
    with tempfile.TemporaryDirectory() as temp:
        job = web_documents.IndexJob(Path(temp) / "raw", Path(temp) / "k.db", Path(temp) / "p", redact=lambda text: text.replace("sk-abcdefghijkl", "***"))
        try:
            web_documents.kb_update, web_documents.kb_rebuild = fake_update, failing_rebuild
            assert job.snapshot()["state"] == "idle"
            assert job.start("update")["started"]
            deadline = time.time() + 5
            while job.snapshot().get("phase") != "reading" and time.time() < deadline:
                time.sleep(0.01)
            state = job.snapshot()
            seen.append(state["phase"])
            assert state["state"] == "running" and state["reading"] == {"done": 0, "total": 2, "file": "a.pdf"}
            # More uploads while it runs: one more update afterwards; another kind is refused.
            assert job.start("update")["queued"]
            assert job.start("rebuild")["ok"] is False
            gate.set()
            final = wait_for(job)
            assert calls == ["update", "update"], calls
            assert final["state"] == "done" and final["summary"]["indexed_documents"] == 2 and final["embedding"]["done"] == 64
            assert final["queued"] is None and final["elapsed_seconds"] >= 0
            assert job.start("rebuild")["started"]
            failed = wait_for(job)
            assert failed["state"] == "failed" and "***" in failed["error"] and "sk-" not in failed["error"]
            try:
                job.start("format-disk")
            except ValueError:
                pass
            else:
                raise AssertionError("unknown job kind accepted")
        finally:
            web_documents.kb_update, web_documents.kb_rebuild = original_update, original_rebuild
            progress.set_reporter(None)
    assert seen == ["reading"]


class Client:
    def __init__(self, base: str) -> None:
        self.base = base
        self.cookie = ""

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None) -> tuple[int, dict, dict]:
        merged = {"Host": self.base.split("//", 1)[1].rstrip("/")}
        if self.cookie:
            merged["Cookie"] = self.cookie
        merged.update(headers or {})
        request = Request(self.base.rstrip("/") + path, data=body, method=method, headers=merged)
        try:
            with urlopen(request, timeout=30) as response:
                status, raw, response_headers = response.status, response.read(), dict(response.headers)
        except HTTPError as error:
            status, raw, response_headers = error.code, error.read(), dict(error.headers)
        cookie = response_headers.get("Set-Cookie", "")
        if cookie:
            self.cookie = cookie.split(";", 1)[0]
        text = raw.decode("utf-8")
        try:
            return status, json.loads(text), response_headers
        except json.JSONDecodeError:
            return status, {"html": text}, response_headers

    def post_json(self, path: str, payload: dict, headers: dict | None = None) -> tuple[int, dict, dict]:
        return self.request("POST", path, json.dumps(payload).encode("utf-8"), {"Content-Type": "application/json", **(headers or {})})


def test_http_server() -> None:
    import query_knowledge_base
    import web_agent

    saved = {name: getattr(web_agent, name) for name in ("RAW_DIR", "DB_PATH", "PROCESSED_DIR", "TRASH_DIR", "JOBS", "ACCESS", "LOCAL_ONLY", "QUESTIONS")}
    saved_query_processed = query_knowledge_base.PROCESSED_DIR
    starts: list[str] = []

    class RecordingJob(web_documents.IndexJob):
        def start(self, kind: str) -> dict:
            starts.append(kind)
            return {"ok": True, "started": True}

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        web_agent.RAW_DIR = root / "raw"
        web_agent.DB_PATH = root / "index" / "knowledge.db"
        web_agent.PROCESSED_DIR = root / "processed"
        web_agent.TRASH_DIR = root / "trash"
        query_knowledge_base.PROCESSED_DIR = root / "processed"  # no FAQ from the real knowledge base
        web_agent.JOBS = RecordingJob(web_agent.RAW_DIR, web_agent.DB_PATH, web_agent.PROCESSED_DIR)
        web_agent.ACCESS = web_access.Access("")
        web_agent.LOCAL_ONLY = True
        import web_stats

        web_agent.QUESTIONS = web_stats.QuestionLog(root / "logs" / "q.jsonl", root / "logs" / "f.jsonl")
        server = ThreadingHTTPServer(("127.0.0.1", 0), web_agent.AgentHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        client = Client(f"http://127.0.0.1:{server.server_address[1]}/")
        try:
            status, page, headers = client.request("GET", "/")
            assert status == 200 and '"admin": true' in page["html"] and "/*LKA_CONFIG*/" not in page["html"]
            assert headers.get("X-Frame-Options") == "DENY"
            status, data, _ = client.request("GET", "/api/status")
            assert status == 200 and data["documents"] == 0 and data["processing"] is False

            # Requests from other sites, other host names, or not as JSON are refused.
            status, data, _ = client.post_json("/api/jobs", {"kind": "update"}, {"Origin": "https://evil.example"})
            assert status == 403, (status, data)
            status, _, _ = client.request("GET", "/api/status", headers={"Host": "evil.example"})
            assert status == 403
            status, _, _ = client.request("POST", "/api/jobs", b'{"kind":"update"}', {"Content-Type": "text/plain"})
            assert status == 415
            assert starts == []

            # Upload, list, delete.
            body = "# 安装\n第一步：解压。".encode("utf-8")
            status, data, _ = client.request(
                "POST", "/api/upload", body,
                {"Content-Type": "application/octet-stream", "X-File-Name": "%E5%AE%89%E8%A3%85.md"},
            )
            assert status == 200 and data["file_path"] == "markdown/安装.md", data
            assert (web_agent.RAW_DIR / "markdown" / "安装.md").read_bytes() == body
            status, data, _ = client.request(
                "POST", "/api/upload", b"MZ", {"Content-Type": "application/octet-stream", "X-File-Name": "setup.exe"}
            )
            assert status == 400 and "不支持" in data["error"], data
            status, data, _ = client.request("GET", "/api/documents")
            assert status == 200 and [item["file_name"] for item in data["documents"]] == ["安装.md"]
            status, data, _ = client.post_json("/api/jobs", {"kind": "update"})
            assert status == 200 and starts == ["update"]
            status, data, _ = client.post_json("/api/documents/delete", {"file_path": "../../etc/passwd"})
            assert status == 400
            status, data, _ = client.post_json("/api/documents/delete", {"file_path": "markdown/安装.md"})
            assert status == 200 and starts == ["update", "update"]
            assert not (web_agent.RAW_DIR / "markdown" / "安装.md").exists()

            # A question on the chat page is recorded; feedback and the statistics follow it.
            status, page, _ = client.request(
                "POST", "/api/query-stream", json.dumps({"question": "保修多久", "allow_api": False}).encode("utf-8"),
                {"Content-Type": "application/json"},
            )
            events = page["html"]
            assert status == 200 and "event: done" in events, events[:300]
            done_line = events.split("event: done", 1)[1].strip().splitlines()[0]
            record_id = json.loads(done_line.removeprefix("data: "))["record_id"]
            status, _, _ = client.post_json("/api/feedback", {"id": record_id, "helpful": False})
            assert status == 200
            status, data, _ = client.request("GET", "/api/stats?days=7")
            assert status == 200 and data["totals"]["questions"] == 1 and data["totals"]["refused"] == 1, data["totals"]
            assert data["unanswered"][0]["question"] == "保修多久" and data["unhelpful"][0]["question"] == "保修多久"

            # The answer API failing (no balance): the answer says so, the status bar and statistics count it.
            original_query, original_enabled = web_agent.kb_query, web_agent.api_enabled
            web_agent.kb_query = lambda *args, **kwargs: {
                "answer": "本地答案", "source_type": "knowledge_base", "sources": [],
                "agent_error": 'HTTP 402: {"error":{"message":"Insufficient Balance"}}',
            }
            web_agent.api_enabled = lambda config: True
            try:
                status, page, _ = client.request(
                    "POST", "/api/query-stream", json.dumps({"question": "保修多久"}).encode("utf-8"),
                    {"Content-Type": "application/json"},
                )
                meta_line = page["html"].split("event: meta", 1)[1].strip().splitlines()[0]
                assert json.loads(meta_line.removeprefix("data: "))["api_failure"]["kind"] == "balance"
                status, data, _ = client.request("GET", "/api/status")
                assert data["api_failure"]["kind"] == "balance"
                status, data, _ = client.request("GET", "/api/stats?days=7")
                assert data["totals"]["api_failures"] == 1 and data["totals"]["api_failure_kinds"] == {"balance": 1}
            finally:
                web_agent.kb_query, web_agent.api_enabled = original_query, original_enabled
                web_agent.note_api_outcome({"api_used": True})
            # Asking with nothing indexed: a polite answer that points to uploading, not an error.
            status, data, _ = client.request("GET", "/api/query?question=%E6%80%8E%E4%B9%88%E5%AE%89%E8%A3%85&allow_api=false")
            assert status == 200 and "上传" in data["answer"] and data["user_guidance"]["kind"] == "no_index_no_faq", data

            # The changed API address does not get the saved key.
            status, data, _ = client.post_json("/api/api-models", {"provider": "openai-compatible", "base_url": "https://collector.example/v1"})
            assert status == 400 and "API Key" in data["error"], data

            # With a password: nothing but the login page until signed in.
            web_agent.ACCESS = web_access.Access("pw-123")
            fresh = Client(client.base)
            status, page, _ = fresh.request("GET", "/")
            assert status == 200 and "请输入访问密码" in page["html"]
            status, _, _ = fresh.request("GET", "/api/status")
            assert status == 401
            original_sleep = web_access.time.sleep
            web_access.time.sleep = lambda seconds: None
            try:
                status, _, _ = fresh.post_json("/api/login", {"password": "nope"})
            finally:
                web_access.time.sleep = original_sleep
            assert status == 401
            status, _, headers = fresh.post_json("/api/login", {"password": "pw-123"})
            assert status == 200 and "HttpOnly" in headers.get("Set-Cookie", "") and "SameSite=Lax" in headers.get("Set-Cookie", "")
            status, data, _ = fresh.request("GET", "/api/status")
            assert status == 200
            # Opened to the network, a colleague's computer: signed in, it may ask but not manage.
            web_agent.LOCAL_ONLY = False
            original_is_local = web_agent.AgentHandler.client_is_local
            web_agent.AgentHandler.client_is_local = lambda handler: False
            try:
                status, page, _ = fresh.request("GET", "/", headers={"Host": "192.168.1.5:8765"})
                assert status == 200 and '"admin": false' in page["html"] and '"loginRequired": true' in page["html"]
                status, data, _ = fresh.request("GET", "/api/status", headers={"Host": "192.168.1.5:8765"})
                assert status == 200
                for method, path in (("GET", "/api/documents"), ("GET", "/api/health"), ("GET", "/api/api-config"), ("GET", "/api/stats")):
                    status, data, _ = fresh.request(method, path, headers={"Host": "192.168.1.5:8765"})
                    assert status == 403 and "本服务的电脑" in data["error"], (path, status, data)
                for path, payload in (("/api/jobs", {"kind": "rebuild"}), ("/api/documents/delete", {"file_path": "x.md"}), ("/api/api-save", {}), ("/api/api-mode", {"enabled": False})):
                    status, data, _ = fresh.post_json(path, payload, {"Host": "192.168.1.5:8765"})
                    assert status == 403, (path, status, data)
                status, data, _ = fresh.request(
                    "POST", "/api/upload", b"# x", {"Content-Type": "application/octet-stream", "X-File-Name": "x.md", "Host": "192.168.1.5:8765"}
                )
                assert status == 403 and not (web_agent.RAW_DIR / "markdown" / "x.md").exists()
                status, data, _ = fresh.request("GET", "/api/query?question=%E6%B5%8B%E8%AF%95&allow_api=false", headers={"Host": "192.168.1.5:8765"})
                assert status == 200 and "answer" in data
                assert starts == ["update", "update"]
            finally:
                web_agent.AgentHandler.client_is_local = original_is_local
                web_agent.LOCAL_ONLY = True
            status, _, _ = fresh.post_json("/api/logout", {})
            assert status == 200
            fresh.cookie = ""
            status, _, _ = fresh.request("GET", "/api/status")
            assert status == 401
        finally:
            server.shutdown()
            server.server_close()
            for name, value in saved.items():
                setattr(web_agent, name, value)
            query_knowledge_base.PROCESSED_DIR = saved_query_processed


def main() -> None:
    test_followup_rules()
    test_followup_api_failure_falls_back_to_rules()
    test_access_rules()
    test_upload_names_and_storage()
    test_review_reasons()
    test_question_log_and_stats()
    test_api_failures()
    test_set_env_value()
    test_index_job()
    test_http_server()
    print("web ui: ok")


if __name__ == "__main__":
    main()
