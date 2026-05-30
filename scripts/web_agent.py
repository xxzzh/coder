#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small local Web UI for the Local Knowledge Base Agent."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
RAW_DIR = ROOT / "knowledge_base" / "raw"
DB_PATH = ROOT / "knowledge_base" / "index" / "knowledge.db"
OCR_EVAL_SCRIPT = SCRIPTS_DIR / "evaluate_ocr_test_materials.py"

os.chdir(ROOT)
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from healthcheck_agent import (  # noqa: E402
    SUPPORTED,
    db_stats,
    extraction_report_stats,
    load_env,
    optional_extractors,
    sqlite_has_fts5,
)
from manage_knowledge_base import rebuild as kb_rebuild  # noqa: E402
from manage_knowledge_base import sources as kb_sources  # noqa: E402
from manage_knowledge_base import update as kb_update  # noqa: E402
from query_knowledge_base import query as kb_query  # noqa: E402


def raw_file_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    if not RAW_DIR.exists():
        return counts
    for path in RAW_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            counts[path.suffix.lower()] = counts.get(path.suffix.lower(), 0) + 1
    return dict(sorted(counts.items()))


def health_payload() -> dict[str, Any]:
    env = load_env(ROOT / ".env")
    stats = db_stats()
    extraction = extraction_report_stats()
    extractors = optional_extractors()
    raw_counts = raw_file_counts()
    chunks = int(stats.get("chunks", 0) or 0) if isinstance(stats, dict) else 0
    ok = (
        sqlite_has_fts5()
        and RAW_DIR.exists()
        and stats.get("exists") is True
        and not stats.get("error")
        and int(stats.get("documents", 0) or 0) > 0
        and chunks > 0
        and int(extraction.get("requires_review", 0) or 0) == 0
    )
    return {
        "ok": ok,
        "python_version": sys.version.split()[0],
        "sqlite_fts5": sqlite_has_fts5(),
        "raw_dir": str(RAW_DIR),
        "raw_supported_files": sum(raw_counts.values()),
        "raw_counts": raw_counts,
        "index": stats,
        "optional_extractors": extractors,
        "extraction_report": extraction,
        "api_configured": env.get("LKA_USE_API", "false").lower() == "true" and bool(env.get("LKA_API_KEY")),
        "api_provider": env.get("LKA_API_PROVIDER", "none"),
        "scale_warning": chunks >= 50000,
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
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


def page_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local Knowledge Base Agent</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1d2329;
      --muted: #66717f;
      --line: #d8dee6;
      --accent: #1769aa;
      --accent-strong: #0f4f82;
      --ok: #177245;
      --warn: #a35d00;
      --bad: #b42318;
      --code: #eef2f6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
    }
    header {
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    .wrap {
      max-width: 1180px;
      margin: 0 auto;
      padding: 18px 20px;
    }
    header .wrap {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 16px;
      letter-spacing: 0;
    }
    button, input, textarea, select {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      min-height: 36px;
      padding: 7px 12px;
      cursor: pointer;
    }
    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }
    button:hover { border-color: var(--accent-strong); }
    button.primary:hover { background: var(--accent-strong); }
    button:disabled { opacity: .55; cursor: wait; }
    main.wrap {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(320px, .8fr);
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }
    .stack { display: grid; gap: 14px; }
    .cards {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      min-height: 78px;
      background: #fbfcfd;
    }
    .metric .label {
      color: var(--muted);
      font-size: 12px;
    }
    .metric .value {
      margin-top: 7px;
      font-size: 20px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 5px 9px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fff;
      font-weight: 600;
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--muted);
      display: inline-block;
    }
    .status.ok .dot { background: var(--ok); }
    .status.warn .dot { background: var(--warn); }
    .status.bad .dot { background: var(--bad); }
    textarea {
      width: 100%;
      min-height: 96px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fff;
      color: var(--text);
    }
    .row {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .row.between { justify-content: space-between; }
    label.check {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      user-select: none;
    }
    .answer {
      white-space: pre-wrap;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfd;
      min-height: 120px;
      padding: 12px;
    }
    .source-list {
      display: grid;
      gap: 10px;
      margin-top: 10px;
    }
    .source-item {
      border-left: 3px solid var(--accent);
      padding: 8px 10px;
      background: #f8fafc;
      overflow-wrap: anywhere;
    }
    .source-title {
      font-weight: 700;
      margin-bottom: 4px;
    }
    .muted { color: var(--muted); }
    .mono {
      font-family: Consolas, "Cascadia Mono", monospace;
      background: var(--code);
      padding: 2px 5px;
      border-radius: 4px;
      overflow-wrap: anywhere;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 7px 6px;
      text-align: left;
      vertical-align: top;
    }
    th { color: var(--muted); font-weight: 600; }
    .log {
      max-height: 260px;
      overflow: auto;
      background: #101418;
      color: #e8edf2;
      border-radius: 6px;
      padding: 10px;
      font-family: Consolas, "Cascadia Mono", monospace;
      font-size: 12px;
      white-space: pre-wrap;
    }
    @media (max-width: 860px) {
      header .wrap, main.wrap { display: block; }
      header .wrap > * + *, main.wrap > * + * { margin-top: 14px; }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 520px) {
      .cards { grid-template-columns: 1fr; }
      .row { align-items: stretch; }
      .row button { flex: 1 1 auto; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>Local Knowledge Base Agent</h1>
      <div id="readyBadge" class="status warn"><span class="dot"></span><span>检查中</span></div>
    </div>
  </header>
  <main class="wrap">
    <section class="stack">
      <div class="panel">
        <div class="row between">
          <h2>提问</h2>
          <div class="row">
            <label class="check"><input id="allowApi" type="checkbox" checked> API 总结</label>
            <label class="check"><input id="allowWeb" type="checkbox"> 联网兜底</label>
          </div>
        </div>
        <textarea id="question" placeholder="输入问题"></textarea>
        <div class="row" style="margin-top:10px">
          <button id="askBtn" class="primary" onclick="ask()">查询</button>
          <button onclick="clearAnswer()">清空</button>
          <span id="askMeta" class="muted"></span>
        </div>
      </div>
      <div class="panel">
        <h2>答案</h2>
        <div id="answer" class="answer muted">等待查询</div>
        <div id="sources" class="source-list"></div>
      </div>
      <div class="panel">
        <div class="row between">
          <h2>索引来源</h2>
          <button onclick="loadSources()">刷新来源</button>
        </div>
        <div id="sourceTable" class="muted">等待加载</div>
      </div>
    </section>
    <aside class="stack">
      <div class="panel">
        <div class="row between">
          <h2>状态</h2>
          <button onclick="loadHealth()">刷新</button>
        </div>
        <div class="cards">
          <div class="metric"><div class="label">资料</div><div id="rawCount" class="value">-</div></div>
          <div class="metric"><div class="label">文档</div><div id="docCount" class="value">-</div></div>
          <div class="metric"><div class="label">分块</div><div id="chunkCount" class="value">-</div></div>
          <div class="metric"><div class="label">待复核</div><div id="reviewCount" class="value">-</div></div>
        </div>
        <div id="healthDetail" style="margin-top:12px"></div>
      </div>
      <div class="panel">
        <h2>操作</h2>
        <div class="stack">
          <button onclick="runAction('/api/update', '增量更新')">增量更新</button>
          <button onclick="runAction('/api/rebuild-strict', '严格重建')">严格重建</button>
          <button onclick="runAction('/api/ocr-eval', 'OCR 评测')">OCR 评测</button>
        </div>
      </div>
      <div class="panel">
        <h2>运行日志</h2>
        <div id="log" class="log">Web UI 已启动。</div>
      </div>
    </aside>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const setBusy = (busy) => {
      document.querySelectorAll('button').forEach((btn) => btn.disabled = busy);
    };
    const log = (message, data) => {
      const line = `[${new Date().toLocaleTimeString()}] ${message}`;
      $('log').textContent = data ? `${line}\\n${JSON.stringify(data, null, 2)}\\n\\n${$('log').textContent}` : `${line}\\n${$('log').textContent}`;
    };
    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: {'Content-Type': 'application/json'},
        ...options
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || response.statusText);
      return data;
    }
    function setBadge(ok, text) {
      const badge = $('readyBadge');
      badge.className = `status ${ok ? 'ok' : 'bad'}`;
      badge.querySelector('span:last-child').textContent = text;
    }
    async function loadHealth() {
      try {
        const data = await api('/api/health');
        setBadge(data.ok, data.ok ? '可使用' : '需处理');
        $('rawCount').textContent = data.raw_supported_files ?? 0;
        $('docCount').textContent = data.index?.documents ?? 0;
        $('chunkCount').textContent = data.index?.chunks ?? 0;
        $('reviewCount').textContent = data.extraction_report?.requires_review ?? 0;
        const lines = [
          `索引时间：${data.index?.indexed_at || '未建立'}`,
          `分块：${data.index?.chunk_strategy || '-'}`,
          `OCR：${data.optional_extractors?.ocr_ready ? '可用' : '不可用'}`,
          `API：${data.api_configured ? data.api_provider : '未配置'}`
        ];
        if (data.scale_warning) lines.push('规模提示：chunk 已超过 50000，建议评估独立向量库。');
        if ((data.extraction_report?.requires_review ?? 0) > 0) lines.push('存在待复核抽取结果，请先处理后再交付使用。');
        $('healthDetail').innerHTML = lines.map((item) => `<div class="muted">${item}</div>`).join('');
      } catch (error) {
        setBadge(false, '检查失败');
        log('状态检查失败', {error: error.message});
      }
    }
    async function ask() {
      const question = $('question').value.trim();
      if (!question) return;
      setBusy(true);
      $('answer').textContent = '查询中...';
      $('sources').textContent = '';
      $('askMeta').textContent = '';
      try {
        const data = await api('/api/query', {
          method: 'POST',
          body: JSON.stringify({
            question,
            use_web: $('allowWeb').checked,
            allow_api: $('allowApi').checked,
            limit: 5
          })
        });
        $('answer').className = 'answer';
        $('answer').textContent = data.answer || '没有返回答案';
        $('askMeta').textContent = `${data.source_type || '-'}${data.cache_hit ? ' · cache' : ''}${data.api_used ? ' · api' : ''}`;
        renderSources(data.sources || []);
        log('查询完成', {source_type: data.source_type, sources: (data.sources || []).length});
      } catch (error) {
        $('answer').className = 'answer muted';
        $('answer').textContent = `查询失败：${error.message}`;
        log('查询失败', {error: error.message});
      } finally {
        setBusy(false);
        loadHealth();
      }
    }
    function renderSources(items) {
      if (!items.length) {
        $('sources').innerHTML = '<div class="muted">没有引用来源</div>';
        return;
      }
      $('sources').innerHTML = '';
      items.forEach((item) => {
        const node = document.createElement('div');
        node.className = 'source-item';
        const title = document.createElement('div');
        title.className = 'source-title';
        title.textContent = `${item.source_id || ''}. ${item.file || item.title || 'source'}`;
        const path = document.createElement('div');
        path.className = 'muted';
        path.textContent = item.file_path || item.url || '';
        const quote = document.createElement('div');
        quote.textContent = item.excerpt || item.citation || item.snippet || '';
        node.append(title, path, quote);
        $('sources').appendChild(node);
      });
    }
    async function loadSources() {
      try {
        const data = await api('/api/sources');
        const rows = data.sources || [];
        if (!rows.length) {
          $('sourceTable').textContent = '没有索引来源';
          return;
        }
        const html = [`<table><thead><tr><th>文件</th><th>类型</th><th>分块</th><th>状态</th></tr></thead><tbody>`];
        rows.slice(0, 80).forEach((row) => {
          html.push(`<tr><td>${escapeHtml(row.file_path || row.file_name || '')}</td><td>${escapeHtml(row.file_type || '')}</td><td>${row.chunk_count || 0}</td><td>${row.duplicate_of ? '重复' : '已索引'}</td></tr>`);
        });
        html.push('</tbody></table>');
        if (rows.length > 80) html.push(`<div class="muted">仅显示前 80 条，共 ${rows.length} 条。</div>`);
        $('sourceTable').innerHTML = html.join('');
      } catch (error) {
        $('sourceTable').textContent = `加载失败：${error.message}`;
      }
    }
    async function runAction(path, name) {
      setBusy(true);
      log(`${name}开始`);
      try {
        const data = await api(path, {method: 'POST', body: '{}'});
        log(`${name}完成`, data.summary || data);
        await loadHealth();
        if (path !== '/api/ocr-eval') await loadSources();
      } catch (error) {
        log(`${name}失败`, {error: error.message});
      } finally {
        setBusy(false);
      }
    }
    function clearAnswer() {
      $('question').value = '';
      $('answer').className = 'answer muted';
      $('answer').textContent = '等待查询';
      $('sources').textContent = '';
      $('askMeta').textContent = '';
    }
    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
    }
    $('question').addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) ask();
    });
    loadHealth();
    loadSources();
  </script>
</body>
</html>
"""


class AgentHandler(BaseHTTPRequestHandler):
    server_version = "LocalKnowledgeAgent/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.send_html(page_html())
        elif parsed.path == "/api/health":
            self.send_json(health_payload())
        elif parsed.path == "/api/sources":
            self.send_json(kb_sources(DB_PATH))
        elif parsed.path == "/api/query":
            params = parse_qs(parsed.query)
            question = params.get("question", [""])[0].strip()
            if not question:
                self.send_error_json("Missing question", HTTPStatus.BAD_REQUEST)
                return
            self.send_json(
                kb_query(
                    DB_PATH,
                    question,
                    limit=int(params.get("limit", ["5"])[0] or 5),
                    use_web=params.get("use_web", ["false"])[0].lower() == "true",
                    allow_api=params.get("allow_api", ["true"])[0].lower() != "false",
                )
            )
        else:
            self.send_error_json("Not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/query":
                payload = self.read_json()
                question = str(payload.get("question", "")).strip()
                if not question:
                    self.send_error_json("Missing question", HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(
                    kb_query(
                        DB_PATH,
                        question,
                        limit=int(payload.get("limit", 5) or 5),
                        use_web=bool(payload.get("use_web", False)),
                        allow_api=bool(payload.get("allow_api", True)),
                    )
                )
            elif parsed.path == "/api/update":
                result = kb_update(RAW_DIR, DB_PATH, ROOT / "knowledge_base" / "processed", strict_extraction=True)
                self.send_json({"ok": True, "summary": compact_ingest_result(result), "result": result})
            elif parsed.path == "/api/rebuild-strict":
                result = kb_rebuild(RAW_DIR, DB_PATH, ROOT / "knowledge_base" / "processed", strict_extraction=True)
                self.send_json({"ok": True, "summary": compact_ingest_result(result), "result": result})
            elif parsed.path == "/api/ocr-eval":
                self.send_json(run_ocr_eval())
            else:
                self.send_error_json("Not found", HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self.send_error_json(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        return json.loads(raw) if raw.strip() else {}

    def send_html(self, content: str) -> None:
        body = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status: HTTPStatus) -> None:
        self.send_json({"ok": False, "error": message}, status)


def compact_ingest_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "documents",
        "indexed_documents",
        "chunks",
        "duplicates",
        "skipped",
        "extraction_warnings",
        "requires_review",
        "indexed_at",
    )
    return {key: result.get(key) for key in keys if key in result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Web UI for Local Knowledge Base Agent.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Default keeps the UI local-only.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the Web UI in the default browser.")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), AgentHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Local Knowledge Base Agent Web UI: {url}")
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
