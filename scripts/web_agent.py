#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small local Web UI for the Local Knowledge Base Agent."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from api_providers import (  # noqa: E402
    default_base_url,
    discover_models,
    infer_provider,
    provider_options,
    read_env as read_api_env,
    validate_chat_model,
    write_env as write_api_env,
)

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
    api_backend_allowed,
    db_stats,
    extraction_report_stats,
    load_env,
    offline_translation_status,
    optional_extractors,
    sqlite_has_fts5,
    unsupported_raw_files,
)
from manage_knowledge_base import rebuild as kb_rebuild  # noqa: E402
from manage_knowledge_base import sources as kb_sources  # noqa: E402
from manage_knowledge_base import update as kb_update  # noqa: E402
from ingest_knowledge_base import approve_review_file, is_ignored_raw_file, pending_review_reports  # noqa: E402
from query_knowledge_base import query as kb_query  # noqa: E402


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
    raw_counts = raw_file_counts()
    unsupported_files = unsupported_raw_files()
    chunks = int(stats.get("chunks", 0) or 0) if isinstance(stats, dict) else 0
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
        "extraction_report": extraction,
        "api_configured": env.get("LKA_USE_API", "false").lower() == "true" and bool(env.get("LKA_API_KEY")),
        "api_backend_usable": api_backend_allowed(env),
        "api_provider": env.get("LKA_API_PROVIDER", "none"),
        "scale_warning": chunks >= 50000,
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def api_config_payload() -> dict[str, Any]:
    env = read_api_env()
    provider = infer_provider(
        env.get("LKA_API_BASE_URL", ""),
        env.get("LKA_API_KEY", ""),
        env.get("LKA_API_PROVIDER", ""),
    )
    configured = env.get("LKA_USE_API", "false").lower() == "true" and bool(env.get("LKA_API_KEY"))
    return {
        "configured": configured,
        "usable": configured and api_backend_allowed(env),
        "provider": provider,
        "base_url": env.get("LKA_API_BASE_URL", ""),
        "model": env.get("LKA_API_MODEL", ""),
        "providers": provider_options(),
    }


def reviews_payload() -> dict[str, Any]:
    reviews = pending_review_reports(RAW_DIR, ROOT / "knowledge_base" / "processed")
    return {
        "pending": len(reviews),
        "reviews": [
            {
                "file_path": item.get("file_path"),
                "file_type": item.get("file_type"),
                "warnings": item.get("warnings", []),
                "text_chars": item.get("text_chars", 0),
            }
            for item in reviews
        ],
    }


def api_key_from_payload(payload: dict[str, Any]) -> str:
    api_key = str(payload.get("api_key", "")).strip()
    if api_key:
        return api_key
    return read_api_env().get("LKA_API_KEY", "")


def save_api_config(payload: dict[str, Any]) -> dict[str, Any]:
    provider = str(payload.get("provider", "openai-compatible")).strip()
    base_url = str(payload.get("base_url", "")).strip() or default_base_url(provider)
    api_key = api_key_from_payload(payload)
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
    input, select {
      width: 100%;
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 9px;
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
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfd;
      min-height: 120px;
      padding: 14px;
      color: var(--text);
      overflow-wrap: anywhere;
    }
    .answer p {
      margin: 0;
      line-height: 1.75;
    }
    .answer p + p,
    .answer p + .answer-list,
    .answer .answer-list + p {
      margin-top: 10px;
    }
    .answer-lead {
      color: #344054;
    }
    .answer-heading {
      margin-top: 10px;
      font-weight: 700;
      color: var(--text);
    }
    .answer-list {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 7px 16px;
      margin: 10px 0 0;
      padding: 0;
      list-style: none;
    }
    .answer-list li {
      position: relative;
      padding-left: 15px;
      line-height: 1.55;
      break-inside: avoid;
    }
    .answer-list li::before {
      content: "";
      position: absolute;
      left: 0;
      top: .68em;
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--accent);
    }
    .citation-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 19px;
      height: 19px;
      margin: 0 2px;
      padding: 0 5px;
      border: 1px solid #b8d3e8;
      border-radius: 999px;
      background: #eaf4fb;
      color: var(--accent-strong);
      font-size: 12px;
      font-weight: 700;
      vertical-align: text-bottom;
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
      .answer-list { grid-template-columns: 1fr; }
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
          <button onclick="openRawFolder()">打开资料目录</button>
          <button onclick="runAction('/api/update', '增量更新')">增量更新</button>
          <button onclick="runAction('/api/rebuild-strict', '严格重建')">严格重建</button>
          <button onclick="runAction('/api/ocr-eval', 'OCR 评测')">OCR 评测</button>
        </div>
      </div>
      <div class="panel">
        <div class="row between">
          <h2>待复核资料</h2>
          <button onclick="loadReviews()">刷新</button>
        </div>
        <div id="reviewList" class="stack muted">等待加载</div>
      </div>
      <div class="panel">
        <h2>API 精炼</h2>
        <div class="stack">
          <select id="apiProvider" onchange="providerChanged()"></select>
          <input id="apiBaseUrl" placeholder="API base URL">
          <input id="apiKey" type="password" placeholder="API key">
          <button onclick="discoverApiModels()">读取可用模型</button>
          <select id="apiModel"><option value="">请先读取模型</option></select>
          <button class="primary" onclick="saveApiConfig()">保存并验证</button>
          <div id="apiConfigMeta" class="muted"></div>
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
    let apiProviders = [];
    async function loadApiConfig() {
      try {
        const data = await api('/api/api-config');
        apiProviders = data.providers || [];
        $('apiProvider').innerHTML = apiProviders.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)}</option>`).join('');
        $('apiProvider').value = data.provider || 'openai-compatible';
        $('apiBaseUrl').value = data.base_url || providerBaseUrl();
        $('apiConfigMeta').textContent = data.usable ? `当前模型：${data.model || '-'}` : (data.configured ? '已有配置不可用于应用后端，请重新配置有效 API key' : '未配置可用 API');
        if (data.model) $('apiModel').innerHTML = `<option value="${escapeHtml(data.model)}">${escapeHtml(data.model)}</option>`;
      } catch (error) {
        $('apiConfigMeta').textContent = `加载失败：${error.message}`;
      }
    }
    function providerBaseUrl() {
      return (apiProviders.find((item) => item.id === $('apiProvider').value) || {}).base_url || '';
    }
    function providerChanged() {
      $('apiBaseUrl').value = providerBaseUrl();
      $('apiModel').innerHTML = '<option value="">请先读取模型</option>';
    }
    async function discoverApiModels() {
      setBusy(true);
      $('apiConfigMeta').textContent = '正在读取可用模型...';
      try {
        const data = await api('/api/api-models', {method: 'POST', body: JSON.stringify({
          provider: $('apiProvider').value,
          base_url: $('apiBaseUrl').value,
          api_key: $('apiKey').value
        })});
        $('apiModel').innerHTML = (data.models || []).map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`).join('');
        $('apiConfigMeta').textContent = data.warning || `已读取 ${(data.models || []).length} 个模型`;
      } catch (error) {
        $('apiConfigMeta').textContent = `读取失败：${error.message}`;
      } finally {
        setBusy(false);
      }
    }
    async function saveApiConfig() {
      setBusy(true);
      $('apiConfigMeta').textContent = '正在验证模型调用...';
      try {
        const data = await api('/api/api-save', {method: 'POST', body: JSON.stringify({
          provider: $('apiProvider').value,
          base_url: $('apiBaseUrl').value,
          api_key: $('apiKey').value,
          model: $('apiModel').value
        })});
        $('apiKey').value = '';
        $('apiConfigMeta').textContent = `已启用：${data.provider} · ${data.model}`;
        log('API 配置已保存', {provider: data.provider, model: data.model});
        loadHealth();
      } catch (error) {
        $('apiConfigMeta').textContent = `保存失败：${error.message}`;
      } finally {
        setBusy(false);
      }
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
          `离线英译中：${data.offline_translation?.en_to_zh_ready ? '可用' : '不可用'}`,
          `API：${data.api_configured ? (data.api_backend_usable ? data.api_provider : '当前配置不可用于应用后端') : '未配置'}`
        ];
        if (data.scale_warning) lines.push('规模提示：chunk 已超过 50000，建议评估独立向量库。');
        if ((data.extraction_report?.requires_review ?? 0) > 0) lines.push('存在待复核抽取结果，请先处理后再交付使用。');
        if ((data.raw_unsupported_files ?? []).length > 0) lines.push(`未进入索引的不支持文件：${data.raw_unsupported_files.join('；')}`);
        $('healthDetail').innerHTML = lines.map((item) => `<div class="muted">${item}</div>`).join('');
      } catch (error) {
        setBadge(false, '检查失败');
        log('状态检查失败', {error: error.message});
      }
    }
    async function loadReviews() {
      try {
        const data = await api('/api/reviews');
        const items = data.reviews || [];
        const list = $('reviewList');
        if (!items.length) {
          list.className = 'stack muted';
          list.textContent = '没有待复核资料';
          return;
        }
        list.className = 'stack';
        list.innerHTML = '';
        items.forEach((item) => {
          const node = document.createElement('div');
          node.className = 'source-item';
          const title = document.createElement('div');
          title.className = 'source-title';
          title.textContent = item.file_path || '';
          const warning = document.createElement('div');
          warning.className = 'muted';
          warning.textContent = (item.warnings || []).join('；');
          const actions = document.createElement('div');
          actions.className = 'row';
          actions.style.marginTop = '8px';
          const approve = document.createElement('button');
          approve.textContent = '确认可用';
          approve.onclick = () => approveReview(item.file_path);
          actions.appendChild(approve);
          node.append(title, warning, actions);
          list.appendChild(node);
        });
      } catch (error) {
        $('reviewList').textContent = `加载失败：${error.message}`;
      }
    }
    async function approveReview(filePath) {
      if (!window.confirm(`确认已检查并允许索引此文件？\\n${filePath}`)) return;
      setBusy(true);
      log('复核确认开始', {file_path: filePath});
      try {
        const data = await api('/api/reviews/approve', {method: 'POST', body: JSON.stringify({file_path: filePath})});
        log('复核确认完成', data.summary || data);
        await loadHealth();
        await loadReviews();
        await loadSources();
      } catch (error) {
        log('复核确认失败', {error: error.message});
      } finally {
        setBusy(false);
      }
    }
    async function openRawFolder() {
      try {
        await api('/api/open-raw-folder', {method: 'POST', body: '{}'});
      } catch (error) {
        log('打开资料目录失败', {error: error.message});
      }
    }
    async function ask() {
      const question = $('question').value.trim();
      if (!question) return;
      setBusy(true);
      $('answer').className = 'answer muted';
      $('answer').textContent = '正在准备查询...';
      $('sources').textContent = '';
      $('askMeta').textContent = '';
      try {
        const response = await fetch('/api/query-stream', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            question,
            use_web: $('allowWeb').checked,
            allow_api: $('allowApi').checked,
            limit: 5
          })
        });
        if (!response.ok || !response.body) {
          const data = await response.json();
          throw new Error(data.error || response.statusText);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let streamedAnswer = '';
        while (true) {
          const {value, done} = await reader.read();
          buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
          const blocks = buffer.split('\\n\\n');
          buffer = blocks.pop() || '';
          blocks.forEach((block) => {
            if (!block.trim()) return;
            const lines = block.split('\\n');
            const eventName = (lines.find((line) => line.startsWith('event: ')) || 'event: message').slice(7);
            const dataLine = lines.find((line) => line.startsWith('data: '));
            if (!dataLine) return;
            const data = JSON.parse(dataLine.slice(6));
            if (eventName === 'status') {
              $('answer').className = 'answer muted';
              $('answer').textContent = data.message;
              $('askMeta').textContent = data.elapsed_seconds ? `已等待 ${data.elapsed_seconds} 秒` : '';
            } else if (eventName === 'meta') {
              $('askMeta').textContent = `${data.source_type || '-'}${data.web_search_provider ? ` · ${data.web_search_provider}` : ''}${data.cache_hit ? ' · cache' : ''}${data.api_used ? ' · api' : ''}`;
            } else if (eventName === 'sources') {
              renderSources(data.sources || []);
            } else if (eventName === 'answer-start') {
              streamedAnswer = '';
              $('answer').className = 'answer';
              $('answer').textContent = '';
            } else if (eventName === 'answer-delta') {
              streamedAnswer += data.text || '';
              renderAnswer(streamedAnswer);
            } else if (eventName === 'error') {
              throw new Error(data.error || '查询失败');
            } else if (eventName === 'done') {
              log('查询完成', {source_type: data.source_type, sources: data.sources_count});
            }
          });
          if (done) break;
        }
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
    function appendAnswerText(parent, text) {
      const parts = String(text || '').split(/(\\[\\d+\\])/g);
      parts.forEach((part) => {
        const citation = part.match(/^\\[(\\d+)\\]$/);
        if (citation) {
          const badge = document.createElement('span');
          badge.className = 'citation-badge';
          badge.textContent = citation[1];
          parent.appendChild(badge);
        } else if (part) {
          parent.appendChild(document.createTextNode(part));
        }
      });
    }
    function addAnswerParagraph(container, text, className = '') {
      const value = String(text || '').trim();
      if (!value) return;
      const paragraph = document.createElement('p');
      if (className) paragraph.className = className;
      appendAnswerText(paragraph, value);
      container.appendChild(paragraph);
    }
    function renderAnswer(text) {
      const container = $('answer');
      container.innerHTML = '';
      const normalized = String(text || '').replace(/\\r\\n/g, '\\n').trim();
      if (!normalized) return;
      const listMarker = '包含以下测试项：';
      const markerIndex = normalized.indexOf(listMarker);
      if (markerIndex >= 0) {
        const lead = normalized.slice(0, markerIndex).trim();
        const remainder = normalized.slice(markerIndex + listMarker.length).trim();
        addAnswerParagraph(container, lead, 'answer-lead');
        addAnswerParagraph(container, '测试项', 'answer-heading');
        const list = document.createElement('ul');
        list.className = 'answer-list';
        remainder.split('；').map((item) => item.trim()).filter(Boolean).forEach((item) => {
          const row = document.createElement('li');
          appendAnswerText(row, item);
          list.appendChild(row);
        });
        container.appendChild(list);
        return;
      }
      normalized.split(/\\n{2,}/).forEach((block) => {
        const lines = block.split('\\n').map((line) => line.trim()).filter(Boolean);
        const bulletLines = lines.filter((line) => /^[-*•]\\s+/.test(line));
        if (bulletLines.length === lines.length && lines.length > 0) {
          const list = document.createElement('ul');
          list.className = 'answer-list';
          lines.forEach((line) => {
            const row = document.createElement('li');
            appendAnswerText(row, line.replace(/^[-*•]\\s+/, ''));
            list.appendChild(row);
          });
          container.appendChild(list);
        } else {
          addAnswerParagraph(container, lines.join(' '));
        }
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
        await loadReviews();
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
    loadReviews();
    loadSources();
    loadApiConfig();
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
        elif parsed.path == "/api/reviews":
            self.send_json(reviews_payload())
        elif parsed.path == "/api/api-config":
            self.send_json(api_config_payload())
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
            if parsed.path == "/api/query-stream":
                payload = self.read_json()
                self.stream_query(payload)
            elif parsed.path == "/api/query":
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
            elif parsed.path == "/api/reviews/approve":
                payload = self.read_json()
                approval = approve_review_file(
                    RAW_DIR,
                    ROOT / "knowledge_base" / "processed",
                    str(payload.get("file_path", "")),
                )
                result = kb_update(RAW_DIR, DB_PATH, ROOT / "knowledge_base" / "processed", strict_extraction=True)
                self.send_json({"ok": True, "approval": approval, "summary": compact_ingest_result(result), "result": result})
            elif parsed.path == "/api/open-raw-folder":
                os.startfile(RAW_DIR)  # type: ignore[attr-defined]
                self.send_json({"ok": True})
            elif parsed.path == "/api/api-models":
                payload = self.read_json()
                provider = str(payload.get("provider", "openai-compatible")).strip()
                base_url = str(payload.get("base_url", "")).strip() or default_base_url(provider)
                api_key = api_key_from_payload(payload)
                if not api_key:
                    raise ValueError("请输入 API key。")
                result = discover_models(provider, base_url, api_key)
                if not result.get("ok"):
                    raise ValueError(str(result.get("error", "没有找到可用模型。")))
                self.send_json(result)
            elif parsed.path == "/api/api-save":
                self.send_json(save_api_config(self.read_json()))
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

    def send_stream_event(self, event: str, data: dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def stream_query(self, payload: dict[str, Any]) -> None:
        question = str(payload.get("question", "")).strip()
        if not question:
            self.send_error_json("Missing question", HTTPStatus.BAD_REQUEST)
            return

        limit = int(payload.get("limit", 5) or 5)
        use_web = bool(payload.get("use_web", False))
        allow_api = bool(payload.get("allow_api", True))
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
                result_box["result"] = kb_query(DB_PATH, question, limit=limit, use_web=use_web, allow_api=allow_api)
            except Exception as exc:  # noqa: BLE001
                error_box["error"] = exc

        started_at = time.monotonic()
        initial_status = (
            "正在检索本地索引和联网资料，并准备整合引用..."
            if use_web
            else "正在检索本地索引并整理引用..."
        )
        self.send_stream_event("status", {"message": initial_status})
        worker = threading.Thread(target=run_query, daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(timeout=1.0)
            if worker.is_alive():
                elapsed = max(1, int(time.monotonic() - started_at))
                self.send_stream_event(
                    "status",
                    {
                        "message": "仍在处理中：正在等待检索、资料更新或 API 总结完成...",
                        "elapsed_seconds": elapsed,
                    },
                )

        if error_box:
            self.send_stream_event("error", {"error": str(error_box["error"])})
            return

        result = result_box["result"]
        sources = result.get("sources", [])
        sys.stderr.write(
            "query "
            + json.dumps(
                {
                    "question": question,
                    "source_type": result.get("source_type"),
                    "web_search_provider": result.get("web_search_provider"),
                    "web_search_reason": result.get("web_search_reason"),
                    "sources": len(sources),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        sys.stderr.flush()
        self.send_stream_event(
            "status",
            {
                "message": f"已找到 {len(sources)} 条引用，正在生成答案...",
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
            },
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
            },
        )


def compact_ingest_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "documents",
        "indexed_documents",
        "chunks",
        "duplicates",
        "skipped",
        "extraction_warnings",
        "requires_review",
        "approved_reviews",
        "unsupported_files",
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
