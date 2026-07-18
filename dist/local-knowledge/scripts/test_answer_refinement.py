#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline regression checks for API-grounded answer refinement."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from query_knowledge_base import chinese_grounded_fallback, grounded_citations, local_translate_to_chinese, synthesize_with_api


class MockHandler(BaseHTTPRequestHandler):
    received: dict[str, object] = {}

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        MockHandler.received = json.loads(self.rfile.read(length).decode("utf-8"))
        body = json.dumps(
            {"choices": [{"message": {"content": "量子纠缠是量子系统中的关联现象 [1]。"}}]},
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> None:
    assert local_translate_to_chinese("Voilà pourquoi tu me verras quelquefois donner cent sous aux vagabonds.") == ""
    result = {
        "source_type": "hybrid_web_knowledge",
        "sources": [
            {"source_id": 1, "title": "量子纠缠", "snippet": "量子纠缠是量子系统中的关联现象。"},
            {"source_id": 2, "title": "重复来源", "snippet": "量子纠缠是量子系统中的关联现象。"},
            {"source_id": 3, "title": "扩展来源", "snippet": "量子纠缠在经典力学中没有直接对应现象。"},
        ],
    }
    citations = grounded_citations(result)
    assert len(citations) == 2
    fallback = chinese_grounded_fallback(
        {
            "source_type": "knowledge_base",
            "sources": [{"source_id": 1, "citation": "Voilà pourquoi tu me verras quelquefois donner cent sous."}],
        }
    )
    assert "当前缺少可用的本地翻译模型" in fallback["answer"]
    assert "python -m pip install -r requirements.txt" in fallback["answer"]
    assert ".\\run-agent.bat translation-setup" in fallback["answer"]
    assert ".\\run-agent.bat verify" in fallback["answer"]

    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        refined = synthesize_with_api(
            "量子纠缠是什么",
            result,
            {
                "LKA_USE_API": "true",
                "LKA_API_PROVIDER": "openai-compatible",
                "LKA_API_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                "LKA_API_KEY": "test-key",
                "LKA_API_MODEL": "mock-chat",
                "LKA_API_TIMEOUT_SECONDS": "10",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
    assert refined["answer"] == "量子纠缠是量子系统中的关联现象 [1]。"
    assert refined["answer_mode"] == "api_grounded_summary"
    assert refined["api_used"] is True
    user_message = MockHandler.received["messages"][1]["content"]  # type: ignore[index]
    sent = json.loads(user_message)
    assert len(sent["citations"]) == 2
    print("answer refinement: ok")


if __name__ == "__main__":
    main()
