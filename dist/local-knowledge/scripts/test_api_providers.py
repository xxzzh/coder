#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline tests for API provider discovery and grounded chat calls."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from api_providers import chat_completion, discover_models, infer_provider, token_plan_rejected, validate_chat_model


class MockHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self.send_json({"data": [{"id": "mock-chat"}, {"id": "text-embedding-3-small"}]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        assert payload["model"] == "mock-chat"
        self.send_json({"choices": [{"message": {"content": "连接成功"}}]})

    def send_json(self, data: object) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> None:
    assert infer_provider("https://api.xiaomimimo.com/v1", "key") == "xiaomi-mimo"
    assert infer_provider("https://api.deepseek.com", "key") == "deepseek"
    assert infer_provider("https://api.openai.com/v1", "key") == "openai"
    assert token_plan_rejected("https://token-plan-cn.xiaomimimo.com/v1", "tp-example")

    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        discovery = discover_models("openai-compatible", base_url, "test-key")
        assert discovery["ok"] is True
        assert discovery["models"] == ["mock-chat"]
        answer = chat_completion(
            "openai-compatible",
            base_url,
            "test-key",
            "mock-chat",
            [{"role": "user", "content": "test"}],
        )
        assert answer == "连接成功"
        validation = validate_chat_model("openai-compatible", base_url, "test-key", "mock-chat")
        assert validation["ok"] is True
    finally:
        server.shutdown()
        server.server_close()
    print("api providers: ok")


if __name__ == "__main__":
    main()
