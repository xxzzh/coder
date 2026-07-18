#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline tests for API provider discovery and grounded chat calls."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from api_providers import chat_completion, discover_models, infer_provider, token_plan_rejected, validate_chat_model
from model_capabilities import embed_texts


class MockHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self.send_json({"data": [{"id": "mock-chat"}, {"id": "text-embedding-3-small"}]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/v1/embeddings":
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            assert payload["model"] == "mock-embedding"
            assert payload["dimensions"] == 3072
            inputs = payload["input"]
            self.send_json(
                {
                    "data": [
                        {"index": index, "embedding": [1.0 + index, 0.5, 0.25, 0.125]}
                        for index, _ in enumerate(inputs)
                    ]
                }
            )
            return
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
        embeddings = embed_texts(
            ["第一条", "第二条"],
            {
                "LKA_EMBEDDING_ENABLED": "true",
                "LKA_EMBEDDING_PROVIDER": "openai-compatible",
                "LKA_EMBEDDING_BASE_URL": base_url,
                "LKA_EMBEDDING_API_KEY": "test-key",
                "LKA_EMBEDDING_MODEL": "mock-embedding",
                "LKA_EMBEDDING_DIMENSIONS": "3072",
                "LKA_EMBEDDING_BATCH_SIZE": "2",
                "LKA_VECTOR_BACKEND": "faiss",
            },
        )
        assert embeddings["ok"] is True
        assert embeddings["dimension"] == 4
        assert len(embeddings["vectors"]) == 2
        disabled = embed_texts(["x"], {"LKA_EMBEDDING_ENABLED": "false"})
        assert disabled["ok"] is False
        assert disabled["error"] == "embedding_not_configured"
    finally:
        server.shutdown()
        server.server_close()
    print("api providers: ok")


if __name__ == "__main__":
    main()
