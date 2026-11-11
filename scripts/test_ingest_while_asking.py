#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asking questions while documents are being processed.

Reading (OCR included), chunking and embedding take minutes; questions asked meanwhile must be
answered at once from the previous index, and the short final write must not make a cached answer
look like a broken index. Everything runs on a temporary knowledge base; the vector index is replaced
by stand-ins, so the real one is never touched and no model is loaded.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

os.environ["LKA_API_USAGE_LOG"] = "off"  # test calls must not land in the real usage log

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ingest_knowledge_base as ingest_module  # noqa: E402
import model_capabilities  # noqa: E402
import query_knowledge_base as query_module  # noqa: E402
import vector_store  # noqa: E402
from kb_ingest import vectors  # noqa: E402
from kb_ingest.store import init_db  # noqa: E402

# An answer that waited for the index lock took 5 seconds; one that did not takes milliseconds.
PROMPT_SECONDS = 2.0
ASKED_BEFORE = "出差报销需要几个工作日内提交发票"
NOT_ASKED_BEFORE = "审批完成后几个工作日内付款"
# The tests replace it with a stand-in; one of them checks the real one.
REAL_TRIGGER_ASYNC_UPDATE = query_module.trigger_async_update


@contextmanager
def temporary_knowledge_base() -> Iterator[tuple[Path, Path, Path]]:
    """(raw folder, index file, processed folder) with one processed document, asked about twice so
    its answer is cached."""
    replaced = {
        (ingest_module, "build_optional_vector_index"): lambda *args, **kwargs: {"ready": False, "reason": "embedding_disabled", "backend": "faiss"},
        (ingest_module, "embed_before_indexing"): lambda *args, **kwargs: {"error": None, "embedded_hashes": set()},
        # A question must not start the real background update against this temporary index.
        (query_module, "trigger_async_update"): lambda *args, **kwargs: False,
        (query_module, "runtime_config"): lambda: {"LKA_USE_API": "false"},
    }
    originals = {key: getattr(*key, None) for key in replaced}
    wait_seconds = query_module.CACHE_WRITE_WAIT_SECONDS
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        raw_dir, db_path, processed_dir = Path(temp_dir) / "raw", Path(temp_dir) / "index" / "knowledge.db", Path(temp_dir) / "processed"
        raw_dir.mkdir()
        (raw_dir / "报销制度.md").write_text(
            "# 报销制度\n\n出差报销需要在返回后 5 个工作日内提交发票，审批完成后 3 个工作日内付款。\n", encoding="utf-8"
        )
        try:
            for (module, name), stand_in in replaced.items():
                setattr(module, name, stand_in)
            ingest_module.ingest(raw_dir, db_path, processed_dir)
            assert ask(db_path, ASKED_BEFORE)[0].get("cache_hit") is False
            assert ask(db_path, ASKED_BEFORE)[0].get("cache_hit") is True
            yield raw_dir, db_path, processed_dir
        finally:
            for (module, name), original in originals.items():
                setattr(module, name, original)
            query_module.CACHE_WRITE_WAIT_SECONDS = wait_seconds


def ask(db_path: Path, question: str) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    result = query_module.query(db_path, question, allow_api=False)
    return result, time.monotonic() - started


def assert_answered(result: dict[str, Any], seconds: float, cached: bool) -> None:
    assert result.get("source_type") == "knowledge_base", result
    assert not result.get("degraded"), result
    assert bool(result.get("cache_hit")) is cached, result
    assert "5 个工作日" in str(result.get("answer")) or "3 个工作日" in str(result.get("answer")), result
    assert seconds < PROMPT_SECONDS, f"answer took {seconds:.1f}s"


def indexed_files(db_path: Path) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return sorted(str(name) for (name,) in conn.execute("SELECT file_name FROM documents"))
    finally:
        conn.close()


def test_questions_while_a_document_is_read() -> None:
    """A slow document (a scanned book) is being read: questions are answered from the previous
    index without waiting, whether asked before or not."""
    with temporary_knowledge_base() as (raw_dir, db_path, processed_dir):
        reading, finish_reading = threading.Event(), threading.Event()
        original_read = ingest_module.read_document_with_report

        def slow_read(path: Path) -> tuple[str, dict[str, Any]]:
            reading.set()
            assert finish_reading.wait(60)
            return original_read(path)

        (raw_dir / "请假制度.md").write_text("# 请假制度\n\n事假需要提前 2 个工作日向主管申请。\n", encoding="utf-8")
        outcome: dict[str, Any] = {}

        def run_update() -> None:
            try:
                outcome["result"] = ingest_module.ingest(raw_dir, db_path, processed_dir)
            except Exception as exc:  # noqa: BLE001  (reported by the assertion below)
                outcome["error"] = exc

        ingest_module.read_document_with_report = slow_read
        worker = threading.Thread(target=run_update, daemon=True)
        try:
            worker.start()
            assert reading.wait(30)
            assert_answered(*ask(db_path, ASKED_BEFORE), cached=True)
            assert_answered(*ask(db_path, NOT_ASKED_BEFORE), cached=False)
            # The second time it comes from the cache: the entry was written during the update.
            assert_answered(*ask(db_path, NOT_ASKED_BEFORE), cached=True)
            assert indexed_files(db_path) == ["报销制度.md"]  # nothing half-written is visible
        finally:
            finish_reading.set()
            worker.join(60)
            ingest_module.read_document_with_report = original_read
        assert "error" not in outcome, outcome
        assert outcome["result"]["changed_documents"] == 1 and outcome["result"]["indexed_documents"] == 2
        assert indexed_files(db_path) == ["报销制度.md", "请假制度.md"]
        # The update emptied the answer cache: the next answer is searched again, in the new index.
        assert_answered(*ask(db_path, ASKED_BEFORE), cached=False)
        answer, _ = ask(db_path, "事假需要提前几个工作日申请")
        assert answer.get("source_type") == "knowledge_base" and "2 个工作日" in str(answer.get("answer")), answer


def test_questions_while_the_index_is_written() -> None:
    """During the update's short final write the answer cache cannot be written: the cached answer
    is still given, a new question is still answered, and neither reports a broken index."""
    with temporary_knowledge_base() as (_, db_path, _):
        query_module.CACHE_WRITE_WAIT_SECONDS = 0.2
        writer = sqlite3.connect(db_path)
        try:
            writer.execute("BEGIN IMMEDIATE")
            assert_answered(*ask(db_path, ASKED_BEFORE), cached=True)
            assert_answered(*ask(db_path, NOT_ASKED_BEFORE), cached=False)
        finally:
            writer.rollback()
            writer.close()
        # With the write over the cache works as usual again.
        assert_answered(*ask(db_path, NOT_ASKED_BEFORE), cached=False)
        assert_answered(*ask(db_path, NOT_ASKED_BEFORE), cached=True)


def test_a_failed_update_writes_nothing() -> None:
    """An update that fails while writing leaves the previous index as it was, and unlocked."""
    with temporary_knowledge_base() as (raw_dir, db_path, processed_dir):
        (raw_dir / "请假制度.md").write_text("# 请假制度\n\n事假需要提前 2 个工作日向主管申请。\n", encoding="utf-8")

        def failing_vector_index(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("disk full")

        ingest_module.build_optional_vector_index = failing_vector_index
        try:
            ingest_module.ingest(raw_dir, db_path, processed_dir)
        except RuntimeError as exc:
            assert str(exc) == "disk full"
        else:
            raise AssertionError("the failing update did not fail")
        assert indexed_files(db_path) == ["报销制度.md"]
        assert_answered(*ask(db_path, ASKED_BEFORE), cached=True)
        writer = sqlite3.connect(db_path, timeout=0.2)
        try:
            writer.execute("BEGIN IMMEDIATE")  # raises "database is locked" if the failed update still held it
            writer.rollback()
        finally:
            writer.close()


def test_a_question_does_not_index_unconfirmed_documents() -> None:
    """The background update a question starts skips a file waiting for confirmation (content that
    could not be read), as the Web UI's processing does; only the update without that rule indexes it."""
    with temporary_knowledge_base() as (raw_dir, db_path, processed_dir):
        original_read = ingest_module.read_document_with_report

        def read_with_unreadable_part(path: Path) -> tuple[str, dict[str, Any]]:
            text, report = original_read(path)
            if path.name == "扫描件.md":
                report = {**report, "requires_review": True, "warnings": ["PDF appears to contain images; OCR text was not available"]}
            return text, report

        (raw_dir / "扫描件.md").write_text("# 扫描件\n\n只读出了这一行。\n", encoding="utf-8")
        started: list[list[str]] = []
        original_popen = query_module.subprocess.Popen
        ingest_module.read_document_with_report = read_with_unreadable_part
        query_module.subprocess.Popen = lambda command, **kwargs: started.append(list(command))
        try:
            assert REAL_TRIGGER_ASYNC_UPDATE(db_path, raw_dir, processed_dir) is True
            assert len(started) == 1 and "--strict-extraction" in started[0], started
            # What that command does, run here: the file is left out and listed for confirmation.
            result = ingest_module.ingest(raw_dir, db_path, processed_dir, strict_extraction=True)
            assert result["skipped"] == 1 and indexed_files(db_path) == ["报销制度.md"], result
            assert [item["file_path"] for item in ingest_module.pending_review_reports(raw_dir, processed_dir)] == ["扫描件.md"]
            assert REAL_TRIGGER_ASYNC_UPDATE(db_path, raw_dir, processed_dir) is False  # not started over and over
            # Without the rule (the command line's plain update) the same file is indexed.
            ingest_module.ingest(raw_dir, db_path, processed_dir)
            assert indexed_files(db_path) == ["扫描件.md", "报销制度.md"]
        finally:
            ingest_module.read_document_with_report = original_read
            query_module.subprocess.Popen = original_popen


def test_chunks_are_embedded_before_the_write() -> None:
    """embed_before_indexing embeds what the vector index will hold while no write transaction is
    open, the write then reuses those vectors, and a failure is not tried a second time."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        db_path = Path(temp_dir) / "knowledge.db"
        conn = init_db(db_path)
        conn.row_factory = sqlite3.Row
        for doc_id, text in (("kept", "kept text"), ("replaced", "replaced text")):
            conn.execute(
                "INSERT INTO documents(doc_id, file_name, file_path, file_type, file_size, modified_at, indexed_at, content_hash, text)"
                " VALUES (?, ?, ?, 'md', 1, 't', 't', ?, ?)",
                (doc_id, f"{doc_id}.md", f"{doc_id}.md", doc_id, text),
            )
            conn.execute("INSERT INTO chunks(chunk_id, doc_id, chunk_index, text, embedding) VALUES (?, ?, 1, ?, '[]')", (f"{doc_id}_0001", doc_id, text))
        conn.commit()

        embedding = {"enabled": True, "usable": True, "provider": "openai-compatible", "model": "fake", "base_url": "", "requested_dimensions": 2}
        calls: list[list[str]] = []
        size = {"value": 2}

        def fake_embed(texts: list[str], config: dict | None = None, **kwargs: object) -> dict:
            calls.append(list(texts))
            # Embedding runs with the index unlocked: another connection can write meanwhile.
            other = sqlite3.connect(db_path, timeout=0.2)
            other.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES ('written_during_embedding', ?)", (str(len(calls)),))
            other.commit()
            other.close()
            return {"ok": True, "vectors": [[float(len(text))] * size["value"] for text in texts]}

        originals = (
            model_capabilities.embed_texts,
            model_capabilities.embedding_config,
            model_capabilities.runtime_config,
            vector_store.VECTOR_DIR,
            vector_store.reset_vector_index,
            vector_store.write_unavailable_metadata,
        )
        try:
            model_capabilities.embed_texts = fake_embed
            model_capabilities.embedding_config = lambda config=None: dict(embedding)
            model_capabilities.runtime_config = lambda *args, **kwargs: {}
            vector_store.VECTOR_DIR = Path(temp_dir) / "vector"
            vector_store.reset_vector_index = lambda *args, **kwargs: None
            vector_store.write_unavailable_metadata = lambda reason, *args, **extra: {"ready": False, "reason": reason, **extra}

            hashes = {text: vector_store.text_hash(text) for text in ("kept text", "new text")}
            first = vectors.embed_before_indexing(conn, {"replaced"}, ["new text"], force=True)
            assert first == {"error": None, "embedded_hashes": set(hashes.values())}, first
            assert calls == [["kept text", "new text"]], calls  # not the chunk about to be replaced
            again = vectors.embed_before_indexing(conn, {"replaced"}, ["new text"], force=True)
            assert again == {"error": None, "embedded_hashes": set()} and len(calls) == 1  # all cached: no model call
            reused = vectors.embed_records_with_cache([{"text": "kept text"}, {"text": "new text"}], embedding, {})
            assert reused["ok"] and reused["embedded_chunks"] == 0 and len(calls) == 1  # the write only reads the cache
            assert reused["vectors"] == [[9.0, 9.0], [8.0, 8.0]]

            # An API that changed its vector size: everything is embedded again here, not during the write.
            size["value"] = 3
            resized = vectors.embed_before_indexing(conn, set(), ["longer new text"], force=True)
            assert resized["error"] is None and calls[-2:] == [["replaced text", "longer new text"], ["kept text", "replaced text", "longer new text"]], calls
            cache = vector_store.EmbeddingCache(model_capabilities.embedding_model_key(embedding))
            assert cache.dimensions() == {3} and cache.missing([hashes["kept text"], "absent"]) == ["absent"]
            cache.close()

            model_capabilities.embedding_config = lambda config=None: {**embedding, "enabled": False, "usable": False}
            assert vectors.embed_before_indexing(conn, set(), ["never embedded"], force=True) == {"error": None, "embedded_hashes": set()}
            model_capabilities.embedding_config = lambda config=None: dict(embedding)

            model_capabilities.embed_texts = lambda texts, config=None, **kwargs: {"ok": False, "vectors": [], "error": "HTTP 500"}
            failed = vectors.embed_before_indexing(conn, set(), ["one more text"], force=True)
            assert failed == {"error": "HTTP 500", "embedded_hashes": set()}, failed
            model_capabilities.embed_texts = fake_embed
            before = len(calls)
            payload = vectors.build_optional_vector_index(conn, "t", "fingerprint", force=True, prepared=failed)
            assert payload["ready"] is False and payload["reason"] == "embedding_api_unavailable" and payload["error"] == "HTTP 500", payload
            assert len(calls) == before  # the failed embedding is not repeated inside the write transaction
        finally:
            (
                model_capabilities.embed_texts,
                model_capabilities.embedding_config,
                model_capabilities.runtime_config,
                vector_store.VECTOR_DIR,
                vector_store.reset_vector_index,
                vector_store.write_unavailable_metadata,
            ) = originals
            conn.close()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    test_questions_while_a_document_is_read()
    test_questions_while_the_index_is_written()
    test_a_failed_update_writes_nothing()
    test_a_question_does_not_index_unconfirmed_documents()
    test_chunks_are_embedded_before_the_write()
    print("ingest while asking: ok")


if __name__ == "__main__":
    main()
