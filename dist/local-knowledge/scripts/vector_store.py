#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional FAISS vector index for local knowledge chunks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = ROOT / "knowledge_base" / "vector"
FAISS_PATH = VECTOR_DIR / "chunks.faiss"
MAP_PATH = VECTOR_DIR / "chunks_map.json"
METADATA_PATH = VECTOR_DIR / "vector_metadata.json"


def faiss_dependency_status() -> dict[str, Any]:
    try:
        import faiss  # type: ignore  # noqa: F401
        import numpy  # type: ignore  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}
    return {"available": True, "error": None}


def read_vector_metadata(vector_dir: Path = VECTOR_DIR) -> dict[str, Any]:
    path = vector_dir / "vector_metadata.json"
    if not path.exists():
        return {"exists": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"exists": True, "ready": False, "error": str(exc)}
    return {"exists": True, **payload}


def vector_status(sqlite_chunk_count: int | None = None, vector_dir: Path = VECTOR_DIR) -> dict[str, Any]:
    dep = faiss_dependency_status()
    metadata = read_vector_metadata(vector_dir)
    faiss_path = vector_dir / "chunks.faiss"
    map_path = vector_dir / "chunks_map.json"
    files_exist = faiss_path.exists() and map_path.exists()
    ready = bool(dep["available"] and metadata.get("ready") and files_exist)
    chunk_count = int(metadata.get("chunk_count", 0) or 0)
    aligned = sqlite_chunk_count is None or chunk_count == int(sqlite_chunk_count or 0)
    reason = "ready"
    if metadata.get("exists") and not metadata.get("ready"):
        reason = str(metadata.get("reason") or "vector_index_not_ready")
    elif not dep["available"]:
        reason = "faiss_dependency_missing"
    elif not metadata.get("exists"):
        reason = "vector_index_missing"
    elif not files_exist:
        reason = "vector_files_missing"
    elif not aligned:
        reason = "chunk_count_mismatch"
    return {
        "backend": "faiss",
        "ready": ready and aligned,
        "reason": reason,
        "dependency": dep,
        "files": {
            "faiss": str(faiss_path),
            "map": str(map_path),
            "metadata": str(vector_dir / "vector_metadata.json"),
            "faiss_exists": faiss_path.exists(),
            "map_exists": map_path.exists(),
        },
        "metadata": metadata,
        "chunk_count": chunk_count,
        "sqlite_chunk_count": sqlite_chunk_count,
        "aligned_with_sqlite": aligned,
    }


def reset_vector_index(vector_dir: Path = VECTOR_DIR) -> None:
    vector_dir.mkdir(parents=True, exist_ok=True)
    for path in (vector_dir / "chunks.faiss", vector_dir / "chunks_map.json", vector_dir / "vector_metadata.json"):
        if path.exists() and path.is_file():
            path.unlink()


def write_unavailable_metadata(reason: str, vector_dir: Path = VECTOR_DIR, **extra: Any) -> dict[str, Any]:
    vector_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ready": False,
        "reason": reason,
        "backend": "faiss",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
    (vector_dir / "vector_metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_vector_index(
    records: list[dict[str, Any]],
    vectors: list[list[float]],
    metadata: dict[str, Any],
    vector_dir: Path = VECTOR_DIR,
) -> dict[str, Any]:
    reset_vector_index(vector_dir)
    if not records:
        return write_unavailable_metadata("no_chunks", vector_dir, chunk_count=0)
    dep = faiss_dependency_status()
    if not dep["available"]:
        return write_unavailable_metadata("faiss_dependency_missing", vector_dir, dependency=dep, chunk_count=len(records), **metadata)
    if len(records) != len(vectors):
        return write_unavailable_metadata("vector_record_count_mismatch", vector_dir, chunk_count=len(records), vector_count=len(vectors), **metadata)
    dimension = len(vectors[0]) if vectors else 0
    if not dimension:
        return write_unavailable_metadata("empty_vectors", vector_dir, chunk_count=len(records), **metadata)

    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore

        matrix = np.array(vectors, dtype="float32")
        index = faiss.IndexFlatIP(dimension)
        index.add(matrix)
        faiss.write_index(index, str(vector_dir / "chunks.faiss"))
    except Exception as exc:  # noqa: BLE001
        return write_unavailable_metadata("faiss_build_failed", vector_dir, error=str(exc), chunk_count=len(records), **metadata)

    map_rows = []
    for faiss_id, record in enumerate(records):
        map_rows.append(
            {
                "faiss_id": faiss_id,
                "chunk_id": record["chunk_id"],
                "doc_id": record["doc_id"],
                "file_name": record["file_name"],
                "file_path": record["file_path"],
                "file_type": record["file_type"],
                "chunk_index": record["chunk_index"],
            }
        )
    (vector_dir / "chunks_map.json").write_text(json.dumps(map_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {
        "ready": True,
        "reason": "ready",
        "backend": "faiss",
        "index_type": "IndexFlatIP",
        "metric": "cosine_on_normalized_vectors",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunk_count": len(records),
        "dimension": dimension,
        **metadata,
    }
    (vector_dir / "vector_metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def search_vector_index(query_vector: list[float], limit: int, vector_dir: Path = VECTOR_DIR) -> dict[str, Any]:
    status = vector_status(vector_dir=vector_dir)
    if not status["ready"]:
        return {"ok": False, "rows": [], "status": status}
    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore

        index = faiss.read_index(str(vector_dir / "chunks.faiss"))
        mapping = json.loads((vector_dir / "chunks_map.json").read_text(encoding="utf-8"))
        vector = np.array([query_vector], dtype="float32")
        scores, ids = index.search(vector, max(1, limit))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "rows": [], "status": {**status, "ready": False, "reason": str(exc)}}

    rows: list[dict[str, Any]] = []
    for score, faiss_id in zip(scores[0].tolist(), ids[0].tolist()):
        if faiss_id < 0 or faiss_id >= len(mapping):
            continue
        rows.append({**mapping[faiss_id], "vector_score": float(score), "semantic_score": max(0.0, float(score))})
    return {"ok": True, "rows": rows, "status": status}
