"""Documents in the Web UI: uploads into knowledge_base/raw, removal into knowledge_base/trash, the
document list with each file's state, and the background job that processes them with progress."""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable

from ingest_knowledge_base import (
    SUPPORTED,
    IngestAlreadyRunningError,
    extraction_reports,
    ingest_in_progress,
    is_ignored_raw_file,
    pending_review_reports,
)
from kb_ingest import progress
from manage_knowledge_base import rebuild as kb_rebuild
from manage_knowledge_base import sources as kb_sources
from manage_knowledge_base import update as kb_update
import vector_store

# Uploaded files go into the folder the sample layout uses for their type.
UPLOAD_SUBDIRS = {
    ".pdf": "pdf",
    ".docx": "word",
    ".doc": "word",
    ".xlsx": "excel",
    ".md": "markdown",
    ".txt": "markdown",
}
DEFAULT_UPLOAD_MAX_MB = 200
UPLOAD_READ_BYTES = 1024 * 1024
FILE_NAME_MAX_CHARS = 150
WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def upload_max_bytes() -> int:
    try:
        megabytes = int(os.environ.get("LKA_UPLOAD_MAX_MB", "") or DEFAULT_UPLOAD_MAX_MB)
    except ValueError:
        megabytes = DEFAULT_UPLOAD_MAX_MB
    return max(1, megabytes) * 1024 * 1024


def safe_upload_name(name: str) -> str:
    """The file name an upload is stored under: no folders, no characters Windows rejects."""
    name = re.split(r"[\\/]", str(name or ""))[-1]
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name).strip().rstrip(".")
    suffix = Path(name).suffix.lower()
    if not name or name.startswith((".", "~$")) or not Path(name).stem.strip():
        raise ValueError("文件名无效。")
    if suffix not in SUPPORTED:
        raise ValueError(f"不支持这种文件（{suffix or '没有扩展名'}）。支持 PDF、Word（.docx/.doc）、Excel（.xlsx）、Markdown（.md）和 TXT。")
    stem = Path(name).stem
    if stem.upper() in WINDOWS_RESERVED_NAMES:
        stem = f"_{stem}"
    if len(stem) + len(suffix) > FILE_NAME_MAX_CHARS:
        stem = stem[: FILE_NAME_MAX_CHARS - len(suffix)]
    return stem + Path(name).suffix


def inside(root: Path, path: Path) -> bool:
    root = root.resolve()
    path = path.resolve()
    return path == root or root in path.parents


def check_upload(file_name: str, length: int) -> str:
    """The stored name of an acceptable upload; ValueError says why one is refused."""
    name = safe_upload_name(file_name)
    if length <= 0:
        raise ValueError("文件是空的。")
    limit = upload_max_bytes()
    if length > limit:
        raise ValueError(f"文件超过 {limit // 1024 // 1024} MB，无法上传。")
    return name


def save_upload(stream: BinaryIO, length: int, file_name: str, raw_dir: Path) -> dict[str, Any]:
    """Store an upload under raw_dir; a file of the same name and type is replaced by the new one."""
    name = check_upload(file_name, length)
    target_dir = raw_dir / UPLOAD_SUBDIRS[Path(name).suffix.lower()]
    target = target_dir / name
    if not inside(raw_dir, target):
        raise ValueError("文件名无效。")
    target_dir.mkdir(parents=True, exist_ok=True)
    replaced = target.exists()
    # «~$» marks the partial file as a temporary one the ingest ignores until it is complete.
    partial = target_dir / f"~$upload-{uuid.uuid4().hex}.part"
    remaining = length
    try:
        with partial.open("wb") as out:
            while remaining > 0:
                block = stream.read(min(UPLOAD_READ_BYTES, remaining))
                if not block:
                    raise ValueError("上传中断，请重试。")
                out.write(block)
                remaining -= len(block)
        os.replace(partial, target)
    finally:
        if partial.exists():
            partial.unlink()
    rel_path = str(target.relative_to(raw_dir)).replace("\\", "/")
    return {"file_path": rel_path, "file_name": name, "size": length, "replaced": replaced}


def remove_document(rel_path: str, raw_dir: Path, trash_dir: Path) -> dict[str, Any]:
    """Move a document out of the knowledge base into trash/<time>/, where it can be restored by hand."""
    rel_path = str(rel_path or "").replace("\\", "/").strip("/")
    source = raw_dir / rel_path
    if not rel_path or not inside(raw_dir, source) or not source.is_file():
        raise ValueError("文件不存在，请刷新列表。")
    target = trash_dir / time.strftime("%Y%m%d-%H%M%S") / rel_path
    copy = 1
    while target.exists():  # the same file removed twice within a second keeps both copies
        copy += 1
        target = target.with_name(f"{source.stem} ({copy}){source.suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    return {"file_path": rel_path, "moved_to": str(target)}


REVIEW_REASONS = (
    (r"^Extracted text is very short", "读出的文字很少，可能是扫描件、加密文件或复杂版式。"),
    (r"^PDF appears to contain images; OCR text was not available", "PDF 里有图片，但文字识别（OCR）不可用，图片中的文字没有读取。"),
    (r"^No text could be extracted from PDF", "没能从 PDF 读出文字。"),
    (r"^DOCX contains (\d+) embedded image\(s\), but OCR returned no image text", r"Word 文档里有 \1 张图片，没能识别出其中的文字。"),
    (r"^OCR skipped", "没有可用的文字识别（OCR）组件，扫描内容没有读取。"),
)
XLSX_PARTS = (
    (r"(\d+) embedded object\(s\) not extracted \((.*)\)", r"\1 个嵌入对象（\2）"),
    (r"(\d+) chart/drawing part\(s\) could not be read", r"\1 个图表或绘图"),
    (r"embedded image\(s\) returned no OCR text", "图片中的文字"),
)


def xlsx_part(part: str) -> str:
    for pattern, replacement in XLSX_PARTS:
        if re.fullmatch(pattern, part):
            return re.sub(pattern, replacement, part)
    return part


def review_reason(warning: str) -> str:
    """An extraction warning in words a user can act on."""
    warning = str(warning).strip()
    if warning.startswith("XLSX content not extracted:"):
        # kb_ingest/xlsx.py joins the parts with ", "; one part lists its objects with "; ".
        details = warning.split(":", 1)[1].strip().rstrip(".")
        return f"Excel 中有部分内容没能读取：{'；'.join(xlsx_part(part) for part in details.split(', '))}。"
    for pattern, replacement in REVIEW_REASONS:
        if re.search(pattern, warning):
            return re.sub(pattern + r".*$", replacement, warning, count=1) if "\\1" in replacement else replacement
    return warning


def _iso_seconds(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")


def documents_payload(raw_dir: Path, db_path: Path, processed_dir: Path, processing: bool) -> dict[str, Any]:
    """Every file in the knowledge base folder with its state:
    ready, processing, outdated (changed since it was processed), needs_review, duplicate, failed, unsupported."""
    indexed = {str(item.get("file_path")): item for item in kb_sources(db_path).get("sources", [])}
    pending = {str(item.get("file_path")): item for item in pending_review_reports(raw_dir, processed_dir)}
    reports = {str(item.get("file_path")): item for item in extraction_reports(processed_dir)}
    items: list[dict[str, Any]] = []
    if raw_dir.exists():
        for path in sorted(raw_dir.rglob("*")):
            if not path.is_file() or is_ignored_raw_file(path):
                continue
            rel_path = str(path.relative_to(raw_dir)).replace("\\", "/")
            stat = path.stat()
            row = indexed.get(rel_path)
            current = bool(row) and int(row.get("file_size") or 0) == stat.st_size and row.get("modified_at") == _iso_seconds(stat.st_mtime)
            reasons: list[str] = []
            if path.suffix.lower() not in SUPPORTED:
                status = "unsupported"
            elif rel_path in pending:
                status = "needs_review"
                reasons = [review_reason(warning) for warning in pending[rel_path].get("warnings", [])]
            elif current and row.get("duplicate_of"):
                status = "duplicate"
                reasons = [f"内容与 {row.get('duplicate_of')} 相同，只保留一份。"]
            elif current:
                status = "ready"
            elif processing:
                status = "processing"
            elif row:
                status = "outdated"
            else:
                status = "failed"
                reasons = [review_reason(warning) for warning in (reports.get(rel_path) or {}).get("warnings", [])]
            items.append(
                {
                    "file_path": rel_path,
                    "file_name": path.name,
                    "file_type": path.suffix.lower().lstrip("."),
                    "size": stat.st_size,
                    "modified_at": _iso_seconds(stat.st_mtime),
                    "status": status,
                    "reasons": reasons,
                    "chunks": int(row.get("chunk_count") or 0) if row else 0,
                }
            )
    counts: dict[str, int] = {}
    for item in items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {"documents": items, "counts": counts, "processing": processing}


JOB_LABELS = {
    "update": "处理资料",
    "rebuild": "全部重新处理",
    "vector-rebuild": "重建向量索引",
}


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "documents",
        "indexed_documents",
        "changed_documents",
        "removed_documents",
        "chunks",
        "skipped",
        "duplicates",
        "requires_review",
        "approved_reviews",
        "vector_index_status",
        "vector_index_reason",
        "indexed_at",
    )
    return {key: result.get(key) for key in keys if key in result}


class IndexJob:
    """One update, rebuild or vector rebuild at a time, run in the background with its progress.

    An update asked for while one runs (more uploads arrive) runs once more right after it, so every
    file is processed; another kind of job is refused until the current one ends.
    """

    def __init__(
        self,
        raw_dir: Path,
        db_path: Path,
        processed_dir: Path,
        redact: Callable[[str], str] = str,
    ) -> None:
        self.raw_dir = raw_dir
        self.db_path = db_path
        self.processed_dir = processed_dir
        self.redact = redact
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {"state": "idle"}
        self._queued: str | None = None

    def running(self) -> bool:
        with self._lock:
            return self._state.get("state") == "running"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
            state["queued"] = self._queued
        if state.get("started_at"):
            end = state.get("finished_at") or time.time()
            state["elapsed_seconds"] = round(end - state["started_at"], 1)
        return state

    def start(self, kind: str) -> dict[str, Any]:
        if kind not in JOB_LABELS:
            raise ValueError(f"unknown job: {kind}")
        with self._lock:
            if self._state.get("state") == "running":
                if kind == "update":
                    self._queued = "update"
                    return {"ok": True, "started": False, "queued": True}
                return {"ok": False, "started": False, "error": "正在处理资料，请等当前任务完成后再试。"}
            self._state = self._running_state(kind)
        threading.Thread(target=self._run, args=(kind,), daemon=True).start()
        return {"ok": True, "started": True}

    @staticmethod
    def _running_state(kind: str) -> dict[str, Any]:
        return {"state": "running", "kind": kind, "label": JOB_LABELS[kind], "phase": "starting", "started_at": time.time()}

    def _progress(self, phase: str, fields: dict[str, Any]) -> None:
        with self._lock:
            self._state["phase"] = phase
            self._state[phase] = fields

    def _run(self, kind: str) -> None:
        while True:
            progress.set_reporter(self._progress)
            try:
                result = self._run_once(kind)
            except Exception as exc:  # noqa: BLE001
                outcome = {"state": "failed", "error": self.redact(str(exc)) or exc.__class__.__name__}
            else:
                outcome = {"state": "done", "summary": compact_result(result)}
            finally:
                progress.set_reporter(None)
            with self._lock:
                if self._queued:
                    kind, self._queued = self._queued, None
                    self._state = self._running_state(kind)
                    continue
                self._state.update(outcome, phase=outcome["state"], finished_at=time.time())
                return

    def _run_once(self, kind: str) -> dict[str, Any]:
        for _ in range(3):
            # Another process (a question's background refresh, the command line) may be updating the
            # same index: wait for it, then run, which picks up whatever it did not.
            while ingest_in_progress(self.db_path):
                self._progress("waiting", {})
                time.sleep(1)
            try:
                if kind == "rebuild":
                    return kb_rebuild(self.raw_dir, self.db_path, self.processed_dir, strict_extraction=True)
                if kind == "vector-rebuild":
                    vector_store.reset_vector_index()
                return kb_update(self.raw_dir, self.db_path, self.processed_dir, strict_extraction=True)
            except IngestAlreadyRunningError:
                continue
        raise RuntimeError("另一个更新一直在进行，请稍后重试。")
