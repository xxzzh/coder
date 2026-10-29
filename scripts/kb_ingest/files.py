"""Raw-file discovery, fingerprints, ingest lock and project paths."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
from pathlib import Path


SUPPORTED = {".md", ".txt", ".doc", ".docx", ".pdf", ".xlsx"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/kb_ingest/files.py -> project root

# From the install folder, not the current folder (see kb/config.py).
DB_PATH = PROJECT_ROOT / "knowledge_base" / "index" / "knowledge.db"

RAW_DIR = PROJECT_ROOT / "knowledge_base" / "raw"

PROCESSED_DIR = PROJECT_ROOT / "knowledge_base" / "processed"

SMALL_FILE_HASH_BYTES = 1024 * 1024

# Part of every file's fingerprint: raised when extraction changes, so the next update reads each
# file again instead of keeping text produced by the old rules. extract-v2: PDF OCR only where the
# text layer lacks content and no repeated text layers, Chinese OCR without spaces between
# characters, passwords and product keys hidden.
EXTRACTION_VERSION = "extract-v2"

# Added for one file type when only its reading changes, so the next update reads just those files
# again. xlsx-rows-v2: a row or item list longer than a chunk is split into lines that each name its
# row's item or group.
EXTRACTION_VERSION_BY_SUFFIX = {".xlsx": "xlsx-rows-v2"}


def portable_error_message(exc: Exception) -> str:
    """Avoid leaking machine-specific temp/runtime paths into generated reports."""
    if isinstance(exc, subprocess.CalledProcessError):
        command = Path(str(exc.cmd[0])).name if isinstance(exc.cmd, list) and exc.cmd else "command"
        return f"{command} exited with status {exc.returncode}"
    return str(exc)


def is_ignored_raw_file(path: Path) -> bool:
    """Ignore transient Office lock files created while documents are open."""
    return path.name.startswith("~$")


class IngestAlreadyRunningError(RuntimeError):
    """Raised when another process is already updating the same index."""


def ingest_lock_path(db_path: Path = DB_PATH) -> Path:
    return db_path.with_suffix(".ingest.lock")


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not process:
                return False
            try:
                exit_code = wintypes.DWORD()
                if not ctypes.windll.kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code)):
                    return False
                return exit_code.value == 259
            finally:
                ctypes.windll.kernel32.CloseHandle(process)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def ingest_in_progress(db_path: Path = DB_PATH) -> bool:
    lock_path = ingest_lock_path(db_path)
    if not lock_path.exists():
        return False
    try:
        lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(lock_data.get("pid", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        pid = 0
    if _pid_is_running(pid):
        return True
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    return False


@contextmanager
def ingestion_lock(db_path: Path):
    lock_path = ingest_lock_path(db_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if ingest_in_progress(db_path):
        raise IngestAlreadyRunningError("知识库索引正在更新，请稍后再试。")
    try:
        handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise IngestAlreadyRunningError("知识库索引正在更新，请稍后再试。") from exc
    try:
        payload = json.dumps({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        os.write(handle, payload.encode("utf-8"))
        os.close(handle)
        yield
    finally:
        try:
            os.close(handle)
        except OSError:
            pass
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def read_plain_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def unsupported_raw_files(raw_dir: Path) -> list[str]:
    if not raw_dir.exists():
        return []
    return [
        str(path.relative_to(raw_dir)).replace("\\", "/")
        for path in sorted(raw_dir.rglob("*"))
        if path.is_file() and not is_ignored_raw_file(path) and path.suffix.lower() not in SUPPORTED
    ]


def raw_file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_file_stat_fingerprint(path: Path, raw_dir: Path | None = None) -> str:
    stat = path.stat()
    rel_path = str(path.relative_to(raw_dir)).replace("\\", "/") if raw_dir else str(path)
    parts = [rel_path, str(stat.st_size), str(stat.st_mtime_ns)]
    if stat.st_size <= SMALL_FILE_HASH_BYTES:
        parts.append(raw_file_fingerprint(path))
    parts.append(EXTRACTION_VERSION)
    if path.suffix.lower() in EXTRACTION_VERSION_BY_SUFFIX:
        parts.append(EXTRACTION_VERSION_BY_SUFFIX[path.suffix.lower()])
    return ":".join(parts)


def source_fingerprint(raw_dir: Path) -> str:
    items: list[str] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and not is_ignored_raw_file(path) and path.suffix.lower() in SUPPORTED:
            items.append(raw_file_stat_fingerprint(path, raw_dir))
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()
