"""Progress of a running ingest, for a caller that shows it (the Web UI's processing bar).

Silent unless a reporter is set. Only one ingest runs at a time (the ingest lock), so one
process-wide reporter is enough.
"""

from __future__ import annotations

from typing import Any, Callable

Reporter = Callable[[str, dict[str, Any]], None]

_reporter: Reporter | None = None


def set_reporter(reporter: Reporter | None) -> None:
    global _reporter
    _reporter = reporter


def report(phase: str, **fields: Any) -> None:
    """phase: "reading" (file of total), "ocr" (page of total), "indexing", "embedding" (chunk of total)."""
    if _reporter is None:
        return
    try:
        _reporter(phase, fields)
    except Exception:  # noqa: BLE001  (a display problem must never stop the ingest)
        pass
