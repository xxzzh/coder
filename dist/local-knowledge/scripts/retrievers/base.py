"""Small plugin abstraction for domain-specific retrieval rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


RetrieverHandler = Callable[[Path, str, dict[str, str], bool], dict[str, Any] | None]


@dataclass(frozen=True)
class DomainRetriever:
    name: str
    handler: RetrieverHandler

    def retrieve(
        self,
        db_path: Path,
        question: str,
        metadata: dict[str, str],
        refresh_scheduled: bool,
    ) -> dict[str, Any] | None:
        return self.handler(db_path, question, metadata, refresh_scheduled)
