"""Usage statistics for the Web UI: every question asked on the chat page, the answer's outcome,
and the users' "有用 / 没用" feedback, kept as JSON lines under logs/ on this computer.

The statistics page shows what the knowledge base does for its users: how much is asked and
answered, the estimated time saved, the most asked questions, and above all the questions it
could not answer, which name the documents still missing. LKA_QUESTION_LOG=off stops recording.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Minutes a person would spend finding one answer in the documents by hand; the estimate of time
# saved uses it and the page says so (LKA_MINUTES_SAVED_PER_ANSWER changes it).
DEFAULT_MINUTES_SAVED_PER_ANSWER = 5
TOP_ITEMS = 10
GAP_ITEMS = 20
REFUSED_SOURCE_TYPES = {"none", None, ""}


def local_day(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")


def question_key(question: str) -> str:
    """Questions that differ only in spacing, punctuation or case count as the same question."""
    return re.sub(r"[\s，。？?！!、,.;；:：\"'“”‘’（）()]+", "", str(question)).lower()


class QuestionLog:
    def __init__(self, path: Path, feedback_path: Path, enabled: bool = True) -> None:
        self.path = path
        self.feedback_path = feedback_path
        self.enabled = enabled
        self._lock = threading.Lock()

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record(
        self,
        question: str,
        understood_as: str | None,
        result: dict[str, Any],
        seconds: float,
        remote: bool,
        api_failure: str | None = None,
    ) -> str:
        """Log one answered (or refused) question; returns its id for later feedback."""
        record_id = uuid.uuid4().hex[:16]
        if not self.enabled:
            return record_id
        sources = result.get("sources") or []
        files = sorted({str(source.get("file") or source.get("title") or "") for source in sources} - {""})
        self._append(
            self.path,
            {
                "id": record_id,
                "time": time.time(),
                "question": question[:500],
                "understood_as": (understood_as or "")[:500] or None,
                "answered": result.get("source_type") not in REFUSED_SOURCE_TYPES,
                "source_type": result.get("source_type"),
                "api": bool(result.get("api_used")),
                "cache": bool(result.get("cache_hit")),
                "seconds": round(seconds, 2),
                "files": files,
                "answer_start": str(result.get("answer") or "")[:200],
                "remote": remote,
                "api_failure": api_failure,
            },
        )
        return record_id

    def feedback(self, record_id: str, helpful: bool) -> None:
        if not self.enabled or not re.fullmatch(r"[0-9a-f]{16}", str(record_id)):
            raise ValueError("无效的记录。")
        self._append(self.feedback_path, {"id": record_id, "helpful": bool(helpful), "time": time.time()})

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as lines:
            for line in lines:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def stats(self, days: int | None = 30, minutes_per_answer: float = DEFAULT_MINUTES_SAVED_PER_ANSWER) -> dict[str, Any]:
        """Totals, a per-day series, top questions, unanswered questions, cited documents, feedback."""
        now = time.time()
        since = now - days * 86400 if days else 0.0
        with self._lock:
            questions = [item for item in self._read(self.path) if float(item.get("time") or 0) >= since]
            feedback_events = self._read(self.feedback_path)
        # The latest feedback on an answer counts (a user may change their mind).
        feedback: dict[str, bool] = {}
        for event in feedback_events:
            feedback[str(event.get("id"))] = bool(event.get("helpful"))

        answered = [item for item in questions if item.get("answered")]
        refused = [item for item in questions if not item.get("answered")]
        rated = [feedback[item["id"]] for item in questions if item.get("id") in feedback]
        seconds = sorted(float(item.get("seconds") or 0) for item in questions)

        day_count = days or max(1, min(90, int((now - min((float(i["time"]) for i in questions), default=now)) // 86400) + 1))
        first_day = datetime.fromtimestamp(now).date() - timedelta(days=day_count - 1)
        per_day: dict[str, dict[str, int]] = {
            (first_day + timedelta(days=offset)).isoformat(): {"answered": 0, "refused": 0} for offset in range(day_count)
        }
        for item in questions:
            day = local_day(float(item["time"]))
            if day in per_day:
                per_day[day]["answered" if item.get("answered") else "refused"] += 1

        def grouped(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in items:
                groups[question_key(item.get("understood_as") or item.get("question") or "")].append(item)
            rows = []
            for members in groups.values():
                latest = max(members, key=lambda item: float(item.get("time") or 0))
                rows.append(
                    {
                        "question": latest.get("understood_as") or latest.get("question"),
                        "count": len(members),
                        "answered": sum(1 for item in members if item.get("answered")),
                        "last_time": datetime.fromtimestamp(float(latest["time"])).strftime("%Y-%m-%d %H:%M"),
                    }
                )
            # Most asked first; among equals, the most recent first.
            rows.sort(key=lambda row: row["last_time"], reverse=True)
            rows.sort(key=lambda row: row["count"], reverse=True)
            return rows[:limit]

        file_counts = Counter(name for item in answered for name in item.get("files") or [])
        unhelpful = [item for item in questions if feedback.get(item.get("id")) is False]
        unhelpful.sort(key=lambda item: float(item.get("time") or 0), reverse=True)

        return {
            "ok": True,
            "days": days,
            "since": datetime.fromtimestamp(since).strftime("%Y-%m-%d") if since else None,
            "recording": self.enabled,
            "totals": {
                "questions": len(questions),
                "answered": len(answered),
                "refused": len(refused),
                "answer_rate": round(len(answered) / len(questions), 3) if questions else None,
                "followups": sum(1 for item in questions if item.get("understood_as")),
                "api_answers": sum(1 for item in answered if item.get("api")),
                "median_seconds": seconds[len(seconds) // 2] if seconds else None,
                "rated": len(rated),
                "helpful": sum(1 for value in rated if value),
                "helpful_rate": round(sum(1 for value in rated if value) / len(rated), 3) if rated else None,
                "minutes_per_answer": minutes_per_answer,
                "hours_saved": round(len(answered) * minutes_per_answer / 60, 1),
                "remote_questions": sum(1 for item in questions if item.get("remote")),
                # Answered on this computer because the answer API failed (no balance, bad key, network).
                "api_failures": sum(1 for item in questions if item.get("api_failure")),
                "api_failure_kinds": dict(Counter(item["api_failure"] for item in questions if item.get("api_failure"))),
            },
            "per_day": [{"day": day, **counts} for day, counts in per_day.items()],
            "top_questions": grouped(questions, TOP_ITEMS),
            "unanswered": grouped(refused, GAP_ITEMS),
            "top_documents": [{"file": name, "count": count} for name, count in file_counts.most_common(TOP_ITEMS)],
            "unhelpful": [
                {
                    "question": item.get("understood_as") or item.get("question"),
                    "answer_start": item.get("answer_start"),
                    "time": datetime.fromtimestamp(float(item["time"])).strftime("%Y-%m-%d %H:%M"),
                }
                for item in unhelpful[:TOP_ITEMS]
            ],
        }
