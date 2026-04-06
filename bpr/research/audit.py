"""Append-only audit logging for research loop activity."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any
from uuid import uuid4


def _event_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AuditEvent:
    """One append-only research log event."""

    event_type: str
    summary: str
    prediction_ids: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_event_timestamp)
    event_id: str = ""

    def __post_init__(self) -> None:
        if not self.event_id:
            digest = sha1(
                (
                    f"{self.timestamp}|{self.event_type}|{self.summary}|"
                    f"{uuid4().hex}"
                ).encode("utf-8")
            ).hexdigest()[:12]
            object.__setattr__(self, "event_id", f"log-{digest}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "summary": self.summary,
            "prediction_ids": list(self.prediction_ids),
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AuditEvent":
        return cls(
            event_type=payload["event_type"],
            summary=payload["summary"],
            prediction_ids=list(payload.get("prediction_ids", [])),
            payload=dict(payload.get("payload", {})),
            timestamp=payload.get("timestamp", _event_timestamp()),
            event_id=payload.get("event_id", ""),
        )


def append_audit_event(path: Path, event: AuditEvent) -> None:
    """Append a JSONL audit event without mutating prior entries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")


def load_audit_events(path: Path) -> list[AuditEvent]:
    """Load the append-only event stream from disk."""
    if not path.exists():
        return []
    events: list[AuditEvent] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            events.append(AuditEvent.from_dict(json.loads(line)))
    return events
