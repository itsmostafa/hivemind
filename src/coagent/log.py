import json
import logging
import os
from datetime import datetime, timezone
from typing import Any


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with clean console format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


class TraceLogger:
    """Writes structured JSONL events to a trace file."""

    def __init__(self, trace_file: str) -> None:
        self.trace_file = trace_file
        # Ensure parent directory exists
        parent = os.path.dirname(trace_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._file = open(trace_file, "a", encoding="utf-8")

    def log(self, event: str, **data: Any) -> None:
        """Write a structured event to the trace file."""
        record = {"event": event, "ts": datetime.now(timezone.utc).isoformat(), **data}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the trace file."""
        self._file.close()

    def __enter__(self) -> "TraceLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class NullTraceLogger:
    """No-op trace logger used when no trace file is configured."""

    def log(self, event: str, **data: Any) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "NullTraceLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        pass
