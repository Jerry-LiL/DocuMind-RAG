"""Token and latency records, one JSON object per line."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from documind.config import Settings

logger = logging.getLogger("documind.usage")


def configure_logging() -> None:
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def log_usage(settings: Settings, record: dict) -> None:
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        **record,
    }
    line = json.dumps(payload, ensure_ascii=False)
    logger.info(line)
    path = settings.log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
