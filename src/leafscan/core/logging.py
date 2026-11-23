"""
src/leafscan/core/logging.py
Structured JSON logging for LeafScan V3.
"""

import sys
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict


class JSONLogFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }

        # Include custom attributes if passed in extra
        for key in ("request_id", "endpoint", "status_code", "duration_ms", "model_bundle", "safety_status", "error_code"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            log_entry["exception"] = record.exc_text

        return json.dumps(log_entry, default=str)


def setup_logging(level: str = "INFO", json_format: bool = True) -> logging.Logger:
    """Configures root LeafScan logger."""
    logger = logging.getLogger("leafscan")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JSONLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = logging.getLogger("leafscan")
