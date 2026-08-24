"""
Centralized Unified Logging Engine for SkyGuard UAV.
Provides colored console output, rotating structured JSON/text file logs,
thread-safe idempotent initialization, and automated sensitive credential redaction.
"""

import os
import re
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from config.settings import settings

# Sensitive credential redaction patterns
REDACTION_PATTERNS = [
    (re.compile(r"0x[a-fA-F0-9]{64}"), "[REDACTED_PRIVATE_KEY]"),
    (re.compile(r"eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}"), "[REDACTED_JWT]"),
    (re.compile(r"(api[-_]?key|secret[-_]?key|password|jwt)[:=]\s*['\"]?[^\s'\"]+", re.IGNORECASE), r"\1=[REDACTED_SECRET]"),
]


class SecretRedactionFilter(logging.Filter):
    """Filters log messages and redacts private keys, JWTs, and API credentials."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            msg = record.msg
            for pattern, replacement in REDACTION_PATTERNS:
                msg = pattern.sub(replacement, msg)
            record.msg = msg
        return True


class ColoredConsoleFormatter(logging.Formatter):
    """Color-coded console formatter for clean CLI output."""

    COLOR_CODES = {
        logging.DEBUG: "\033[36m",    # Cyan
        logging.INFO: "\033[32m",     # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",    # Red
        logging.CRITICAL: "\033[35m"  # Magenta
    }
    RESET_CODE = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLOR_CODES.get(record.levelno, self.RESET_CODE)
        levelname = f"[{record.levelname}]"
        name = f"[{record.name}]"
        msg = record.getMessage()
        time_str = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        return f"{time_str} {color}{levelname:<9}{self.RESET_CODE} {name} {msg}"


class JSONFormatter(logging.Formatter):
    """Formats log records as structured JSON for file storage and audit trails."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_data.update(record.extra_data)
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def setup_logger(
    name: str = "skyguard",
    log_file: Optional[str] = None,
    level: Optional[int] = None
) -> logging.Logger:
    """
    Authoritative Logger Factory for SkyGuard UAV.
    Configures console and rotating file handlers with secret redaction and idempotency.
    """
    logger = logging.getLogger(name)

    # Determine log level
    if level is None:
        level_str = getattr(settings, "LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_str, logging.INFO)

    logger.setLevel(level)

    # Idempotent guard: if handlers already configured on this logger, do not duplicate
    if logger.handlers:
        return logger

    # Attach secret redaction filter to logger
    redaction_filter = SecretRedactionFilter()
    logger.addFilter(redaction_filter)

    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredConsoleFormatter())
    console_handler.addFilter(redaction_filter)
    logger.addHandler(console_handler)

    # 2. Rotating File Handler
    log_dir = settings.BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_name = log_file or "skyguard_uav.log"
    file_path = log_dir / Path(file_name).name

    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(JSONFormatter())
    file_handler.addFilter(redaction_filter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Convenience accessor for child loggers."""
    return setup_logger(name)


# Global Application Default Logger
app_logger = setup_logger("skyguard", "skyguard_uav.log")
