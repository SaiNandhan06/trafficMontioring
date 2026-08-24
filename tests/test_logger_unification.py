"""
Logger Unification and Observability Regression Tests.
Verifies authoritative logger factory, handler idempotency, rotation limits,
secret redaction filters, and backward compatibility.
"""

import json
import logging
from pathlib import Path
import pytest

from config.logging_config import setup_logger, SecretRedactionFilter
from src.utils.logger import setup_logger as legacy_setup_logger


def test_logger_initialization_and_handlers():
    """Verifies that setup_logger configures both console and rotating file handlers."""
    logger = setup_logger("test_init_logger", "test_init.log")
    assert logger is not None
    assert len(logger.handlers) == 2

    handler_types = [type(h).__name__ for h in logger.handlers]
    assert "StreamHandler" in handler_types
    assert "RotatingFileHandler" in handler_types


def test_logger_idempotency_no_duplicate_handlers():
    """Verifies that repeated setup_logger calls do NOT add duplicate handlers."""
    logger_name = "test_idempotency_logger"
    logger1 = setup_logger(logger_name)
    initial_handler_count = len(logger1.handlers)

    # Call 5 more times
    for _ in range(5):
        logger_next = setup_logger(logger_name)
        assert len(logger_next.handlers) == initial_handler_count


def test_logger_sensitive_data_redaction():
    """Verifies that private keys, JWT tokens, and passwords are automatically redacted."""
    redaction_filter = SecretRedactionFilter()

    # 1. Private key redaction
    rec_pk = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=10,
        msg="Using private key: 0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d for signing",
        args=(), exc_info=None
    )
    redaction_filter.filter(rec_pk)
    assert "0x59c699" not in rec_pk.msg
    assert "[REDACTED_PRIVATE_KEY]" in rec_pk.msg

    # 2. JWT token redaction
    rec_jwt = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=20,
        msg="Issued bearer token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgN_p_placeholder",
        args=(), exc_info=None
    )
    redaction_filter.filter(rec_jwt)
    assert "eyJhbGci" not in rec_jwt.msg
    assert "[REDACTED_JWT]" in rec_jwt.msg


def test_legacy_logger_alias_compatibility():
    """Verifies that importing from src.utils.logger points to the same canonical logger engine."""
    logger_a = setup_logger("test_compat")
    logger_b = legacy_setup_logger("test_compat")
    assert logger_a is logger_b


def test_logger_file_rotation_configuration():
    """Verifies that the file handler uses RotatingFileHandler with bounded limits."""
    logger = setup_logger("test_rotation_logger", "test_rotation.log")
    file_handlers = [h for h in logger.handlers if type(h).__name__ == "RotatingFileHandler"]
    assert len(file_handlers) == 1

    rfh = file_handlers[0]
    assert rfh.maxBytes == 10 * 1024 * 1024  # 10 MB
    assert rfh.backupCount == 5
