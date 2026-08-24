"""
Legacy logger compatibility layer.
Redirects all logging requests to the canonical authoritative config.logging_config engine.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import setup_logger, get_logger, app_logger, ColoredConsoleFormatter, JSONFormatter

__all__ = ["setup_logger", "get_logger", "app_logger", "ColoredConsoleFormatter", "JSONFormatter"]
