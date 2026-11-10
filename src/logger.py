"""Centralized logging for hotel-oversell.

Usage in any module:

    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Reading %s", path)

Behaviour:
- One consistent format: timestamp | severity | module | message
- Logs to the console and to a rotating file under logs/
- Level comes from the LOG_LEVEL environment variable (default INFO)
- Obvious secrets (tokens, passwords, AWS keys) are redacted from messages

Conventions:
- INFO    stage progress, input/output paths, row counts, date ranges
- WARNING recoverable data-quality or configuration concerns
- ERROR   failed stages (use logger.exception(...) inside except blocks)
Do not log raw guest records or any credentials.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "hotel_oversell.log"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

ROOT_LOGGER_NAME = "hotel_oversell"

_REDACTIONS = [
    # key=value or key: value for common secret names
    (re.compile(r"(?i)\b(token|secret|password|passwd|api[_-]?key|access[_-]?key)\b(\s*[=:]\s*)\S+"),
     r"\1\2***REDACTED***"),
    # AWS access key ids
    (re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b"), "***REDACTED***"),
]


class RedactSecretsFilter(logging.Filter):
    """Masks obvious secrets in log messages before they are written."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = message
        for pattern, replacement in _REDACTIONS:
            redacted = pattern.sub(replacement, redacted)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


_configured = False


def _configure() -> None:
    """Set up handlers on the project root logger once."""
    global _configured
    if _configured:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(level)
    root.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    redact = RedactSecretsFilter()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.addFilter(redact)
    root.addHandler(console)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redact)
        root.addHandler(file_handler)
    except OSError as exc:  # e.g. read-only filesystem in CI
        root.warning("File logging disabled (%s); using console only", exc)

    _configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger that uses the shared configuration.

    Pass __name__ from the calling module. Names such as "src.validate" are
    attached under the project root logger, so they share its handlers.
    """
    _configure()
    if not name or name == "__main__":
        return logging.getLogger(ROOT_LOGGER_NAME)
    if name.startswith(ROOT_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def log_dataframe_summary(
    logger: logging.Logger,
    df,
    label: str,
    date_col: Optional[str] = None,
) -> None:
    """Log row/column counts and, if given, the date range of a DataFrame.

    Logs counts and ranges only, never row contents.
    """
    logger.info("%s: %d rows, %d columns", label, len(df), df.shape[1])
    if date_col and date_col in df.columns:
        try:
            import pandas as pd

            dates = pd.to_datetime(df[date_col], errors="coerce")
            if dates.notna().any():
                logger.info("%s: %s range %s to %s", label, date_col,
                            dates.min().date(), dates.max().date())
            else:
                logger.warning("%s: no valid dates found in %s", label, date_col)
        except Exception:
            logger.warning("%s: could not compute date range for %s", label, date_col)
