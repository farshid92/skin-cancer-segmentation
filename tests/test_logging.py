"""Tests for project logging helpers."""

import logging

from src.utils.logging import get_logger


def test_get_logger_is_configured_once() -> None:
    """Logger setup is idempotent and uses a stream handler."""
    logger = get_logger("tests.logging")

    assert logger is get_logger("tests.logging")
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
