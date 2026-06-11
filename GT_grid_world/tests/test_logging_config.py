"""Unit tests for ``GT_grid_world.src.logging_config``.

These tests live alongside the rest of the ``GT_grid_world`` test suite.
They exercise the small surface area of ``setup_logging`` / ``get_logger``
and document the contract: idempotent setup, per-logger overrides, level
coercion from strings.
"""

from __future__ import annotations

import io
import logging

import pytest

from GT_grid_world.src.logging_config import (
    ROOT_LOGGER_NAME,
    get_logger,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _reset_logging():
    """Drop ``m2m`` handlers and level between tests so order doesn't matter."""
    yield
    root = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.NOTSET)


def test_get_logger_prepends_namespace():
    logger = get_logger("simulate")

    assert logger.name == f"{ROOT_LOGGER_NAME}.simulate"


def test_get_logger_accepts_already_namespaced_name():
    logger = get_logger("m2m.path")

    assert logger.name == "m2m.path"


def test_setup_logging_attaches_exactly_one_handler():
    setup_logging("INFO")

    handlers = logging.getLogger(ROOT_LOGGER_NAME).handlers
    assert len(handlers) == 1


def test_setup_logging_is_idempotent():
    setup_logging("INFO")
    setup_logging("DEBUG")
    setup_logging("WARNING")

    handlers = logging.getLogger(ROOT_LOGGER_NAME).handlers
    assert len(handlers) == 1


def test_setup_logging_emits_messages():
    stream = io.StringIO()
    setup_logging("INFO", stream=stream)

    get_logger("crg").info("hello world")

    output = stream.getvalue()
    assert "hello world" in output
    assert "m2m.crg" in output
    assert "INFO" in output


def test_setup_logging_does_not_propagate_to_root():
    stream = io.StringIO()
    setup_logging("INFO", stream=stream)

    assert logging.getLogger(ROOT_LOGGER_NAME).propagate is False


def test_setup_logging_per_logger_overrides_silence_subsystem():
    stream = io.StringIO()
    setup_logging("INFO", stream=stream, per_logger_levels={"path": "WARNING"})

    get_logger("path").info("should be silent")
    get_logger("path").warning("should be visible")

    output = stream.getvalue()
    assert "should be silent" not in output
    assert "should be visible" in output


def test_setup_logging_rejects_unknown_level_string():
    with pytest.raises(ValueError, match="Unknown log level"):
        setup_logging("CHATTY")


def test_setup_logging_accepts_int_level():
    setup_logging(logging.DEBUG)

    assert logging.getLogger(ROOT_LOGGER_NAME).level == logging.DEBUG
