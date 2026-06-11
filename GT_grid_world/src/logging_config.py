"""Project-wide logging configuration for the M2M proactive rearrangement work.

Convention
----------
Use one named logger per subsystem so that levels can be tuned independently
without hunting through code. The conventional names are:

    m2m.crg              Case Request Generator and task arrival logic
    m2m.alloc            Initial allocation (greedy, fast_greedy, FCF, ...)
    m2m.lns              LNS improvement (Shaw removal, repair, acceptance)
    m2m.path             Path planning (PBS, EECBS, pybind11 boundary)
    m2m.simulate         Per-timestep agent state machine, pickup/dropoff
    m2m.repair           Solution repair detection and BnB/HA repair
    m2m.cost             Cost tensor construction
    m2m.experiment       Experiment runner / config / I/O

Usage
-----
At the top of any module:

    from .logging_config import get_logger
    logger = get_logger("simulate")    # becomes m2m.simulate

    logger.info("Allocated %d tasks across %d agents in %.2fs",
                len(allocations), len(agents), elapsed)

At program entry (GT_grid_world.py main):

    from src.logging_config import setup_logging
    setup_logging(level="INFO")        # or read from CLI

Design notes
------------
- We use the stdlib ``logging`` module rather than a third-party structured
  logger. ``structlog``/``loguru`` are nicer ergonomically but add a dependency
  and convention-shift for collaborators (Ethan, Symbotic engineers) who are
  used to vanilla logging.
- ``setup_logging`` is idempotent so it is safe to call from tests, notebooks,
  or repeated CLI runs without doubling up handlers.
- Per-logger level overrides let us silence chatty subsystems (e.g. set
  ``m2m.path`` to WARNING when stepping through allocation logic).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Mapping, Optional

ROOT_LOGGER_NAME = "m2m"
DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"


def get_logger(subsystem: str) -> logging.Logger:
    """Return the named logger for a subsystem (e.g. ``simulate`` -> ``m2m.simulate``).

    Subsystem names that already start with ``m2m.`` are returned unchanged.
    """
    if subsystem.startswith(f"{ROOT_LOGGER_NAME}."):
        name = subsystem
    else:
        name = f"{ROOT_LOGGER_NAME}.{subsystem}"
    return logging.getLogger(name)


def setup_logging(
    level: str = "INFO",
    *,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
    stream=sys.stderr,
    per_logger_levels: Optional[Mapping[str, str]] = None,
) -> logging.Logger:
    """Configure the ``m2m.*`` logger tree once for the whole process.

    Parameters
    ----------
    level
        Default level for the ``m2m`` root logger. Accepts standard names
        (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
    fmt, datefmt
        Format strings passed to ``logging.Formatter``.
    stream
        Output stream. Defaults to stderr so that stdout stays usable for
        machine-readable simulation output.
    per_logger_levels
        Optional mapping of subsystem names to per-logger levels, e.g.
        ``{"path": "WARNING", "simulate": "DEBUG"}``. Names are normalised
        through :func:`get_logger`.

    Returns
    -------
    logging.Logger
        The configured ``m2m`` root logger.

    Notes
    -----
    Idempotent: calling more than once does not stack handlers. Existing
    handlers on the ``m2m`` logger are removed and replaced.
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(handler)
    root.setLevel(_coerce_level(level))
    # Don't bubble to the root Python logger; that would duplicate output if
    # the application or a notebook also configured logging.
    root.propagate = False

    if per_logger_levels:
        for name, lvl in per_logger_levels.items():
            get_logger(name).setLevel(_coerce_level(lvl))

    return root


def _coerce_level(level) -> int:
    """Accept either an int level or a name like ``"DEBUG"``."""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        normalised = level.upper()
        if normalised not in logging._nameToLevel:
            raise ValueError(
                f"Unknown log level {level!r}; expected one of "
                f"{sorted(logging._nameToLevel)}"
            )
        return logging._nameToLevel[normalised]
    raise TypeError(f"level must be int or str, got {type(level).__name__}")


def setup_logging_from_env(default: str = "INFO") -> logging.Logger:
    """Convenience entry point: read level from ``M2M_LOG_LEVEL`` env var.

    Useful for tests and notebooks where editing the CLI is awkward.
    """
    return setup_logging(level=os.environ.get("M2M_LOG_LEVEL", default))
