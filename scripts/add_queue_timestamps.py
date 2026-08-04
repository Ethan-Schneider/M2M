#!/usr/bin/env python3
"""Stamp an arrival-time column onto a precomputed task queue.

The 2-column queue format ``[sku_id, task_type]`` carries no clock: release is
demand-driven, so the file is only an *ordering* of tasks. The buffer-aware
crM2M slack model needs an outbound arrival rate ``alpha``, which
``OutputBuffer.estimate_arrival_rate`` derives from a per-row timestamp column.

This utility appends that third column using an evenly-spaced arrival model at a
chosen total task rate ``R`` (tasks/min), matching Ethan's task_queue format:

    timestamp(i) = ceil((i + 1) * 60 / R)   # 1-based arrival second

At ``R = 60`` this is simply ``1, 2, 3, ...`` (one task per second). Because the
estimator counts outbound tasks over the window's time span, the effective
outbound arrival rate seen by the buffer is ``outbound_fraction * R`` -- so ``R``
sets the offered load independently of the buffer drain rate ``mu``.

Columns 0 (sku_id) and 1 (task_type) are preserved exactly, so an augmented
queue stays byte-for-byte comparable in task content to the original and remains
readable by 2-column consumers (they ignore the extra column).
"""

from __future__ import annotations

import argparse

import numpy as np


def assign_arrival_seconds(num_tasks: int, rate_per_min: float) -> np.ndarray:
    """1-based arrival second for each 0-based queue index at ``rate_per_min``."""
    if rate_per_min <= 0:
        raise ValueError(f"rate_per_min must be positive, got {rate_per_min}")
    indices = np.arange(num_tasks)
    return np.ceil((indices + 1) * 60.0 / rate_per_min).astype(int)


def stamp_queue(src: str, dst: str, rate_per_min: float) -> None:
    queue = np.atleast_2d(np.loadtxt(src, dtype=int))
    if queue.size == 0:
        raise ValueError(f"queue {src} is empty")

    sku_type = queue[:, :2]  # preserve sku_id, task_type exactly
    timestamps = assign_arrival_seconds(queue.shape[0], rate_per_min)
    stamped = np.column_stack([sku_type, timestamps])
    np.savetxt(dst, stamped, fmt="%d")
    print(
        f"stamped {queue.shape[0]} rows at R={rate_per_min:g} tasks/min "
        f"({src} -> {dst})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue_file", help="path to the 2-column queue .txt")
    parser.add_argument(
        "--rate", type=float, default=60.0,
        help="total task arrival rate R in tasks/min (default: 60 = 1/sec)",
    )
    parser.add_argument(
        "--out", default=None,
        help="output path (default: overwrite the input queue in place)",
    )
    args = parser.parse_args()
    stamp_queue(args.queue_file, args.out or args.queue_file, args.rate)


if __name__ == "__main__":
    main()
