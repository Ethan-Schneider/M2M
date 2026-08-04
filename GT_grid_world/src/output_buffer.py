"""Shared capacity buffer for outbound (driveway) output nodes.

Ported from Ethan's ``task_queue`` branch (commit 58bb27f). Behaviour is
identical; the only change from the source is that the debug ``print``
statements inside ``consumption_tick`` were removed (they fired every tick and
would flood the simulation log) and a module logger is used instead.

Model: a single shared pool drains ``consumption_rate_per_min / 60`` items per
simulation tick (1 tick == 1 second). Each completed *outbound* driveway
delivery adds one item via ``record_outbound_delivery``. When the pool is at
capacity, outbound deliveries are blocked (``outbound_delivery_blocked``) and
the delivering agent waits, creating backpressure that caps outbound
throughput at the consumption rate. Inbound and shuffle tasks never touch the
buffer.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .logging_config import get_logger

_log = get_logger("output_buffer")

TASK_TYPE_OUTBOUND = 0
QUEUE_TASK_TYPE = 1
QUEUE_DEADLINE = 2


def tasks_per_min_to_per_tick(tasks_per_min: float) -> float:
    """Convert tasks/min to tasks per simulation tick (1 tick = 1 second)."""
    return tasks_per_min / 60.0


class OutputBuffer:
    """Pool of buffer slots shared across all outbound output nodes.

    ``consumption_rate_per_min`` is set directly as a hyperparameter. Each
    simulation tick drains ``consumption_rate_per_min / 60`` items from the
    buffer. Outbound driveway deliveries increment ``level`` via
    ``record_outbound_delivery``.
    """

    def __init__(
        self,
        capacity: int,
        consumption_rate_per_min: float,
        *,
        level: float = 0.0,
    ) -> None:
        if capacity < 0:
            raise ValueError(f"capacity must be non-negative, got {capacity}")
        if consumption_rate_per_min < 0:
            raise ValueError(
                f"consumption_rate_per_min must be non-negative, got {consumption_rate_per_min}"
            )
        if level < 0 or level > capacity:
            raise ValueError(f"level must be in [0, {capacity}], got {level}")

        self.capacity = capacity
        self.consumption_rate_per_min = float(consumption_rate_per_min)
        self.consumption_rate_per_tick = tasks_per_min_to_per_tick(consumption_rate_per_min)
        self.level = float(level)

    @property
    def consumption_rate(self) -> float:
        """Drain rate in items per simulation tick."""
        return self.consumption_rate_per_tick

    @property
    def available_slots(self) -> float:
        return max(0.0, self.capacity - self.level)

    @property
    def is_at_capacity(self) -> bool:
        return self.level >= self.capacity

    def can_accept(self, amount: float = 1.0) -> bool:
        return amount >= 0 and self.level + amount <= self.capacity

    def record_outbound_delivery(self, amount: float = 1.0) -> None:
        """Increase buffer occupancy when an outbound item is delivered."""
        if amount <= 0:
            raise ValueError(f"amount must be positive, got {amount}")
        if not self.can_accept(amount):
            raise ValueError(
                f"cannot record outbound delivery of {amount}: "
                f"level={self.level}, capacity={self.capacity}"
            )
        self.level += amount

    def consumption_tick(self) -> float:
        """Drain the buffer by one tick of consumption. Returns amount removed."""
        drained = min(self.level, self.consumption_rate_per_tick)
        self.level = max(0.0, self.level - self.consumption_rate_per_tick)
        _log.debug(
            "consumption_tick: drained=%.4f level=%.4f (rate/tick=%.4f)",
            drained, self.level, self.consumption_rate_per_tick,
        )
        return drained

    @staticmethod
    def estimate_outbound_fraction(queue: NDArray, W: int) -> float:
        """Fraction of the next ``W`` queue rows that are outbound tasks.

        Works with the timeless demand-driven queue format ``[sku_id, task_type]``
        (no deadline / arrival-time column): it only reads the task-type column,
        so unlike ``estimate_arrival_rate`` it does not require temporal data.
        Returns a value in ``[0, 1]``; ``0.0`` for an empty queue.
        """
        if queue is None or queue.size == 0:
            return 0.0
        window = np.atleast_2d(queue[:W])
        if window.shape[1] <= QUEUE_TASK_TYPE:
            return 0.0
        n = window.shape[0]
        if n == 0:
            return 0.0
        outbound_count = int((window[:, QUEUE_TASK_TYPE] == TASK_TYPE_OUTBOUND).sum())
        return outbound_count / n

    @staticmethod
    def estimate_arrival_rate(queue: NDArray, W: int) -> float:
        """Estimate outbound arrival rate (tasks/min) from the first ``W`` queue rows."""
        if queue is None or queue.size == 0:
            return 0.0

        window = np.atleast_2d(queue[:W])
        if window.shape[1] <= QUEUE_DEADLINE:
            raise ValueError(
                "queue must include a deadline column for arrival rate estimation"
            )

        outbound_count = int((window[:, QUEUE_TASK_TYPE] == TASK_TYPE_OUTBOUND).sum())
        if outbound_count == 0:
            return 0.0

        deadlines = window[:, QUEUE_DEADLINE].astype(float)
        span_seconds = max(float(deadlines.max() - deadlines.min()), 1.0)
        span_minutes = span_seconds / 60.0
        return outbound_count / span_minutes


def consumption_tick(output_buffer: Optional[OutputBuffer]) -> None:
    """Apply one simulation timestep of shared output-buffer consumption."""
    if output_buffer is not None:
        output_buffer.consumption_tick()


def outbound_delivery_blocked(
    output_buffer: Optional[OutputBuffer],
    amount: float = 1.0,
) -> bool:
    """Return True when an outbound delivery of ``amount`` cannot enter the buffer."""
    if output_buffer is None:
        return False
    return not output_buffer.can_accept(amount)
