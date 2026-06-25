import numpy as np
from numpy.typing import NDArray

### Functions for how tasks are generated (i.e. comes in waves (sinusoidal) or constant, etc.)

def constant(frequency : float, total_timesteps : int) -> NDArray:
    if frequency > 1:
        frequency = 1.0
    elif frequency < 0:
        frequency = 0.0
    else:
        pass
    return np.full(total_timesteps, frequency, dtype=float)


def sigmoid(k : float, t0 : int, inverse : int, min_value : float, max_value : float, total_timesteps : int) -> NDArray:
    t = np.arange(total_timesteps)
    frequency = (1 / (1 + np.exp(-k * (t - t0))))
    if inverse:
        frequency = 1 - frequency
    frequency = np.asarray(frequency).clip(min_value, max_value)
    return frequency.astype(float)


def sinusoid(amplitude : float, frequency : float, phase : float, vertical_shift : float, total_timesteps : int) -> NDArray:
    t = np.arange(total_timesteps)
    # interpret `frequency` as the number of complete cycles across the whole sequence
    omega = 2 * np.pi * frequency * t / total_timesteps
    frequency = amplitude * np.sin(omega + phase) + vertical_shift
    frequency = frequency.astype(float).clip(0.0, 1.0)
    return frequency.astype(float)


def poisson(lam : float, total_timesteps : int) -> NDArray:
    ts = np.random.poisson(lam=lam, size=total_timesteps).astype(float)
    ts_min = ts.min()
    ts_max = ts.max()
    if ts_max > ts_min:
        ts = (ts - ts_min) / (ts_max - ts_min)
    else:
        ts = np.zeros_like(ts)
    return ts.astype(float)


def constant_weight(value: float, total_timesteps: int) -> NDArray:
    """Return a constant tasking weight for every task index."""
    return np.full(total_timesteps, max(float(value), 1e-6), dtype=float)


def sinusoid_weight(
    min_weight: float,
    max_weight: float,
    period_tasks: float,
    phase: float,
    total_timesteps: int,
) -> NDArray:
    """Return tasking weights that oscillate between ``min_weight`` and ``max_weight``.

    ``period_tasks`` is the number of queue indices per full cycle and does not
    depend on ``total_timesteps``. Generating fewer tasks simply truncates the
    curve rather than compressing more cycles into the queue.
    """
    if period_tasks <= 0:
        raise ValueError(f"period_tasks must be positive, got {period_tasks}")

    lo = float(min_weight)
    hi = float(max_weight)
    if hi < lo:
        lo, hi = hi, lo

    t = np.arange(total_timesteps, dtype=float)
    omega = 2 * np.pi * t / float(period_tasks)
    sine = np.sin(omega + phase)
    weights = lo + (hi - lo) * (sine + 1.0) / 2.0
    return np.clip(weights, 1e-6, None).astype(float)
