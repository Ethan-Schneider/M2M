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
