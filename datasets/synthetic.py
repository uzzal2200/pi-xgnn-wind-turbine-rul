"""Synthetic bearing data generator for demo / CI testing."""
import numpy as np


def generate_bearing(T: int = 120,
                      n_samples: int = 2560,
                      seed: int = 42) -> tuple:
    """Simulate one run-to-failure bearing with realistic degradation.

    Returns
    -------
    cycles_h : np.ndarray  (T, n_samples)
    cycles_v : np.ndarray  (T, n_samples)
    """
    rng = np.random.default_rng(seed)

    cycles_h, cycles_v = [], []
    for t in range(T):
        deg  = (t / T) ** 1.5                         # accelerating degradation
        amp  = 1.0 + 6.0 * deg                        # amplitude growth
        freq = 1.0 + 2.0 * deg                        # spectral shift

        t_axis = np.linspace(0, 1, n_samples)
        base   = amp * np.sin(2 * np.pi * freq * t_axis)
        noise  = rng.normal(0, 0.5 * amp, n_samples)
        impulse = rng.standard_t(max(2.5, 20 * (1 - deg)), n_samples) * deg * 2

        cycles_h.append((base + noise + impulse).astype(np.float32))
        cycles_v.append(
            (0.8 * amp * np.cos(2 * np.pi * freq * t_axis)
             + rng.normal(0, 0.4 * amp, n_samples)).astype(np.float32)
        )

    return np.array(cycles_h), np.array(cycles_v)
