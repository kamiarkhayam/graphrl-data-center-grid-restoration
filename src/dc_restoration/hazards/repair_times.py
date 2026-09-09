from __future__ import annotations

import numpy as np


def sample_conductor_repair_times(
    damaged_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    damaged_mask = np.asarray(damaged_mask, dtype=bool)
    repair_times = np.zeros(damaged_mask.shape[0], dtype=float)
    repair_times[damaged_mask] = rng.uniform(3.0, 5.0, size=int(damaged_mask.sum()))
    return repair_times


def sample_substation_repair_times(
    severity_states: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    severity_states = np.asarray(severity_states, dtype=object)
    repair_times = np.zeros(severity_states.shape[0], dtype=float)
    params = {
        "Severe": (7 * 24, 3.5 * 24),
        "Complete": (30 * 24, 15 * 24),
    }
    for state, (mean_val, std_val) in params.items():
        mask = severity_states == state
        if mask.any():
            samples = rng.normal(loc=mean_val, scale=std_val, size=int(mask.sum()))
            repair_times[mask] = np.maximum(samples, 0.0)
    return repair_times
