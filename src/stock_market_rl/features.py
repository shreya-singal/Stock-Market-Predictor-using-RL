from __future__ import annotations

import numpy as np


def build_state_vector(prices: np.ndarray | list[float], window_size: int) -> np.ndarray:
    series = np.asarray(prices, dtype=float)
    if series.size < window_size + 1:
        raise ValueError(f"Need at least {window_size + 1} points to build a state vector with window_size={window_size}.")
    window = series[-(window_size + 1) :]
    diffs = np.diff(window) / np.maximum(np.abs(window[:-1]), 1e-6)
    return diffs.astype(float)


def build_state_history(prices: np.ndarray | list[float], window_size: int) -> list[np.ndarray]:
    series = np.asarray(prices, dtype=float)
    states = []
    for idx in range(window_size, len(series)):
        state = build_state_vector(series[: idx + 1], window_size)
        states.append(state)
    return states
