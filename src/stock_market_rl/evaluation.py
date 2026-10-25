from __future__ import annotations

import numpy as np


def compute_metrics(prices: np.ndarray | list[float], initial_cash: float = 10_000.0) -> dict[str, float]:
    series = np.asarray(prices, dtype=float)
    if series.size < 2:
        raise ValueError("At least two price points are required to compute metrics.")

    total_return = float((series[-1] / series[0]) - 1.0)
    buy_hold_value = initial_cash * (1.0 + total_return)
    daily_returns = np.diff(series) / np.maximum(np.abs(series[:-1]), 1e-8)
    annualized = float((1.0 + total_return) ** (1.0 / max(len(series) / 252.0, 1e-6)) - 1.0)

    return {
        "total_return": float(total_return),
        "buy_hold_value": float(buy_hold_value),
        "annualized_return": annualized,
        "mean_daily_return": float(np.mean(daily_returns)),
        "volatility": float(np.std(daily_returns)),
    }
