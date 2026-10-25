from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .agent import QTradingAgent
from .config import TradingConfig
from .data import load_price_data, split_train_test
from .evaluation import compute_metrics


def train_and_evaluate(config: TradingConfig | None = None) -> dict[str, object]:
    config = config or TradingConfig()
    df = load_price_data(config.ticker, config.start_date, config.end_date, cache=True)
    train_df, test_df = split_train_test(df, config.train_ratio)

    train_prices = train_df["close"].to_numpy(dtype=float)
    test_prices = test_df["close"].to_numpy(dtype=float)

    agent = QTradingAgent(
        window_size=config.window_size,
        action_space=config.action_space,
        learning_rate=config.learning_rate,
        discount=config.discount,
        epsilon=config.epsilon,
        epsilon_min=config.epsilon_min,
        epsilon_decay=config.epsilon_decay,
        random_seed=config.random_seed,
    )

    training_summary = agent.train(train_prices, episodes=config.episodes, initial_cash=config.initial_cash, transaction_cost=config.transaction_cost)
    evaluation_summary = agent.evaluate_strategy(test_prices, initial_cash=config.initial_cash, transaction_cost=config.transaction_cost)
    baseline_summary = compute_metrics(test_prices, initial_cash=config.initial_cash)

    result = {
        "ticker": config.ticker,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "training_summary": training_summary,
        "strategy_summary": evaluation_summary,
        "baseline_summary": baseline_summary,
    }

    metrics_dir = config.metrics_dir
    metrics_path = metrics_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    plot_path = config.plots_dir / f"{config.ticker.lower()}_strategy.png"
    plt.figure(figsize=(12, 5))
    plt.plot(test_prices, label="Close price", color="tab:blue")
    plt.title(f"{config.ticker} test-period strategy")
    plt.xlabel("Time step")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return result
