from __future__ import annotations

import argparse
import json

from .config import TradingConfig
from .training import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a simple stock-trading Q-learning agent.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TradingConfig(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        episodes=args.episodes,
        window_size=args.window_size,
        initial_cash=args.initial_cash,
        train_ratio=args.train_ratio,
    )
    result = train_and_evaluate(config)
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
