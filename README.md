# Stock Market RL

A compact reinforcement-learning project for trading a single stock using a Q-learning style policy. The code is intentionally small and auditable, with a reproducible pipeline that downloads data, trains on a past segment, evaluates on a later segment, and writes out metrics and a plot.

## Why this project exists

This repository explores a simple trading-agent setup in a controlled way. It is meant to demonstrate:

- how market data can be loaded and validated,
- how a trading problem can be expressed as a state/action/reward loop,
- how a basic Q-learning agent behaves on price data,
- how to evaluate a strategy against a buy-and-hold baseline.

It is a prototype and research baseline, not a production trading system or financial recommendation engine.

## What is implemented

- Historical market data download from Yahoo Finance with local cache support
- Time-based train/test split to reduce leakage risk
- Rolling price-difference feature construction
- Q-learning style trading policy with discrete actions
- Evaluation metrics and baseline comparison
- CLI entry point and JSON metric output
- Plot generation under the results directory
- Automated smoke tests for core logic

## Methodology

### State representation

The agent uses a rolling vector derived from recent point-to-point price changes:

$$
\Delta p_i = \frac{p_i - p_{i-1}}{\max(|p_{i-1}|, \epsilon)}
$$

This gives a compact representation of short-term trend behavior.

### Action space

The environment supports three actions:

- 0 = hold
- 1 = buy
- 2 = sell

### Q-learning update

The agent updates its Q-values using the Bellman equation:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma\max_{a'}Q(s',a') - Q(s,a)\right]
$$

where $\alpha$ is the learning rate and $\gamma$ is the discount factor.

### Evaluation

The pipeline uses a chronological split:

1. Download historical price data.
2. Split into train and test periods.
3. Train on the earlier section.
4. Evaluate on the later out-of-sample section.
5. Compare results against a buy-and-hold baseline.

## Repository structure

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── pytest.ini
├── src/
│   └── stock_market_rl/
│       ├── __init__.py
│       ├── __main__.py
│       ├── agent.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── features.py
│       ├── pipeline.py
│       └── training.py
├── tests/
│   └── test_pipeline.py
├── data/
│   └── raw/
├── results/
│   ├── metrics/
│   └── plots/
└── stock_market_predict.ipynb
```

## Installation

### Prerequisites

- Python 3.10+
- pip
- network access to download data from Yahoo Finance

### Install dependencies

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Usage

### Run the full pipeline

```bash
python -m stock_market_rl --ticker AAPL --start-date 2024-01-01 --end-date 2024-12-31 --episodes 40 --window-size 10 --initial-cash 10000
```

### Run tests

```bash
python -m pytest -q
```

This creates a metrics file under `results/metrics/training_metrics.json` and a plot under `results/plots/aapl_strategy.png`.

## Verified results

The project was executed in this environment using the command above. The actual output was:

```json
{
  "ticker": "AAPL",
  "train_rows": 200,
  "test_rows": 51,
  "training_summary": {
    "mean_return": 0.004634701109809463,
    "best_return": 0.007196142060851729,
    "final_return": 0.005090159889220791
  },
  "strategy_summary": {
    "final_cash": 9998.214321228026,
    "total_return": -0.00017856787719738865,
    "trades": 1,
    "portfolio_value": 9998.214321228026
  },
  "baseline_summary": {
    "total_return": 0.08636658875252046,
    "buy_hold_value": 10863.665887525205,
    "annualized_return": 0.5057946269692426,
    "mean_daily_return": 0.0017104062079485664,
    "volatility": 0.01021354086291388
  }
}
```

### Interpretation

This sample run shows the agent does not outperform the buy-and-hold baseline in the tested period. That is useful information: the project is working as a reproducible prototype, but it is not a market-beating strategy in this configuration.

### Automated validation

```bash
python -m pytest -q
```

Observed result:

```text
4 passed in 11.08s
```

## Demo

The project includes a small end-to-end demo via the CLI. Running the command above downloads market data, trains the agent, evaluates it, and writes the resulting metrics and plot into the results directory.

## Limitations

- This is a single-asset prototype, not a production trading system.
- The action space is intentionally minimal.
- The feature set is simple and does not include macro or fundamental signals.
- Q-learning is a lightweight baseline and may not compete with modern RL or statistical approaches.
- Results are sensitive to the selected date range and hyperparameters.

## Technologies used

- Python
- NumPy
- pandas
- Matplotlib
- yfinance
- pytest

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.
- Yahoo Finance for historical market data.

## Notes on reliability

This project is useful as a learning and demo pipeline. It is intentionally transparent about its limitations and does not claim to be a production-grade trading strategy.

