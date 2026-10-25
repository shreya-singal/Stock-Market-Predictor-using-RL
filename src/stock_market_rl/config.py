from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"


@dataclass(frozen=True)
class TradingConfig:
    ticker: str = "AAPL"
    start_date: str = "2018-01-01"
    end_date: str = "2024-12-31"
    initial_cash: float = 10_000.0
    window_size: int = 20
    train_ratio: float = 0.8
    episodes: int = 150
    learning_rate: float = 0.15
    discount: float = 0.95
    epsilon: float = 0.2
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.995
    transaction_cost: float = 0.001
    random_seed: int = 42
    action_space: tuple[int, ...] = (0, 1, 2)

    @property
    def raw_data_path(self) -> Path:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        return RAW_DATA_DIR / f"{self.ticker.lower()}.csv"

    @property
    def metrics_dir(self) -> Path:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        return METRICS_DIR

    @property
    def plots_dir(self) -> Path:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        return PLOTS_DIR
