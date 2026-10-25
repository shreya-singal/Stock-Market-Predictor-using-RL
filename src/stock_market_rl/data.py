from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import PROJECT_ROOT, TradingConfig


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()

    if isinstance(renamed.columns, pd.MultiIndex):
        flattened = []
        for column in renamed.columns:
            if isinstance(column, tuple):
                flattened.append(str(column[0]).strip().lower())
            else:
                flattened.append(str(column).strip().lower())
        renamed.columns = flattened
    else:
        renamed.columns = [str(col).strip().lower() for col in renamed.columns]

    renamed = renamed.rename(columns={"adj close": "adj_close"})
    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        if column not in renamed.columns:
            raise ValueError(f"Required column '{column}' missing from downloaded data.")
    return renamed


def load_price_data(
    ticker: str = "AAPL",
    start_date: str = "2018-01-01",
    end_date: str | None = None,
    cache: bool = True,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    if data_dir is None:
        target_dir = PROJECT_ROOT / "data" / "raw"
    else:
        target_dir = Path(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{ticker.lower()}.csv"

    if cache and target_path.exists():
        df = pd.read_csv(target_path, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        actions=False,
    )

    if data.empty:
        raise ValueError(f"No market data found for ticker '{ticker}' between {start_date} and {end_date}.")

    df = data.reset_index().rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
    df = _standardize_columns(df)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(target_path, index=False)
    return df


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    split_idx = max(1, int(len(df) * train_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def default_config() -> TradingConfig:
    return TradingConfig()
