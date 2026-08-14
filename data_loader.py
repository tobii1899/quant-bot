"""
data_loader.py
---------------
Lädt OHLCV-Daten (lokal aus CSV oder via yfinance), validiert sie und
stellt sie den anderen Modulen in einem einheitlichen Format bereit.

Erwartetes Spalten-Schema (Index = DatetimeIndex, UTC oder lokale TZ):
    open, high, low, close, volume
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, EXECUTION

logger = logging.getLogger("data_loader")

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataLoadError(Exception):
    pass


def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Wirft DataLoadError bei strukturell kaputten Daten, sonst bereinigt."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if any(not isinstance(c, str) for c in df.columns):
        raise DataLoadError(
            f"Unerwartete Spalten-Typen (kein einfacher String): {list(df.columns)}. "
            "Vermutlich ein MultiIndex-DataFrame -- Spalten vor dem Laden flachklopfen."
        )
    df.columns = [c.lower() for c in df.columns]

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise DataLoadError(f"Fehlende Spalten im OHLCV-Datensatz: {missing}")

    df = df[REQUIRED_COLUMNS]
    df = df.dropna()

    bad_rows = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (df[REQUIRED_COLUMNS[:-1]] <= 0).any(axis=1)
    )
    if bad_rows.any():
        logger.warning("%d fehlerhafte Kerzen entfernt (High/Low-Inkonsistenz).", bad_rows.sum())
        df = df[~bad_rows]

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_from_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"CSV nicht gefunden: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return _validate_ohlcv(df)


def load_from_yfinance(ticker: str, period: str | None = None, interval: str = "1d") -> pd.DataFrame:
    """
    Lädt Daten via yfinance. Erfordert `pip install yfinance` und Internetzugriff
    zur Laufzeit auf dem Zielsystem des Nutzers.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise DataLoadError(
            "yfinance ist nicht installiert. `pip install yfinance` ausführen "
            "oder stattdessen load_from_csv() mit eigenen Daten nutzen."
        ) from e

                                                                                  
    if period is None:
        if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
            period = "60d"
        else:
            period = "2y"

    raw = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if raw.empty:
        raise DataLoadError(f"Keine Daten für Ticker '{ticker}' erhalten.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df = _validate_ohlcv(raw)

    cache_path = DATA_DIR / f"{ticker}_{interval}.csv"
    df.to_csv(cache_path)
    logger.info("Daten für %s gecacht unter %s", ticker, cache_path)
    return df


def load_data(ticker: str | None = None, source: str = "auto") -> pd.DataFrame:
    """
    Einheitlicher Einstiegspunkt.
    source: "csv" | "yfinance" | "auto" (versucht Cache -> yfinance -> Fehler)
    """
    ticker = ticker or EXECUTION.ticker
    cache_path = DATA_DIR / f"{ticker}_{EXECUTION.timeframe}.csv"

    if source == "csv":
        return load_from_csv(cache_path)

    if source == "yfinance":
        return load_from_yfinance(ticker, interval=EXECUTION.timeframe)

    if cache_path.exists():
        logger.info("Lade gecachte Daten: %s", cache_path)
        return load_from_csv(cache_path)

    logger.info("Kein Cache gefunden, versuche yfinance-Download für %s", ticker)
    return load_from_yfinance(ticker, interval=EXECUTION.timeframe)


def train_test_walk_forward_split(df: pd.DataFrame, n_splits: int, train_pct: float):
    """
    Erzeugt n_splits aufeinanderfolgende (nicht überlappende) Walk-Forward-Fenster.
    """
    n = len(df)
    fold_size = n // n_splits
    for i in range(n_splits):
        start = i * fold_size
        end = start + fold_size if i < n_splits - 1 else n
        fold = df.iloc[start:end]
        split_point = int(len(fold) * train_pct)
        train_df = fold.iloc[:split_point]
        test_df = fold.iloc[split_point:]
        if len(train_df) < 30 or len(test_df) < 10:
            continue
        yield train_df, test_df