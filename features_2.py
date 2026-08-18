"""
features.py
------------
Feature Engineering für das ML-Modell. Der Optimizer kann per `feature_flags`
Gruppen ein-/ausschalten, um verschiedene Feature-Kombinationen zu testen.

Alle Funktionen sind Look-Ahead-frei: es wird ausschließlich mit Daten bis
inklusive der aktuellen (bereits geschlossenen) Kerze gerechnet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


                                                                            
                                                                          
                                                                            
def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def _bollinger(close: pd.Series, period: int, n_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, pct_b.fillna(0.5)


def add_trend_features(df: pd.DataFrame, ema_fast: int, ema_slow: int) -> pd.DataFrame:
    df["ema_fast"] = df["close"].ewm(span=ema_fast, min_periods=ema_fast).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow, min_periods=ema_slow).mean()
    df["ema_spread_pct"] = (df["ema_fast"] - df["ema_slow"]) / df["ema_slow"]
    df["price_vs_ema_slow"] = (df["close"] - df["ema_slow"]) / df["ema_slow"]
    df["ema_slope"] = df["ema_fast"].pct_change(5)
    return df


def add_momentum_features(df: pd.DataFrame, rsi_period: int) -> pd.DataFrame:
    df["rsi"] = _rsi(df["close"], rsi_period)
    df["rsi_delta"] = df["rsi"].diff()
    df["roc_5"] = df["close"].pct_change(5)
    df["roc_10"] = df["close"].pct_change(10)
    macd_fast = df["close"].ewm(span=12).mean()
    macd_slow = df["close"].ewm(span=26).mean()
    df["macd"] = macd_fast - macd_slow
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_volatility_features(df: pd.DataFrame, atr_period: int, bb_period: int) -> pd.DataFrame:
    df["atr"] = _atr(df, atr_period)
    df["atr_pct"] = df["atr"] / df["close"]
    _, _, _, pct_b = _bollinger(df["close"], bb_period)
    df["bb_pct_b"] = pct_b
    df["realized_vol_20"] = df["close"].pct_change().rolling(20).std()
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
                                                                                 
                                                                              
                                                                                  
                                                                    
    vol_sum_5 = df["volume"].rolling(5).sum().replace(0, np.nan)
    df["obv_slope"] = df["obv"].diff(5) / vol_sum_5
    return df


def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    body = (df["close"] - df["open"]).abs()
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    df["candle_body_pct"] = body / range_
    df["upper_wick_pct"] = (df["high"] - df[["open", "close"]].max(axis=1)) / range_
    df["lower_wick_pct"] = (df[["open", "close"]].min(axis=1) - df["low"]) / range_
    df["bullish_candle"] = (df["close"] > df["open"]).astype(int)
    df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
                                                       
    df["higher_high"] = (df["high"] > df["high"].rolling(5).max().shift(1)).astype(int)
    df["lower_low"] = (df["low"] < df["low"].rolling(5).min().shift(1)).astype(int)
    return df.fillna(0)


def add_news_sentiment_placeholder(df: pd.DataFrame) -> pd.DataFrame:
    """
    Platzhalter-Schnittstelle für News-Sentiment-Scores (z.B. von einem
    externen NLP-Service). Ohne angebundene Quelle wird neutral (0.0) gesetzt,
    damit das Feature-Schema stabil bleibt, wenn die Gruppe aktiviert ist.
    Ersetze diese Funktion durch einen echten Feed (z.B. NewsAPI + FinBERT).
    """
    if "news_sentiment" not in df.columns:
        df["news_sentiment"] = 0.0
    df["news_sentiment_ma3"] = df["news_sentiment"].rolling(3).mean().fillna(0)
    return df


FEATURE_GROUP_FUNCS = {
    "trend": lambda df, p: add_trend_features(df, p.ema_fast, p.ema_slow),
    "momentum": lambda df, p: add_momentum_features(df, p.rsi_period),
    "volatility": lambda df, p: add_volatility_features(df, p.atr_period, p.bb_period),
    "volume": lambda df, p: add_volume_features(df),
    "price_action": lambda df, p: add_price_action_features(df),
    "news_sentiment": lambda df, p: add_news_sentiment_placeholder(df),
}


def build_feature_matrix(df: pd.DataFrame, params: "TrialParams") -> pd.DataFrame:
    """
    params: Objekt/Namespace mit den vom Optimizer gesampelten Werten
    (rsi_period, ema_fast, ema_slow, atr_period, bb_period, active_features: list[str])
    """
    out = df.copy()
    for group in params.active_features:
        func = FEATURE_GROUP_FUNCS.get(group)
        if func:
            out = func(out, params)

                                                                
    if "atr" not in out.columns:
        out["atr"] = _atr(out, params.atr_period)

    out = out.dropna()
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Alle Spalten außer den rohen OHLCV-Werten -> das sind die Modell-Inputs."""
    excluded = {"open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in excluded]


def build_target_labels(df: pd.DataFrame, forward_bars: int, sl_atr_mult: float, tp_atr_mult: float) -> pd.Series:
    """
    Erzeugt Klassifikations-Labels für das ML-Modell (überwachtes Lernen):
    1 = TP wird vor SL erreicht innerhalb `forward_bars` (long-Setup profitabel)
    0 = sonst (SL zuerst, kein Treffer, oder Zeit läuft ab)

    Wichtig: Nutzt ausschließlich zukünftige High/Low ab t+1 -> beim Training
    kein Look-Ahead, da das Label erst nach Feature-Berechnung an Zeitpunkt t
    berechnet und im Fit-Schritt korrekt versetzt (shift) verwendet wird.
    """
    n = len(df)
    labels = np.zeros(n, dtype=int)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atr = df["atr"].values

    for i in range(n - 1):
        entry = closes[i]
        sl = entry - sl_atr_mult * atr[i]
        tp = entry + tp_atr_mult * atr[i]
        end = min(i + 1 + forward_bars, n)
        hit = 0
        for j in range(i + 1, end):
            sl_hit = lows[j] <= sl
            tp_hit = highs[j] >= tp
            if sl_hit and tp_hit:
                hit = 0                                          
                break
            elif sl_hit:
                hit = 0
                break
            elif tp_hit:
                hit = 1
                break
        labels[i] = hit

    return pd.Series(labels, index=df.index, name="target")