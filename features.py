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


# ==============================================================================
# HILFSFUNKTIONEN (Indikatoren)
# ==============================================================================

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


# ==============================================================================
# STANDARD FEATURE-GRUPPEN
# ==============================================================================

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
    if "news_sentiment" not in df.columns:
        df["news_sentiment"] = 0.0
    df["news_sentiment_ma3"] = df["news_sentiment"].rolling(3).mean().fillna(0)
    return df


# ==============================================================================
# SMC / ICT FEATURE-GRUPPEN (Neu hinzugefügt)
# ==============================================================================

def add_fvg_features(df: pd.DataFrame) -> pd.DataFrame:
    """1. Fair Value Gaps (Bullish/Bearish Unfilled FVG + Inversion/Reserve FVG)."""
    atr = df["atr"] if "atr" in df.columns else _atr(df, 14)
    
    # 3-Kerzen Imbalance (Bullish: Low[t] > High[t-2] | Bearish: High[t] < Low[t-2])
    bull_fvg_top = df["low"]
    bull_fvg_bottom = df["high"].shift(2)
    has_bull_fvg = (bull_fvg_top > bull_fvg_bottom).astype(int)
    
    bear_fvg_top = df["low"].shift(2)
    bear_fvg_bottom = df["high"]
    has_bear_fvg = (bear_fvg_top > bear_fvg_bottom).astype(int)
    
    # FVG Größen in ATR
    fvg_size_bull = ((bull_fvg_top - bull_fvg_bottom) / atr).clip(lower=0) * has_bull_fvg
    fvg_size_bear = ((bear_fvg_top - bear_fvg_bottom) / atr).clip(lower=0) * has_bear_fvg
    
    # Unfilled & Inversion FVG Tracking über ein Rolling Window
    # Distanz vom aktuellen Close zum letzten aktiven Bullish FVG Bottom
    fvg_bot_series = bull_fvg_bottom.where(has_bull_fvg == 1)
    last_fvg_bot = fvg_bot_series.ffill().rolling(20, min_periods=1).max()
    dist_to_unfilled_fvg_atr = ((df["close"] - last_fvg_bot) / atr).fillna(0)
    
    # Inversion/Reserve FVG (Wenn ein Bearish FVG nach oben durchbrochen wird)
    inversion_fvg_bull = ((df["close"] > bear_fvg_top.shift(1)) & (df["close"].shift(1) <= bear_fvg_top.shift(1))).astype(int)

    df["smc_has_bull_fvg"] = has_bull_fvg
    df["smc_has_bear_fvg"] = has_bear_fvg
    df["smc_fvg_size_bull_atr"] = fvg_size_bull
    df["smc_fvg_size_bear_atr"] = fvg_size_bear
    df["smc_dist_to_unfilled_fvg_atr"] = dist_to_unfilled_fvg_atr
    df["smc_inversion_fvg_bull"] = inversion_fvg_bull
    return df


def add_order_block_features(df: pd.DataFrame) -> pd.DataFrame:
    """2. Order Blocks (Letzte bärische Kerze vor einem starken bullischen Impuls)."""
    atr = df["atr"] if "atr" in df.columns else _atr(df, 14)
    
    # Impulsives Erholungssignal (Kerzenkörper > 1.5x ATR)
    body = df["close"] - df["open"]
    strong_bull_move = body > (1.5 * atr)
    
    # Der Bullish Order Block ist die vorherige bärische Kerze [t-1]
    is_ob_candidate = (df["close"].shift(1) < df["open"].shift(1)) & strong_bull_move
    
    ob_top = df["high"].shift(1).where(is_ob_candidate)
    ob_bottom = df["low"].shift(1).where(is_ob_candidate)
    
    last_ob_top = ob_top.ffill().rolling(30, min_periods=1).max()
    last_ob_bottom = ob_bottom.ffill().rolling(30, min_periods=1).min()
    
    # Ist der aktuelle Kurs nahe/innerhalb des letzten Order Blocks?
    in_ob_zone = ((df["low"] <= last_ob_top) & (df["high"] >= last_ob_bottom)).astype(int)
    dist_to_ob_top_atr = ((df["close"] - last_ob_top) / atr).fillna(0)
    
    df["smc_is_ob_candidate"] = is_ob_candidate.astype(int)
    df["smc_in_ob_zone"] = in_ob_zone
    df["smc_dist_to_ob_top_atr"] = dist_to_ob_top_atr
    return df


def add_market_structure_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """3. Market Structure Shifts (BOS = Break of Structure / CHoCH = Change of Character)."""
    atr = df["atr"] if "atr" in df.columns else _atr(df, 14)
    
    # Lokale Swing Highs und Lows ermitteln
    swing_high = df["high"].rolling(window * 2 + 1, center=True).max() == df["high"]
    swing_low = df["low"].rolling(window * 2 + 1, center=True).min() == df["low"]
    
    # Shift um `window`, damit es absolut look-ahead-frei ist
    last_swing_high = df["high"].where(swing_high).shift(window).ffill()
    last_swing_low = df["low"].where(swing_low).shift(window).ffill()
    
    # Break of Structure (BOS) / CHoCH
    bullish_bos = (df["close"] > last_swing_high) & (df["close"].shift(1) <= last_swing_high.shift(1))
    bearish_bos = (df["close"] < last_swing_low) & (df["close"].shift(1) >= last_swing_low.shift(1))
    
    df["smc_bullish_bos"] = bullish_bos.astype(int)
    df["smc_bearish_bos"] = bearish_bos.astype(int)
    df["smc_dist_to_swing_high_atr"] = ((last_swing_high - df["close"]) / atr).fillna(0)
    df["smc_dist_to_swing_low_atr"] = ((df["close"] - last_swing_low) / atr).fillna(0)
    return df


def add_liquidity_pool_features(df: pd.DataFrame, tolerance_pct: float = 0.0015) -> pd.DataFrame:
    """4. Liquidity Pools (Equal Highs / Equal Lows als Liquiditäts-Magneten)."""
    atr = df["atr"] if "atr" in df.columns else _atr(df, 14)
    
    # Prüft, ob vorherige Highs/Lows nahezu identisch sind (Equal Highs / Lows)
    prev_high = df["high"].shift(1)
    prev_high_2 = df["high"].shift(2)
    equal_highs = (prev_high - prev_high_2).abs() / prev_high <= tolerance_pct
    
    prev_low = df["low"].shift(1)
    prev_low_2 = df["low"].shift(2)
    equal_lows = (prev_low - prev_low_2).abs() / prev_low <= tolerance_pct
    
    # Sweept der aktuelle Kurs diese Liquidität? (Liquidity Sweep)
    bsl_sweep = (df["high"] > prev_high) & equal_highs  # Buy-side Liquidity Sweep
    ssl_sweep = (df["low"] < prev_low) & equal_lows     # Sell-side Liquidity Sweep
    
    df["smc_equal_highs"] = equal_highs.astype(int)
    df["smc_equal_lows"] = equal_lows.astype(int)
    df["smc_bsl_sweep"] = bsl_sweep.astype(int)
    df["smc_ssl_sweep"] = ssl_sweep.astype(int)
    return df


def add_premium_discount_features(df: pd.DataFrame, window: int = 40) -> pd.DataFrame:
    """5. Premium / Discount Zones (Fibonacci 50% Equilibrium eines Sweeps/Ranges)."""
    range_high = df["high"].rolling(window).max()
    range_low = df["low"].rolling(window).min()
    equilibrium = (range_high + range_low) / 2.0
    
    # Relativer Stand in der Range: 0.0 = Discount Bottom, 0.5 = Equilibrium, 1.0 = Premium Top
    range_pct = (df["close"] - range_low) / (range_high - range_low).replace(0, np.nan)
    
    # Binares Flag: Ist der Kurs in der Discount-Zone (< 0.5) = Ideal für Longs
    in_discount = (range_pct < 0.5).astype(int)
    
    df["smc_range_pct"] = range_pct.fillna(0.5)
    df["smc_in_discount_zone"] = in_discount
    df["smc_dist_to_eq_pct"] = (df["close"] - equilibrium) / equilibrium
    return df


def add_killzone_features(df: pd.DataFrame) -> pd.DataFrame:
    """6. Session Killzones (London Open, New York Open, Asia Session)."""
    # Berücksichtigt Timezone des DataFrames
    if df.index.tzinfo is None:
        times = df.index
    else:
        times = df.index.tz_convert("America/New_York")
        
    hours = times.hour
    
    # ICT Killzones in New York Zeit (EST)
    london_open = ((hours >= 2) & (hours < 5)).astype(int)      # 02:00 - 05:00 EST
    ny_open = ((hours >= 7) & (hours < 10)).astype(int)         # 07:00 - 10:00 EST
    london_close = ((hours >= 10) & (hours < 12)).astype(int)   # 10:00 - 12:00 EST
    
    df["smc_killzone_london"] = london_open
    df["smc_killzone_ny"] = ny_open
    df["smc_killzone_london_close"] = london_close
    return df


# ==============================================================================
# FEATURE REGISTRY & BUILDER
# ==============================================================================

FEATURE_GROUP_FUNCS = {
    # Standard-Gruppen
    "trend": lambda df, p: add_trend_features(df, p.ema_fast, p.ema_slow),
    "momentum": lambda df, p: add_momentum_features(df, p.rsi_period),
    "volatility": lambda df, p: add_volatility_features(df, p.atr_period, p.bb_period),
    "volume": lambda df, p: add_volume_features(df),
    "price_action": lambda df, p: add_price_action_features(df),
    "news_sentiment": lambda df, p: add_news_sentiment_placeholder(df),
    
    # Neue SMC / ICT Feature-Gruppen
    "smc_fvg": lambda df, p: add_fvg_features(df),
    "smc_order_blocks": lambda df, p: add_order_block_features(df),
    "smc_market_structure": lambda df, p: add_market_structure_features(df),
    "smc_liquidity": lambda df, p: add_liquidity_pool_features(df),
    "smc_premium_discount": lambda df, p: add_premium_discount_features(df),
    "smc_killzones": lambda df, p: add_killzone_features(df),
}


def build_feature_matrix(df: pd.DataFrame, params: "TrialParams") -> pd.DataFrame:
    out = df.copy()
    for group in params.active_features:
        func = FEATURE_GROUP_FUNCS.get(group)
        if func:
            out = func(out, params)

    if "atr" not in out.columns:
        out["atr"] = _atr(out, getattr(params, "atr_period", 14))

    # Infs durch NaNs ersetzen und saubere Werte garantieren
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(0)
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
    """
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
    """
    n = len(df)
    labels = np.zeros(n, dtype=int)
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    atr = df['atr'].values

    # Erstelle ein Array mit den Tagen, um das EOD-Ende der aktuellen Session zu bestimmen
    dates = pd.to_datetime(df.index).date

    for i in range(n - 1):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        entry = closes[i]
        sl = entry - sl_atr_mult * atr[i]
        tp = entry + tp_atr_mult * atr[i]
        current_date = dates[i]

        hit = 0
        j = i + 1

        # Suche nur vorwärts bis zum Ende desselben Handelstages (EOD)
        while j < n and dates[j] == current_date:
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

            j += 1

        # Falls weder TP noch SL vor Tagesende getroffen wurden:
        # Evaluierung zum EOD-Close-Kurs
        if hit == 0 and not (sl_hit or tp_hit) if 'sl_hit' in locals() else True:
            if j > i + 1:
                eod_close = closes[j - 1]
                hit = 1 if eod_close > entry else 0

        labels[i] = hit

    return pd.Series(labels, index=df.index, name='target')