"""
backtester.py
--------------
Realistischer Event-basierter Backtester.

Kernregeln (wie gefordert):
- Entry IMMER zum Open-Kurs der Signal-Kerze (+ Spread + Slippage).
- SL/TP werden intraday über High/Low derselben und folgender Kerzen geprüft.
- Falls in einer Kerze sowohl Low <= SL als auch High >= TP erreicht werden,
  gilt IMMER der SL (konservative Worst-Case-Annahme).
- Gebühren werden auf Entry UND Exit angewendet.
- Liefert alle Kennzahlen, die für die Zielkriterien (config.StrategyCriteria)
  gebraucht werden: Winrate, CRV, Total Return, Max Drawdown, Profit Factor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import EXECUTION


@dataclass
class Trade:
    entry_time: pd.Timestamp
    direction: str                            
    entry_price: float
    sl_price: float
    tp_price: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None                            
    pnl_pct: float | None = None
    pnl_abs: float | None = None
    confidence: float | None = None


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series | None = None
    benchmark_curve: pd.Series | None = None

                
    winrate: float = 0.0
    avg_crv: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0

    def meets_criteria(self, criteria) -> bool:
        return (
            self.n_trades >= criteria.min_trades
            and self.winrate >= criteria.min_winrate
            and self.avg_crv >= criteria.min_crv
            and self.total_return_pct >= criteria.min_total_return_pct
            and self.max_drawdown_pct <= criteria.max_drawdown_pct
            and self.profit_factor >= criteria.min_profit_factor
        )

    def to_dict(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "winrate": round(self.winrate, 4),
            "avg_crv": round(self.avg_crv, 4),
            "total_return_pct": round(self.total_return_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "profit_factor": round(self.profit_factor, 4),
        }


def _apply_entry_costs(price: float, direction: str) -> float:
    """Wendet Spread + Slippage auf den Open-Preis an (Entry wird ungünstiger)."""
    cost_pct = EXECUTION.spread_pct + EXECUTION.slippage_pct
    if direction == "long":
        return price * (1 + cost_pct)
    return price * (1 - cost_pct)


def _session_end_idx(df: pd.DataFrame, entry_idx: int) -> int:
    """
    Letzter Bar-Index desselben Handelstags wie entry_idx. Bei Daily-Daten
    (ein Bar = ein Tag) ist das immer entry_idx selbst -- die Funktion greift
    dann effektiv nicht ein. Bei Intraday-Daten (5m/1h) verhindert das ein
    Halten über den Handelsschluss hinweg (Daytrading-Anforderung).
    """
    entry_date = df.index[entry_idx].date()
    j = entry_idx
    n = len(df)
    while j + 1 < n and df.index[j + 1].date() == entry_date:
        j += 1
    return j


def simulate_trade(df: pd.DataFrame, entry_idx: int, direction: str,
                    sl_price: float, tp_price: float, max_hold_bars: int = 20,
                    force_eod_close: bool = True) -> Trade:
    """
    Simuliert einen einzelnen Trade ab entry_idx (Entry zum Open dieser Kerze).
    Prüft ab derselben Kerze (Intraday-High/Low kann SL/TP der Entry-Kerze
    selbst treffen) bis max_hold_bars nach vorne.

    force_eod_close=True (Default): Position wird spätestens zum letzten Bar
    des Handelstags zwangsweise geschlossen (EOD-Exit zum Close-Preis dieses
    Bars) -- kein Overnight-Halten. Wichtig für Daytrading-Strategien auf
    Intraday-Timeframes (5m/1h). Bei Daily-Daten hat das keinen Effekt.
    """
    raw_open = df["open"].iloc[entry_idx]
    entry_price = _apply_entry_costs(raw_open, direction)
    entry_time = df.index[entry_idx]

    trade = Trade(
        entry_time=entry_time, direction=direction,
        entry_price=entry_price, sl_price=sl_price, tp_price=tp_price,
    )

    end_idx = min(entry_idx + max_hold_bars, len(df) - 1)
    if force_eod_close:
        end_idx = min(end_idx, _session_end_idx(df, entry_idx))

    for j in range(entry_idx, end_idx + 1):
        high, low = df["high"].iloc[j], df["low"].iloc[j]

        if direction == "long":
            sl_hit = low <= sl_price
            tp_hit = high >= tp_price
        else:         
            sl_hit = high >= sl_price
            tp_hit = low <= tp_price

        if sl_hit and tp_hit:
                                                               
            trade.exit_price, trade.exit_reason = sl_price, "SL"
            trade.exit_time = df.index[j]
            break
        elif sl_hit:
            trade.exit_price, trade.exit_reason = sl_price, "SL"
            trade.exit_time = df.index[j]
            break
        elif tp_hit:
            trade.exit_price, trade.exit_reason = tp_price, "TP"
            trade.exit_time = df.index[j]
            break
    else:
                                                                              
                                                                             
                                                                             
        trade.exit_price = df["close"].iloc[end_idx]
        trade.exit_reason = "EOD" if force_eod_close else "TIMEOUT"
        trade.exit_time = df.index[end_idx]

    fee = EXECUTION.fee_rate
    if direction == "long":
        gross_pnl_pct = (trade.exit_price - entry_price) / entry_price
    else:
        gross_pnl_pct = (entry_price - trade.exit_price) / entry_price

    net_pnl_pct = gross_pnl_pct - 2 * fee                         
    trade.pnl_pct = net_pnl_pct
    return trade


def run_backtest(df: pd.DataFrame, signals: pd.Series, confidences: pd.Series,
                  sl_atr_mult: float, tp_atr_mult: float,
                  max_hold_bars: int | None = None,
                  force_eod_close: bool | None = None) -> BacktestResult:
    """
    signals: pd.Series mit Werten {-1, 0, 1} (short/none/long) je Kerze,
             ausgerichtet auf df.index (Signal basiert nur auf bis inkl. t
             bekannten Daten -> Entry erfolgt am OPEN von t+1 in dieser Funktion).
    confidences: Wahrscheinlichkeit/Konfidenz des Modells je Signal (0-1).
    max_hold_bars/force_eod_close: None = Werte aus config.EXECUTION übernehmen.
    """
    assert "atr" in df.columns, "ATR-Spalte fehlt -- build_feature_matrix() zuvor aufrufen."

    max_hold_bars = EXECUTION.max_hold_bars if max_hold_bars is None else max_hold_bars
    force_eod_close = EXECUTION.no_overnight_hold if force_eod_close is None else force_eod_close

    capital = EXECUTION.initial_capital
    equity = [capital]
    equity_index = [df.index[0]]
    trades: list[Trade] = []

    i = 0
    n = len(df)
    while i < n - 1:
        sig = signals.iloc[i]
        if sig == 0:
            i += 1
            continue

        entry_idx = i + 1                                                      
        if entry_idx >= n:
            break

        atr_at_signal = df["atr"].iloc[i]
        direction = "long" if sig == 1 else "short"
        raw_open = df["open"].iloc[entry_idx]

        if direction == "long":
            sl_price = raw_open - sl_atr_mult * atr_at_signal
            tp_price = raw_open + tp_atr_mult * atr_at_signal
        else:
            sl_price = raw_open + sl_atr_mult * atr_at_signal
            tp_price = raw_open - tp_atr_mult * atr_at_signal

        trade = simulate_trade(df, entry_idx, direction, sl_price, tp_price, max_hold_bars, force_eod_close)
        trade.confidence = float(confidences.iloc[i]) if confidences is not None else None

                                                                                       
        risk_amount = capital * EXECUTION.risk_per_trade_pct
        risk_pct_of_price = abs(trade.entry_price - sl_price) / trade.entry_price
        position_value = risk_amount / max(risk_pct_of_price, 1e-6)
        position_value = min(position_value, capital)                         

        trade.pnl_abs = position_value * trade.pnl_pct
        capital += trade.pnl_abs

        trades.append(trade)
        equity.append(capital)
        equity_index.append(trade.exit_time)

                                                                                         
        exit_pos = df.index.get_loc(trade.exit_time)
        i = exit_pos + 1

    equity_curve = pd.Series(equity, index=equity_index).drop_duplicates()
    equity_curve = equity_curve.reindex(df.index, method="ffill").fillna(EXECUTION.initial_capital)

    benchmark_curve = EXECUTION.initial_capital * (df["close"] / df["close"].iloc[0])

    result = BacktestResult(trades=trades, equity_curve=equity_curve, benchmark_curve=benchmark_curve)
    _compute_metrics(result)

    trades_df = pd.DataFrame([vars(t) for t in result.trades])
    trades_df.to_csv("backtest_trades.csv", index=False)
    print("Trades erfolgreich in 'backtest_trades.csv' gespeichert!")
    return result


def _compute_metrics(result: BacktestResult) -> None:
    trades = result.trades
    result.n_trades = len(trades)
    if not trades:
        return

    wins = [t for t in trades if t.pnl_abs > 0]
    losses = [t for t in trades if t.pnl_abs <= 0]

    result.winrate = len(wins) / len(trades)

                                                                                 
    avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0.0
    avg_loss_pct = abs(np.mean([t.pnl_pct for t in losses])) if losses else 0.0

    if avg_loss_pct > 1e-5:
        raw_crv = avg_win_pct / avg_loss_pct
        result.avg_crv = float(np.clip(raw_crv, 0.0, 10.0))            
    else:
        result.avg_crv = 10.0 if wins else 0.0

                                                
    gross_profit = sum(t.pnl_abs for t in wins)
    gross_loss = abs(sum(t.pnl_abs for t in losses))

    if gross_loss > 0:
        result.profit_factor = min(gross_profit / gross_loss, 10.0)
    else:
        result.profit_factor = 10.0 if gross_profit > 0 else 0.0

    start_cap = EXECUTION.initial_capital
    end_cap = result.equity_curve.iloc[-1]
    result.total_return_pct = (end_cap - start_cap) / start_cap

    running_max = result.equity_curve.cummax()
    drawdown = (result.equity_curve - running_max) / running_max
    result.max_drawdown_pct = abs(drawdown.min())

if __name__ == "__main__":
    import os
    import joblib
    import pandas as pd
    import yfinance as yf
    from features import build_feature_matrix

    class Params:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    model_file = "aapl_3055_model.pkl"
    config_file = "aapl_3055_config.pkl"

    if not (os.path.exists(model_file) and os.path.exists(config_file)):
        print("❌ 'aapl_3055_model.pkl' oder 'aapl_3055_config.pkl' fehlt im Ordner!")
    else:
        print("============================================================")
        print(" Lade Modell-Dateien für Trial #3055 (XGBoost)...")
        print("============================================================")
        
        model = joblib.load(model_file)
        config = joblib.load(config_file)
        
        params = Params(config["params"])
        feature_cols = config["feature_cols"]

        # 15m Kerzen für die letzten 60 Tage laden
        df_raw = yf.download("AAPL", period="60d", interval="15m", progress=False, auto_adjust=False)
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
        df_raw = df_raw.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        df = build_feature_matrix(df_raw, params)
        X = df[feature_cols]
        
        # XGBoost verarbeitet X direkt ohne Scaler
        probs = model.predict_proba(X)[:, 1]

        signals = pd.Series(0, index=df.index)
        signals[probs >= params.signal_threshold] = 1
        confidences = pd.Series(probs, index=df.index)

        res = run_backtest(
            df=df,
            signals=signals,
            confidences=confidences,
            sl_atr_mult=params.sl_atr_mult,
            tp_atr_mult=params.tp_atr_mult,
        )

        print("\n==================================================")
        print(" BACKTEST ERGEBNISSE TRIAL #3055 (LETZTE 60 TAGE):")
        print("==================================================")
        print(f"  • Ausgeführte Trades: {res.n_trades}")
        print(f"  • Winrate:            {res.winrate * 100:.1f}%")
        print(f"  • Ø CRV:              {res.avg_crv:.2f}")
        print(f"  • Total Return:       +{res.total_return_pct * 100:.2f}%")
        print(f"  • Max Drawdown:       {res.max_drawdown_pct * 100:.2f}%")
        print(f"  • Profit Factor:      {res.profit_factor:.2f}")
        print("==================================================\n")