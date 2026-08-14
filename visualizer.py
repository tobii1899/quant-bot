"""
visualizer.py
--------------
Erzeugt die drei geforderten PNG-Reports im /output-Ordner, basierend auf der
aktuell gespeicherten "Active Strategy". Wird automatisch vom Optimizer nach
jedem neuen Fund aufgerufen, kann aber auch manuell (main.py --visualize) an-
gestoßen werden.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")                                         
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from backtester import run_backtest
from config import OUTPUT_DIR
from data_loader import load_data
from features import build_feature_matrix, get_feature_columns
from strategy_store import load_model, load_strategy

logger = logging.getLogger("visualizer")
sns.set_theme(style="darkgrid")


def _prepare_signals(df: pd.DataFrame, strategy: dict, bundle: dict):
    params = SimpleNamespace(**strategy["params"])
    feat_df = build_feature_matrix(df, params)
    feature_cols = strategy["feature_columns"]

    X = bundle["scaler"].transform(feat_df[feature_cols])
    proba = bundle["model"].predict_proba(X)[:, 1]

    signals = pd.Series(np.where(proba >= params.signal_threshold, 1, 0), index=feat_df.index)
    confidences = pd.Series(proba, index=feat_df.index)
    return feat_df, signals, confidences, params


def plot_equity_curve(result, out_path=None) -> None:
    out_path = out_path or OUTPUT_DIR / "equity_curve.png"
    fig, ax = plt.subplots(figsize=(12, 6))

    strat_norm = result.equity_curve / result.equity_curve.iloc[0] * 100
    bench_norm = result.benchmark_curve / result.benchmark_curve.iloc[0] * 100

    ax.plot(strat_norm.index, strat_norm.values, label="Strategie", linewidth=2, color="#2ecc71")
    ax.plot(bench_norm.index, bench_norm.values, label="Buy & Hold (Benchmark)",
            linewidth=1.5, color="#95a5a6", linestyle="--")

    ax.set_title("Equity Curve: Strategie vs. Buy & Hold (Basis = 100)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Indexierter Wert")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Gespeichert: %s", out_path)


def plot_trade_analysis(df: pd.DataFrame, result, out_path=None) -> None:
    out_path = out_path or OUTPUT_DIR / "trade_analysis.png"
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [2, 1]})

    ax1 = axes[0]
    ax1.plot(df.index, df["close"], color="#34495e", linewidth=1, alpha=0.7, label="Close")

    for t in result.trades:
        color = "#2ecc71" if t.exit_reason == "TP" else ("#e74c3c" if t.exit_reason == "SL" else "#f39c12")
        marker = "^" if t.direction == "long" else "v"
        ax1.scatter(t.entry_time, t.entry_price, marker=marker, color="#3498db", s=40, zorder=3)
        if t.exit_time is not None:
            ax1.scatter(t.exit_time, t.exit_price, marker="x", color=color, s=40, zorder=3)

    ax1.set_title("Trades: Entry (blau) / TP-Exit (grün) / SL-Exit (rot) / Timeout (orange)",
                   fontsize=13, fontweight="bold")
    ax1.legend(["Close-Preis"])

    ax2 = axes[1]
    if result.trades:
        monthly = pd.Series(
            {t.exit_time: t.pnl_abs for t in result.trades if t.exit_time is not None}
        ).groupby(pd.Grouper(freq="ME")).sum()
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in monthly.values]
        ax2.bar(monthly.index, monthly.values, width=20, color=colors)
    ax2.set_title("Monats-Performance (P&L in Kontowährung)", fontsize=12)
    ax2.axhline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Gespeichert: %s", out_path)


def plot_strategy_summary(result, strategy: dict, out_path=None) -> None:
    out_path = out_path or OUTPUT_DIR / "strategy_summary.png"
    metrics = result.to_dict()

    rows = [
        ["Ticker", strategy.get("params", {}).get("ticker", "N/A")],
        ["Winrate", f"{metrics['winrate']*100:.1f} %"],
        ["Chance-Risiko (CRV)", f"{metrics['avg_crv']:.2f}"],
        ["Total Return", f"{metrics['total_return_pct']*100:.1f} %"],
        ["Max Drawdown", f"{metrics['max_drawdown_pct']*100:.1f} %"],
        ["Profit Factor", f"{metrics['profit_factor']:.2f}"],
        ["Anzahl Trades", f"{metrics['n_trades']}"],
        ["Modelltyp", strategy["params"].get("model_type", "N/A")],
        ["Gespeichert am", strategy.get("saved_at", "N/A")[:19].replace("T", " ")],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Kennzahl", "Wert"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#ecf0f1" if row % 2 == 0 else "white")

    ax.set_title("Strategy Summary", fontsize=15, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Gespeichert: %s", out_path)


def generate_all_visuals(ticker: str | None = None) -> bool:
    """Rendert alle drei PNGs basierend auf der aktuell besten Strategie. Gibt False zurück, falls keine existiert."""
    strategy = load_strategy()
    bundle = load_model()
    if not strategy or not bundle:
        logger.warning("Keine gespeicherte Strategie gefunden -- Visualisierung übersprungen.")
        return False

    df = load_data(ticker)
    feat_df, signals, confidences, params = _prepare_signals(df, strategy, bundle)

    result = run_backtest(
        feat_df, signals, confidences,
        sl_atr_mult=params.sl_atr_mult, tp_atr_mult=params.tp_atr_mult,
    )

    plot_equity_curve(result)
    plot_trade_analysis(feat_df, result)
    plot_strategy_summary(result, strategy)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_all_visuals()
