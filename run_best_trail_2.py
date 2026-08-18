import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from backtester import run_backtest
from data_loader import load_data
from features import build_feature_matrix


class Params:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def main():
    N_SIMULATIONS = 1000

    # Exakte Parameter aus Trial #2523 der quant_strategy_search_v2
# Exakte Parameter aus Trial #2523 der quant_strategy_search_v2
    best_params_dict = {
        'rsi_period': 25,
        'ema_fast': 12,
        'ema_slow': 84,
        'atr_period': 7,
        'bb_period': 38,
        'sl_atr_mult': 0.8252061389868222,
        'tp_atr_mult': 2.2970554553757583,
        'signal_threshold': 0.6144108793646803,
        'model_type': 'logistic',
        'active_features': ['trend', 'price_action', 'smc_order_blocks', 'smc_killzones']
    }

    feature_cols = [
        'ema_fast', 'ema_slow', 'ema_spread_pct', 'price_vs_ema_slow', 'ema_slope',
        'candle_body_pct', 'upper_wick_pct', 'lower_wick_pct', 'bullish_candle',
        'gap_pct', 'higher_high', 'lower_low', 'smc_is_ob_candidate', 'smc_in_ob_zone',
        'smc_dist_to_ob_top_atr', 'smc_killzone_london', 'smc_killzone_ny',
        'smc_killzone_london_close', 'atr'
    ]

    trial_params = Params(best_params_dict)

    print("=" * 60)
    print(" Lade AAPL 15m Daten & erstelle Features für Trial #2523 (v2)...")
    print("=" * 60)
    
    df_raw = load_data(ticker="AAPL")
    df = build_feature_matrix(df_raw, trial_params)

    # Bereinigung fehlender Werte durch Indikator-Lookbacks
    df = df.dropna(subset=feature_cols).copy()

    X = df[feature_cols]

    # Target-Simulation für das Training des LogisticRegression-Modells auf der 1. Hälfte
    split_idx = int(len(df) * 0.5)
    
    # Vereinfachte Target-Generierung analog zum Optimizer
    future_returns = df['close'].shift(-4) / df['close'] - 1.0
    y = (future_returns > 0).astype(int)

    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

    print(" Trainiere LogisticRegression Modell auf der 1. Hälfte...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Vorhersagen für den gesamten Datensatz
    probs = model.predict_proba(X)[:, 1]
    confidences_full = pd.Series(probs, index=df.index)
    signals_full = (confidences_full >= trial_params.signal_threshold).astype(int)

    # Filter: Keine Trades nach 21:00 Uhr
    times = pd.to_datetime(df.index).time
    no_trade_mask = times >= pd.to_datetime("21:00").time()
    signals_full[no_trade_mask] = 0

    # Evaluation ausschließlich auf der 2. Hälfte (Out-of-Sample Test)
    df_test = df.iloc[split_idx:].copy()
    base_signals = signals_full.iloc[split_idx:].copy()
    base_confidences = confidences_full.iloc[split_idx:].copy()

    print(f"\n Starte {N_SIMULATIONS:,} Out-of-Sample Stresstest-Simulationen (Trial #2523)...")
    start_time = time.time()
    results = []

    for run_id in range(1, N_SIMULATIONS + 1):
        np.random.seed(42 + run_id)

        df_sim = df_test.copy()
        noise = 1.0 + np.random.uniform(-0.0003, 0.0003, size=len(df_sim))
        df_sim["open"] = df_sim["open"] * noise

        res = run_backtest(
            df=df_sim,
            signals=base_signals,
            confidences=base_confidences,
            sl_atr_mult=trial_params.sl_atr_mult,
            tp_atr_mult=trial_params.tp_atr_mult,
            max_hold_bars=999,
            force_eod_close=True,
        )

        results.append({
            'return': res.total_return_pct * 100,
            'winrate': res.winrate * 100,
            'max_dd': res.max_drawdown_pct * 100,
            'pf': res.profit_factor,
            'trades': res.n_trades,
        })

        if run_id % (N_SIMULATIONS // 10) == 0:
            pct = (run_id / N_SIMULATIONS) * 100
            print(f"  • Progress: {run_id:,} / {N_SIMULATIONS:,} Runs ({pct:.0f}%) nach {time.time() - start_time:.1f}s")

    elapsed_time = time.time() - start_time
    print(f"\n Fertig! {N_SIMULATIONS:,} Runs in {elapsed_time:.2f} Sekunden berechnet.")

    returns = [r['return'] for r in results]
    winrates = [r['winrate'] for r in results]
    dds = [r['max_dd'] for r in results]
    pfs = [r['pf'] for r in results]
    trades = [r['trades'] for r in results]

    avg_return = np.mean(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)
    profitable_runs = sum(1 for r in returns if r > 0)
    win_pct_runs = (profitable_runs / N_SIMULATIONS) * 100

    print("\n" + "=" * 50)
    print(f" ERGEBNISSE TRIAL #2523 (v2 Study) AUS {N_SIMULATIONS:,} RUNS:")
    print("=" * 50)
    print(f"  • Profitable Runs:    {profitable_runs:,} / {N_SIMULATIONS:,} ({win_pct_runs:.2f}%)")
    print(f"  • Ø Total Return:     {avg_return:+.2f}% (Min: {min_return:+.2f}%, Max: {max_return:+.2f}%)")
    print(f"  • Ø Winrate:          {np.mean(winrates):.1f}%")
    print(f"  • Ø Max Drawdown:     {np.mean(dds):.2f}%")
    print(f"  • Ø Profit Factor:    {np.mean(pfs):.2f}")
    print(f"  • Ø Trades pro Run:   {np.mean(trades):.1f}")
    print("=" * 50)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.hist(returns, bins=25, color='#3498db', edgecolor='black', alpha=0.75)
    ax1.axvline(avg_return, color='red', linestyle='--', linewidth=2, label=f'Ø Return ({avg_return:+.2f}%)')
    ax1.set_title(f"Verteilung der Returns (Trial #2523 - v2) über {N_SIMULATIONS:,} Out-of-Sample Runs", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Total Return (%)")
    ax1.set_ylabel("Anzahl Durchläufe")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    ax2.axis('off')
    summary_data = [
        ["Metrik", f"Durchschnitt ({N_SIMULATIONS:,} Runs)", "Best-Case", "Worst-Case"],
        ["Profitable Runs", f"{win_pct_runs:.2f}%", "-", "-"],
        ["Total Return", f"{avg_return:+.2f}%", f"{max_return:+.2f}%", f"{min_return:+.2f}%"],
        ["Winrate", f"{np.mean(winrates):.1f}%", f"{np.max(winrates):.1f}%", f"{np.min(winrates):.1f}%"],
        ["Max Drawdown", f"{np.mean(dds):.2f}%", f"{np.min(dds):.2f}%", f"{np.max(dds):.2f}%"],
        ["Profit Factor", f"{np.mean(pfs):.2f}", f"{np.max(pfs):.2f}", f"{np.min(pfs):.2f}"],
        ["Trades pro Run", f"{np.mean(trades):.1f}", f"{np.max(trades)}", f"{np.min(trades)}"],
    ]

    table = ax2.table(cellText=summary_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        elif row == 1:
            cell.set_facecolor('#27ae60' if win_pct_runs > 50 else '#e74c3c')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f2f2f2')

    plt.tight_layout()
    out_file = f"stresstest_trial_2523_v2_{N_SIMULATIONS}_runs.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\n Stresstest-Chart gespeichert unter: {out_file}")


if __name__ == "__main__":
    main()