import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtester import run_backtest
from data_loader import load_data
from features import build_feature_matrix


class Params:

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def main():
    N_SIMULATIONS = 1000

    try:
        model = joblib.load("aapl_3055_model.pkl")
        config = joblib.load("aapl_3055_config.pkl")
        print("'aapl_3055_model.pkl' & 'aapl_3055_config.pkl' erfolgreich geladen!")
    except FileNotFoundError:
        print("Model- oder Config-Datei nicht gefunden! Bitte zuerst export_model.py ausführen.")
        return

    trial_params = Params(config['params'])
    feature_cols = config['feature_cols']

    print("Lade AAPL 15m Daten & erstelle Features...")
    df_raw = load_data(ticker="AAPL")
    df = build_feature_matrix(df_raw, trial_params)

    X = df[feature_cols]
    probs = model.predict_proba(X)[:, 1]

    confidences_full = pd.Series(probs, index=df.index)
    signals_full = (
        confidences_full >= trial_params.signal_threshold
    ).astype(int)

    times = pd.to_datetime(df.index).time
    no_trade_mask = times >= pd.to_datetime("21:00").time()
    signals_full[no_trade_mask] = 0

    split_idx = int(len(df) * 0.5)

    df_test = df.iloc[split_idx:].copy()

    base_signals = signals_full.iloc[split_idx:].copy()
    base_confidences = confidences_full.iloc[split_idx:].copy()

    print(
        f"🔄 Starte {N_SIMULATIONS:,} ehrliche (Out-of-Sample) Stresstest-Simulationen zuschneidend ab der 2. Hälfte...\n"
    )
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
            print(
                f"  • Progress: {run_id:,} / {N_SIMULATIONS:,} Runs ({pct:.0f}%) nach {time.time() - start_time:.1f}s"
            )

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
    print(f" EHRLICHE ERGEBNISSE AUS {N_SIMULATIONS:,} SIMULATIONEN (.pkl Modell):")
    print("=" * 50)
    print(
        f"  • Profitable Runs:    {profitable_runs:,} / {N_SIMULATIONS:,} ({win_pct_runs:.2f}%)"
    )
    print(
        f"  • Ø Total Return:     {avg_return:+.2f}% (Min: {min_return:+.2f}%, Max: {max_return:+.2f}%)"
    )
    print(f"  • Ø Winrate:          {np.mean(winrates):.1f}%")
    print(f"  • Ø Max Drawdown:     {np.mean(dds):.2f}%")
    print(f"  • Ø Profit Factor:    {np.mean(pfs):.2f}")
    print(f"  • Ø Trades pro Run:   {np.mean(trades):.1f}")
    print("=" * 50)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.hist(returns, bins=25, color='#27ae60', edgecolor='black', alpha=0.75)
    ax1.axvline(
        avg_return,
        color='blue',
        linestyle='--',
        linewidth=2,
        label=f'Ø Return ({avg_return:+.2f}%)',
    )
    ax1.set_title(
        f"Ehrliche Verteilung der Returns über {N_SIMULATIONS:,} Out-of-Sample Runs (.pkl Model)",
        fontsize=12,
        fontweight='bold',
    )
    ax1.set_xlabel("Total Return (%)")
    ax1.set_ylabel("Anzahl Durchläufe")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    ax2.axis('off')
    summary_data = [
        [
            "Metrik",
            f"Durchschnitt ({N_SIMULATIONS:,} Runs)",
            "Best-Case",
            "Worst-Case",
        ],
        ["Profitable Runs", f"{win_pct_runs:.2f}%", "-", "-"],
        [
            "Total Return",
            f"{avg_return:+.2f}%",
            f"{max_return:+.2f}%",
            f"{min_return:+.2f}%",
        ],
        [
            "Winrate",
            f"{np.mean(winrates):.1f}%",
            f"{np.max(winrates):.1f}%",
            f"{np.min(winrates):.1f}%",
        ],
        [
            "Max Drawdown",
            f"{np.mean(dds):.2f}%",
            f"{np.min(dds):.2f}%",
            f"{np.max(dds):.2f}%",
        ],
        [
            "Profit Factor",
            f"{np.mean(pfs):.2f}",
            f"{np.max(pfs):.2f}",
            f"{np.min(pfs):.2f}",
        ],
        [
            "Trades pro Run",
            f"{np.mean(trades):.1f}",
            f"{np.max(trades)}",
            f"{np.min(trades)}",
        ],
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
            cell.set_facecolor('#27ae60')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f2f2f2')

    plt.tight_layout()
    out_file = f"honest_pkl_simulation_{N_SIMULATIONS}_runs.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\n Stresstest-Chart gespeichert unter: {out_file}")


if __name__ == "__main__":
    main()