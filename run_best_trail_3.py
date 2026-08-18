import os
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


def run_model_stresstest(model_prefix, n_simulations=1000, test_days=60, df_raw=None):
    model_file = f"{model_prefix}_model.pkl"
    config_file = f"{model_prefix}_config.pkl"
    scaler_file = f"{model_prefix}_scaler.pkl"

    if not (os.path.exists(model_file) and os.path.exists(config_file)):
        raise FileNotFoundError(f"Fehler: '{model_file}' oder '{config_file}' fehlt.")

    print(f"\n" + "=" * 60)
    print(f" STARTE STRESSTEST FÜR MODELL: {model_prefix.upper()}")
    print("=" * 60)

    model = joblib.load(model_file)
    config = joblib.load(config_file)
    scaler = joblib.load(scaler_file) if os.path.exists(scaler_file) else None

    feature_cols = config['feature_cols']
    params_dict = config['params']
    trial_params = Params(params_dict)

    df = build_feature_matrix(df_raw, trial_params)
    df = df.dropna(subset=feature_cols).copy()

    cutoff_date = df.index.max() - pd.Timedelta(days=test_days)
    df_test = df[df.index >= cutoff_date].copy()
    X_test = df_test[feature_cols]

    if scaler is not None:
        X_test_in = scaler.transform(X_test)
    else:
        X_test_in = X_test

    probs = model.predict_proba(X_test_in)[:, 1]
    base_confidences = pd.Series(probs, index=df_test.index)
    
    thresh = trial_params.signal_threshold
    base_signals = (base_confidences >= thresh).astype(int)

    times = pd.to_datetime(df_test.index).time
    no_trade_mask = times >= pd.to_datetime("21:00").time()
    base_signals[no_trade_mask] = 0

    if base_signals.sum() == 0:
        thresh = base_confidences.quantile(0.85)
        base_signals = (base_confidences >= thresh).astype(int)
        base_signals[no_trade_mask] = 0

    results = []
    start_time = time.time()

    for run_id in range(1, n_simulations + 1):
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

    elapsed = time.time() - start_time
    print(f" {n_simulations:,} Runs in {elapsed:.2f}s abgeschlossen.")

    # Einzelnes Chart generieren
    returns = [r['return'] for r in results]
    avg_return = np.mean(returns)
    win_pct_runs = (sum(1 for r in returns if r > 0) / n_simulations) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.hist(returns, bins=25, color='#3498db', edgecolor='black', alpha=0.75)
    ax1.axvline(avg_return, color='red', linestyle='--', linewidth=2, label=f'Ø Return ({avg_return:+.2f}%)')
    ax1.set_title(f"Return-Verteilung {model_prefix.upper()} (Letzte {test_days} Tage - {n_simulations:,} Runs)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Total Return (%)")
    ax1.set_ylabel("Anzahl Durchläufe")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    ax2.axis('off')
    summary_data = [
        ["Metrik", f"Durchschnitt ({n_simulations:,} Runs)", "Best-Case", "Worst-Case"],
        ["Profitable Runs", f"{win_pct_runs:.2f}%", "-", "-"],
        ["Total Return", f"{avg_return:+.2f}%", f"{np.max(returns):+.2f}%", f"{np.min(returns):+.2f}%"],
        ["Winrate", f"{np.mean([r['winrate'] for r in results]):.1f}%", f"{np.max([r['winrate'] for r in results]):.1f}%", f"{np.min([r['winrate'] for r in results]):.1f}%"],
        ["Max Drawdown", f"{np.mean([r['max_dd'] for r in results]):.2f}%", f"{np.min([r['max_dd'] for r in results]):.2f}%", f"{np.max([r['max_dd'] for r in results]):.2f}%"],
        ["Profit Factor", f"{np.mean([r['pf'] for r in results]):.2f}", f"{np.max([r['pf'] for r in results]):.2f}", f"{np.min([r['pf'] for r in results]):.2f}"],
        ["Trades in 60 Tagen", f"{np.mean([r['trades'] for r in results]):.1f}", f"{np.max([r['trades'] for r in results])}", f"{np.min([r['trades'] for r in results])}"],
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
    out_file = f"stresstest_{model_prefix}_60d.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()

    return results


def plot_comparison(res_3055, res_7798, n_simulations=1000, test_days=60):
    ret_3055 = [r['return'] for r in res_3055]
    ret_7798 = [r['return'] for r in res_7798]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Histogramm-Vergleich
    ax1.hist(ret_3055, bins=25, color='#e74c3c', alpha=0.6, label=f'Trial #3055 (Alt - XGBoost) Ø {np.mean(ret_3055):+.2f}%', edgecolor='black')
    ax1.hist(ret_7798, bins=25, color='#2ecc71', alpha=0.6, label=f'Trial #7798 (Neu - RandomForest) Ø {np.mean(ret_7798):+.2f}%', edgecolor='black')
    ax1.set_title(f"VERGLEICH: Trial #3055 vs. Trial #7798 (Letzte {test_days} Tage - {n_simulations:,} Runs)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Total Return (%)")
    ax1.set_ylabel("Anzahl Durchläufe")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Vergleichstabelle
    ax2.axis('off')
    comp_data = [
        ["Metrik", "Trial #3055 (Alt / XGBoost)", "Trial #7798 (Neu / RandomForest)", "Differenz / Delta"],
        ["Profitable Runs", f"{(sum(1 for r in ret_3055 if r > 0)/n_simulations)*100:.1f}%", f"{(sum(1 for r in ret_7798 if r > 0)/n_simulations)*100:.1f}%", f"{((sum(1 for r in ret_7798 if r > 0)-sum(1 for r in ret_3055 if r > 0))/n_simulations)*100:+.1f}%"],
        ["Ø Total Return", f"{np.mean(ret_3055):+.2f}%", f"{np.mean(ret_7798):+.2f}%", f"{np.mean(ret_7798)-np.mean(ret_3055):+.2f}%"],
        ["Ø Winrate", f"{np.mean([r['winrate'] for r in res_3055]):.1f}%", f"{np.mean([r['winrate'] for r in res_7798]):.1f}%", f"{np.mean([r['winrate'] for r in res_7798])-np.mean([r['winrate'] for r in res_3055]):+.1f}%"],
        ["Ø Max Drawdown", f"{np.mean([r['max_dd'] for r in res_3055]):.2f}%", f"{np.mean([r['max_dd'] for r in res_7798]):.2f}%", f"{np.mean([r['max_dd'] for r in res_7798])-np.mean([r['max_dd'] for r in res_3055]):+.2f}%"],
        ["Ø Profit Factor", f"{np.mean([r['pf'] for r in res_3055]):.2f}", f"{np.mean([r['pf'] for r in res_7798]):.2f}", f"{np.mean([r['pf'] for r in res_7798])-np.mean([r['pf'] for r in res_3055]):+.2f}"],
        ["Ø Trades in 60 Tagen", f"{np.mean([r['trades'] for r in res_3055]):.1f}", f"{np.mean([r['trades'] for r in res_7798]):.1f}", f"{np.mean([r['trades'] for r in res_7798])-np.mean([r['trades'] for r in res_3055]):+.1f}"],
    ]

    table = ax2.table(cellText=comp_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f8f9fa')

    plt.tight_layout()
    out_comp = "vergleich_3055_vs_7798_60d.png"
    plt.savefig(out_comp, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n Vergleichs-Chart gespeichert unter: {out_comp}")


def main():
    df_raw = load_data(ticker="AAPL")

    # 1. Altes Modell (Trial #3055) durchsimulieren
    res_3055 = run_model_stresstest("aapl_3055", n_simulations=1000, test_days=60, df_raw=df_raw)

    # 2. Neues Modell (Trial #7798) durchsimulieren
    res_7798 = run_model_stresstest("aapl_7798", n_simulations=1000, test_days=60, df_raw=df_raw)

    # 3. Direktvergleich erstellen
    plot_comparison(res_3055, res_7798, n_simulations=1000, test_days=60)


if __name__ == "__main__":
    main()