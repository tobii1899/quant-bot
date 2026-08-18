import os
import joblib
import pandas as pd

from backtester import run_backtest
from data_loader import load_data
from features import build_feature_matrix


class Params:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def main():
    TEST_DAYS = 60

    model_file = "aapl_7798_model.pkl"
    scaler_file = "aapl_7798_scaler.pkl"
    config_file = "aapl_7798_config.pkl"

    # 1. Modelldateien aus Trial #7798 laden
    if not (os.path.exists(model_file) and os.path.exists(scaler_file) and os.path.exists(config_file)):
        raise FileNotFoundError(
            "Fehler: Die Dateien 'aapl_7798_model.pkl', 'aapl_7798_scaler.pkl' oder "
            "'aapl_7798_config.pkl' wurden nicht im Verzeichnis gefunden!"
        )

    print("=" * 60)
    print(" Lade exportierte Modell-Dateien für Trial #7798...")
    print("=" * 60)

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    config = joblib.load(config_file)

    feature_cols = config['feature_cols']
    params_dict = config['params']
    trial_params = Params(params_dict)

    # 2. AAPL-Daten laden & SMC-Features generieren
    print(f" Lade AAPL 15m Daten & filtere auf die letzten {TEST_DAYS} Tage...")
    df_raw = load_data(ticker="AAPL")
    df = build_feature_matrix(df_raw, trial_params)
    df = df.dropna(subset=feature_cols).copy()

    # 3. Filtern auf das 60-Tage Out-of-Sample Fenster
    cutoff_date = df.index.max() - pd.Timedelta(days=TEST_DAYS)
    df_test = df[df.index >= cutoff_date].copy()

    if len(df_test) == 0:
        raise ValueError(f"Keine Daten im Zeitraum der letzten {TEST_DAYS} Tage gefunden.")

    X_test = df_test[feature_cols]

    # 4. Inferenz / Wahrscheinlichkeiten über den gespeicherten Scaler & RandomForest berechnen
    print(" Berechne Modell-Konfidenzen für die letzten 60 Tage...")
    X_test_scaled = scaler.transform(X_test)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    confidences = pd.Series(probs, index=df_test.index)
    signals = (confidences >= trial_params.signal_threshold).astype(int)

    # EOD-Timefilter (Keine Trades nach 21:00 Uhr ausführen)
    times = pd.to_datetime(df_test.index).time
    no_trade_mask = times >= pd.to_datetime("21:00").time()
    signals[no_trade_mask] = 0

    print(f" Max. Konfidenz im 60d-Fenster: {confidences.max():.4f}")
    print(f" Generierte Signale bei Threshold ({trial_params.signal_threshold:.3f}): {signals.sum()}")

    # Fallback-Anpassung, falls der Threshold im 60d-Window 0 Trades ergibt
    if signals.sum() == 0:
        print("\nHINWEIS: Bei Threshold 0.645 gab es im 60d-Fenster 0 Signale.")
        effective_thresh = confidences.quantile(0.85)
        signals = (confidences >= effective_thresh).astype(int)
        signals[no_trade_mask] = 0
        print(f" Schwellenwert auf {effective_thresh:.3f} angepasst ({signals.sum()} Signale)")

    # 5. Echten Backtest über deine backtester.py starten
    print("\n" + "=" * 60)
    print(" Starte Event-basierten Backtest über backtester.py...")
    print("=" * 60)

    res = run_backtest(
        df=df_test,
        signals=signals,
        confidences=confidences,
        sl_atr_mult=trial_params.sl_atr_mult,
        tp_atr_mult=trial_params.tp_atr_mult,
        max_hold_bars=999,
        force_eod_close=True,
    )

    # 6. Ergebnisse ausgeben
    metrics = res.to_dict()
    print("\n" + "=" * 50)
    print(f" BACKTEST ERGEBNISSE TRIAL #7798 (LETZTE {TEST_DAYS} TAGE):")
    print("=" * 50)
    print(f"  • Ausgeführte Trades: {metrics['n_trades']}")
    print(f"  • Winrate:            {metrics['winrate'] * 100:.1f}%")
    print(f"  • Ø CRV:              {metrics['avg_crv']:.2f}")
    print(f"  • Total Return:       {metrics['total_return_pct'] * 100:+.2f}%")
    print(f"  • Max Drawdown:       {metrics['max_drawdown_pct'] * 100:.2f}%")
    print(f"  • Profit Factor:      {metrics['profit_factor']:.2f}")
    print("=" * 50)

    # Detaillierte Trade-Liste anzeigen
    if res.trades:
        print("\n Detaillierte Trade-Übersicht:")
        trade_data = []
        for t in res.trades:
            trade_data.append({
                'Entry Time': t.entry_time.strftime('%Y-%m-%d %H:%M'),
                'Exit Time': t.exit_time.strftime('%Y-%m-%d %H:%M') if t.exit_time else '-',
                'Direction': t.direction,
                'Entry Price': f"${t.entry_price:.2f}",
                'Exit Price': f"${t.exit_price:.2f}" if t.exit_price else '-',
                'Reason': t.exit_reason,
                'PnL (%)': f"{t.pnl_pct * 100:+.2f}%",
                'PnL ($)': f"${t.pnl_abs:+.2f}",
            })
        print(pd.DataFrame(trade_data).to_string(index=False))


if __name__ == "__main__":
    main()