import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo
from features import build_feature_matrix


class Params:

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def run_backtest_with_account_table():
    # 1. Modell & Config laden
    model = joblib.load('aapl_3055_model.pkl')
    config = joblib.load('aapl_3055_config.pkl')
    params = Params(config['params'])

    # 2. Daten laden
    df_raw = yf.download(
        'AAPL', period='60d', interval='15m', progress=False, auto_adjust=False
    )
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    df_raw = df_raw.rename(
        columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        }
    )

    # Zeitzone auf lokale Zeit (Wien/Berlin) umstellen
    if df_raw.index.tz is None:
        df_raw.index = df_raw.index.tz_localize('UTC')
    df_raw.index = df_raw.index.tz_convert('Europe/Vienna')

    # 3. Features & Probabilities
    df = build_feature_matrix(df_raw, params)
    X = df[config['feature_cols']]
    df['prob'] = model.predict_proba(X)[:, 1]

    # 4. STARTWERTE FÜR KONTOSIMULATION
    START_BALANCE = 6000.0
    RISK_DOLLARS = 60.0  # Konstant 1% von $6.000
    current_balance = START_BALANCE

    trades = []
    in_position = False
    entry_price, sl_price, tp_price = 0.0, 0.0, 0.0
    sl_dist, tp_dist = 0.0, 0.0
    entry_time = None

    # 5. BACKTEST LOOP
    for i in range(len(df)):
        row = df.iloc[i]
        current_time = df.index[i]
        high, low, close = row['high'], row['low'], row['close']

        if in_position:
            # STOP LOSS HIT
            if low <= sl_price:
                shares = int(RISK_DOLLARS / sl_dist) if sl_dist > 0 else 0
                pnl = -RISK_DOLLARS
                current_balance += pnl
                growth_pct = (
                    (current_balance - START_BALANCE) / START_BALANCE
                ) * 100

                trades.append({
                    'Trade #': len(trades) + 1,
                    'Entry Zeit': entry_time.strftime('%d.%m.%Y %H:%M'),
                    'Exit Zeit': current_time.strftime('%d.%m.%Y %H:%M'),
                    'Entry Preis ($)': round(entry_price, 2),
                    'SL Preis ($)': round(sl_price, 2),
                    'TP Preis ($)': round(tp_price, 2),
                    'Exit Preis ($)': round(sl_price, 2),
                    'Ergebnis': '❌ LOSS (SL)',
                    'PnL ($)': round(pnl, 2),
                    'Kontostand ($)': round(current_balance, 2),
                    'Wachstum (%)': f'{growth_pct:+.2f}%',
                })
                in_position = False

            # TAKE PROFIT HIT
            elif high >= tp_price:
                shares = int(RISK_DOLLARS / sl_dist) if sl_dist > 0 else 0
                pnl = shares * tp_dist
                current_balance += pnl
                growth_pct = (
                    (current_balance - START_BALANCE) / START_BALANCE
                ) * 100

                trades.append({
                    'Trade #': len(trades) + 1,
                    'Entry Zeit': entry_time.strftime('%d.%m.%Y %H:%M'),
                    'Exit Zeit': current_time.strftime('%d.%m.%Y %H:%M'),
                    'Entry Preis ($)': round(entry_price, 2),
                    'SL Preis ($)': round(sl_price, 2),
                    'TP Preis ($)': round(tp_price, 2),
                    'Exit Preis ($)': round(tp_price, 2),
                    'Ergebnis': '✅ WIN (TP)',
                    'PnL ($)': round(pnl, 2),
                    'Kontostand ($)': round(current_balance, 2),
                    'Wachstum (%)': f'{growth_pct:+.2f}%',
                })
                in_position = False

        elif row['prob'] >= params.signal_threshold:
            in_position = True
            entry_price = close
            entry_time = current_time
            atr = row['atr']
            sl_dist = params.sl_atr_mult * atr
            tp_dist = params.tp_atr_mult * atr
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist

    # 6. TABELLE SPEICHERN
    df_trades = pd.DataFrame(trades)

    if not df_trades.empty:
        md_table = df_trades.to_markdown(index=False)
        with open('backtest_account_history.md', 'w', encoding='utf-8') as f:
            f.write('# AAPL Trading Bot (Trial #3055) — Kontoverlauf\n\n')
            f.write(
                '**Startkapital:** $6,000.00 | **Risiko pro Trade:** $60.00'
                ' (1%)\n\n'
            )
            f.write(md_table)

        df_trades.to_csv(
            'backtest_account_history.csv', index=False, encoding='utf-8'
        )

        print('====================================================')
        print('✅ Backtest erfolgreich beendet!')
        print(f'• Gesamtergebnis Kontostand: ${current_balance:,.2f}')
        print(
            '• Gesamtwachstum:'
            f' {((current_balance - START_BALANCE) / START_BALANCE) * 100:+.2f}%'
        )
        print('====================================================')
    else:
        print('Keine Trades im Zeitraum gefunden.')

    # 7. CHART ANZEIGEN
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['close'], color='gray', alpha=0.5, label='AAPL Price')

    if not df_trades.empty:
        entry_times = pd.to_datetime(df_trades['Entry Zeit'])
        plt.scatter(
            entry_times,
            df_trades['Entry Preis ($)'],
            color='green',
            marker='^',
            s=120,
            label='Trade Entry',
            zorder=5,
        )

        wins = df_trades[df_trades['Ergebnis'] == '✅ WIN (TP)']
        losses = df_trades[df_trades['Ergebnis'] == '❌ LOSS (SL)']

        plt.scatter(
            pd.to_datetime(wins['Exit Zeit']),
            wins['Exit Preis ($)'],
            color='lime',
            marker='o',
            s=80,
            label='TP Hit (Win)',
            zorder=5,
        )
        plt.scatter(
            pd.to_datetime(losses['Exit Zeit']),
            losses['Exit Preis ($)'],
            color='red',
            marker='x',
            s=80,
            label='SL Hit (Loss)',
            zorder=5,
        )

    plt.title('Realistischer Backtest — Chart mit Einstiegen')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == '__main__':
    run_backtest_with_account_table()