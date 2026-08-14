import os
from datetime import datetime
import joblib
import pandas as pd
import yfinance as yf
from features import build_feature_matrix


class Params:

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


ACCOUNT_SIZE = 6000.0
RISK_PCT = 0.01


def log_signal(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)

    with open("trade_signals.log", "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")


def main():
    if not os.path.exists("aapl_3055_model.pkl") or not os.path.exists(
        "aapl_3055_config.pkl"
    ):
        print("❌ 'aapl_3055_model.pkl' oder 'aapl_3055_config.pkl' fehlt!")
        return

    model = joblib.load("aapl_3055_model.pkl")
    config = joblib.load("aapl_3055_config.pkl")
    params = Params(config["params"])

    # Daten laden
    df_raw = yf.download(
        "AAPL", period="5d", interval="15m", progress=False, auto_adjust=False
    )
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

    # Features berechnen
    df = build_feature_matrix(df_raw, params)
    latest_row = df.iloc[-1:]
    latest_price = latest_row["close"].values[0]
    atr = latest_row["atr"].values[0]

    # Modell-Wahrscheinlichkeit
    X = latest_row[config["feature_cols"]]
    prob = model.predict_proba(X)[0, 1]

    print(
        f"🔍 [{datetime.now().strftime('%H:%M:%S')}] AAPL: ${latest_price:.2f} |"
        f" Prob: {prob:.4f} (Threshold: {params.signal_threshold:.4f})"
    )

    if prob >= params.signal_threshold:
        sl_distance = params.sl_atr_mult * atr
        tp_distance = params.tp_atr_mult * atr

        sl_price = latest_price - sl_distance
        tp_price = latest_price + tp_distance

        max_risk_dollars = ACCOUNT_SIZE * RISK_PCT
        shares = (
            int(max_risk_dollars / sl_distance) if sl_distance > 0 else 0
        )
        position_value = shares * latest_price
        expected_profit = shares * tp_distance

        msg = (
            f"\n🚀 === LONG SIGNAL GENERIERT! ===\n"
            f"  • Symbol:            AAPL (15m)\n"
            f"  • Entry Preis:       ${latest_price:.2f}\n"
            f"  • Stop Loss:         ${sl_price:.2f} (-${sl_distance:.2f})\n"
            f"  • Take Profit:       ${tp_price:.2f} (+${tp_distance:.2f})\n"
            f"  • Wahrscheinlichkeit: {prob*100:.1f}%\n"
            f"  ---------------------------------\n"
            f"  • Kontostand:        ${ACCOUNT_SIZE:,.2f}\n"
            f"  • Max. Risiko:       {RISK_PCT*100:.1f}% (${max_risk_dollars:.2f})\n"
            f"  • Kauf-Anzahl:       {shares} Shares\n"
            f"  • Positionsgröße:    ${position_value:,.2f}\n"
            f"  • Mgl. Gewinn (TP):  +${expected_profit:.2f}\n"
            f"================================="
        )
        log_signal(msg)


if __name__ == "__main__":
    main()