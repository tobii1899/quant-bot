import os
from datetime import datetime
import joblib
import pandas as pd
import requests
import yfinance as yf
from features import build_feature_matrix

# ==============================================================================
# DISCORD CONFIGURATION
# ==============================================================================
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1537949378126286879/k1RsdmZjARU3ZlGcdGe_WwgdOMkCXHV7dd1Nz_v3j1qyZ2VetKshjzC516dSUeBVa1vi"

ACCOUNT_SIZE = 6000.0  # Kontostand in USD
RISK_PCT = 0.01        # 1% Risiko pro Trade ($60.00)
# ==============================================================================


class Params:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def send_discord_embed(latest_price, sl_price, tp_price, sl_distance, tp_distance, prob, shares, position_value, expected_profit, max_risk_dollars):
    """Verschickt eine hochauflösende Rich-Embed-Nachricht direkt an den Discord-Kanal."""
    timestamp_str = datetime.now().strftime("%d.%m.%Y %H:%M")

    embed = {
        "title": "🚀 AAPL LONG SIGNAL GENERIERT",
        "description": f"**Strategie Trial #3055** — *{timestamp_str} CEST*",
        "color": 3066993,  # Grüner Farbcode (Hex #2ECC71)
        "fields": [
            {
                "name": "📈 Trade Details",
                "value": (
                    f"• **Symbol:** AAPL (15m)\n"
                    f"• **Entry Preis:** `${latest_price:.2f}`\n"
                    f"• **Stop Loss:** `${sl_price:.2f}` (-${sl_distance:.2f})\n"
                    f"• **Take Profit:** `${tp_price:.2f}` (+${tp_distance:.2f})\n"
                    f"• **Modell-Konfidenz:** `{prob*100:.1f}%`"
                ),
                "inline": False
            },
            {
                "name": "💰 Risikomanagement & Sizing",
                "value": (
                    f"• **Empf. Stückzahl:** `{shares} Shares`\n"
                    f"• **Positionsgröße:** `${position_value:,.2f}`\n"
                    f"• **Max. Risiko (1%):** `${max_risk_dollars:.2f}`\n"
                    f"• **Möglicher Gewinn:** `+${expected_profit:.2f}`"
                ),
                "inline": False
            }
        ],
        "footer": {
            "text": "Quant Trading Bot • Automated Execution System"
        }
    }

    payload = {
        "username": "AAPL Trading Bot",
        "avatar_url": "https://cdn-icons-png.flaticon.com/512/25/25231.png",
        "embeds": [embed]
    }

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code in [200, 204]:
            print("📲 Discord-Benachrichtigung erfolgreich gesendet!")
        else:
            print(f"⚠️ Discord Fehler: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Fehler beim Senden an Discord: {e}")


def main():
    if not os.path.exists("aapl_3055_model.pkl") or not os.path.exists("aapl_3055_config.pkl"):
        print("❌ 'aapl_3055_model.pkl' oder 'aapl_3055_config.pkl' fehlt!")
        return

    model = joblib.load("aapl_3055_model.pkl")
    config = joblib.load("aapl_3055_config.pkl")
    params = Params(config["params"])

    # 1. Neueste 15m Daten laden
    df_raw = yf.download("AAPL", period="5d", interval="15m", progress=False, auto_adjust=False)
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    df_raw = df_raw.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
    })

    # 2. Features berechnen
    df = build_feature_matrix(df_raw, params)
    latest_row = df.iloc[-1:]
    latest_price = latest_row["close"].values[0]
    atr = latest_row["atr"].values[0]

    # 3. Modell-Wahrscheinlichkeit auswerten
    X = latest_row[config["feature_cols"]]
    prob = model.predict_proba(X)[0, 1]

    print(f"🔍 [{datetime.now().strftime('%H:%M:%S')}] AAPL: ${latest_price:.2f} | Prob: {prob:.4f} (Threshold: {params.signal_threshold:.4f})")

    # 4. Signal-Prüfung
    if prob >= params.signal_threshold:
        sl_distance = params.sl_atr_mult * atr
        tp_distance = params.tp_atr_mult * atr

        sl_price = latest_price - sl_distance
        tp_price = latest_price + tp_distance

        max_risk_dollars = ACCOUNT_SIZE * RISK_PCT
        shares = int(max_risk_dollars / sl_distance) if sl_distance > 0 else 0
        position_value = shares * latest_price
        expected_profit = shares * tp_distance

        send_discord_embed(
            latest_price, sl_price, tp_price, sl_distance, tp_distance,
            prob, shares, position_value, expected_profit, max_risk_dollars
        )


if __name__ == "__main__":
    main()