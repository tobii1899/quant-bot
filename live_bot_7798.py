from datetime import datetime
import os
from zoneinfo import ZoneInfo
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import (
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)
import joblib
import pandas as pd
import requests
import yfinance as yf
from features import build_feature_matrix
from dotenv import load_dotenv

load_dotenv("keys.env")

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

ACCOUNT_SIZE = 6000.0  # Kontostand in USD
RISK_PCT = 0.01        # 1% Risiko pro Trade ($60.00)

ONE_TIME_TEST_RUN = False

alpaca_client = TradingClient(
    ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True
)

class Params:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def execute_paper_trade(symbol, shares, sl_price, tp_price):
    """Sendet eine automatisierte Bracket-Order mit eindeutiger Client-Order-ID für Trial #7798."""
    try:
        # Eindeutige ID zur Unterscheidung im Alpaca Dashboard
        order_custom_id = f"7798_{int(datetime.now().timestamp())}"
        
        req = MarketOrderRequest(
            symbol=symbol,
            qty=shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            client_order_id=order_custom_id,
            stop_loss=StopLossRequest(stop_price=round(sl_price, 2)),
            take_profit=TakeProfitRequest(limit_price=round(tp_price, 2)),
        )
        order = alpaca_client.submit_order(order_data=req)
        print(f"Alpaca Paper Trade [#7798] platziert! Order ID: {order.id} | Client ID: {order_custom_id}")
        return True
    except Exception as e:
        print(f"Fehler bei der Alpaca Order-Ausführung: {e}")
        return False


def send_discord_embed(
    latest_price,
    sl_price,
    tp_price,
    sl_distance,
    tp_distance,
    prob,
    shares,
    position_value,
    expected_profit,
    max_risk_dollars,
    is_test=False,
):
    local_now = datetime.now(ZoneInfo("Europe/Vienna"))
    timestamp_str = local_now.strftime("%d.%m.%Y %H:%M")

    title = (
        "🧪 AAPL TEST-SIGNAL (Trial #7798)"
        if is_test
        else "🚀 AAPL LONG SIGNAL GENERIERT (Trial #7798)"
    )
    color = 3447003 if is_test else 3066993  # Grün für Signal, Blau für Test

    embed = {
        "title": title,
        "description": f"**RandomForest SMC-Strategie Trial #7798** — *{timestamp_str} CEST*",
        "color": color,
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
                "inline": False,
            },
            {
                "name": "💰 Risikomanagement & Sizing",
                "value": (
                    f"• **Empf. Stückzahl:** `{shares} Shares`\n"
                    f"• **Positionsgröße:** `${position_value:,.2f}`\n"
                    f"• **Max. Risiko (1%):** `${max_risk_dollars:.2f}`\n"
                    f"• **Möglicher Gewinn:** `+${expected_profit:.2f}`"
                ),
                "inline": False,
            },
        ],
        "footer": {"text": "Quant Trading Bot v2 • RandomForest SMC Execution System"},
    }

    payload = {
        "username": "AAPL Bot #7798",
        "avatar_url": "https://cdn-icons-png.flaticon.com/512/25/25231.png",
        "embeds": [embed],
    }

    try:
        response = requests.post(
            DISCORD_WEBHOOK_URL, json=payload, timeout=10
        )
        if response.status_code in [200, 204]:
            print("📲 Discord-Benachrichtigung erfolgreich gesendet!")
        else:
            print(f"⚠️ Discord Fehler: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Fehler beim Senden an Discord: {e}")


def main():
    model_file = "aapl_7798_model.pkl"
    scaler_file = "aapl_7798_scaler.pkl"
    config_file = "aapl_7798_config.pkl"

    if not (os.path.exists(model_file) and os.path.exists(scaler_file) and os.path.exists(config_file)):
        print("❌ 'aapl_7798_model.pkl', 'aapl_7798_scaler.pkl' oder 'aapl_7798_config.pkl' fehlt!")
        return

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    config = joblib.load(config_file)
    
    params = Params(config["params"])
    feature_cols = config["feature_cols"]

    # 15m Kerzen für AAPL abrufen
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

    df = build_feature_matrix(df_raw, params)
    latest_row = df.iloc[-1:]
    latest_price = latest_row["close"].values[0]
    atr = latest_row["atr"].values[0]

    # Skalierung & Inferenz
    X = latest_row[feature_cols]
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0, 1]

    print(
        f"🔍 [{datetime.now().strftime('%H:%M:%S')}] AAPL: ${latest_price:.2f}"
        f" | Prob: {prob:.4f} (Threshold: {params.signal_threshold:.4f})"
    )

    if (prob >= params.signal_threshold) or ONE_TIME_TEST_RUN:
        sl_distance = params.sl_atr_mult * atr
        tp_distance = params.tp_atr_mult * atr

        sl_price = latest_price - sl_distance
        tp_price = latest_price + tp_distance

        max_risk_dollars = ACCOUNT_SIZE * RISK_PCT
        shares = int(max_risk_dollars / sl_distance) if sl_distance > 0 else 0
        position_value = shares * latest_price
        expected_profit = shares * tp_distance

        is_test = ONE_TIME_TEST_RUN and (prob < params.signal_threshold)

        send_discord_embed(
            latest_price,
            sl_price,
            tp_price,
            sl_distance,
            tp_distance,
            prob,
            shares,
            position_value,
            expected_profit,
            max_risk_dollars,
            is_test=is_test,
        )

        if ((prob >= params.signal_threshold) or ONE_TIME_TEST_RUN) and shares > 0:
            execute_paper_trade("AAPL", shares, sl_price, tp_price)


if __name__ == "__main__":
    main()