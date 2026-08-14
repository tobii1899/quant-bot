"""
notifier.py
------------
Modulares Benachrichtigungssystem. Prüft vor Handelsbeginn die neuesten
Marktdaten gegen die aktuell gespeicherte "Active Strategy" und verschickt
bei validem Signal eine Benachrichtigung über einen oder mehrere Kanäle.

Kanäle sind als separate Funktionen implementiert (Strategy-Pattern) --
neue Kanäle einfach als weitere `send_via_*`-Funktion ergänzen und in
NOTIFIER.enabled_channels registrieren.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

import requests

from config import EXECUTION, NOTIFIER
from data_loader import load_data
from features import build_feature_matrix
from strategy_store import load_model, load_strategy

logger = logging.getLogger("notifier")


@dataclass
class Signal:
    ticker: str
    direction: str                                   
    confidence: float
    entry_price: float
    sl_price: float
    tp_price: float
    crv: float
    timestamp: datetime


                                                                            
                   
                                                                            
def check_for_signal(ticker: str | None = None) -> Signal | None:
    """Lädt aktuellste Daten, wendet die gespeicherte Strategie an, gibt Signal zurück (oder None)."""
    strategy = load_strategy()
    bundle = load_model()
    if not strategy or not bundle:
        logger.warning("Keine aktive Strategie vorhanden -- kein Signal-Check möglich.")
        return None

    ticker = ticker or strategy.get("params", {}).get("ticker", EXECUTION.ticker)
    df = load_data(ticker)

    params = SimpleNamespace(**strategy["params"])
    feat_df = build_feature_matrix(df, params)
    feature_cols = strategy["feature_columns"]

    latest = feat_df.iloc[[-1]]
    X = bundle["scaler"].transform(latest[feature_cols])
    proba = float(bundle["model"].predict_proba(X)[0, 1])

    last_close = float(df["close"].iloc[-1])
    atr = float(feat_df["atr"].iloc[-1])

    if proba >= params.signal_threshold:
        direction = "BUY"
        sl = last_close - params.sl_atr_mult * atr
        tp = last_close + params.tp_atr_mult * atr
    elif proba <= (1 - params.signal_threshold):
        direction = "SELL"
        sl = last_close + params.sl_atr_mult * atr
        tp = last_close - params.tp_atr_mult * atr
    else:
        logger.info("Kein valides Signal (Konfidenz %.1f%% unter Schwelle).", proba * 100)
        return None

    crv = abs(tp - last_close) / abs(last_close - sl) if last_close != sl else 0.0

    return Signal(
        ticker=ticker, direction=direction, confidence=proba,
        entry_price=last_close, sl_price=sl, tp_price=tp, crv=crv,
        timestamp=datetime.now(),
    )


                                                                            
                     
                                                                            
def format_message(signal: Signal) -> str:
    emoji = "🟢" if signal.direction == "BUY" else "🔴"
    return (
        f"{emoji} **{signal.direction} SIGNAL** — {signal.ticker}\n"
        f"Konfidenz: {signal.confidence*100:.1f}%\n"
        f"Entry: {signal.entry_price:.2f}\n"
        f"Stop-Loss: {signal.sl_price:.2f}\n"
        f"Take-Profit: {signal.tp_price:.2f}\n"
        f"CRV: {signal.crv:.2f}\n"
        f"Zeit: {signal.timestamp.strftime('%Y-%m-%d %H:%M')}"
    )


def send_via_discord(signal: Signal) -> bool:
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", NOTIFIER.discord_webhook_url)
    if not webhook_url:
        logger.warning("Kein Discord-Webhook konfiguriert (ENV DISCORD_WEBHOOK_URL).")
        return False
    try:
        resp = requests.post(webhook_url, json={"content": format_message(signal)}, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException:
        logger.exception("Discord-Benachrichtigung fehlgeschlagen.")
        return False


def send_via_telegram(signal: Signal) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", NOTIFIER.telegram_bot_token)
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", NOTIFIER.telegram_chat_id)
    if not token or not chat_id:
        logger.warning("Kein Telegram Bot-Token/Chat-ID konfiguriert.")
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": chat_id, "text": format_message(signal), "parse_mode": "Markdown",
        }, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException:
        logger.exception("Telegram-Benachrichtigung fehlgeschlagen.")
        return False


def send_via_desktop(signal: Signal) -> bool:
    """Lokale Desktop-Notification. Nutzt `plyer` falls verfügbar, sonst Konsolen-Fallback."""
    message = format_message(signal)
    try:
        from plyer import notification
        notification.notify(
            title=f"{signal.direction} Signal: {signal.ticker}",
            message=message, timeout=15,
        )
        return True
    except ImportError:
        logger.info("plyer nicht installiert -- Fallback auf Konsolen-Ausgabe:\n%s", message)
        print(message)
        return True
    except Exception:
        logger.exception("Desktop-Benachrichtigung fehlgeschlagen.")
        return False


CHANNEL_DISPATCH = {
    "discord": send_via_discord,
    "telegram": send_via_telegram,
    "desktop": send_via_desktop,
}


def notify(signal: Signal) -> None:
    for channel in NOTIFIER.enabled_channels:
        func = CHANNEL_DISPATCH.get(channel)
        if not func:
            logger.warning("Unbekannter Notification-Kanal: %s", channel)
            continue
        success = func(signal)
        logger.info("Kanal '%s': %s", channel, "OK" if success else "FEHLGESCHLAGEN")


def run_daily_check(ticker: str | None = None) -> None:
    """Einstiegspunkt für den täglichen Vor-Handels-Check (z.B. via Cron/Scheduler)."""
    logger.info("Starte täglichen Signal-Check für %s ...", ticker or EXECUTION.ticker)
    signal = check_for_signal(ticker)
    if signal is None:
        logger.info("Kein Trade-Signal heute.")
        return
    logger.info("Signal gefunden: %s", signal)
    notify(signal)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_daily_check()
