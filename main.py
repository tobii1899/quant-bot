"""
main.py
--------
Einstiegspunkt des autonomen Trading-Backends.

Startet:
  1. Den Optimizer als Background-Thread (Endlos-Loop, siehe optimizer.run_forever).
  2. Einen Scheduler-Thread, der täglich zur konfigurierten Uhrzeit den
     Notifier-Check ausführt (notifier.run_daily_check).

Beendet sich sauber über CTRL+C (SIGINT) -- stop_event wird gesetzt, beide
Threads laufen ihren aktuellen Zyklus zu Ende und terminieren.

Nutzung:
    python main.py                     # startet vollen Hintergrundbetrieb
    python main.py --ticker MSFT       # anderer Ticker
    python main.py --once              # nur EIN Optimierungszyklus, dann Exit
    python main.py --visualize-only    # nur PNGs aus bestehender Strategie neu erzeugen
    python main.py --notify-only       # nur einmaligen Signal-Check + Notify
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime

from config import EXECUTION, NOTIFIER, OPTIMIZER
from notifier import run_daily_check
from optimizer import run_forever, run_optimization_cycle
from visualizer import generate_all_visuals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

stop_event = threading.Event()


def _optimizer_thread(ticker: str) -> None:
    run_forever(ticker=ticker, stop_event=stop_event)


def _notifier_scheduler_thread(ticker: str) -> None:
    """Prüft minütlich, ob die konfigurierte tägliche Check-Uhrzeit erreicht ist."""
    target_h, target_m = map(int, NOTIFIER.check_time_local.split(":"))
    last_run_date = None

    while not stop_event.is_set():
        now = datetime.now()
        if now.hour == target_h and now.minute == target_m and last_run_date != now.date():
            try:
                run_daily_check(ticker)
            except Exception:
                logger.exception("Notifier-Check fehlgeschlagen.")
            last_run_date = now.date()
        stop_event.wait(30)                                      


def _handle_shutdown(signum, frame) -> None:
    logger.info("Shutdown-Signal empfangen, beende Threads nach aktuellem Zyklus ...")
    stop_event.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomes Quant-Trading-Backend")
    parser.add_argument("--ticker", type=str, default=EXECUTION.ticker)
    parser.add_argument("--once", action="store_true", help="Nur ein Optimierungszyklus, dann Exit.")
    parser.add_argument("--visualize-only", action="store_true", help="Nur PNGs neu generieren.")
    parser.add_argument("--notify-only", action="store_true", help="Nur einmaliger Signal-Check.")
    args = parser.parse_args()

    if args.visualize_only:
        ok = generate_all_visuals(args.ticker)
        sys.exit(0 if ok else 1)

    if args.notify_only:
        run_daily_check(args.ticker)
        sys.exit(0)

    if args.once:
        logger.info("Führe einmaligen Optimierungszyklus aus (Ticker=%s) ...", args.ticker)
        run_optimization_cycle(args.ticker)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    logger.info("=== Autonomes Trading-Backend startet (Ticker=%s) ===", args.ticker)
    logger.info("Optimizer: %d Trials/Zyklus, %ds Pause zwischen Zyklen.",
                OPTIMIZER.n_trials_per_cycle, OPTIMIZER.sleep_between_cycles_sec)
    logger.info("Notifier: täglicher Check um %s Uhr, Kanäle=%s",
                NOTIFIER.check_time_local, NOTIFIER.enabled_channels)

    opt_thread = threading.Thread(target=_optimizer_thread, args=(args.ticker,), daemon=True)
    notify_thread = threading.Thread(target=_notifier_scheduler_thread, args=(args.ticker,), daemon=True)

    opt_thread.start()
    notify_thread.start()

    while not stop_event.is_set():
        time.sleep(1)

    opt_thread.join(timeout=10)
    notify_thread.join(timeout=10)
    logger.info("Backend sauber beendet.")


if __name__ == "__main__":
    main()
