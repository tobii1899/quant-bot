"""
strategy_store.py
------------------
Verwaltet Persistierung der "Active Strategy": Parameter als JSON,
trainiertes Modell als Pickle. Thread-safe, da Optimizer (Schreiber) und
Notifier (Leser) parallel darauf zugreifen.
"""

from __future__ import annotations

import json
import pickle
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import BEST_MODEL_PATH, BEST_STRATEGY_PATH, STRATEGY_HISTORY_PATH

_lock = threading.Lock()


def save_strategy(params: dict, metrics: dict, model: Any, feature_columns: list[str]) -> None:
    """Speichert eine neue "beste" Strategie atomar (write-then-rename)."""
    with _lock:
        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "params": params,
            "metrics": metrics,
            "feature_columns": feature_columns,
        }

        tmp_json = BEST_STRATEGY_PATH.with_suffix(".json.tmp")
        tmp_json.write_text(json.dumps(payload, indent=2))
        tmp_json.replace(BEST_STRATEGY_PATH)

        tmp_pkl = BEST_MODEL_PATH.with_suffix(".pkl.tmp")
        with open(tmp_pkl, "wb") as f:
            pickle.dump(model, f)
        tmp_pkl.replace(BEST_MODEL_PATH)

        with open(STRATEGY_HISTORY_PATH, "a") as f:
            f.write(json.dumps(payload) + "\n")


def load_strategy() -> Optional[dict]:
    with _lock:
        if not BEST_STRATEGY_PATH.exists():
            return None
        return json.loads(BEST_STRATEGY_PATH.read_text())


def load_model() -> Optional[Any]:
    with _lock:
        if not BEST_MODEL_PATH.exists():
            return None
        with open(BEST_MODEL_PATH, "rb") as f:
            return pickle.load(f)


def get_current_best_score() -> float:
    """Liefert den Composite-Score der aktuell gespeicherten Strategie, 0 falls keine existiert."""
    strat = load_strategy()
    if not strat:
        return float("-inf")
    return strat["metrics"].get("composite_score", float("-inf"))


def strategy_age_seconds() -> Optional[float]:
    if not BEST_STRATEGY_PATH.exists():
        return None
    return time.time() - BEST_STRATEGY_PATH.stat().st_mtime
