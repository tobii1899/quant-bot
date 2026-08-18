"""
optimizer.py
-------------
Optuna-getriebene Optimierungsschleife.

Pro Trial:
  1. Sample Feature-Kombination, Indikator-Parameter, SL/TP (in ATR-Multiplen),
     Signal-Schwelle und Modelltyp.
  2. Baue Feature-Matrix + Zielvariable.
  3. Trainiere über Walk-Forward-Splits (verhindert simples In-Sample-Overfitting).
  4. Backteste auf den Out-of-Sample-Test-Folds mit realistischer Execution.
  5. Composite-Score aus Winrate, CRV, Return, Drawdown, Profit-Factor.
  6. Falls Zielkriterien in ALLEN Folds im Mittel erreicht werden UND der Score
     besser ist als die aktuell gespeicherte Strategie -> als neue "Active
     Strategy" speichern.

Der Loop läuft potenziell endlos im Hintergrund (siehe run_forever) während
die zuletzt gespeicherte gute Strategie parallel vom Notifier genutzt wird.
"""

from __future__ import annotations

import logging
import time
import warnings
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from backtester import run_backtest
from config import CRITERIA, EXECUTION, OPTIMIZER, SEARCH_SPACE
from data_loader import load_data, train_test_walk_forward_split
from features import build_feature_matrix, build_target_labels, get_feature_columns
from strategy_store import get_current_best_score, save_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("optimizer")

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def _build_model(model_type: str, seed: int = 42):
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                random_state=seed, n_jobs=-1,
            )
        except ImportError:
            logger.warning("xgboost nicht installiert, fallback auf RandomForest.")
            model_type = "random_forest"

    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=200, max_depth=6, random_state=seed, n_jobs=-1)

    return LogisticRegression(max_iter=500)


def _sample_trial_params(trial: optuna.Trial) -> SimpleNamespace:
    ss = SEARCH_SPACE
    active_features = [
        g for g in ss.feature_flags
        if trial.suggest_categorical(f"use_{g}", [True, False])
    ]
    if not active_features:
        active_features = ["trend"]

    return SimpleNamespace(
        rsi_period=trial.suggest_int("rsi_period", *ss.rsi_period),
        ema_fast=trial.suggest_int("ema_fast", *ss.ema_fast),
        ema_slow=trial.suggest_int("ema_slow", *ss.ema_slow),
        atr_period=trial.suggest_int("atr_period", *ss.atr_period),
        bb_period=trial.suggest_int("bb_period", *ss.bb_period),
        sl_atr_mult=trial.suggest_float("sl_atr_mult", *ss.sl_atr_mult),
        tp_atr_mult=trial.suggest_float("tp_atr_mult", *ss.tp_atr_mult),
        # Untergrenze für Signal-Threshold auf mindestens 0.60 anheben
        signal_threshold=trial.suggest_float("signal_threshold", 0.60, 0.82),
        # Nur XGBoost & Random Forest verwenden
        model_type=trial.suggest_categorical("model_type", ["xgboost", "random_forest"]),
        active_features=active_features,
    )


def _composite_score(metrics: dict) -> float:
    """Gewichtete Score-Funktion mit angepassten Strafen gegen Low-Trade-Overfitting."""
    n_trades = metrics.get("n_trades", 0)
    total_return = metrics.get("total_return_pct", 0.0)
    avg_crv = metrics.get("avg_crv", 0.0)
    
                                                  
                                                                                         
    trade_penalty = 0.0
    if n_trades < 20:
        trade_penalty = (20 - n_trades) * 0.50

                      
    return_penalty = 0.0
    if total_return < 0:
        return_penalty = abs(total_return) * 10.0

                   
    crv_penalty = 0.0
    if avg_crv < 1.0:
        crv_penalty = (1.0 - avg_crv) * 2.0

    score = (
        2.0 * metrics["winrate"]
        + 1.5 * (min(avg_crv, 4.0) / 4.0)
        + 3.0 * np.tanh(total_return)
        - 3.0 * metrics["max_drawdown_pct"]
        + 1.0 * (min(metrics["profit_factor"], 3.0) / 3.0)
        - trade_penalty
        - return_penalty
        - crv_penalty
    )

    return float(score)


def objective(trial: optuna.Trial, df: pd.DataFrame) -> float:
    params = _sample_trial_params(trial)

    feat_df = build_feature_matrix(df, params)
    if len(feat_df) < 100:
        raise optuna.TrialPruned("Zu wenig Daten nach Feature-Berechnung.")

    labels = build_target_labels(
        feat_df, forward_bars=EXECUTION.max_hold_bars,
        sl_atr_mult=params.sl_atr_mult, tp_atr_mult=params.tp_atr_mult,
    )
    feature_cols = get_feature_columns(feat_df)

    fold_scores, fold_metrics = [], []

    for train_df, test_df in train_test_walk_forward_split(
        feat_df, OPTIMIZER.walk_forward_splits, OPTIMIZER.train_test_split_pct
    ):
        X_train = train_df[feature_cols]
        y_train = labels.loc[train_df.index]
        X_test = test_df[feature_cols]

        if y_train.nunique() < 2:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = _build_model(params.model_type)
        model.fit(X_train_scaled, y_train)

        proba = model.predict_proba(X_test_scaled)[:, 1]
        signals = pd.Series(
            np.where(proba >= params.signal_threshold, 1, 0), index=test_df.index
        )
        confidences = pd.Series(proba, index=test_df.index)

        result = run_backtest(
            test_df, signals, confidences,
            sl_atr_mult=params.sl_atr_mult, tp_atr_mult=params.tp_atr_mult,
        )
        if result.n_trades == 0:
            continue

        metrics = result.to_dict()
        metrics["composite_score"] = _composite_score(metrics)
        fold_scores.append(metrics["composite_score"])
        fold_metrics.append(metrics)

    if not fold_scores:
        raise optuna.TrialPruned("Keine validen Trades in irgendeinem Fold.")

    avg_score = float(np.mean(fold_scores))

                                                                                           
    trial.set_user_attr("avg_metrics", {
        k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0] if k != "composite_score"
    })
    trial.set_user_attr("params", vars(params))
    trial.set_user_attr("feature_cols", feature_cols)

    return avg_score


def _maybe_save_best(study: optuna.Study, df: pd.DataFrame) -> None:
    """Prüft besten Trial des Study-Objekts gegen Zielkriterien und speichert ggf."""
    best_trial = study.best_trial
    avg_metrics = best_trial.user_attrs.get("avg_metrics")
    params = best_trial.user_attrs.get("params")
    feature_cols = best_trial.user_attrs.get("feature_cols")
    if not avg_metrics or not params:
        return

    criteria_ok = (
        avg_metrics["n_trades"] >= CRITERIA.min_trades
        and avg_metrics["winrate"] >= CRITERIA.min_winrate
        and avg_metrics["avg_crv"] >= CRITERIA.min_crv
        and avg_metrics["total_return_pct"] >= CRITERIA.min_total_return_pct
        and avg_metrics["max_drawdown_pct"] <= CRITERIA.max_drawdown_pct
        and avg_metrics["profit_factor"] >= CRITERIA.min_profit_factor
    )
    if not criteria_ok:
        logger.info("Bester Trial erfüllt Zielkriterien noch nicht: %s", avg_metrics)
        return

    if best_trial.value <= get_current_best_score():
        logger.info("Bester Trial erfüllt Kriterien, aber kein Verbesserung ggü. gespeicherter Strategie.")
        return

                                                                          
    ns_params = SimpleNamespace(**params)
    feat_df = build_feature_matrix(df, ns_params)
    labels = build_target_labels(feat_df, EXECUTION.max_hold_bars, ns_params.sl_atr_mult, ns_params.tp_atr_mult)

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df[feature_cols])
    final_model = _build_model(ns_params.model_type)
    final_model.fit(X, labels.loc[feat_df.index])

    avg_metrics["composite_score"] = best_trial.value
    save_strategy(
        params=params,
        metrics=avg_metrics,
        model={"model": final_model, "scaler": scaler},
        feature_columns=feature_cols,
    )
    logger.info("✅ Neue Active Strategy gespeichert! Score=%.3f Metrics=%s", best_trial.value, avg_metrics)

                                                    
    try:
        from visualizer import generate_all_visuals
        generate_all_visuals()
    except Exception:
        logger.exception("Visualisierung nach neuem Strategie-Fund fehlgeschlagen.")


def run_optimization_cycle(ticker: str | None = None) -> optuna.Study:
    """Führt EINEN Optimierungszyklus (n_trials_per_cycle Trials) aus."""
    df = load_data(ticker)
    study = optuna.create_study(
        study_name=OPTIMIZER.study_name,
        storage=OPTIMIZER.storage_path,
        direction=OPTIMIZER.direction,
        load_if_exists=True,
                                                                                     
                                                                                    
                                                                                         
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, group=True, n_startup_trials=20),
    )
    logger.info("Study enthält bereits %d Trials aus vorherigen Zyklen.", len(study.trials))
    # study.optimize(lambda t: objective(t, df), n_trials=OPTIMIZER.n_trials_per_cycle, show_progress_bar=False)
    study.optimize(
        lambda t: objective(t, df), 
        n_trials=OPTIMIZER.n_trials_per_cycle, 
        show_progress_bar=False,
        catch=(ValueError, Exception) 
    )
    _log_progress_towards_criteria(study)
    _maybe_save_best(study, df)
    return study


def _log_progress_towards_criteria(study: optuna.Study) -> None:
    """Zeigt an, wie weit der beste bisherige Trial von JEDEM einzelnen Kriterium entfernt ist."""
    try:
        best = study.best_trial
        m = best.user_attrs.get("avg_metrics")
        if not m:
            return
        gaps = {
            "winrate": f"{m['winrate']*100:.1f}% (Ziel {CRITERIA.min_winrate*100:.0f}%)",
            "crv": f"{m['avg_crv']:.2f} (Ziel {CRITERIA.min_crv})",
            "drawdown": f"{m['max_drawdown_pct']*100:.1f}% (Limit {CRITERIA.max_drawdown_pct*100:.0f}%)",
            "profit_factor": f"{m['profit_factor']:.2f} (Ziel {CRITERIA.min_profit_factor})",
            "trades": f"{m['n_trades']:.0f} (Ziel {CRITERIA.min_trades})",
        }
        logger.info("Bester Trial bisher (Score=%.3f) -- %s", study.best_value, gaps)
    except ValueError:
        pass                                   


def run_forever(ticker: str | None = None, stop_event=None) -> None:
    """
    Endlos-Loop für den Hintergrundbetrieb. `stop_event` (threading.Event)
    erlaubt sauberes Beenden von außen (z.B. aus main.py bei Shutdown-Signal).
    """
    cycle = 0
    while stop_event is None or not stop_event.is_set():
        cycle += 1
        logger.info("=== Optimierungszyklus #%d startet (%d Trials) ===", cycle, OPTIMIZER.n_trials_per_cycle)
        try:
            run_optimization_cycle(ticker)
        except Exception:
            logger.exception("Fehler im Optimierungszyklus #%d -- Loop läuft weiter.", cycle)
        logger.info("Zyklus #%d beendet, Pause %ds.", cycle, OPTIMIZER.sleep_between_cycles_sec)
        time.sleep(OPTIMIZER.sleep_between_cycles_sec)