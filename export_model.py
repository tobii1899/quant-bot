import joblib
import pandas as pd
from xgboost import XGBClassifier

from data_loader import load_data
from features import build_feature_matrix, build_target_labels, get_feature_columns

PARAMS = {
    'sl_atr_mult': 0.828353939228746,
    'tp_atr_mult': 1.718289853662821,
    'signal_threshold': 0.565695054049191,
    'ema_fast': 7,
    'ema_slow': 29,
    'atr_period': 26,
    'bb_period': 16,
    'rsi_period': 22,
    'active_features': ['trend', 'price_action']
}

class Params:
    def __init__(self, d):
        for k, v in d.items(): setattr(self, k, v)

def export():
    print("Trainiere und speichere Trial #3055 Modell...")
    df_raw = load_data(ticker="AAPL")
    p = Params(PARAMS)
    df = build_feature_matrix(df_raw, p)
    
    df["target"] = build_target_labels(df, forward_bars=20, sl_atr_mult=p.sl_atr_mult, tp_atr_mult=p.tp_atr_mult)
    feature_cols = [c for c in get_feature_columns(df) if c != "target"]
    
    X, y = df[feature_cols], df["target"]
    
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    
                                                
    joblib.dump(model, "aapl_3055_model.pkl")
    joblib.dump({'feature_cols': feature_cols, 'params': PARAMS}, "aapl_3055_config.pkl")
    print("Dateien 'aapl_3055_model.pkl' und 'aapl_3055_config.pkl' erfolgreich erstellt!")

if __name__ == "__main__":
    export()