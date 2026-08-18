import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from data_loader import load_data
from features import build_feature_matrix, build_target_labels, get_feature_columns

PARAMS = {
    'rsi_period': 24,
    'ema_fast': 15,
    'ema_slow': 86,
    'atr_period': 29,
    'bb_period': 30,
    'sl_atr_mult': 0.9482251770062909,
    'tp_atr_mult': 2.1861398368585454,
    'signal_threshold': 0.6450257142067576,
    'model_type': 'random_forest',
    'active_features': ['volatility', 'smc_order_blocks', 'smc_liquidity', 'smc_killzones']
}

FEATURE_COLS = [
    'atr', 'atr_pct', 'bb_pct_b', 'realized_vol_20',
    'smc_is_ob_candidate', 'smc_in_ob_zone', 'smc_dist_to_ob_top_atr',
    'smc_equal_highs', 'smc_equal_lows', 'smc_bsl_sweep', 'smc_ssl_sweep',
    'smc_killzone_london', 'smc_killzone_ny', 'smc_killzone_london_close'
]

class Params:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

def export():
    print("Trainiere und speichere Trial #7798 Modell (RandomForest)...")
    df_raw = load_data(ticker="AAPL")
    p = Params(PARAMS)
    df = build_feature_matrix(df_raw, p)
    
    # Target-Labels mit den EOD-Triple-Barrier Parameter von Trial #7798 erstellen
    df["target"] = build_target_labels(
        df, 
        forward_bars=20, 
        sl_atr_mult=p.sl_atr_mult, 
        tp_atr_mult=p.tp_atr_mult
    )
    
    # Bereinigung von NaNs aus Indikator-Lookbacks
    df = df.dropna(subset=FEATURE_COLS + ["target"]).copy()
    
    X, y = df[FEATURE_COLS], df["target"]
    
    # Features skalieren
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # RandomForest-Modell mit den exakten Optimizer-Settings instanziieren
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    
    # Modell, Scaler und Config abspeichern
    joblib.dump(model, "aapl_7798_model.pkl")
    joblib.dump(scaler, "aapl_7798_scaler.pkl")
    joblib.dump({'feature_cols': FEATURE_COLS, 'params': PARAMS}, "aapl_7798_config.pkl")
    
    print("Dateien 'aapl_7798_model.pkl', 'aapl_7798_scaler.pkl' und 'aapl_7798_config.pkl' erfolgreich erstellt!")

if __name__ == "__main__":
    export()