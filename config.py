"""
config.py
---------
Zentrale Konfiguration für das gesamte Trading-System.
Alle Module importieren ausschließlich aus dieser Datei -> ein einziger Ort
für Parameter-Änderungen (kein Hardcoding in den Modulen selbst).
"""

from dataclasses import dataclass, field
from pathlib import Path


                                                                            
       
                                                                            
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
STRATEGY_DIR = BASE_DIR / "strategies"
LOG_DIR = BASE_DIR / "logs"

for d in (DATA_DIR, OUTPUT_DIR, STRATEGY_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

BEST_STRATEGY_PATH = STRATEGY_DIR / "best_strategy.json"
BEST_MODEL_PATH = STRATEGY_DIR / "best_model.pkl"
STRATEGY_HISTORY_PATH = STRATEGY_DIR / "strategy_history.jsonl"


                                                                            
                   
                                                                            
@dataclass
class ExecutionConfig:
    ticker: str = "AAPL"
    timeframe: str = "15m"                                                                            
                                                                                                
    max_hold_bars: int = 16                                                                              
    no_overnight_hold: bool = True                                                                                
    session_open_time: str = "15:30"                                                     
    fee_rate: float = 0.001                                                      
    spread_pct: float = 0.0005                                                               
    slippage_pct: float = 0.0002                                     
    conservative_sl_tp: bool = True                                                          
    initial_capital: float = 10_000.0
    risk_per_trade_pct: float = 0.01                                                  


                                                                            
                                         
                                                                            
@dataclass
class StrategyCriteria:
    min_winrate: float = 0.55                              
    min_crv: float = 1.3                                  
    min_total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.07
    min_trades: int = 30                                                                
    min_profit_factor: float = 1.3


                                                                            
                                        
                                                                            
@dataclass
class SearchSpace:
                                                            
    rsi_period: tuple = (7, 30)
    ema_fast: tuple = (5, 20)
    ema_slow: tuple = (21, 100)
    atr_period: tuple = (7, 30)
    bb_period: tuple = (10, 40)

                                                              
    sl_atr_mult: tuple = (0.8, 2.5)
    tp_atr_mult: tuple = (1.2, 4.0)

                                                                     
    signal_threshold: tuple = (0.50, 0.75)

                                                                
    feature_flags: tuple = ("trend", "momentum", "volatility", "volume", "price_action")

                       
    model_types: tuple = ("xgboost", "random_forest", "logistic")


                                                                            
                
                                                                            
@dataclass
class OptimizerConfig:
    n_trials_per_cycle: int = 150                                 
    sleep_between_cycles_sec: int = 10                                                          
                                                                                           
    walk_forward_splits: int = 2                                                              
                                                                                                       
    train_test_split_pct: float = 0.7                                   
    study_name: str = "quant_strategy_search"
                                                                               
                                                                              
                                                                                     
    storage_path: str = f"sqlite:///{STRATEGY_DIR / 'optuna_study.db'}"
    direction: str = "maximize"                                                 


                                                                            
          
                                                                            
@dataclass
class NotifierConfig:
    enabled_channels: tuple = ("desktop",)                                     
    discord_webhook_url: str = ""                                                            
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    check_time_local: str = "08:30"                                        


EXECUTION = ExecutionConfig()
CRITERIA = StrategyCriteria()
SEARCH_SPACE = SearchSpace()
OPTIMIZER = OptimizerConfig()
NOTIFIER = NotifierConfig()