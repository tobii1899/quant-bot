import optuna

                                              
STORAGE_URL = "sqlite:///strategies/optuna_study.db"
TARGET_STUDY_NAME = "quant_strategy_smc_fine_tune_v1"

try:
                                                      
    summaries = optuna.study.get_all_study_summaries(storage=STORAGE_URL)
    
    if not summaries:
        print("Keine Studies in der Datenbank gefunden!")
    else:
        print("Gefundene Studies in der DB:")
        for s in summaries:
            print(f"  • {s.study_name} ({s.n_trials} Trials)")
        
                                                     
        target_study_name = summaries[0].study_name
        # study = optuna.load_study(study_name=target_target_study_name if 'target_target_study_name' in locals() else target_study_name, storage=STORAGE_URL)
        study = optuna.load_study(study_name=TARGET_STUDY_NAME, storage=STORAGE_URL)
        best_trial = study.best_trial

        print("\n" + "=" * 60)
        print(f"BESTER TRIAL REKORD (Trial #{best_trial.number})")
        print(f"Score: {best_trial.value:.4f}")
        print("=" * 60)
        
        print("\n ERZIELTE METRIKEN (User Attrs):")
        for key, val in best_trial.user_attrs.items():
            print(f"  • {key}: {val}")

        print("\n GEWÄHLTE HYPERPARAMETER:")
        for key, val in sorted(best_trial.params.items()):
            print(f"  • {key}: {val}")
            
        print("=" * 60)

except Exception as e:
    print(f"Fehler beim Laden der Study: {e}")