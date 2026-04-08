"""
scripts/ablation_study.py
─────────────────────────
Performs an automated Recursive Feature Elimination (RFE) loop.
It trains a walk-forward model on all X features, logs it to MLflow, 
pulls the SHAP values, drops the lowest performing feature, and repeats until 1 feature remains.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import mlflow
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.data.loader import load_full_dataset
from src.data.features import build_features
from src.models.elo import EloSystem
from src.models.xgboost_model import F1LGBMModel
from src.validation.walk_forward import WalkForwardValidator

import os

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ablation")

def run_ablation():
    HISTORICAL_START = int(os.getenv("HISTORICAL_START_YEAR", 2010))
    HISTORICAL_END   = int(os.getenv("HISTORICAL_END_YEAR",   2017))
    FASTF1_START     = int(os.getenv("FASTF1_START_YEAR",     2018))
    FASTF1_END       = int(os.getenv("FASTF1_END_YEAR",       2024))
    MIN_TRAIN_RACES  = int(os.getenv("MIN_TRAIN_RACES",       50))
    DECAY            = float(os.getenv("SAMPLE_WEIGHT_DECAY", 0.15))
    PROCESSED_DIR    = Path("data/processed")

    logger.info("Loading full dataset for Ablation Study...")
    df = load_full_dataset(HISTORICAL_START, HISTORICAL_END, FASTF1_START, FASTF1_END)
    elo = EloSystem().fit(df)
    X_full, y_full = build_features(df, elo)

    meta_cols = ["season", "round", "race_index", "driver_id", "constructor_id"]
    current_features = [c for c in X_full.columns if c not in meta_cols]
    
    experiment_id = "F1_Ablation_Study_RFE"
    mlflow.set_experiment(experiment_id)
    
    # Track the globally dropped features for the descriptive log
    dropped_log = []
    
    # We will loop until we run out of features
    while len(current_features) > 0:
        feat_cnt = len(current_features)
        
        timestamp_iso = datetime.utcnow().isoformat() + "Z"
        run_name = f"RFE_Step_{feat_cnt:02d}_features"
        
        logger.info("\n" + "="*55)
        logger.info(f" Starting RFE Run: {run_name}")
        logger.info("="*55)
        
        model = F1LGBMModel()
        validator = WalkForwardValidator(
            min_train_races=MIN_TRAIN_RACES,
            sample_weight_decay=DECAY,
        )
        
        results = validator.run(model, X_full, y_full, current_features)
        agg_metrics = results.summary()
        
        weights = validator.sample_weights(X_full)
        model.fit(X_full[current_features], y_full, sample_weight=weights)
        
        try:
            importance_df = model.explain(X_full[current_features].head(500))
        except Exception:
            importance_df = model.feature_importance()
            
        run_dir = PROCESSED_DIR / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        results_df = pd.DataFrame(results.fold_metrics)
        results_df.insert(0, "run_name", run_name)
        results_df.to_csv(run_dir / "fold_history.csv", index=False)
        importance_df.to_csv(run_dir / "feature_importance.csv", index=False)
        
        with mlflow.start_run(run_name=run_name):
            desc = (
                "Automated Recursive Feature Elimination Run.\n"
                f"Features Dropped Historically: {dropped_log}"
            )
            mlflow.set_tag("mlflow.note.content", desc)
            mlflow.log_param("features_count", feat_cnt)
            
            mlflow.log_metric("spearman_rho", float(agg_metrics.loc["spearman_rho", "mean"]))
            mlflow.log_metric("mae", float(agg_metrics.loc["mae", "mean"]))
            mlflow.log_metric("top1_acc", float(agg_metrics.loc["top1_acc", "mean"]))
            mlflow.log_metric("top3_acc", float(agg_metrics.loc["top3_acc", "mean"]))
            
            mlflow.log_artifact(str(run_dir / "fold_history.csv"))
            mlflow.log_artifact(str(run_dir / "feature_importance.csv"))
            
        # Find least important feature and drop it
        if len(current_features) > 1:
            if "mean_abs_shap" in importance_df.columns:
                worst_feature = importance_df.sort_values("mean_abs_shap", ascending=True).iloc[0]["feature"]
                val = importance_df.sort_values("mean_abs_shap", ascending=True).iloc[0]["mean_abs_shap"]
            else:
                worst_feature = importance_df.sort_values("importance", ascending=True).iloc[0]["feature"]
                val = importance_df.sort_values("importance", ascending=True).iloc[0]["importance"]
                
            logger.info(f"  [RFE RESULT] -> Dropping '{worst_feature}' (Score: {val:.4f})")
            
            current_features.remove(worst_feature)
            dropped_log.append(worst_feature)
        else:
            logger.info("  [RFE COMPLETE] Reached 1 feature remaining.")
            break

if __name__ == "__main__":
    run_ablation()
