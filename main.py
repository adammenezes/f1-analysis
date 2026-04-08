"""
main.py
────────
CLI entry point for the F1 Predictor pipeline.

Commands
────────
  fetch      Download and cache race data
  train      Train the LightGBM model with walk-forward validation
  predict    Predict finishing order for an upcoming race
  simulate   Run Monte Carlo simulation for an upcoming race
  evaluate   Print held-out walk-forward evaluation metrics

Examples
────────
  python main.py fetch --start-year 2018 --end-year 2024
  python main.py train
  python main.py predict --season 2025 --round 5
  python main.py simulate --season 2025 --round 5 --n-sims 10000
  python main.py evaluate --plot
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("f1-predictor")

# ── Env config ────────────────────────────────────────────────────────────────
HISTORICAL_START = int(os.getenv("HISTORICAL_START_YEAR", 2010))
HISTORICAL_END   = int(os.getenv("HISTORICAL_END_YEAR",   2017))
FASTF1_START     = int(os.getenv("FASTF1_START_YEAR",     2018))
FASTF1_END       = int(os.getenv("FASTF1_END_YEAR",       2024))
ROLLING_WINDOW   = int(os.getenv("ROLLING_WINDOW",        5))
MIN_TRAIN_RACES  = int(os.getenv("MIN_TRAIN_RACES",       50))
MC_N_RUNS        = int(os.getenv("MONTE_CARLO_N_RUNS",    10_000))
DECAY            = float(os.getenv("SAMPLE_WEIGHT_DECAY", 0.15))
SEED             = int(os.getenv("RANDOM_SEED", 42))

MODEL_PATH = Path("data/processed/lgbm_model.txt")
PROCESSED_DIR = Path("data/processed")


# ── Sub-commands ──────────────────────────────────────────────────────────────

def cmd_fetch(args: argparse.Namespace) -> None:
    """Download and cache race data from Jolpica and FastF1."""
    from src.data.loader import fetch_historical_results, fetch_fastf1_results

    logger.info("Fetching historical data (%d–%d)…", args.start_year, args.end_year)
    fetch_historical_results(args.start_year, min(args.end_year, FASTF1_START - 1), force_refresh=args.refresh)

    if args.end_year >= FASTF1_START:
        logger.info("Fetching FastF1 data (%d–%d)…", FASTF1_START, args.end_year)
        fetch_fastf1_results(FASTF1_START, args.end_year, force_refresh=args.refresh)

    logger.info("✅ Data fetch complete.")


def cmd_train(args: argparse.Namespace) -> None:
    """Train LightGBM model with walk-forward validation and save."""
    from src.data.loader import load_full_dataset
    from src.data.features import build_features
    from src.models.elo import EloSystem
    from src.models.xgboost_model import F1LGBMModel
    from src.validation.walk_forward import WalkForwardValidator

    logger.info("Loading dataset…")
    df = load_full_dataset(HISTORICAL_START, HISTORICAL_END, FASTF1_START, FASTF1_END)

    logger.info("Fitting Elo system…")
    elo = EloSystem().fit(df)

    logger.info("Building features…")
    X, y = build_features(df, elo)

    # Feature columns passed to the model (exclude metadata)
    meta_cols = ["season", "round", "race_index", "driver_id", "constructor_id"]
    feature_cols = [c for c in X.columns if c not in meta_cols]

    if getattr(args, "basic", False):
        feature_cols = [
            "grid_position",
            "team_rolling_pos_10r",
            "team_circuit_avg_pos",
            "driver_rolling_pos_10r",
            "constructor_elo"
        ]
        logger.info("Basic mode enabled. Using feature set: %s", feature_cols)

    logger.info("Starting walk-forward validation…")
    model = F1LGBMModel()
    validator = WalkForwardValidator(
        min_train_races=MIN_TRAIN_RACES,
        sample_weight_decay=DECAY,
    )
    results = validator.run(model, X, y, feature_cols)

    print("\n" + "=" * 55)
    print("  Walk-Forward Evaluation Summary")
    print("=" * 55)
    print(results.summary().to_string())
    print()

    import json
    import mlflow
    from datetime import datetime

    timestamp_iso = datetime.utcnow().isoformat() + "Z"
    timestamp_fs = datetime.now().strftime("%Y%m%d_%H%M")
    
    feat_cnt = len(feature_cols)
    rho_val = float(results.summary().loc["spearman_rho", "mean"])
    
    # Run Naming Logic
    if hasattr(args, "name") and args.name:
        run_name = args.name
    else:
        run_name = f"f1_{feat_cnt}f_{timestamp_fs}_rho{rho_val:.3f}"
        
    run_desc = getattr(args, "desc", "None provided")

    # ── Option B: Custom CSV Folder Logging ──
    run_dir = PROCESSED_DIR / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    with open(run_dir / "description.txt", "w", encoding="utf-8") as dfdesc:
        dfdesc.write(run_desc)

    results_df = pd.DataFrame(results.fold_metrics)
    results_df.insert(0, "run_name", run_name)
    results_df.insert(1, "timestamp", timestamp_iso)
    results_df.to_csv(run_dir / "fold_history.csv", index=False)

    logger.info("Re-training final model on full dataset…")
    weights = validator.sample_weights(X)
    model.fit(X[feature_cols], y, sample_weight=weights)
    
    local_model_path = run_dir / "lgbm_model.txt"
    model.save(str(local_model_path))

    # Calculate feature importances
    try:
        importance_df = model.explain(X[feature_cols].head(500))
    except Exception:
        importance_df = model.feature_importance()
        
    importance_df.to_csv(run_dir / "feature_importance.csv", index=False)
    print("\n  Top Features:")
    print(importance_df.head(15).to_string(index=False))
    
    config_record = {
        "run_name": run_name,
        "timestamp": timestamp_iso,
        "features_count": feat_cnt,
        "basic_mode": getattr(args, "basic", False),
        **{f"out_{k}": v for k, v in results.summary().loc[:, "mean"].to_dict().items()}
    }
    with open(run_dir / "config.json", "w") as jf:
        json.dump(config_record, jf, indent=2)
        
    global_summary_path = PROCESSED_DIR / "runs" / "global_runs_summary.csv"
    summary_df = pd.DataFrame([{**config_record, "description": run_desc}])
    hdr = not global_summary_path.exists()
    summary_df.to_csv(global_summary_path, mode='a', index=False, header=hdr)
    
    # ── Option A: MLflow Logging ──
    mlflow.set_experiment("F1_Predictor")
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("mlflow.note.content", run_desc)
        mlflow.log_param("features_count", feat_cnt)
        mlflow.log_param("basic_mode", getattr(args, "basic", False))
        
        # Log aggregated metrics
        agg_metrics = results.summary()
        mlflow.log_metric("spearman_rho", float(agg_metrics.loc["spearman_rho", "mean"]))
        mlflow.log_metric("mae", float(agg_metrics.loc["mae", "mean"]))
        mlflow.log_metric("top1_acc", float(agg_metrics.loc["top1_acc", "mean"]))
        mlflow.log_metric("top3_acc", float(agg_metrics.loc["top3_acc", "mean"]))
        
        # MLflow saves the local model file dynamically inside the artifact payload
        mlflow.log_artifact(str(local_model_path))
        mlflow.log_artifact(str(run_dir / "fold_history.csv"))
        mlflow.log_artifact(str(run_dir / "feature_importance.csv"))

    logger.info("✅ Hybrid Logging complete. Custom isolated folder created at: %s", run_dir.name)


def cmd_predict(args: argparse.Namespace) -> None:
    """Predict finishing order for a specified race."""
    from src.data.loader import load_full_dataset
    from src.data.features import build_features
    from src.models.elo import EloSystem
    from src.models.xgboost_model import F1LGBMModel

    logger.info("Loading data and model…")
    df = load_full_dataset(HISTORICAL_START, HISTORICAL_END, FASTF1_START, FASTF1_END)
    elo = EloSystem().fit(df)
    X, y = build_features(df, elo)

    meta_cols = ["season", "round", "race_index", "driver_id", "constructor_id"]
    feature_cols = [c for c in X.columns if c not in meta_cols]

    model = F1LGBMModel()
    if MODEL_PATH.exists():
        model.load(str(MODEL_PATH))
    else:
        logger.warning("No saved model found. Training on all available data first…")
        from src.validation.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator(min_train_races=MIN_TRAIN_RACES, sample_weight_decay=DECAY)
        weights = validator.sample_weights(X)
        model.fit(X[feature_cols], y, sample_weight=weights)

    race_mask = (X["season"] == args.season) & (X["round"] == args.round)
    X_race = X[race_mask]

    if X_race.empty:
        logger.error(
            "No data found for season=%d round=%d. "
            "Check that the race has qualifying data loaded.",
            args.season, args.round,
        )
        sys.exit(1)

    driver_ids = X_race["driver_id"].tolist()
    ranking = model.predict_ranking(X_race[feature_cols], driver_ids)

    print(f"\n  🏁 Predicted Finishing Order — {args.season} Round {args.round}")
    print("  " + "─" * 45)
    for _, row in ranking.iterrows():
        print(f"  P{int(row['predicted_position']):2d}  {row['driver_id']}")


def cmd_simulate(args: argparse.Namespace) -> None:
    """Run Monte Carlo simulation for a specified race."""
    from src.data.loader import load_full_dataset
    from src.data.features import build_features
    from src.models.elo import EloSystem
    from src.models.xgboost_model import F1LGBMModel
    from src.models.monte_carlo import RaceParams, RaceSimulator, params_from_model_output

    logger.info("Preparing simulation…")
    df = load_full_dataset(HISTORICAL_START, HISTORICAL_END, FASTF1_START, FASTF1_END)
    elo = EloSystem().fit(df)
    X, y = build_features(df, elo)

    meta_cols = ["season", "round", "race_index", "driver_id", "constructor_id"]
    feature_cols = [c for c in X.columns if c not in meta_cols]

    model = F1LGBMModel()
    if MODEL_PATH.exists():
        model.load(str(MODEL_PATH))
    else:
        from src.validation.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator(min_train_races=MIN_TRAIN_RACES, sample_weight_decay=DECAY)
        weights = validator.sample_weights(X)
        model.fit(X[feature_cols], y, sample_weight=weights)

    race_mask = (X["season"] == args.season) & (X["round"] == args.round)
    X_race = X[race_mask]
    if X_race.empty:
        logger.error("No race data found for season=%d round=%d", args.season, args.round)
        sys.exit(1)

    driver_ids = X_race["driver_id"].tolist()
    ranking_df = model.predict_ranking(X_race[feature_cols], driver_ids)
    ranking_df["constructor_id"] = X_race["constructor_id"].values

    # Build circuit params (defaults — override per circuit in production)
    circuit_row = df[(df["season"] == args.season) & (df["round"] == args.round)]
    circuit_id = circuit_row["circuit_id"].iloc[0] if not circuit_row.empty else "unknown"
    circuit_params = RaceParams(circuit_id=circuit_id, total_laps=56)

    driver_params = params_from_model_output(ranking_df, circuit_params, df)
    sim = RaceSimulator(circuit_params, driver_params)

    logger.info("Running %d simulations for %s…", args.n_sims, circuit_id)
    results = sim.run(n_simulations=args.n_sims, seed=SEED)

    print(f"\n  🎲 Monte Carlo Results — {args.season} Round {args.round} ({circuit_id})")
    print(f"  N = {args.n_sims:,} simulations")
    print("  " + "─" * 50)
    summary = results.to_dataframe()[
        ["driver_id", "win_prob", "podium_prob", "points_prob", "expected_position"]
    ]
    for _, row in summary.iterrows():
        print(
            f"  {row['driver_id']:>5s}  "
            f"Win:{row['win_prob']:5.1%}  "
            f"Podium:{row['podium_prob']:5.1%}  "
            f"Points:{row['points_prob']:5.1%}  "
            f"Exp.Pos:{row['expected_position']:5.1f}"
        )


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Print walk-forward evaluation results, optionally with plots."""
    results_path = PROCESSED_DIR / "walk_forward_results.csv"
    if not results_path.exists():
        logger.error("No evaluation results found. Run `python main.py train` first.")
        sys.exit(1)

    df = pd.read_csv(results_path)

    print("\n  📊 Walk-Forward Evaluation Results")
    print("  " + "─" * 50)
    print(df[["fold", "season", "round", "spearman_rho", "top3_acc", "mae"]].to_string(index=False))

    agg = df[["spearman_rho", "top3_acc", "mae"]].agg(["mean", "std"])
    print("\n  Aggregated:")
    print(agg.to_string())

    if args.plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Walk-Forward Validation Results", fontweight="bold")

        for ax, col, label in zip(
            axes,
            ["spearman_rho", "top3_acc", "mae"],
            ["Spearman ρ", "Top-3 Accuracy", "MAE (positions)"],
        ):
            ax.plot(df["fold"], df[col], alpha=0.5, lw=1)
            ax.axhline(df[col].mean(), color="red", linestyle="--", label=f"Mean={df[col].mean():.3f}")
            ax.set_title(label)
            ax.set_xlabel("Fold (race)")
            ax.legend()
            sns.despine(ax=ax)

        plt.tight_layout()
        out_path = PROCESSED_DIR / "walk_forward_plot.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved to %s", out_path)
        plt.show()


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch MLflow UI."""
    import subprocess
    import webbrowser
    logger.info("Launching MLflow Dashboard on http://localhost:5000...")
    webbrowser.open("http://localhost:5000")
    try:
        subprocess.run([sys.executable, "-m", "mlflow", "ui"], check=True)
    except KeyboardInterrupt:
        logger.info("MLflow dashboard shutdown.")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="f1-predictor",
        description="Formula 1 race result predictor — data pipeline, ML model, and Monte Carlo simulation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # fetch
    p_fetch = sub.add_parser("fetch", help="Download and cache race data")
    p_fetch.add_argument("--start-year", type=int, default=HISTORICAL_START)
    p_fetch.add_argument("--end-year",   type=int, default=FASTF1_END)
    p_fetch.add_argument("--refresh",    action="store_true", help="Force re-download even if cached")

    # train
    p_train = sub.add_parser("train", help="Train model with walk-forward validation")
    p_train.add_argument("--basic", action="store_true", help="Train using only the top 5 basic features")
    p_train.add_argument("--name", type=str, help="Custom readable name for this specific model run")
    p_train.add_argument("--desc", type=str, default="None provided", help="Detailed description of what this model run is testing")

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Launch the MLflow web UI in browser")

    # predict
    p_pred = sub.add_parser("predict", help="Predict race finishing order")
    p_pred.add_argument("--season", type=int, required=True)
    p_pred.add_argument("--round",  type=int, required=True)

    # simulate
    p_sim = sub.add_parser("simulate", help="Monte Carlo race simulation")
    p_sim.add_argument("--season", type=int, required=True)
    p_sim.add_argument("--round",  type=int, required=True)
    p_sim.add_argument("--n-sims", type=int, default=MC_N_RUNS)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Show walk-forward evaluation metrics")
    p_eval.add_argument("--plot", action="store_true", help="Generate and show metric plots")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    command_map = {
        "fetch":    cmd_fetch,
        "train":    cmd_train,
        "predict":  cmd_predict,
        "simulate": cmd_simulate,
        "evaluate": cmd_evaluate,
        "dashboard": cmd_dashboard,
    }
    command_map[args.command](args)
