"""
src/validation/walk_forward.py
───────────────────────────────
Walk-forward (expanding window) validation for F1 race prediction.

The split is done at the *race* level, not the row level.  This prevents
any data from a given race weekend leaking into the model that predicts it.

Example
-------
    from src.validation.walk_forward import WalkForwardValidator
    wfv = WalkForwardValidator(min_train_races=50, step=1)
    for X_train, y_train, X_test, y_test, meta in wfv.split(X, y):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = wfv.evaluate(y_test, preds)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


@dataclass
class FoldMeta:
    """Metadata about a single walk-forward fold."""
    fold_index: int
    train_race_count: int
    test_season: int
    test_round: int
    test_race_index: int


@dataclass
class EvaluationResult:
    """Aggregated evaluation metrics across all walk-forward folds."""
    fold_metrics: list[dict] = field(default_factory=list)

    # Aggregate helpers
    def mean(self, metric: str) -> float:
        vals = [f[metric] for f in self.fold_metrics if metric in f]
        return float(np.mean(vals)) if vals else float("nan")

    def std(self, metric: str) -> float:
        vals = [f[metric] for f in self.fold_metrics if metric in f]
        return float(np.std(vals)) if vals else float("nan")

    def summary(self) -> pd.DataFrame:
        if not self.fold_metrics:
            return pd.DataFrame()
        df = pd.DataFrame(self.fold_metrics)
        numeric = df.select_dtypes(include="number")
        return numeric.agg(["mean", "std"]).T.rename(
            columns={"mean": "mean", "std": "std"}
        )


class WalkForwardValidator:
    """
    Walk-forward expanding-window validator.

    Parameters
    ----------
    min_train_races : int
        Minimum number of races required before making the first prediction.
    step : int
        Number of races to advance the test window each fold.
    regulation_break_reset : bool
        If True, whenever a regulation-break season starts the training window
        is reset to only include data from the new era.  Trades data quantity
        for relevance.
    sample_weight_decay : float
        Per-year exponential decay applied to training sample weights.
        0 = no decay.  0.15 means races 5 years ago have weight ~0.49.
    """

    def __init__(
        self,
        min_train_races: int = 50,
        step: int = 1,
        regulation_break_reset: bool = True,
        sample_weight_decay: float = 0.15,
    ) -> None:
        self.min_train_races = min_train_races
        self.step = step
        self.regulation_break_reset = regulation_break_reset
        self.sample_weight_decay = sample_weight_decay

        self._regulation_breaks = {2009, 2014, 2017, 2022}

    # ── Splitting ─────────────────────────────────────────────────────────────

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Generator[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, FoldMeta], None, None]:
        """
        Yields (X_train, y_train, X_test, y_test, meta) for each fold.
        X must contain columns: race_index, season, round.
        """
        assert "race_index" in X.columns, "X must have a 'race_index' column"

        race_indices = sorted(X["race_index"].unique())
        total_races = len(race_indices)

        fold_idx = 0
        test_ptr = self.min_train_races  # first test race index position

        while test_ptr < total_races:
            test_race_idx = race_indices[test_ptr]
            test_race_meta = X[X["race_index"] == test_race_idx].iloc[0]
            test_season = int(test_race_meta["season"])

            # ── Determine training window ──────────────────────────────────
            if self.regulation_break_reset and test_season in self._regulation_breaks:
                # Only train on data from the current era start
                era_start_idx = race_indices[max(0, test_ptr - 1)]
                train_mask = X["race_index"] < test_race_idx
                # Find where this era began — first race in break year
                era_first = X[X["season"] == test_season]["race_index"].min()
                train_mask = train_mask & (X["race_index"] >= era_first)
                # Fall back to full window if era just started
                if train_mask.sum() < 5:
                    train_mask = X["race_index"] < test_race_idx
            else:
                train_mask = X["race_index"] < test_race_idx

            test_mask = X["race_index"] == test_race_idx

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            if len(X_train) == 0 or len(X_test) == 0:
                test_ptr += self.step
                continue

            meta = FoldMeta(
                fold_index=fold_idx,
                train_race_count=int(X_train["race_index"].nunique()),
                test_season=test_season,
                test_round=int(test_race_meta["round"]),
                test_race_index=test_race_idx,
            )
            yield X_train, y_train, X_test, y_test, meta

            fold_idx += 1
            test_ptr += self.step

    def sample_weights(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        Compute per-sample exponential decay weights based on season.
        Most recent season's races receive weight = 1.0; older seasons decay.
        """
        if self.sample_weight_decay == 0:
            return np.ones(len(X_train))

        max_season = X_train["season"].max()
        age = max_season - X_train["season"]
        weights = np.exp(-self.sample_weight_decay * age.values.astype(float))
        return weights / weights.mean()  # normalise to mean = 1

    # ── Evaluation metrics ────────────────────────────────────────────────────

    @staticmethod
    def evaluate(
        y_true: pd.Series,
        y_pred: np.ndarray,
        top_n: list[int] = [1, 3, 5],
    ) -> dict:
        """
        Compute evaluation metrics for one race fold.

        Metrics:
            spearman_rho    - Spearman rank correlation between predicted and actual order
            top_N_acc       - fraction of top-N drivers correctly identified (any order)
            mae             - mean absolute error on position
        """
        y_true_arr = np.array(y_true, dtype=float)
        y_pred_arr = np.array(y_pred, dtype=float)

        results: dict = {}

        # Spearman rank correlation
        if len(y_true_arr) > 1:
            rho, _ = spearmanr(y_true_arr, y_pred_arr)
            results["spearman_rho"] = float(rho)
        else:
            results["spearman_rho"] = float("nan")

        # Mean Absolute Error
        results["mae"] = float(np.abs(y_true_arr - y_pred_arr).mean())

        # Top-N accuracy
        for n in top_n:
            actual_topn = set(np.argsort(y_true_arr)[:n])
            pred_topn = set(np.argsort(y_pred_arr)[:n])
            overlap = len(actual_topn & pred_topn)
            results[f"top{n}_acc"] = overlap / n
        return results

    def run(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: list[str],
    ) -> EvaluationResult:
        """
        Full walk-forward evaluation loop.

        Parameters
        ----------
        model       : Any object with fit(X, y, sample_weight=...) and predict(X).
        X           : Full feature DataFrame (must have race_index, season, round).
        y           : Full target series.
        feature_cols: Model input columns (excludes metadata columns).

        Returns EvaluationResult with per-fold and aggregate metrics.
        """
        evaluation = EvaluationResult()

        for X_train, y_train, X_test, y_test, meta in self.split(X, y):
            weights = self.sample_weights(X_train)

            # Drop metadata columns before passing to model
            X_tr = X_train[feature_cols]
            X_te = X_test[feature_cols]

            model.fit(X_tr, y_train, sample_weight=weights)

            # Extract feature importances (e.g., gain) assigned by the model for this specific fold
            try:
                imp_df = model.feature_importance()
                importance_dict = {f"imp_{row['feature']}": row["importance"] for _, row in imp_df.iterrows()}
            except Exception:
                importance_dict = {}

            preds = model.predict(X_te)
            metrics = self.evaluate(y_test, preds)
            metrics.update(importance_dict)
            metrics.update(
                {
                    "fold": meta.fold_index,
                    "season": meta.test_season,
                    "round": meta.test_round,
                    "train_races": meta.train_race_count,
                }
            )
            evaluation.fold_metrics.append(metrics)

            if meta.fold_index % 20 == 0:
                logger.info(
                    "Fold %3d | S%d R%2d | Spearman=%.3f | Top3=%.1f%% | MAE=%.2f",
                    meta.fold_index,
                    meta.test_season,
                    meta.test_round,
                    metrics.get("spearman_rho", float("nan")),
                    metrics.get("top3_acc", 0) * 100,
                    metrics.get("mae", float("nan")),
                )

        logger.info(
            "Walk-forward complete. %d folds. Mean Spearman=%.3f, Mean Top3=%.1f%%",
            len(evaluation.fold_metrics),
            evaluation.mean("spearman_rho"),
            evaluation.mean("top3_acc") * 100,
        )
        return evaluation
