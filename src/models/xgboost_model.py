"""
src/models/xgboost_model.py
────────────────────────────
LightGBM-based race position predictor.

Architecture
────────────
• LGBMRegressor trained to predict raw finish position (1–20).
  Position is then argsorted per-race to produce a final ranking.
• Categorical features (regulation_era) handled natively by LightGBM.
• Sample weights decay exponentially with age (older seasons down-weighted).
• SHAP values exposed via explain() for interpretability.

Walk-forward usage
──────────────────
    from src.models.xgboost_model import F1LGBMModel
    from src.validation.walk_forward import WalkForwardValidator

    model = F1LGBMModel()
    validator = WalkForwardValidator(min_train_races=50)
    results = validator.run(model, X, y, feature_cols)
    print(results.summary())

Single-race prediction
──────────────────────
    model.fit(X_train, y_train, sample_weight=weights)
    ranked = model.predict_ranking(X_race)   # returns driver ids sorted 1st→last
"""

from __future__ import annotations

import logging
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Default hyperparameters ────────────────────────────────────────────────────
# Tuned conservatively to avoid overfitting on F1's small dataset.
# Run a proper Optuna sweep on a hold-out season before going to production.
DEFAULT_PARAMS = {
    "objective": "regression_l1",   # MAE objective — robust to outliers (DNFs etc.)
    "metric": "mae",
    "n_estimators": 800,
    "learning_rate": 0.03,
    "num_leaves": 31,                # Low complexity — F1 data is small
    "max_depth": 5,
    "min_child_samples": 15,         # Prevents tiny leaf nodes
    "subsample": 0.75,               # Row sampling for randomness
    "colsample_bytree": 0.70,        # Feature sampling
    "reg_alpha": 0.1,                # L1 regularisation
    "reg_lambda": 1.5,               # L2 regularisation
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

CATEGORICAL_FEATURES = ["regulation_era"]


class F1LGBMModel:
    """
    LightGBM regressor wrapped to produce ranked predictions per race.

    The underlying model predicts a continuous score for each driver.
    Lower score = predicted earlier finish position.
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self._model: Optional[lgb.LGBMRegressor] = None
        self._feature_cols: list[str] = []

    # ── scikit-learn compatible interface ──────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[list] = None,
    ) -> "F1LGBMModel":
        """
        Fit the model on the training data.

        Parameters
        ----------
        X              : Feature DataFrame (metadata columns already removed).
        y              : Finish position target.
        sample_weight  : Per-row sample weights (from WalkForwardValidator).
        eval_set       : [(X_val, y_val)] for early stopping — optional.
        """
        self._feature_cols = list(X.columns)

        # Identify categorical columns present in X
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

        callbacks = []
        if eval_set:
            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]

        self._model = lgb.LGBMRegressor(**self.params)
        self._model.fit(
            X,
            y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            categorical_feature=cat_cols if cat_cols else "auto",
            callbacks=callbacks if callbacks else None,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw score per driver (lower = predicted better position)."""
        assert self._model is not None, "Model not fitted yet"
        return self._model.predict(X)

    def predict_ranking(
        self, X: pd.DataFrame, driver_ids: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Predict a ranked finishing order for a single race.

        Parameters
        ----------
        X          : Feature DataFrame for all drivers in the race.
        driver_ids : Optional list of driver IDs aligned with X rows.

        Returns
        -------
        DataFrame with columns [driver_id, predicted_position, score].
        """
        scores = self.predict(X)
        order = np.argsort(scores)   # ascending: lowest score = P1

        ids = driver_ids or list(range(len(scores)))
        return pd.DataFrame(
            {
                "driver_id": [ids[i] for i in order],
                "predicted_position": range(1, len(order) + 1),
                "score": scores[order],
            }
        )

    # ── Interpretability ──────────────────────────────────────────────────────

    def explain(self, X: pd.DataFrame, max_display: int = 15) -> pd.DataFrame:
        """
        Return a DataFrame of mean |SHAP| values per feature.
        Requires the ``shap`` package.
        """
        try:
            import shap
        except ImportError:
            raise ImportError("Install shap: pip install shap")

        assert self._model is not None, "Model not fitted yet"
        explainer = shap.TreeExplainer(self._model)
        shap_values = explainer.shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        result = pd.DataFrame(
            {"feature": X.columns, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False).head(max_display)
        return result

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Return LightGBM built-in feature importance (gain or split)."""
        assert self._model is not None, "Model not fitted yet"
        imp = self._model.booster_.feature_importance(importance_type=importance_type)
        return (
            pd.DataFrame({"feature": self._feature_cols, "importance": imp})
            .sort_values("importance", ascending=False)
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        assert self._model is not None
        self._model.booster_.save_model(path)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> "F1LGBMModel":
        self._model = lgb.LGBMRegressor(**self.params)
        self._model._Booster = lgb.Booster(model_file=path)
        logger.info("Model loaded from %s", path)
        return self
