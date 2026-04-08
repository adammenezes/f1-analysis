"""
src/utils/normalization.py
───────────────────────────
Utilities for cross-regulation-era normalization of F1 performance metrics.

The fundamental rule: never compare absolute lap times across eras.
Always convert to relative metrics within each race or regulation era.

Functions
─────────
  era_percentile_rank    - rank driver within era on any metric
  within_race_zscore     - z-score a feature within each race
  normalize_to_pole      - express time as % gap to pole
  regulation_era_encode  - encode era as learnable sequential integers
"""

from __future__ import annotations

import pandas as pd
import numpy as np


REGULATION_ERA_ORDER = [
    "pre_wing",
    "wing_era",
    "turbo_v1",
    "na_early",
    "v10_era",
    "v8_era",
    "hybrid_v1",
    "hybrid_v2",
    "ground_effect",
]


def within_race_zscore(
    df: pd.DataFrame,
    value_col: str,
    new_col: Optional[str] = None,
    group_cols: list[str] = ["season", "round"],
) -> pd.DataFrame:
    """
    Z-score *value_col* within each (season, round) group.
    Produces a feature that is inherently comparable across eras and circuits.
    """
    new = new_col or f"{value_col}_zscore"
    grp = df.groupby(group_cols)[value_col]
    df[new] = (df[value_col] - grp.transform("mean")) / (grp.transform("std") + 1e-9)
    return df


def normalize_to_pole(
    df: pd.DataFrame,
    time_col: str,
    new_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert raw qualifying/lap times to percentage gap to session best.
    Works for qualifying (session best = pole) or race (session best = fastest lap).

    safe for NaN — rows with NaN time_col will have NaN in the output.
    """
    new = new_col or f"{time_col}_pct_gap"
    session_best = df.groupby(["season", "round"])[time_col].transform("min")
    df[new] = (df[time_col] - session_best) / (session_best + 1e-9) * 100.0
    return df


def era_percentile_rank(
    df: pd.DataFrame,
    value_col: str,
    era_col: str = "regulation_era",
    new_col: Optional[str] = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Rank each driver by *value_col* within their regulation era (percentile 0–1).
    Useful for comparing career-level performance across eras.

    ascending=True means lower value → higher (better) percentile rank.
    """
    new = new_col or f"{value_col}_era_pct"
    df[new] = df.groupby(era_col)[value_col].transform(
        lambda s: s.rank(pct=True, ascending=ascending)
    )
    return df


def regulation_era_encode(
    df: pd.DataFrame,
    era_col: str = "regulation_era",
    new_col: str = "regulation_era_idx",
) -> pd.DataFrame:
    """
    Encode the regulation era string as a sequential integer (ordinal encoding).
    Preserves the chronological ordering of eras for tree-based models.
    """
    era_to_idx = {era: i for i, era in enumerate(REGULATION_ERA_ORDER)}
    df[new_col] = df[era_col].map(era_to_idx).fillna(-1).astype(int)
    return df


def decay_sample_weights(
    df: pd.DataFrame,
    season_col: str = "season",
    decay_rate: float = 0.15,
) -> np.ndarray:
    """
    Compute exponential decay sample weights.
    Most recent season receives weight 1.0; older seasons decay by decay_rate per year.

    Returns a 1D numpy array aligned with df rows.
    """
    max_season = df[season_col].max()
    age = (max_season - df[season_col]).astype(float)
    weights = np.exp(-decay_rate * age)
    return (weights / weights.mean()).values  # normalise to mean=1


def clip_outliers_iqr(
    df: pd.DataFrame,
    cols: list[str],
    factor: float = 3.0,
) -> pd.DataFrame:
    """
    Clip extreme outliers in *cols* using IQR method.
    Values beyond median ± factor × IQR are clipped.
    Helps gradient boosting models avoid learning from DNF-induced anomalous laps.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - factor * iqr
        hi = q3 + factor * iqr
        df[col] = df[col].clip(lo, hi)
    return df
