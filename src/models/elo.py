"""
src/models/elo.py
─────────────────
Dual-track Elo rating system for F1 drivers and constructors.

Design decisions:
  • Separate ratings for driver skill and constructor (car) performance.
    Driver Elo is updated from head-to-head pairwise results within a race.
    Constructor Elo uses mean team finish position per race.
  • K-factor is higher early in each season (uncertainty is higher) and
    decays as the season progresses.
  • On a regulation break the ratings decay toward the prior mean —
    no one knows whose car will be fast on new rules.
  • All ratings are stored in a history dict keyed by (season, round)
    so they can be retrieved as point-in-time features for ML models.

Usage:
    from src.models.elo import EloSystem
    elo = EloSystem()
    elo.fit(df)          # df = load_full_dataset() output
    df = elo.add_features(df)   # adds driver_elo, constructor_elo columns
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_RATING = 1500.0
K_MAX = 48.0           # K-factor early in season / after regulation break
K_MIN = 16.0           # K-factor late in a stable season
ROUNDS_TO_K_MIN = 12   # Number of races before K settles to K_MIN

# After a regulation break, ratings regress toward the mean by this factor.
# 0.0 = full reset to DEFAULT_RATING, 1.0 = no regression.
REGULATION_BREAK_REGRESSION = 0.5

REGULATION_BREAKS = {2009, 2014, 2017, 2022}


def _k_factor(round_in_season: int) -> float:
    """Linearly decay K from K_MAX to K_MIN over the first ROUNDS_TO_K_MIN races."""
    t = min(round_in_season - 1, ROUNDS_TO_K_MIN) / ROUNDS_TO_K_MIN
    return K_MAX + t * (K_MIN - K_MAX)


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


class EloSystem:
    """
    Maintains running Elo ratings for every driver and constructor seen in the
    training data, and produces a point-in-time snapshot per race for use as
    ML features.
    """

    def __init__(self) -> None:
        self.driver_ratings: dict[str, float] = defaultdict(lambda: DEFAULT_RATING)
        self.constructor_ratings: dict[str, float] = defaultdict(lambda: DEFAULT_RATING)

        # history[season][round] = {driver_id: rating, ...}
        self.driver_history: dict[tuple, dict] = {}
        self.constructor_history: dict[tuple, dict] = {}

    # ── Rating update helpers ─────────────────────────────────────────────────

    def _apply_regulation_break(self, year: int) -> None:
        """Regress all ratings toward the mean on a regulation reset year."""
        mean_d = np.mean(list(self.driver_ratings.values())) if self.driver_ratings else DEFAULT_RATING
        mean_c = np.mean(list(self.constructor_ratings.values())) if self.constructor_ratings else DEFAULT_RATING

        self.driver_ratings = defaultdict(
            lambda: DEFAULT_RATING,
            {
                did: r * REGULATION_BREAK_REGRESSION + mean_d * (1 - REGULATION_BREAK_REGRESSION)
                for did, r in self.driver_ratings.items()
            },
        )
        self.constructor_ratings = defaultdict(
            lambda: DEFAULT_RATING,
            {
                cid: r * REGULATION_BREAK_REGRESSION + mean_c * (1 - REGULATION_BREAK_REGRESSION)
                for cid, r in self.constructor_ratings.items()
            },
        )
        logger.debug("Applied regulation break regression for year %d", year)

    def _update_driver_elo(self, race_df: pd.DataFrame, k: float) -> None:
        """
        Update driver ratings using pairwise comparisons within a race.
        Every driver is compared against every other driver who finished:
          score = 1 if driver finished ahead, 0 if behind.
        """
        finished = race_df[race_df["finish_position"] > 0].copy()
        finished = finished.sort_values("finish_position")
        drivers = finished["driver_id"].tolist()
        positions = dict(zip(finished["driver_id"], finished["finish_position"]))

        for i, d_a in enumerate(drivers):
            delta = 0.0
            for d_b in drivers[i + 1:]:
                # d_a finished ahead of d_b
                e_a = _expected(self.driver_ratings[d_a], self.driver_ratings[d_b])
                delta += k * (1.0 - e_a)
                self.driver_ratings[d_b] += k * (0.0 - (1.0 - e_a))
            self.driver_ratings[d_a] += delta

    def _update_constructor_elo(self, race_df: pd.DataFrame, k: float) -> None:
        """
        Update constructor ratings from mean finishing position per team.
        Team pairs compared on whether their mean position was better/worse.
        """
        team_avg = (
            race_df[race_df["finish_position"] > 0]
            .groupby("constructor_id")["finish_position"]
            .mean()
            .to_dict()
        )
        teams = sorted(team_avg.keys(), key=lambda t: team_avg[t])  # best first

        for i, t_a in enumerate(teams):
            delta = 0.0
            for t_b in teams[i + 1:]:
                e_a = _expected(
                    self.constructor_ratings[t_a], self.constructor_ratings[t_b]
                )
                delta += k * (1.0 - e_a)
                self.constructor_ratings[t_b] += k * (0.0 - (1.0 - e_a))
            self.constructor_ratings[t_a] += delta

    # ── Public interface ──────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "EloSystem":
        """
        Replay all races in chronological order, updating ratings after each.
        Snapshots are stored so they can be attached to features *before* each race.
        """
        races = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
        prev_season: Optional[int] = None

        for _, (season, round_num) in races.iterrows():
            season, round_num = int(season), int(round_num)

            # ── Snapshot BEFORE this race (pre-race ratings = feature values) ──
            self.driver_history[(season, round_num)] = dict(self.driver_ratings)
            self.constructor_history[(season, round_num)] = dict(self.constructor_ratings)

            # ── Apply regulation-break regression at first race of break year ──
            if season in REGULATION_BREAKS and season != prev_season:
                self._apply_regulation_break(season)
            prev_season = season

            k = _k_factor(round_num)
            race_df = df[(df["season"] == season) & (df["round"] == round_num)]
            self._update_driver_elo(race_df, k)
            self._update_constructor_elo(race_df, k)

        return self

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach pre-race Elo ratings to every row in *df*.
        Uses the snapshot taken just before each race.
        """
        df = df.copy()
        df["driver_elo"] = df.apply(
            lambda r: self.driver_history.get(
                (int(r["season"]), int(r["round"])), {}
            ).get(r["driver_id"], DEFAULT_RATING),
            axis=1,
        )
        df["constructor_elo"] = df.apply(
            lambda r: self.constructor_history.get(
                (int(r["season"]), int(r["round"])), {}
            ).get(r["constructor_id"], DEFAULT_RATING),
            axis=1,
        )
        return df

    def predict_winner(self, entrants: list[dict]) -> list[dict]:
        """
        Given a list of {'driver_id': ..., 'constructor_id': ...} entrant dicts,
        return them sorted by combined Elo score (descending = favoured to win).
        """
        scored = [
            {
                **e,
                "driver_elo": self.driver_ratings[e["driver_id"]],
                "constructor_elo": self.constructor_ratings[e["constructor_id"]],
                "combined_elo": (
                    self.driver_ratings[e["driver_id"]] * 0.45
                    + self.constructor_ratings[e["constructor_id"]] * 0.55
                ),
            }
            for e in entrants
        ]
        return sorted(scored, key=lambda x: x["combined_elo"], reverse=True)
