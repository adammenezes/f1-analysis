"""
src/models/monte_carlo.py
──────────────────────────
Monte Carlo race simulator.

Architecture
────────────
Each simulation run models a full race lap-by-lap:

  1. Starting grid positions are noised slightly to reflect first-corner chaos.
  2. Every lap:
       a. Driver base pace is sampled from a Normal distribution (mean, std).
       b. Tire degradation increases lap time exponentially with tire age.
       c. Safety car fires stochastically (Poisson rate per circuit).
       d. DNFs are drawn per team's reliability rate.
       e. Pit stop decisions are evaluated:
            – Mandatory when tires hit cliff lap.
            – Opportunistic undercut if driver is within gap threshold.
            – Safety car window if SC is active and tires are half-spent.

  3. Lap times determine position delta each lap.
  4. After N laps, final positions are tallied.

Running 10,000 simulations produces a probability distribution over all
finishing positions for every driver — the key output.

Usage:
    from src.models.monte_carlo import RaceSimulator, RaceParams, DriverParams

    params = RaceParams(circuit_id="monza", total_laps=53)
    drivers = [
        DriverParams(driver_id="ver", constructor_id="red_bull",
                     base_pace_mean=1.235, base_pace_std=0.008,
                     start_position=1, pit_mean_s=2.5, pit_std_s=0.3,
                     dnf_prob_per_lap=0.002, cliff_lap=28, tire_deg_k=0.0004),
        ...
    ]
    sim = RaceSimulator(params, drivers)
    results = sim.run(n_simulations=10_000)
    print(results.win_probabilities())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RaceParams:
    """Circuit-level parameters for a single race simulation."""
    circuit_id: str
    total_laps: int
    sc_prob_per_lap: float = 0.04          # P(safety car starts this lap) = 4% default
    vsc_prob_per_lap: float = 0.03         # P(virtual SC)
    sc_duration_laps: int = 4              # How many laps SC typically runs
    undercut_gap_threshold_s: float = 2.0  # Seconds within which undercut is viable
    pit_loss_base_s: float = 21.0          # Time lost in pit lane (circuit-specific)
    max_stints: int = 3                    # Maximum pit stops allowed by strategy


@dataclass
class DriverParams:
    """Per-driver/constructor parameters for one race."""
    driver_id: str
    constructor_id: str
    start_position: int                    # Grid position (1 = pole)
    base_pace_mean: float                  # Mean lap time in seconds
    base_pace_std: float                   # Lap time consistency (lower = more consistent)
    pit_mean_s: float = 2.5               # Mean pit stop stationary time
    pit_std_s: float = 0.3                # Pit stop standard deviation
    dnf_prob_per_lap: float = 0.002       # P(mechanical failure this lap)
    cliff_lap: int = 30                   # Lap at which tires degrades sharply
    tire_deg_k: float = 0.0003           # Exponential tire degradation coefficient
    tire_compound: str = "medium"         # Starting tire compound


# ── Simulation results ────────────────────────────────────────────────────────

@dataclass
class SimulationResults:
    """Aggregated results from N Monte Carlo runs."""
    driver_ids: list[str]
    # position_matrix[i, j] = fraction of runs where driver i finished in position j+1
    position_matrix: np.ndarray

    def win_probabilities(self) -> pd.Series:
        return pd.Series(
            self.position_matrix[:, 0], index=self.driver_ids, name="win_prob"
        ).sort_values(ascending=False)

    def podium_probabilities(self) -> pd.Series:
        return pd.Series(
            self.position_matrix[:, :3].sum(axis=1),
            index=self.driver_ids,
            name="podium_prob",
        ).sort_values(ascending=False)

    def points_probabilities(self) -> pd.Series:
        """Probability of finishing in the points (top 10)."""
        return pd.Series(
            self.position_matrix[:, :10].sum(axis=1),
            index=self.driver_ids,
            name="points_prob",
        ).sort_values(ascending=False)

    def expected_position(self) -> pd.Series:
        positions = np.arange(1, self.position_matrix.shape[1] + 1)
        exp = (self.position_matrix * positions).sum(axis=1)
        return pd.Series(exp, index=self.driver_ids, name="expected_position").sort_values()

    def to_dataframe(self) -> pd.DataFrame:
        n = len(self.driver_ids)
        pos_cols = {f"p{i+1}_prob": self.position_matrix[:, i] for i in range(n)}
        return pd.DataFrame(
            {
                "driver_id": self.driver_ids,
                "win_prob": self.position_matrix[:, 0],
                "podium_prob": self.position_matrix[:, :3].sum(axis=1),
                "points_prob": self.position_matrix[:, :10].sum(axis=1),
                "expected_position": (
                    self.position_matrix * np.arange(1, n + 1)
                ).sum(axis=1),
                **pos_cols,
            }
        ).sort_values("expected_position")


# ── Core single-race simulation ───────────────────────────────────────────────

def _simulate_race_once(
    params: RaceParams,
    drivers: list[DriverParams],
    rng: np.random.Generator,
) -> list[str]:
    """
    Simulate a single race.  Returns driver IDs sorted by finish position
    (index 0 = winner), with DNFs appended at the end.
    """
    n = len(drivers)

    # ── Mutable race state ────────────────────────────────────────────────────
    race_pos = np.array([d.start_position for d in drivers], dtype=float)
    # Slight noise on grid to simulate first-corner chaos
    race_pos += rng.normal(0, 0.8, size=n)

    cumulative_time = np.zeros(n)     # total time elapsed per driver
    tire_age = np.zeros(n)            # laps on current tire set
    pit_count = np.zeros(n, dtype=int)
    retired = np.zeros(n, dtype=bool)

    sc_active = False
    sc_laps_remaining = 0

    for lap in range(1, params.total_laps + 1):
        # ── Safety car logic ───────────────────────────────────────────────
        if not sc_active:
            if rng.random() < params.sc_prob_per_lap:
                sc_active = True
                sc_laps_remaining = params.sc_duration_laps
                logger.debug("SC deployed on lap %d", lap)
        else:
            sc_laps_remaining -= 1
            if sc_laps_remaining <= 0:
                sc_active = False

        vsc_active = (not sc_active) and rng.random() < params.vsc_prob_per_lap

        # ── Per-driver lap ────────────────────────────────────────────────
        for i, drv in enumerate(drivers):
            if retired[i]:
                continue

            # DNF check
            if rng.random() < drv.dnf_prob_per_lap:
                retired[i] = True
                cumulative_time[i] += 1e6   # effectively last place
                continue

            # Base lap time
            base = rng.normal(drv.base_pace_mean, drv.base_pace_std)

            # Tire degradation — exponential beyond cliff lap
            deg = 0.0
            if tire_age[i] > drv.cliff_lap:
                overshoot = tire_age[i] - drv.cliff_lap
                deg = drv.tire_deg_k * overshoot ** 2
            else:
                deg = drv.tire_deg_k * tire_age[i] * 0.1   # mild pre-cliff deg

            lap_time = base + deg

            # Safety car / VSC pace delta (everyone bunches up)
            if sc_active:
                lap_time += 20.0 + rng.uniform(0, 3)   # SC lap pace
            elif vsc_active:
                lap_time += 8.0 + rng.uniform(0, 2)

            cumulative_time[i] += lap_time
            tire_age[i] += 1

        # ── Pit stop decisions ────────────────────────────────────────────
        for i, drv in enumerate(drivers):
            if retired[i]:
                continue
            if pit_count[i] >= params.max_stints:
                continue

            must_pit = tire_age[i] >= drv.cliff_lap + 3

            # Undercut check: am I within threshold of driver ahead?
            undercut_viable = False
            if not sc_active:
                # Find driver directly ahead
                my_time = cumulative_time[i]
                times_ahead = [
                    cumulative_time[j]
                    for j in range(n)
                    if not retired[j] and cumulative_time[j] < my_time
                ]
                if times_ahead:
                    gap_to_ahead = my_time - max(times_ahead)
                    undercut_viable = gap_to_ahead < params.undercut_gap_threshold_s

            # SC window opportunity
            sc_window = (
                sc_active
                and tire_age[i] > drv.cliff_lap * 0.4
                and pit_count[i] == 0
            )

            if must_pit or (undercut_viable and rng.random() < 0.6) or sc_window:
                pit_duration = rng.normal(drv.pit_mean_s, drv.pit_std_s)
                pit_duration = max(pit_duration, 1.8)  # floor
                cumulative_time[i] += params.pit_loss_base_s + pit_duration
                tire_age[i] = 0
                pit_count[i] += 1

        # ── Update position order ─────────────────────────────────────────
        # Positions determined by cumulative time (lower = ahead)
        order = np.argsort(cumulative_time)
        race_pos = np.empty(n)
        race_pos[order] = np.arange(1, n + 1)

    # ── Final order ───────────────────────────────────────────────────────────
    final_order = np.argsort(cumulative_time)
    return [drivers[i].driver_id for i in final_order]


# ── Parallel runner ───────────────────────────────────────────────────────────

def _run_batch(
    params: RaceParams,
    drivers: list[DriverParams],
    n: int,
    seed: int,
) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    return [_simulate_race_once(params, drivers, rng) for _ in range(n)]


class RaceSimulator:
    """Runs Monte Carlo race simulations and aggregates results."""

    def __init__(self, params: RaceParams, drivers: list[DriverParams]) -> None:
        self.params = params
        self.drivers = drivers
        self._driver_ids = [d.driver_id for d in drivers]

    def run(
        self,
        n_simulations: int = 10_000,
        n_jobs: int = -1,
        seed: int = 42,
    ) -> SimulationResults:
        """
        Run *n_simulations* races in parallel and aggregate finishing distributions.

        Returns a SimulationResults object with win/podium/points probabilities.
        """
        n_drivers = len(self.drivers)
        n_jobs_actual = n_jobs if n_jobs > 0 else 4

        # Split work into batches (one per worker)
        batch_size = max(1, n_simulations // n_jobs_actual)
        n_batches = (n_simulations + batch_size - 1) // batch_size
        batch_sizes = [batch_size] * (n_batches - 1) + [
            n_simulations - batch_size * (n_batches - 1)
        ]
        seeds = [seed + i * 1000 for i in range(n_batches)]

        logger.info(
            "Running %d Monte Carlo simulations (%d batches × ~%d runs)…",
            n_simulations, n_batches, batch_size,
        )

        all_races: list[list[str]] = []
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(_run_batch)(self.params, self.drivers, bs, s)
            for bs, s in zip(batch_sizes, seeds)
        )
        for batch in batch_results:
            all_races.extend(batch)

        # ── Build position matrix ─────────────────────────────────────────────
        # position_matrix[i, j] = count of times driver i finished in position j+1
        id_to_idx = {did: i for i, did in enumerate(self._driver_ids)}
        counts = np.zeros((n_drivers, n_drivers), dtype=np.float64)

        for ordering in all_races:
            for pos_idx, did in enumerate(ordering):
                if did in id_to_idx:
                    counts[id_to_idx[did], pos_idx] += 1

        position_matrix = counts / len(all_races)

        return SimulationResults(
            driver_ids=self._driver_ids,
            position_matrix=position_matrix,
        )


# ── Helper: build DriverParams from model output ──────────────────────────────

def params_from_model_output(
    ranking_df: pd.DataFrame,
    circuit_params: RaceParams,
    historical_df: pd.DataFrame,
) -> list[DriverParams]:
    """
    Convert a model ranking DataFrame + historical data into DriverParams
    for the Monte Carlo simulation.

    ranking_df columns: driver_id, constructor_id, predicted_position, score
    historical_df: full dataset for calibrating pace and reliability stats.
    """
    drivers = []
    n = len(ranking_df)

    for _, row in ranking_df.iterrows():
        did = row["driver_id"]
        cid = row["constructor_id"]

        # Base pace from score (lower model score → faster driver)
        # Normalise model scores to realistic lap time range
        score_min = ranking_df["score"].min()
        score_max = ranking_df["score"].max()
        pace_range = 3.5  # seconds gap between P1 and P20 on typical circuit
        norm_score = (row["score"] - score_min) / max(score_max - score_min, 1e-6)
        base_pace = circuit_params.total_laps * 0 + 90.0 + norm_score * pace_range  # placeholder

        # Reliability from historical DNF rate
        drv_data = historical_df[historical_df["driver_id"] == did].tail(20)
        if len(drv_data) > 0:
            dnf_rate = drv_data["is_dnf"].mean()
            total_laps_approx = len(drv_data) * 55   # rough laps per race
            dnf_per_lap = dnf_rate / 55
        else:
            dnf_per_lap = 0.002

        # Pit stop speed from constructor history
        con_data = historical_df[historical_df["constructor_id"] == cid]
        pit_mean = 2.5  # fallback — would be filled from actual pit stop data

        drivers.append(
            DriverParams(
                driver_id=did,
                constructor_id=cid,
                start_position=int(row.get("predicted_position", n)),
                base_pace_mean=base_pace,
                base_pace_std=0.008 + (1 - norm_score) * 0.005,
                pit_mean_s=pit_mean,
                pit_std_s=0.3,
                dnf_prob_per_lap=float(np.clip(dnf_per_lap, 0.001, 0.015)),
            )
        )

    return drivers
