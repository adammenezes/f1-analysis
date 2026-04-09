# F1 Race Predictor

A machine learning pipeline for predicting Formula 1 race finishing positions. The system ingests historical results via two data sources, engineers 30 hand-crafted features, trains a LightGBM regressor using walk-forward validation to prevent data leakage, and includes a Monte Carlo race simulator for probabilistic outcome analysis.

An automated Recursive Feature Elimination (RFE) ablation study identified a mathematically optimal **"Elite 10"** feature subset, validated by a Unified Performance Index (UPI) achieving **68.8% Top-3 (Podium) Accuracy**.

---

## Project Architecture

```
f1-predictor/
├── main.py                         # CLI entry point
├── requirements.txt
├── .env.example                    # Copy to .env — configure year ranges here
├── mlflow.db                       # Local MLflow SQLite tracking store
│
├── data/
│   ├── raw/
│   │   └── fastf1_cache/           # FastF1 session cache (auto-populated)
│   └── processed/
│       ├── hist_results_2014_2017.parquet   # Jolpica cache
│       ├── ff1_results_2018_2024.parquet    # FastF1 cache
│       ├── ablation_final_results.csv       # UPI results from RFE study
│       └── runs/                   # Per-run logs: fold CSVs, SHAP, model checkpoints
│           ├── RFE_Step_30_features/
│           ├── RFE_Step_10_features/   ← winning model artefacts live here
│           └── ...
│
├── src/
│   ├── data/
│   │   ├── loader.py               # Jolpica API + FastF1 data ingestion
│   │   ├── features.py             # Feature engineering pipeline (30 features)
│   │   └── constructor_lineage.py  # Normalises constructor name changes across eras
│   ├── models/
│   │   ├── elo.py                  # Dual-track Elo rating system (driver + constructor)
│   │   ├── xgboost_model.py        # LightGBM regressor wrapper + SHAP explanation
│   │   └── monte_carlo.py          # Lap-by-lap Monte Carlo race simulator
│   ├── validation/
│   │   └── walk_forward.py         # Expanding-window walk-forward validator
│   └── utils/
│       └── normalization.py
│
└── scripts/
    ├── ablation_study.py           # Automated RFE loop (runs 30 walk-forward trainings)
    └── analyze_winner.py           # Computes Unified Performance Index (UPI) from ablation CSVs
```

---

## Data Sources & Training Window

| Source | Training Years | What it provides |
|---|---|---|
| **Jolpica API** (`api.jolpi.ca/ergast/f1`) | 2014–2017 | Race results, grid positions, finish status, fastest lap, points, driver/constructor metadata |
| **FastF1** (`docs.fastf1.dev`) | 2018–2024 | All of the above, plus: Q1/Q2/Q3 qualifying times, race weather (track temp, air temp, wind speed, humidity, rainfall) |

### Why 2014 onwards only?
Training data is strictly limited to **2014–present** (the V6 Turbo-Hybrid era). Formula 1 underwent a fundamental technical reset in 2014, changing engine architecture, aerodynamic rules, and energy recovery systems. Training on V8-era data (pre-2014) would introduce **concept drift** — the historical dominance hierarchies, reliability profiles, and strategic tendencies are not transferable to the modern formula.

### Data Split by Source
- **2014–2017:** Fetched from the Jolpica API (an open-source Ergast mirror). These seasons lack granular qualifying sector times and FastF1 telemetry, so weather features are filled with era-median defaults for those rows.
- **2018–2024:** Fetched from FastF1, which provides session-level qualifying and race weather data via its official F1 timing cache.

### Missing Data Handling
- **Qualifying times (pre-2018 / missing sessions):** `quali_gap_to_pole_pct` and `quali_gap_zscore` are set to `NaN`. LightGBM handles `NaN` natively without imputation.
- **Weather (pre-2018):** Filled with fixed conservative defaults: `is_wet_race=0`, `mean_track_temp=30.0°C`, `mean_air_temp=25.0°C`, `mean_wind_speed=2.0 m/s`, `mean_humidity=55.0%`.
- **New / rookie drivers:** Elo ratings default to 1500 (the system-wide mean). Rolling averages impute `NaN` for the first race; LightGBM handles these natively.
- **Constructor renames** (e.g., Force India → Racing Point → Aston Martin): resolved to a canonical lineage ID by `constructor_lineage.py` so rolling averages remain consistent across rebrands.

### Regulation Era Encoding
The loader assigns every race row a `regulation_era` string and a `seasons_since_reg_change` integer based on hardcoded FIA regulation break years (`{2009, 2014, 2017, 2022}`). On regulation-break seasons, the walk-forward validator optionally resets to train only on the new-era data (controlled by `regulation_break_reset=True`).

---

## Feature Engineering

`src/data/features.py` builds **30 features** per driver × race row. All are computed using shift-by-1 (lag-1) logic — the prediction target race's result is **never included** in the feature for that race.

### Complete Feature List

| # | Feature | Category | Description |
|---|---|---|---|
| 1 | `grid_position` | Grid | Starting position (1 = pole). Most predictive single feature. |
| 2 | `quali_gap_to_pole_pct` | Grid | `(driver_best_Q_time - pole_time) / pole_time × 100`. Era-safe relative metric. Best Q-time used (Q3 → Q2 → Q1 cascade). |
| 3 | `quali_gap_zscore` | Grid | Z-score of `quali_gap_to_pole_pct` within the race. Normalises for varying circuit lap lengths. |
| 4 | `driver_rolling_pos_3r` | Driver Form | Rolling mean finish position over last 3 races (hyper-recent form). |
| 5 | `driver_rolling_pos_5r` | Driver Form | Rolling mean finish position over last 5 races. |
| 6 | `driver_rolling_pos_10r` | Driver Form | Rolling mean finish position over last 10 races (long-term momentum). |
| 7 | `team_rolling_pos_3r` | Constructor Form | Constructor rolling mean finish over last 3 races. |
| 8 | `team_rolling_pos_5r` | Constructor Form | Constructor rolling mean finish over last 5 races. |
| 9 | `team_rolling_pos_10r` | Constructor Form | Constructor rolling mean finish over last 10 races. |
| 10 | `driver_career_races` | Experience | Total starts prior to this race (cumulative, lagged). |
| 11 | `driver_is_rookie` | Experience | Binary: 1 if `driver_career_races < 20`. |
| 12 | `driver_career_wins` | Experience | Cumulative wins prior to this race (lagged). |
| 13 | `driver_career_podiums` | Experience | Cumulative podiums prior to this race (lagged). |
| 14 | `teammate_outquali_rate_10r` | Intra-Team | Fraction of last 10 races where driver achieved a better qualifying position than their teammate. Defaults to 0.5 for first race. |
| 15 | `teammate_beat_rate_10r` | Intra-Team | Fraction of last 10 races where driver finished ahead of their teammate in the race. |
| 16 | `driver_circuit_avg_pos` | Track Affinity | Expanding historical mean finish position at this specific circuit (lagged). Captures "circuit specialists." |
| 17 | `team_circuit_avg_pos` | Track Affinity | Constructor's expanding historical mean finish at this circuit (lagged). |
| 18 | `driver_dnf_rate_10r` | Reliability | DNF rate over last 10 races for the driver. `is_dnf = True` for any non-"Finished / +N Laps" status. |
| 19 | `team_dnf_rate_10r` | Reliability | Constructor DNF rate over last 10 races. Captures mechanical reliability trends. |
| 20 | `driver_elo` | Skill Rating | Pre-race driver Elo rating (see Elo System below). |
| 21 | `constructor_elo` | Skill Rating | Pre-race constructor Elo rating. |
| 22 | `seasons_since_reg_change` | Context | Seasons elapsed since last major FIA regulation reset. Correlates with inter-team performance gaps narrowing / widening. |
| 23 | `championship_gap_driver` | Context | Points gap between this driver and the WDC leader going into this race. |
| 24 | `championship_gap_team` | Context | Points gap between this constructor and the WCC leader going into this race. |
| 25 | `regulation_era` | Context | Categorical string (`"hybrid_v1"`, `"hybrid_v2"`, `"ground_effect"`). Passed as a native LightGBM categorical. |
| 26 | `is_wet_race` | Weather | 1 if any lap had `Rainfall = True`, else 0. |
| 27 | `mean_track_temp` | Weather | Mean `TrackTemp` [°C] over the race session. Primary proxy for tire degradation rate. |
| 28 | `mean_air_temp` | Weather | Mean `AirTemp` [°C]. Secondary engine/aero reference. |
| 29 | `mean_wind_speed` | Weather | Mean `WindSpeed` [m/s]. Elevated wind increases overtake difficulty and crash probability. |
| 30 | `mean_humidity` | Weather | Mean `Humidity` [%]. Secondary grip / tire temperature interaction signal. |

> **Excluded weather features:** `WindDirection` (requires per-sector circuit geometry modelling to be meaningful) and `Pressure` (redundant with per-circuit altitude — a fixed constant).

---

## The Elo Rating System (`src/models/elo.py`)

Two independent Elo tracks are maintained — one for **drivers**, one for **constructors**.

- **Default Rating:** 1500 (both tracks)
- **Driver Elo:** Updated after each race via pairwise comparisons. Each driver is matched against every other finisher. Finishing ahead = win (score 1.0), finishing behind = loss (score 0.0).
- **Constructor Elo:** Updated from each team's mean finish position per race. Teams are paired and compared similarly.
- **K-factor:** Decays from 48 (early season / post-regulation break) to 16 (stable mid-season) over the first 12 rounds.
- **Regulation Break Regression:** At the first race of each break year (`{2009, 2014, 2017, 2022}`), all ratings regress 50% toward the mean (factor = 0.5). This models genuine uncertainty about which team's car will be fast under new rules.
- **Snapshot Architecture:** Ratings are snapshotted *before* each race. The `driver_elo` and `constructor_elo` features attached to a race row are always the **pre-race** rating, preventing leakage.
- **Combined Elo (predict_winner):** For standalone ranking (not used in ML training), combined score = `driver_elo × 0.45 + constructor_elo × 0.55`.

---

## Model: LightGBM (`src/models/xgboost_model.py`)

The core predictor is an `LGBMRegressor` with an **L1 (MAE) objective**, wrapped in `F1LGBMModel`.

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `objective` | `regression_l1` | MAE loss — robust to outliers (DNFs, safety cars). Does not catastrophically penalise extreme position errors. |
| `n_estimators` | 800 | Sufficient trees for the dataset size with the chosen learning rate. |
| `learning_rate` | 0.03 | Conservative — reduces overfitting risk. |
| `num_leaves` | 31 | Low complexity. F1 dataset is ~3,800 rows; high `num_leaves` risks memorising noise. |
| `max_depth` | 5 | Limits overfitting. |
| `min_child_samples` | 15 | Prevents leaf nodes on tiny subgroups. |
| `subsample` | 0.75 | Row sampling per tree for variance reduction. |
| `colsample_bytree` | 0.70 | Feature sampling per tree. |
| `reg_alpha` | 0.1 | L1 regularisation — promotes feature sparsity. |
| `reg_lambda` | 1.5 | L2 regularisation — shrinks weights further. |

### Why LightGBM over other architectures?
1. **Tabular data dominance:** Gradient-boosted trees consistently outperform neural networks on medium-scale tabular regression.
2. **Native categoricals:** Handles `regulation_era` as a categorical directly, without one-hot encoding.
3. **Native NaN handling:** Rows with missing qualifying times or weather data require no pre-imputation.
4. **SHAP compatibility:** Exact SHAP (Shapley Additive Explanations) values are computable via TreeExplainer, enabling the RFE study.
5. **L1 objective suitability:** F1 finishes have frequent outliers (crashes, retirements). MAE is preferable to MSE because a single DNF from pole does not disproportionately distort the entire model.

### Prediction Mechanism
The model outputs a continuous **score** per driver (lower = predicted better finishing position). Predictions are converted to a ranked order using `numpy.argsort()`. This is a **deterministic** argsort — not stochastic. The Monte Carlo simulator is a separate component (see below).

---

## Walk-Forward Validation (`src/validation/walk_forward.py`)

Standard K-fold cross-validation is invalid for F1 prediction because it can train on future races and test on past races. The `WalkForwardValidator` enforces strict temporal ordering.

### Mechanism
1. Sort all races globally by `(season, round)`, assign a monotonic `race_index`.
2. Require `min_train_races = 50` races before making the first prediction.
3. For each fold:
   - **Train:** All races with `race_index < test_race_index`.
   - **Predict:** The single next race (`race_index == test_race_index`).
   - **Advance:** Move test pointer forward by `step = 1`.
4. On regulation-break first races, the training window optionally resets to only include races from the new era (`regulation_break_reset=True`).

### Sample Weight Decay
Training samples are weighted with `weight = exp(-0.15 × age_in_seasons)`. A race from 5 seasons ago receives weight ≈ 0.47 relative to the current season. This balances having enough data with prioritising recent relevance.

### Evaluation Metrics (per fold)

| Metric | Formula | Interpretation |
|---|---|---|
| `spearman_rho` | Spearman rank correlation between predicted and actual positions | Overall rank ordering quality (1.0 = perfect) |
| `top1_acc` | Fraction of predicted P1 drivers that actually finished P1 | Winner prediction accuracy |
| `top3_acc` | Overlap of predicted top-3 set vs. actual top-3 set, divided by 3 | Podium prediction accuracy |
| `mae` | Mean absolute error in positions | Average position error across the 20-driver grid |

---

## Experiment Tracking (Hybrid System)

Every training run produces two parallel audit trails:

1. **MLflow** (`mlflow.db`): High-level metrics (`spearman_rho`, `top3_acc`, `top1_acc`, `mae`) and parameters are logged to a local SQLite database. Launch the visual UI with:
   ```bash
   python main.py dashboard
   # or directly:
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

2. **Local CSV folders** (`data/processed/runs/<run_name>/`): Each run writes:
   - `fold_history.csv` — per-fold metrics and per-fold feature importances
   - `feature_importance.csv` — mean absolute SHAP or LightGBM gain importance
   - `lgbm_model.txt` — saved LightGBM booster
   - `config.json` — parameter summary
   - `description.txt` — free-text run description

---

## Ablation Study & Feature Selection (`scripts/ablation_study.py`)

### The Problem: Curse of Dimensionality
A model trained on 30 features risks memorising noise in the 3,800-row dataset (≈190 rows per feature). Regularisation helps, but noisy features still consume model capacity and add variance to predictions.

### Automated RFE Loop
`ablation_study.py` implements full automated Recursive Feature Elimination:

1. Start with all `N` features (N=30).
2. Run a complete walk-forward validation.
3. Compute **Mean Absolute SHAP** for each feature over a 500-row sample:
   - SHAP values are measured in **units of finishing positions** (the model's target variable). A SHAP value of +2.0 means that feature pushed the prediction 2 positions worse than the baseline.
   - Mean Absolute SHAP = average of |SHAP| across all rows. Represents the typical magnitude of influence.
   - The feature with the lowest Mean Absolute SHAP is the weakest contributor.
4. Drop the single weakest feature.
5. Repeat from step 2 until 1 feature remains.

This produces 30 trained models, each with a different feature count (30 → 29 → ... → 1), logged to MLflow and saved to `data/processed/runs/`.

> **Fallback:** If the `shap` package is not installed or SHAP computation fails, the script falls back to LightGBM's built-in `gain`-based feature importance for the elimination criterion.

### Unified Performance Index (`scripts/analyze_winner.py`)

Choosing the optimal model purely by Spearman Rho, MAE, or Top-3 accuracy individually produces conflicting winners. The UPI resolves this via **Z-Score normalization across the study**:

```
z_rho   = (spearman_rho - μ_rho)   / σ_rho     # higher is better
z_top3  = (top3_acc    - μ_top3)   / σ_top3    # higher is better
z_mae   = (mae         - μ_mae)    / σ_mae × -1 # lower MAE is better → invert

UPI = mean(z_rho, z_top3, z_mae)
```

The model with the highest UPI is Pareto-optimal across all three dimensions simultaneously.

**Result:** The **10-feature model** achieved UPI = +1.061, the highest of all 30 models.

### The Elite 10 (Winning Feature Set)

| Rank | Feature | Mean Abs SHAP (full 30-feature model) |
|---|---|---|
| 1 | `grid_position` | 2.226 positions |
| 2 | `team_rolling_pos_10r` | 1.176 positions |
| 3 | `team_circuit_avg_pos` | 0.786 positions |
| 4 | `driver_rolling_pos_10r` | 0.608 positions |
| 5 | `constructor_elo` | 0.243 positions |
| 6 | `driver_circuit_avg_pos` | 0.171 positions |
| 7 | `championship_gap_team` | 0.128 positions |
| 8 | `driver_elo` | 0.111 positions |
| 9 | `teammate_outquali_rate_10r` | 0.099 positions |
| 10 | `team_rolling_pos_3r` | 0.069 positions |

Features such as weather, direct career statistics (`driver_career_wins`), and the qualifying Z-score were all mathematically eliminated by the SHAP-based RFE.

---

## Monte Carlo Race Simulator (`src/models/monte_carlo.py`)

The simulator is a **separate, optional component** used for probabilistic forecasting of upcoming races. It is **not used during training or validation**.

### How It Works
Each simulation models one full race lap-by-lap:
1. **Grid noise:** Start positions are perturbed by `Normal(0, 0.8)` to model first-corner variability.
2. **Lap time sampling:** Each driver's lap time is drawn from `Normal(base_pace_mean, base_pace_std)`.
3. **Tire degradation:** Lap time penalty increases exponentially beyond the `cliff_lap` using coefficient `tire_deg_k`.
4. **Safety car / VSC:** Deployed stochastically per lap (default: 4% SC, 3% VSC). Bunches the field.
5. **DNF:** Each driver has a `dnf_prob_per_lap` drawn from their historical DNF rate over the last 20 races.
6. **Pit stops:** Triggered by mandatory tire cliff, opportunistic undercut (if gap to car ahead < threshold), or safety car window.
7. **Cumulative time** determines final positions.

### Driver Parameters
`params_from_model_output()` converts the LightGBM ranking into `DriverParams` by:
- Normalising LightGBM scores into a realistic 3.5-second pace spread across the field.
- Computing per-driver `dnf_prob_per_lap` from their last 20 races in the historical dataset.

### Parallel Execution
Simulations are batched across CPU cores using `joblib.Parallel`. Default: 10,000 simulations in 4 batches.

### Output
```
🎲 Monte Carlo Results — 2024 Round 1 (sakhir)
  N = 1,000 simulations
  ──────────────────────────────────────────────────
    ver  Win:93.2%  Podium:95.9%  Points:96.1%  Exp.Pos:  1.7
    per  Win: 2.6%  Podium:65.3%  Points:67.1%  Exp.Pos:  7.2
    sai  Win: 1.8%  Podium:65.9%  Points:86.1%  Exp.Pos:  5.0
    ...
```

`Exp.Pos` is the **expected finishing position** (probability-weighted mean across all simulations). Using `Exp.Pos` instead of the raw LightGBM score as the final prediction encodes stochastic risk (e.g., DNF probability) into the ranking.

---

## Setup & Installation

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env if you want to change year ranges or cache directory
```

> The FastF1 cache (`data/raw/fastf1_cache/`) is populated automatically on the first `fetch` run. Subsequent runs load from the local cache. Store it on an SSD — reading from a network drive significantly increases training time.

---

## CLI Usage

### Fetch data
```bash
python main.py fetch --start-year 2014 --end-year 2024
```
Downloads Jolpica (2014–2017) and FastF1 (2018–2024) data. Cached to Parquet on first run; subsequent runs load from cache instantly.

### Train the model
```bash
python main.py train
# With a named run and description for MLflow:
python main.py train --name "elite10_baseline" --desc "Elite 10 feature set, default hyperparams"
```
Runs full walk-forward validation and saves all artefacts to `data/processed/runs/<run_name>/`.

### Predict a specific race
```bash
python main.py predict --season 2025 --round 5
```
Outputs deterministic predicted finishing order (LightGBM argsort).

### Monte Carlo simulation
```bash
python main.py simulate --season 2025 --round 5 --n-sims 10000
```
Outputs win/podium/points probabilities and expected positions for all drivers.

### Evaluate and plot results
```bash
python main.py evaluate --plot
```

### Launch MLflow dashboard
```bash
python main.py dashboard
# Then open: http://localhost:5000
```

### Run the RFE ablation study
```bash
python -m scripts.ablation_study
```
Trains 30 models iteratively. Results logged to MLflow and saved to `data/processed/runs/`.

### Compute the Unified Performance Index
```bash
python scripts/analyze_winner.py
```
Reads from `data/processed/runs/` and outputs a ranked table of all RFE models by UPI.

---

## Sharing MLflow Results

To share the experiment results with a peer:
1. Zip `mlflow.db` and `data/processed/runs/`.
2. The peer installs MLflow (`pip install mlflow pandas`) and runs:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
3. Navigate to `http://localhost:5000`.

> MLflow is fully open-source and free. No license is required.

---

## Key Design Principles

| Principle | Implementation |
|---|---|
| No data leakage | All rolling/expanding features use `shift(1)`. Walk-forward validation is strictly chronological. Elo ratings are snapshotted pre-race. |
| Era safety | Qualifying gaps are expressed as `%` of pole time or Z-score within the race session, never raw seconds. |
| Robustness to outliers | L1 (MAE) loss objective. Outliers from DNFs do not disproportionately distort training. |
| Handling rebrands | `constructor_lineage.py` maps Force India → Racing Point → Aston Martin (and all other renames) to a canonical ID. |
| Reproducibility | `RANDOM_SEED=42` throughout. All data is cached to Parquet. |
