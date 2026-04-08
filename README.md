# F1 Predictor

A machine learning system for predicting Formula 1 race results using historical performance data.

## Architecture

```
f1-predictor/
├── main.py                    # CLI entry point
├── requirements.txt
├── .env.example               # Copy to .env and configure
├── data/
│   ├── raw/
│   │   └── fastf1_cache/      # FastF1 session cache (auto-populated, ~15–20 GB)
│   └── processed/
│       ├── features.parquet   # Engineered feature matrix (~50–200 MB)
│       ├── elo_snapshots.parquet
│       └── lgbm_model.txt     # Saved LightGBM model
├── src/
│   ├── data/
│   │   ├── loader.py          # Jolpica API + FastF1 data ingestion
│   │   └── features.py        # Feature engineering pipeline
│   ├── models/
│   │   ├── elo.py             # Dual-track Elo rating system
│   │   ├── xgboost_model.py   # LightGBM race predictor
│   │   └── monte_carlo.py     # Monte Carlo race simulator
│   ├── validation/
│   │   └── walk_forward.py    # Walk-forward expanding-window validator
│   └── utils/
│       └── normalization.py   # Cross-era normalization utilities
├── notebooks/                 # Jupyter exploration notebooks
└── scripts/                   # One-off utility scripts
```

## Data Sources

| Source | Coverage | What it provides |
|--------|----------|--------------------|
| [Jolpica API](https://api.jolpi.ca/ergast/f1) | 1950–present | Race results, standings, pit stops, qualifying |
| [FastF1](https://docs.fastf1.dev) | 2018–present | Lap times, sector times, tire data, weather, safety car events |

## Storage

**Local storage is recommended** — no cloud drive needed.

| Layer | Location | Estimated Size |
|-------|----------|----------------|
| FastF1 session cache (raw) | `data/raw/fastf1_cache/` | ~15–20 GB (2018–2024, Race + Qualifying sessions only) |
| Processed feature matrix | `data/processed/features.parquet` | ~50–200 MB |
| Elo snapshots | `data/processed/elo_snapshots.parquet` | ~5 MB |
| Saved model | `data/processed/lgbm_model.txt` | < 10 MB |
| **Total** | | **~16–21 GB** |

> **Why local, not Google Drive?**  
> FastF1's cache is accessed on every run. Network storage (Google Drive, OneDrive) adds significant I/O latency that would slow training loops dramatically. A modern laptop SSD handles this comfortably. Only back up the small `data/processed/` directory to cloud — **never** the raw FastF1 cache (it can always be re-downloaded).

Set `FASTF1_CACHE_DIR` in your `.env` to point to a different drive if your primary disk is tight on space:
```
FASTF1_CACHE_DIR=D:\f1_cache   # e.g. a secondary SSD
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env to set FASTF1_CACHE_DIR if needed, adjust year ranges, etc.
```

## Usage

### Download race data
```bash
python main.py fetch --start-year 2018 --end-year 2024
```
FastF1 session data is cached in `data/raw/fastf1_cache/`. Subsequent runs skip the API for already-cached sessions. Jolpica data is saved as Parquet to `data/processed/`.

### Train the model
```bash
python main.py train
```
Runs walk-forward validation across all races, printing Spearman ρ, Top-3 accuracy, and MAE per fold. Saves final model to `data/processed/lgbm_model.txt`.

### Predict a race
```bash
python main.py predict --season 2024 --round 5
```

### Monte Carlo simulation
```bash
python main.py simulate --season 2024 --round 5 --n-sims 10000
```
Outputs win, podium, and points-finish probabilities for all drivers.

### Evaluate walk-forward results
```bash
python main.py evaluate --plot    # --plot saves a diagnostic chart
```

## Feature Set

The model uses ~28 engineered features, all era-safe (no absolute lap times ever used raw).

### Qualifying / Grid
| Feature | Source | Notes |
|---------|--------|-------|
| `quali_gap_to_pole_pct` | Q1/Q2/Q3 times | % gap behind pole — the #1 most predictive feature |
| `quali_gap_zscore` | Above, normalised | Z-score within the race field |
| `grid_position` | Race result | Starting slot (1 = pole) |
| `sector1/2/3_gap_pct` | Sector times | % gap to best sector in session |

### Form & History
| Feature | Source | Notes |
|---------|--------|-------|
| `driver_rolling_pos_{3,5,10}r` | Historic results | Lag-1 shifted rolling avg position |
| `team_rolling_pos_{3,5,10}r` | Historic results | Same for constructor |
| `driver_circuit_avg_pos` | Historic results | Driver's avg at this specific circuit |
| `team_circuit_avg_pos` | Historic results | Constructor's avg at this circuit |
| `driver_dnf_rate_10r` | Historic results | DNF rate over last 10 races |
| `team_dnf_rate_10r` | Historic results | Same for team |

### Ratings
| Feature | Source | Notes |
|---------|--------|-------|
| `driver_elo` | Elo system | Pre-race driver skill rating |
| `constructor_elo` | Elo system | Pre-race constructor performance rating |

### Championship Context
| Feature | Source | Notes |
|---------|--------|-------|
| `championship_gap_driver` | Standings | Points behind WDC leader |
| `championship_gap_team` | Standings | Points behind WCC leader |
| `season_round_pct` | Event schedule | How far through the season (0–1) |

### Weather (race-level aggregates)
| Feature | Source | Notes |
|---------|--------|-------|
| `is_wet_race` | `Rainfall` boolean | Binary — triggers alternate strategy model |
| `mean_track_temp` | `TrackTemp` mean | Best single proxy for tire degradation rate |
| `mean_air_temp` | `AirTemp` mean | Engine/aero reference temperature |
| `mean_wind_speed` | `WindSpeed` mean | Nonlinear risk factor; relevant above ~12 m/s |
| `mean_humidity` | `Humidity` mean | Secondary tire/grip feature; L2-regularised |

> **Excluded weather**: `WindDirection` (requires per-sector circuit geometry modeling), `Pressure` (signal fully absorbed by circuit identity + altitude constant).

### Race Conditions
| Feature | Source | Notes |
|---------|--------|-------|
| `n_safety_cars` | `track_status` | Count of SC deployments |
| `n_vsc_periods` | `track_status` | Count of VSC deployments |
| `regulation_era` | Derived from season | Categorical: era of technical regulations |
| `seasons_since_reg_change` | Derived | How stable the current rules are |

## Key Design Decisions

### Overfitting Prevention
- **Walk-forward validation**: Train on races 1→N, predict race N+1. Never uses future data.
- **Exponential sample decay**: Older races receive lower weight (`SAMPLE_WEIGHT_DECAY=0.15`).
- **Regulation-break reset**: Training window resets at major rule changes (2009, 2014, 2017, 2022).
- **Conservative hyperparameters**: `max_depth=5`, `num_leaves=31`, L1+L2 regularization.

### Cross-Era Normalization
All time-based features are era-safe by design:
- `quali_gap_to_pole_pct`: % gap to the fastest qualifier in the **same session**.
- `quali_gap_zscore`: Z-score of the gap within the **same race** field.
- Rolling position averages: relative finishing order, not absolute lap times.
- Speed trap readings: z-score within each session, not raw km/h values.

### Weather Feature Rationale
- **Included**: `Rainfall`, `TrackTemp`, `AirTemp`, `WindSpeed`, `Humidity`
- **Excluded**: `WindDirection` (needs per-sector circuit model to be meaningful), `Pressure` (redundant with circuit altitude, which is a fixed per-circuit constant)

### Pit Stop Modeling
- **LightGBM**: Encodes historical strategy patterns as rolling features.
- **Monte Carlo**: Explicitly models the pit stop decision tree including undercut viability, mandatory stops at tire cliff, and safety car windows.

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Spearman ρ | Rank correlation between predicted and actual order | > 0.7 |
| Top-3 accuracy | % of top-3 drivers correctly identified | > 50% |
| MAE | Mean absolute error in positions | < 3.0 |

## Extending the Project

- **Hyperparameter tuning**: Add an Optuna study targeting Spearman ρ (swap `F1LGBMModel.fit()`).
- **Bayesian layer**: Replace static Elo with a PyMC hierarchical model for honest uncertainty intervals.
- **Telemetry features**: Batch-aggregate `SpeedFL` and `% full throttle` per circuit as constructor power unit proxies.
- **Live prediction**: Connect to FastF1's live timing client to update predictions lap-by-lap during a race.
