# F1 Predictor (1.0 Production)

A robust, machine-learning-based system capable of predicting Formula 1 race outcomes. Through aggressive optimization, hybrid MLOps tracking, and an automated Recursive Feature Elimination (RFE) study, this pipeline uses an "Elite 10" feature set to achieve mathematically proven peak efficiency (68.8% Podium Accuracy).

## Architecture

```
f1-predictor/
├── main.py                    # CLI entry point (fetch, train, dashboard)
├── requirements.txt           # Python dependencies
├── .env.example               # Copy to .env and configure
├── data/
│   ├── raw/
│   │   └── fastf1_cache/      # FastF1 session cache
│   └── processed/
│       ├── features.parquet   # Engineered feature matrix
│       ├── elo_snapshots.parquet
│       ├── lgbm_model.txt     # Saved LightGBM model
│       └── runs/              # Hybrid CSV logging for RFE & SHAP
├── src/
│   ├── data/
│   │   ├── loader.py          # Jolpica API + FastF1 data ingestion
│   │   └── features.py        # Feature engineering pipeline
│   ├── models/
│   │   ├── elo.py             # Dual-track Elo rating system
│   │   ├── xgboost_model.py   # LightGBM race predictor (L1 Loss)
│   │   └── monte_carlo.py     # Monte Carlo simulator
│   ├── validation/
│   │   └── walk_forward.py    # Walk-forward expanding-window validator
│   └── utils/
│       └── normalization.py   # Z-Score normalization routines
└── scripts/
    ├── ablation_study.py      # Automated RFE loop (SHAP)
    └── analyze_winner.py      # Unified Performance Index calculation
```

## Data Sources & The "Turbo-Hybrid" Cutoff

| Source | Timeframe | What it provides |
|--------|----------|--------------------|
| [Jolpica API](https://api.jolpi.ca/ergast/f1) | **2014–Present** | Race results, standings, driver bios, grid positions |
| [FastF1](https://docs.fastf1.dev) | **2014–Present** | Weather conditions, basic timing configurations |

> **Critical Note on the 2014 Cutoff:**  
> Training data is strictly limited to the 2014 season onwards. In 2014, Formula 1 transitioned to the V6 Turbo-Hybrid regulation era. Training on V8 era data (2010-2013) introduces "concept drift" and actively harms the model's ability to predict modern F1 dominance hierarchies.

## Storage (Local Required)

**Local storage is recommended** — no cloud drive needed. Network storage (Google Drive, OneDrive) adds significant I/O latency that would slow training loops dramatically. Set `FASTF1_CACHE_DIR` in your `.env` to point to a different drive if needed.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
```

## Usage

### Download Race Data
```bash
python main.py fetch --start-year 2014 --end-year 2024
```
FastF1 session data is safely cached, making subsequent runs lightning fast.

### Train the Production Model
```bash
python main.py train
```
Runs walk-forward validation across all races. Saves the final production model to `data/processed/lgbm_model.txt`.

### Evaluate & Dashboard
We track experiments using a hybrid system. Run this to review iterations visually:
```bash
python main.py dashboard
```
You can also manually audit the timestamped CSV SHAP and fold histories located in `data/processed/runs/`.

---

## Methodology & Feature Engineering

The project initially engineered **30 distinct features**. However, Formula 1 modeling is extremely prone to the "Curse of Dimensionality" (overfitting to noise). To solve this, we ran an **Automated Recursive Feature Elimination (RFE) Study**, governed by **Mean Absolute SHAP**, dropping the weakest link across 30 iterative training runs.

Models were evaluated using a **Unified Performance Index (UPI)**, converting Overall Rank Correlation (Spearman Rho), Prediction Error (MAE), and Podium Accuracy (Top-3) into an aggregated Z-Score.

This definitively proved that a dense **"Elite 10" feature model** is the mathematical peak, stripping away 20 variables of noise.

### The Elite 10 Features
| Feature | Source | Why it Matters |
|---------|--------|-------|
| `grid_position` | Race result | Starting slot (1 = pole) - The ultimate baseline |
| `team_rolling_pos_10r`| Historic results | Constructor's average over the last 10 races |
| `team_circuit_avg_pos`| Historic results | Constructor's historical affinity for the current track |
| `driver_rolling_pos_10r` | Historic results | Driver's long-term recent form |
| `championship_gap_team`| Standings | Points behind the WCC leader (maps financial dominance) |
| `driver_circuit_avg_pos` | Historic results | Driver's historical affinity for the current venue |
| `constructor_elo` | Elo system | Pre-race intrinsic car pace |
| `driver_elo` | Elo system | Pre-race intrinsic driver skill level |
| `teammate_outquali_rate_10r` | Qualifying | % the driver out-qualifies their teammate (Head-to-head metric) |
| `team_rolling_pos_3r` | Historic results | Constructor's hyper-recent (3 race) form |

*(Note: Features previously believed to be critical, such as Practice times, Raw Weather, or Career Race Starts, were mathematically eliminated as noise by the SHAP analysis for this specific architecture).*

### Model Selection
The core predictor is **LightGBM (LGBM)** configured with an **L1 Objective (Mean Absolute Error)**. 
- **Tabular Superiority:** LGBM outperforms deep learning for medium-scale tabular regressions.
- **Robustness:** L1 loss prevents catastrophic penalties from stochastic events (e.g., a driver crashing from 1st place).
- **Interpretability:** Exact SHAP calculation allows us to perfectly explain every prediction.

## Future Enhancements
To break the ~70% precision ceiling (achieved with strictly historical variables), future architectural phases demand:
1. **Long Short-Term Memory (LSTM) Networks:** To ingest raw, massive sequential telemetry (throttle traces, corner G-forces) directly from weekend Free Practice sessions.
2. **DNF Sub-Models:** Classification splits for Mechanical Failures vs. Crashes. 
3. **Brier Calibration:** Advanced probability scaling for expected win/podium percentages.
