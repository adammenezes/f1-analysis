# F1 Predictor: Project Evolution & Results Report

## 1. Executive Summary
The F1 Predictor project was initiated to build a robust, machine-learning-based system capable of predicting Formula 1 race outcomes. Over the course of its development, the project evolved from basic data ingestion scripts into a production-grade MLOps pipeline featuring hybrid experiment tracking, walk-forward validation, and automated feature selection. The current 1.0 production model relies on an ultra-refined "Elite 10" feature set, optimized mathematically to balance overall rank correlation, podium accuracy, and mean absolute error.

---

## 2. Phase 1: Inception and Data Foundation
The first challenge in predicting F1 is the sheer complexity and fragmentation of the data. The project established a dual-source data ingestion pipeline:
*   **Historical Results (Ergast API / Kaggle):** Used to establish long-term baselines, stretching back to 2010 (the start of the Pirelli tire era) to build historical context.
*   **Modern Telemetry (FastF1):** Integrated to capture hyper-granular, session-level data from the modern Turbo-Hybrid era (2014-Present).

A significant early decision was restricting the active training dataset to the **2014 season onwards**. This isolated the model to the modern V6 Turbo-Hybrid engine regulations, ensuring the algorithm wasn't chasing irrelevant engineering trends from the V8 or V10 eras.

---

## 3. Phase 2: Feature Engineering & Modeling
Formula 1 is a sport where absolute speed is less important than relative momentum and historical context. To capture this, the pipeline was engineered with a specific set of features:
*   **Rolling Averages:** Tracking a driver and constructor's performance over the last 3 and 10 races to capture "current form."
*   **Circuit Specifics:** Tracking how well a specific team or driver historically performs at the current venue (e.g., "Monaoco Specialists").
*   **Elo Rating System:** Implemented a chess-style Elo rating system for both Drivers and Constructors. This was a critical innovation to mathematically quantify the "skill" of a driver relative to the machinery they are given.
*   **The Algorithm:** The project utilizes **LightGBM** configured for `regression_l1` (Mean Absolute Error). 

---

## 4. Phase 3: Validation & Experiment Tracking
To prevent data leakage (a common trap where models accidentally "see" the future), we implemented **Walk-Forward Validation**. The model starts by training on a minimum set of historical races (e.g., 50 races) and then predicts the *very next* race. It then absorbs that race's true results and predicts the next one, exactly mimicking how the model will be used in reality.

To track our progress, we built a **Hybrid MLOps Tracking System**:
*   **MLflow:** A local SQLite-backed MLflow server tracks high-level metrics (Spearman Rho, Top-3 Accuracy) across every single run.
*   **Local CSV Logging:** Deep, granular data (like fold-by-fold history and SHAP feature importance) is saved to timestamped CSV folders for manual audit.

---

## 5. Phase 4: The Ablation Study and the "Elite 10"
With over 30 features engineered, the model was at risk of overfitting to noise. We executed an **Automated Recursive Feature Elimination (RFE) Study**. The script trained 30 separate models, using SHAP (Shapley Additive exPlanations) to identify and drop the weakest feature at every step.

To determine the true winner, we calculated a **Unified Performance Index (UPI)**. By converting Overall Accuracy (Spearman Rho), Podium Accuracy (Top-3), and Error Rate (MAE) into Z-Scores, we found the mathematical point of diminishing returns.

**The result was the "Elite 10" Feature Model.** It destroyed heavier models by discarding 20 variables of noise.
It achieved a **68.8% Top-3 Accuracy** and the second-lowest MAE (3.02) across all 30 tests.

**The 10 Surviving Features:**
1. `grid_position` (The ultimate baseline)
2. `team_rolling_pos_10r`
3. `team_circuit_avg_pos`
4. `driver_rolling_pos_10r`
5. `championship_gap_team`
6. `driver_circuit_avg_pos`
7. `constructor_elo`
8. `driver_elo`
9. `teammate_outquali_rate_10r`
10. `team_rolling_pos_3r`

Variables like weather, explicit pit times, and practice lap aggregates were mathematically proven to be noise for this specific architecture.

---

## 6. Phase 5: Future Directions & Nuanced Analysis
The current 1.0 architecture is a rock-solid, highly defensible foundation based on historical and rolling statistics. However, to push past the ~70% accuracy ceiling, future iterations must transition from *historical* analysis to *live event* analysis, while expanding how we measure success.

### 1. Nuanced Evaluation Metrics
Relying solely on MAE and Top-3 Accuracy is reductive. Future models will be evaluated against:
*   **Top 10 Accuracy (Points Finishers):** Crucial for evaluating mid-field predictions.
*   **Position Tolerance ($\pm$ 1, $\pm$ 2):** Treating a prediction of 4th when the driver finished 3rd as a near-success rather than a failure.
*   **Teammate H2H Accuracy:** Isolating the car's performance to test if the model truly understands driver skill.
*   **RMSE & Brier Scores:** Evaluating probability calibration and heavily penalizing catastrophic prediction misses.

### 2. DNF (Did Not Finish) Classification
Currently, the model struggles with the inherent chaos of crashes and mechanical failures. Future work requires a dedicated sub-model to predict DNFs.
*   **Mechanical vs. Crash Separation:** Differentiating reliability issues (engine age/penalties) from crashes (track type/driver aggression).
*   Tracking **DNF Precision vs. Recall**.

### 3. Transitioning to Neural Networks & Raw Telemetry
The ultimate future state of this project is moving beyond tabular data (LightGBM) and into deep learning.
*   **Random Forests / LSTMs:** Exploring models capable of ingesting time-series telemetry.
*   **FP1, FP2, FP3 Integration:** Instead of relying on how a team performed *last* week, the model should ingest the live telemetry (cornering speeds, tire deg curves, straight-line speed) from the current weekend's Free Practice sessions to determine the *true* pace of the car before qualifying even begins.
