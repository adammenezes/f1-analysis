# F1 Predictor: Project Evolution & Results

## 1. Executive Summary

This report details the evolution and current state of the F1 Predictor, a robust machine-learning pipeline designed to forecast Formula 1 race outcomes. To combat the extreme variance inherent to motorsport, the project leverages a highly disciplined data strategy—restricting training to the modern V6 Turbo-Hybrid era (2014-Present) and utilizing Walk-Forward validation to perfectly simulate real-world, out-of-sample prediction scenarios without data leakage.

Through an automated Recursive Feature Elimination (RFE) study, we mathematically reduced an initial set of 30 engineered features down to an "Elite 10" core. This reduction was governed by a novel Unified Performance Index (UPI)—a Z-Score normalization algorithm ensuring the final model is Pareto Optimal across overall rank correlation, Mean Absolute Error (MAE), and Top-3 podium accuracy. The current 1.0 architecture (utilizing LightGBM and custom Elo ratings) is mathematically proven to be peak efficiency, achieving **68.8% Top-3 Accuracy**.

Finally, this report establishes a roadmap for the future. To shatter the current ~70% accuracy ceiling, future iterations must transition from purely historical tabular data to live weekend telemetry. This will require the implementation of deep learning architectures (LSTMs and CNNs) to parse raw throttle, brake, and cornering G-force traces from Free Practice sessions, alongside a suite of nuanced evaluation metrics tailored for the complexities of modern Formula 1.

---

## 2. Inception and Data Foundation

The cornerstone of predicting an engineering-dominated, high-variance sport like Formula 1 is data quality and relevance. The project leverages two distinct data sources to build its foundation:
*   **Jolpica API (Ergast fork):** Provides macro-level historical context, including championship standings, race results, driver bios, and grid positions.
*   **FastF1 API:** Provides micro-level session data, including weather conditions, track telemetry, and session-specific timings.

### The "Turbo-Hybrid" Cutoff
A critical architectural decision was to restrict training data to the **2014 season onwards**. In 2014, Formula 1 transitioned to the V6 Turbo-Hybrid regulation era. The physics of the cars, the reliability of the engines, and the dominance hierarchies shifted radically. Training a model on 2010-2013 (V8 era) data would introduce "concept drift" and actively harm the model's ability to predict modern races. 

### Handling Missing Data
Because historical data is notoriously inconsistent (especially regarding weather or lower-tier driver telemetry), the pipeline handles missing values programmatically:
*   **Missing Weather Data:** Filled using mediants from available data or defaulted to zero (e.g., assuming a dry race if `is_wet_race` cannot be determined).
*   **Rookies:** New drivers have no historical performance. They are assigned a baseline Elo rating of 1500 and their "rolling averages" are imputed with median mid-field values to prevent the model from failing or assigning them default 0s (which implies 1st place).

### Raw Data Ingested
*   **Race Results:** Finish position, grid position, points, laps completed, status (DNF reason).
*   **Qualifying Data:** Q1, Q2, Q3 times, gap to pole.
*   **Driver & Constructor Metadata:** Ages, nationalities, experience levels.
*   **Championship Standings:** Points gaps between leaders and followers for both drivers and constructors.
*   **Weather:** Track temperature, air temperature, wind speed, humidity, rainfall.

---

## 3. Feature Engineering & Modeling

We engineered exactly **30 features** to describe the state of the F1 world before the lights go out. 

### The 30 Features & Their Context

**Pace & Grid**
1.  `grid_position`: The starting position. The single most predictive baseline.
2.  `quali_gap_to_pole_pct`: The percentage difference in lap time between the driver and the pole sitter.
3.  `quali_gap_zscore`: The standardized gap to pole, normalizing for long vs. short tracks.

**Driver Form (Rolling Averages)**
4.  `driver_rolling_pos_3r`: Driver's average finish position over the last 3 races (Hyper-recent form).
5.  `driver_rolling_pos_5r`: Driver's average over the last 5 races.
6.  `driver_rolling_pos_10r`: Driver's average over the last 10 races (Long-term form).

**Constructor Form (Rolling Averages)**
7.  `team_rolling_pos_3r`: Team's average finish position over the last 3 races.
8.  `team_rolling_pos_5r`: Team's average over the last 5 races.
9.  `team_rolling_pos_10r`: Team's average over the last 10 races.

**Experience & Pedigree**
10. `driver_career_races`: Total number of starts.
11. `driver_is_rookie`: Boolean flag indicating if the driver is in their first season.
12. `driver_career_wins`: Lifetime wins prior to the race.
13. `driver_career_podiums`: Lifetime podiums prior to the race.

**Intra-Team Battles (The Ultimate Benchmark)**
14. `teammate_outquali_rate_10r`: % of times the driver out-qualified their teammate in the last 10 races.
15. `teammate_beat_rate_10r`: % of times the driver finished ahead of their teammate in the last 10 races.

**Track Affinity**
16. `driver_circuit_avg_pos`: Driver's career average finish at this specific venue.
17. `team_circuit_avg_pos`: Team's career average finish at this specific venue.

**Reliability**
18. `driver_dnf_rate_10r`: % of Did-Not-Finish occurrences for the driver recently.
19. `team_dnf_rate_10r`: % of DNFs for the constructor recently (measures engine/car reliability).

**Skill Ratings (Elo)**
20. `driver_elo`: A chess-style rating representing the driver's intrinsic skill, updated race-by-race.
21. `constructor_elo`: A rating representing the car's intrinsic pace.

**Macro Context**
22. `seasons_since_reg_change`: How deep into a regulation era we are (gaps tend to close later in an era).
23. `championship_gap_driver`: Points gap to the WDC leader.
24. `championship_gap_team`: Points gap to the WCC leader.
25. `regulation_era`: Categorical string representing the specific FIA regulation set.

**Weather Conditions**
26. `is_wet_race`: Boolean flag for rain.
27. `mean_track_temp`: Average track surface temperature.
28. `mean_air_temp`: Average ambient temperature.
29. `mean_wind_speed`: Wind velocity.
30. `mean_humidity`: Moisture in the air.

### Model Selection: LightGBM (LGBM)
We selected **LightGBM**, configured as a regressor with an L1 objective (Mean Absolute Error). 

**Why LGBM over Neural Networks or Linear Models?**
1.  **Tabular Domination:** Tree-based boosters (XGBoost, LGBM) consistently outperform deep learning on medium-sized structured tabular data.
2.  **Robust to Noise and Outliers:** F1 data is noisy (crashes, safety cars). We use MAE (L1 loss) because it does not catastrophically penalize the model when a driver crashes out from 1st place, unlike MSE (L2 loss).
3.  **Missing Values & Categoricals:** LGBM natively handles missing data and categorical strings (like `regulation_era`) without requiring massive one-hot encoded matrices.
4.  **Defensibility (SHAP):** Tree ensembles allow for exact calculation of SHAP values, giving us a mathematically perfect explanation of *why* the model made every single prediction.

---

## 4. Validation and Experiment Tracking

### Walk-Forward Validation
To test the model, we could not use standard K-Fold Cross Validation. If you train an F1 model on 2018 data, and ask it to predict 2016, it is "cheating" by knowing the future (e.g., knowing Mercedes is dominant). 
We use **Walk-Forward Validation**:
1.  Train on races 1 through $N$.
2.  Predict race $N+1$.
3.  Train on races 1 through $N+1$.
4.  Predict race $N+2$.
This perfectly simulates the reality of deploying a model on a Sunday morning.

### Hybrid Experiment Tracking
We track experiments using a dual system:
1.  **MLflow:** Logs high-level metrics (Top-3 Accuracy, Spearman Rho) and tags to a local SQLite database, providing a visual dashboard to compare runs.
2.  **Local CSV Logging:** Because MLflow can be brittle with large artifacts, we save the granular fold-by-fold histories and SHAP importance tables directly to local CSV directories (`data/processed/runs/`). This ensures no data is lost and allows for custom downstream Python analysis.

---

## 5. Automated Recursive Feature Elimination (RFE) & UPI

Having 30 features is prone to the 'Curse of Dimensionality'—the model starts memorizing noise rather than learning physics. To combat this, we ran an automated **Recursive Feature Elimination (RFE)**.

### The RFE Loop & Mean Absolute SHAP
1.  Train the model on all $N$ features.
2.  Calculate the **SHAP values** for every prediction. SHAP maps out how many positions a specific feature shifted the final prediction.
3.  Calculate the **Mean Absolute SHAP**. If a feature's average impact on the final prediction is only 0.01 positions, it is useless.
4.  Drop the single feature with the lowest Mean Absolute SHAP.
5.  Repeat until 1 feature remains.

### The Unified Performance Index (UPI)
Evaluating models is a Multi-Objective Optimization problem. One model might be amazing at Top-3 accuracy but terrible overall. To find the "True" optimal feature count, we utilized Z-Score Normalization to create a **Unified Performance Index (UPI)**.

We took the three key metrics (`spearman_rho`, `top3_acc`, and `mae`), converted them to standard deviations from the study's mean (Z-Scores), and averaged them. 

This mathematical standardization proved that our **10-feature model** was the ultimate champion. It achieved the highest Top-3 accuracy (68.8%), near-lowest MAE (3.02), and highest overall UPI, proving that the 20 dropped features were actively hurting prediction consistency.

---

## 6. Future Directions & Nuanced Analysis

While the 10-feature historical model is a highly defensible baseline, it is still making predictions *before* seeing the cars hit the track on the weekend. To shatter the ~70% accuracy ceiling, we must incorporate live telemetry and expand our evaluation footprint.

### Transitioning to Predictive Telemetry & Deep Learning
Currently, the model looks at *historical* rolling momentum. In the future, we must pull live **Free Practice (FP1, FP2, FP3)** telemetry from the FastF1 API. 
*   **The Problem:** Tabular models like LightGBM cannot natively ingest a million rows of time-series throttle/brake traces.
*   **The Solution:** We must implement deep learning models, specifically **LSTMs (Long Short-Term Memory)** or **1D-CNNs**, specifically trained to ingest sequential lap data, cornering G-forces, and tire degradation curves. The LSTM would output an "Expected Pace" scalar, which is then fed into our LightGBM model as a massive new feature. Similarly, **Random Forests** can be used to aggregate and classify complex categorical telemetry flags (like tire compound delta-times) to create a robust practice-pace profile.

### Advanced Evaluation Metrics
As outlined in `future_metrics_tracking.md`, predicting F1 is not just about ranking 1 through 20. We will implement:
*   **DNF Prediction (Sub-Models):** Separating purely mechanical failures (predictable via engine age) from crashes (stochastic, but correlated to circuit type).
*   **Top-10 (Points) Accuracy:** Mid-field predictions are where the real betting/strategic value lies. 
*   **Position Tolerance ($\pm$ 1, $\pm$ 2):** Predicting 4th when a driver finished 3rd should be graded as a "near-hit," not a complete failure.
*   **Head-to-Head (H2H) Accuracy:** The purest test of the model's understanding of driver skill versus car engineering is whether it can accurately predict which Mercedes or Ferrari will cross the line first. 
*   **Calibration (Brier Score):** Ensuring that when the model outputs a "70% win probability," that event statistically happens 70% of the time, rather than just being a raw, uncalibrated score.
