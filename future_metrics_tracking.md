# F1 Model Evaluation: Advanced Metrics & Statistics

To gain a truly nuanced understanding of how your model is performing against reality, we must expand our evaluation suite beyond just "Average Error" (MAE) and "Podium Accuracy" (Top 3). Formula 1 is a highly complex, multi-objective sport, and your model evaluation should reflect all facets of the race.

Here is an exhaustive list of derived statistics and metrics we should implement to compare our model against reality in future iterations:

## 1. Positional & Tier Accuracy (Classification)
Instead of just whether we got the exact position right, how often do we place drivers in the correct "Bucket" or "Tier"?

*   **Top 10 Accuracy (Points Finishers):** The percentage of time the model correctly predicted if a driver would score points. This is crucial for evaluating mid-field predictions, where MAE can be misleading.
*   **Exact Position Match Rate ($Acc_{Exact}$):** The percentage of drivers predicted in their *exact* finishing position.
*   **Position Tolerance Accuracy ($\pm$ 1, $\pm$ 2):** The percentage of predictions that were within 1 or 2 positions of the actual result. A driver finishing 4th instead of 3rd is still a very good prediction.
*   **Tier Classification:** Did we correctly predict the driver would finish in the "Podium Tier" (1-3), "Points Tier" (4-10), "Backmarker Tier" (11-15), or "At Risk Tier" (16-20)?
*   **Teammate Head-to-Head (H2H) Accuracy:** How often did the model correctly predict which driver in the same team would finish ahead? This isolates car performance and tests if the model truly understands driver skill.

## 2. Advanced Ranking & Error Metrics
Mathematical distances between predicted and actual results.

*   **RMSE (Root Mean Squared Error):** Unlike MAE (which treats all errors equally), RMSE heavily penalizes *large* errors. If the model predicts Verstappen 1st and he finishes 20th, RMSE flags this as a massive failure, whereas MAE is more forgiving.
*   **NDCG (Normalized Discounted Cumulative Gain):** A search engine metric that is perfect for F1. It heavily weights getting the top positions (1st, 2nd, 3rd) right, and cares very little if you mix up who finished 17th vs 18th.
*   **Kendall Rank Correlation Coefficient ($\tau$):** Measures the number of concordant and discordant pairs. It's often more robust than Spearman for smaller lists (like a 20-car grid).
*   **Directional Accuracy (Quali vs. Race):** If a driver qualifies 5th, does the model correctly predict if they will move *Forward* or *Backward* during the race, regardless of the exact end position?

## 3. Reliability & Event Predictions (Sub-Models)
F1 is highly volatile. Predicting retirements requires a different approach than predicting pure speed.

*   **DNF (Did Not Finish) Accuracy:** 
    *   *Precision:* When the model predicts a DNF, how often do they actually DNF?
    *   *Recall:* Out of all the actual DNFs, how many did the model catch?
*   **Mechanical vs. Crash Separation:** Differentiating between reliability issues (can be predicted by engine age/grid penalties) and crashes (highly random, but correlated with track type and driver aggression).
*   **Fastest Lap Prediction:** Can the model predict who will take the extra point?
*   **Lap 1 Position Change:** Who gains or loses the most positions on lap 1? (Requires lap-by-lap data).

## 4. Probability & Calibration
If your model outputs probabilities (e.g., "Lando Norris has a 65% chance to win"), you must track if those probabilities reflect reality.

*   **Brier Score:** Measures the accuracy of probabilistic predictions. If the model says 10 different drivers have a 10% chance to win, exactly 1 of them should win. If 0 or 5 win, your model is poorly calibrated.
*   **Log Loss (Cross-Entropy Loss):** Heavily penalizes the model for being confidently wrong (e.g., giving a driver a 99% chance to win, but they finish 5th).
*   **Calibration Curves (Reliability Diagrams):** A visual plot comparing predicted probabilities against actual outcomes. If the model says "70% chance of a Top 3", does that event happen 70% of the time over the season?

## 5. Contextual Segmented Analysis
Global metrics hide specific weaknesses. We must track performance sliced by context:

*   **Performance by Circuit Type:** Does the model perform better on Street Circuits (Monaco, Baku) vs. High-Downforce Tracks (Silverstone, Suzuka)?
*   **Performance by Weather:** What is the MAE in Dry races vs. Wet races? (Models usually perform significantly worse in the wet).
*   **Performance by Team:** Are we consistently over-predicting Ferrari and under-predicting McLaren?
*   **Performance by Grid Position:** Is the model great at predicting the front row, but terrible at predicting the midfield shuffle?

## 6. Betting & Business Metrics
If the model is used for financial modelling or gamification.

*   **Implied Probability ROI:** If you compare your model's win probability to Vegas odds, what is the simulated Return on Investment over the season?
*   **Underdog Strike Rate:** How often did the model correctly predict a driver outside the "Big 4" teams to score a podium?
*   **Longshot Value:** Identifying drivers whose predicted finish position is significantly higher than their qualifying position.
