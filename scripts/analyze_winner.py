import pandas as pd
import numpy as np
from pathlib import Path
import os

# Paths
runs_dir = Path("data/processed/runs")
output_path = Path("data/processed/ablation_final_results.csv")

if not runs_dir.exists():
    print(f"Error: {runs_dir} not found.")
    exit(1)

records = []
folders = list(runs_dir.glob("RFE_Step_*_features"))

# Scan for RFE directories
for run_folder in folders:
    fold_csv = run_folder / "fold_history.csv"
    if fold_csv.exists():
        df_fold = pd.read_csv(fold_csv)
        
        # Calculate aggregate metrics
        record = {
            "run_name": run_folder.name,
            "features_count": int(run_folder.name.split('_')[2]),
            "out_spearman_rho": df_fold['spearman_rho'].mean(),
            "out_top1_acc": df_fold['top1_acc'].mean(),
            "out_top3_acc": df_fold['top3_acc'].mean(),
            "out_mae": df_fold['mae'].mean()
        }
        records.append(record)

if not records:
    print("No RFE run data found.")
    exit(1)

df_rfe = pd.DataFrame(records)

# 1. Z-Score Normalization
def z_score(series):
    if len(series) <= 1 or series.std() == 0:
        return series * 0 
    return (series - series.mean()) / series.std()

df_rfe['z_rho'] = z_score(df_rfe['out_spearman_rho'])
df_rfe['z_top3'] = z_score(df_rfe['out_top3_acc'])
df_rfe['z_mae'] = z_score(df_rfe['out_mae']) * -1 # negative because lower MAE is better

# 2. UPI
df_rfe['unified_score'] = (df_rfe['z_rho'] + df_rfe['z_top3'] + df_rfe['z_mae']) / 3

# 3. Sort
df_results = df_rfe.sort_values('unified_score', ascending=False)
df_results.to_csv(output_path, index=False)

# Print Top 5
print("\n" + "="*80)
print(f"{'TOP 5 MODELS - UNIFIED PERFORMANCE INDEX (UPI)':^80}")
print("="*80)
cols_to_print = ['run_name', 'features_count', 'unified_score', 'out_spearman_rho', 'out_top3_acc', 'out_mae']
print(df_results[cols_to_print].head(5).to_string(index=False))

winner = df_results.iloc[0]
print("\n" + "*"*80)
print(f" MATHEMATICAL WINNER: {winner['run_name']}")
print(f" This model utilizes {winner['features_count']} features.")
print("*"*80)

specialist = df_rfe.sort_values('out_top3_acc', ascending=False).iloc[0]
print(f" TOP-3 SPECIALIST:   {specialist['run_name']} ({specialist['out_top3_acc']:.2%})")
print("*"*80)
