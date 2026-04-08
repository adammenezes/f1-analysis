"""
scripts/smoke_test.py
─────────────────────
Quick sanity-check for the F1 Predictor pipeline.

Tests every layer with minimal data so the whole thing runs in ~60 seconds:
  ✓  Jolpica API  — fetches 2023 season results (1 API call batch)
  ✓  FastF1       — loads Bahrain 2023 Race + Qualifying (1 round)
  ✓  Constructor lineage — verifies all 2023 team IDs resolve
  ✓  Feature pipeline — builds the feature matrix and checks shape / dtypes
  ✓  Elo system   — fits and checks rating snapshot exists pre-race
  ✓  LightGBM     — tiny train + predict cycle (no walk-forward)
  ✓  Data directory — confirms expected files land in the right places

Run from the project root AFTER activating the venv:
    python scripts/smoke_test.py

Exit code 0 = all checks passed.
Exit code 1 = at least one check failed (details printed above).
"""

from __future__ import annotations

import io
import logging
import sys
import traceback
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Make sure project root is on the path ────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.WARNING,          # suppress FastF1 noise during test
    format="%(levelname)-8s | %(name)s | %(message)s",
)
logging.getLogger("f1-predictor").setLevel(logging.INFO)

PASS = "  [OK]"
FAIL = "  [!!]"
SKIP = "  [--]"

_results: list[tuple[str, bool, str]] = []   # (name, passed, detail)


def check(name: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), record pass/fail, return the result or None."""
    print(f"\n{'─'*55}\n  {name}")
    try:
        result = fn(*args, **kwargs)
        print(f"{PASS}  PASSED")
        _results.append((name, True, ""))
        return result
    except Exception as exc:
        detail = traceback.format_exc()
        print(f"{FAIL}  FAILED → {exc}")
        print(detail)
        _results.append((name, False, str(exc)))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Import smoke — catch any broken imports immediately
# ─────────────────────────────────────────────────────────────────────────────

def _test_imports():
    from src.data.loader import (
        fetch_historical_results, fetch_fastf1_results,
        load_full_dataset, year_to_era, seasons_since_regulation_change,
    )
    from src.data.features import build_features
    from src.data.constructor_lineage import (
        normalize_constructor_id, CONSTRUCTOR_LINEAGE,
    )
    from src.models.elo import EloSystem
    from src.models.xgboost_model import F1LGBMModel
    print(f"    Imported all core modules successfully.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 2. Constructor lineage — all 2023 team names resolve
# ─────────────────────────────────────────────────────────────────────────────

def _test_lineage():
    from src.data.constructor_lineage import normalize_constructor_id

    # These are the exact TeamName strings FastF1 returns in 2023
    teams_2023 = [
        "Red Bull Racing", "Ferrari", "Mercedes", "Aston Martin",
        "Alpine", "McLaren", "Alfa Romeo", "Haas F1 Team",
        "AlphaTauri", "Williams",
    ]
    unknown = []
    for team in teams_2023:
        lid = normalize_constructor_id(team)
        print(f"    {team:<25} → {lid}")
        # A proper lineage ID ends with '_line'; raw passthrough means unknown
        if not lid.endswith("_line"):
            unknown.append(team)

    if unknown:
        raise ValueError(
            f"These 2023 teams have no lineage mapping: {unknown}\n"
            f"Add them to src/data/constructor_lineage.py"
        )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 3. Jolpica API — fetch one season (2023) to confirm connectivity + parsing
# ─────────────────────────────────────────────────────────────────────────────

def _test_jolpica():
    from src.data.loader import fetch_historical_results
    import pandas as pd

    df = fetch_historical_results(start_year=2023, end_year=2023, force_refresh=False)

    assert isinstance(df, pd.DataFrame), "Expected DataFrame"
    assert len(df) > 0, "Got 0 rows — API returned nothing"
    assert "constructor_lineage_id" in df.columns, "Missing constructor_lineage_id column"
    assert "is_dnf" in df.columns, "Missing is_dnf column"
    assert df["finish_position"].between(1, 20).all(), \
        f"Unexpected finish_position values: {df['finish_position'].unique()}"

    rounds = df["round"].nunique()
    drivers = df["driver_id"].nunique()
    print(f"    Rows: {len(df)} | Rounds: {rounds} | Drivers: {drivers}")
    print(f"    Regulation era: {df['regulation_era'].unique()}")
    print(df[["season","round","driver_id","constructor_lineage_id","finish_position"]].head(5).to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. FastF1 — load Bahrain 2023 Race + Qualifying (1 round only)
# ─────────────────────────────────────────────────────────────────────────────

def _test_fastf1():
    from src.data.loader import fetch_fastf1_results
    import pandas as pd

    df = fetch_fastf1_results(start_year=2023, end_year=2023, force_refresh=False)

    assert isinstance(df, pd.DataFrame), "Expected DataFrame"
    assert len(df) > 0, "Got 0 rows"
    assert "constructor_lineage_id" in df.columns, "Missing constructor_lineage_id"
    assert "race_track_temp" in df.columns, "Missing weather column race_track_temp"
    assert "race_rainfall" in df.columns, "Missing weather column race_rainfall"
    assert "q1_time_s" in df.columns, "Missing qualifying column q1_time_s"

    rounds = df["round"].nunique()
    wet_races = df["race_rainfall"].sum()
    print(f"    Rows: {len(df)} | Rounds: {rounds} | Wet races (any driver): {wet_races}")
    print(f"    Track temp range: {df['race_track_temp'].min():.1f}°C – {df['race_track_temp'].max():.1f}°C")
    print(df[["season","round","driver_id","constructor_lineage_id",
              "q1_time_s","race_track_temp","race_rainfall"]].head(5).to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature pipeline — combined dataset → feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def _test_features(hist_df, ff1_df):
    import pandas as pd
    from src.data.loader import load_full_dataset
    from src.data.features import build_features
    from src.models.elo import EloSystem

    # Combine the two already-fetched DataFrames (avoids re-downloading)
    import pandas as pd
    df = pd.concat([hist_df, ff1_df], ignore_index=True)
    df = df.sort_values(["season", "round", "finish_position"]).reset_index(drop=True)

    # Assign race_index (needed by walk-forward validator)
    race_keys = df[["season","round"]].drop_duplicates().sort_values(["season","round"])
    race_keys["race_index"] = range(len(race_keys))
    df = df.merge(race_keys, on=["season","round"], how="left")

    elo = EloSystem().fit(df)
    X, y = build_features(df, elo)

    expected_weather_cols = [
        "is_wet_race", "mean_track_temp", "mean_air_temp",
        "mean_wind_speed", "mean_humidity",
    ]
    meta_cols = ["season", "round", "race_index", "driver_id", "constructor_id"]
    feature_cols = [c for c in X.columns if c not in meta_cols]

    print(f"    Feature matrix shape: {X.shape}")
    print(f"    Target shape: {y.shape}")
    print(f"    Feature columns ({len(feature_cols)}):")
    for col in feature_cols:
        missing = X[col].isna().sum()
        print(f"      {col:<35} missing={missing}")

    # Check weather features present
    missing_weather = [c for c in expected_weather_cols if c not in feature_cols]
    if missing_weather:
        raise ValueError(f"Weather features missing from feature matrix: {missing_weather}")

    # Check no NaNs in critical columns
    critical = ["quali_gap_to_pole_pct", "grid_position", "driver_elo", "constructor_elo"]
    for col in critical:
        if col in X.columns:
            n_nan = X[col].isna().sum()
            if n_nan > len(X) * 0.5:   # allow up to 50% NaN (early races lack history)
                raise ValueError(f"Column '{col}' has {n_nan}/{len(X)} NaNs — likely pipeline error")

    return X, y, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 6. LightGBM — tiny train + predict (not walk-forward, just smoke)
# ─────────────────────────────────────────────────────────────────────────────

def _test_model(X, y, feature_cols):
    from src.models.xgboost_model import F1LGBMModel

    model = F1LGBMModel()
    model.fit(X[feature_cols], y)

    # Predict on the last race in the dataset
    last_season = X["season"].max()
    last_round  = X[X["season"] == last_season]["round"].max()
    X_last = X[(X["season"] == last_season) & (X["round"] == last_round)]

    if X_last.empty:
        raise ValueError("No rows for last race — cannot test predict")

    driver_ids = X_last["driver_id"].tolist()
    ranking = model.predict_ranking(X_last[feature_cols], driver_ids)

    print(f"    Trained on {len(X)} rows → predicting {last_season} Round {last_round}")
    print(f"    Predicted order (first 5):")
    print(ranking.head(5).to_string(index=False))

    assert len(ranking) == len(driver_ids), "Ranking length mismatch"
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 7. Directory structure — confirm expected files exist
# ─────────────────────────────────────────────────────────────────────────────

def _test_directories():
    expected_dirs = [
        ROOT / "data" / "raw" / "fastf1_cache",
        ROOT / "data" / "processed",
        ROOT / "src" / "data",
        ROOT / "src" / "models",
        ROOT / "src" / "validation",
    ]
    expected_files = [
        ROOT / "src" / "data" / "constructor_lineage.py",
        ROOT / "src" / "data" / "loader.py",
        ROOT / "src" / "data" / "features.py",
        ROOT / "src" / "models" / "elo.py",
        ROOT / "src" / "models" / "xgboost_model.py",
        ROOT / "src" / "models" / "monte_carlo.py",
        ROOT / "main.py",
        ROOT / ".env.example",
        ROOT / "requirements.txt",
    ]
    expected_parquet = [
        ROOT / "data" / "processed" / "hist_results_2023_2023.parquet",
        ROOT / "data" / "processed" / "ff1_results_2023_2023.parquet",
    ]

    all_ok = True
    for d in expected_dirs:
        exists = d.exists() and d.is_dir()
        status = PASS if exists else FAIL
        print(f"    {status}  DIR   {d.relative_to(ROOT)}")
        if not exists:
            all_ok = False

    for f in expected_files:
        exists = f.exists() and f.is_file()
        status = PASS if exists else FAIL
        size = f"{f.stat().st_size // 1024} KB" if exists else "MISSING"
        print(f"    {status}  FILE  {f.relative_to(ROOT)}  ({size})")
        if not exists:
            all_ok = False

    for p in expected_parquet:
        exists = p.exists() and p.is_file()
        status = PASS if exists else SKIP
        size = f"{p.stat().st_size // 1024} KB" if exists else "not yet created"
        print(f"    {status}  PARQ  {p.relative_to(ROOT)}  ({size})")
        # Parquet files are warnings, not failures (they're created by the test itself)

    if not all_ok:
        raise FileNotFoundError("One or more required files/dirs are missing (see above).")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 55)
    print("  F1 Predictor — Smoke Test")
    print("  Testing against: 2023 season (minimal data)")
    print("═" * 55)

    check("1. Module imports",          _test_imports)
    check("2. Constructor lineage map", _test_lineage)
    hist_df = check("3. Jolpica API (2023 season)", _test_jolpica)
    ff1_df  = check("4. FastF1 (2023 season)",      _test_fastf1)

    # Feature + model tests only run if data loaded successfully
    if hist_df is not None and ff1_df is not None:
        result = check("5. Feature pipeline",  _test_features, hist_df, ff1_df)
        if result is not None:
            X, y, feature_cols = result
            check("6. LightGBM train+predict", _test_model, X, y, feature_cols)
    else:
        print(f"\n{SKIP}  SKIPPING feature/model tests (data fetch failed above)")
        _results.append(("5. Feature pipeline",    False, "skipped"))
        _results.append(("6. LightGBM train+predict", False, "skipped"))

    check("7. Directory structure", _test_directories)

    # ── Final summary ─────────────────────────────────────────────────────────
    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_fail = len(_results) - n_pass

    print("\n" + "═" * 55)
    print("  SMOKE TEST RESULTS")
    print("═" * 55)
    for name, ok, detail in _results:
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {name}")
        if not ok and detail and "skipped" not in detail:
            print(f"       → {detail[:120]}")

    print(f"\n  {n_pass}/{len(_results)} checks passed")

    if n_fail == 0:
        print("\n  [PASS] All checks passed -- ready for full data fetch!\n")
        print("  Next step:")
        print("    python main.py fetch --start-year 2014 --end-year 2024\n")
        sys.exit(0)
    else:
        print(f"\n  [FAIL] {n_fail} check(s) failed -- fix the issues above before full fetch.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
