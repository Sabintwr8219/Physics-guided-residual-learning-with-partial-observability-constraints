"""
Standalone robustness checks for the CV-constrained queue reconstruction model.

This script performs two robustness checks:

1. Data size robustness:
   Retrains the global XGBoost residual model using different fractions of the
   training data and evaluates on the fixed test runs.

2. Noise / perturbation robustness:
   Adds controlled noise to selected input features and evaluates whether the
   trained model remains stable.

Outputs:
    output/results/robustness_data_size.csv
    output/results/robustness_noise.csv

This script does not generate plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import (
    TRAIN_RUN_IDS,
    TEST_RUN_IDS,
    RANDOM_SEED,
    EVENT_FEATURES_ALLRATES_CSV,
    XGB_MODEL_FILE,
    RESULTS_DIR,
    XGB_N_ESTIMATORS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_SUBSAMPLE,
    XGB_COLSAMPLE_BYTREE,
    XGB_OBJECTIVE,
    ensure_project_directories,
)


# ============================================================
# Robustness-check configuration
# ============================================================

DATA_SIZE_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
DATA_SIZE_REPEATS = 3

NOISE_LEVELS = [0.00, 0.02, 0.05, 0.10]
NOISE_REPEATS = 3

ROBUSTNESS_DATA_SIZE_CSV = RESULTS_DIR / "robustness_data_size.csv"
ROBUSTNESS_NOISE_CSV = RESULTS_DIR / "robustness_noise.csv"

# Noise is added only to continuous/context features, not IDs, labels, target, or binary flags.
NOISE_FEATURE_CANDIDATES = [
    "N",
    "N_norm",
    "tB_sec",
    "dt_from_prev_B_sec",
    "dt_to_next_B_sec",
    "prev_cv_event_N",
    "next_cv_event_N",
    "count_gap_from_prev_cv",
    "count_gap_to_next_cv",
    "time_from_prev_cv_sec",
    "time_to_next_cv_sec",
    "prev_cv_join_sec",
    "time_since_prev_cv_join_sec",
    "latest_cv_N",
    "N_base_at_t",
]


# ============================================================
# Helpers
# ============================================================

def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    """Raise an error if required columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def safe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert selected columns to numeric."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_metrics(true_times: np.ndarray, predicted_times: np.ndarray) -> dict:
    """Compute MAE and RMSE."""
    true_times = np.asarray(true_times, dtype=float)
    predicted_times = np.asarray(predicted_times, dtype=float)

    mask = np.isfinite(true_times) & np.isfinite(predicted_times)

    if not np.any(mask):
        return {
            "mae_sec": np.nan,
            "rmse_sec": np.nan,
            "n_eval": 0,
        }

    y_true = true_times[mask]
    y_pred = predicted_times[mask]

    return {
        "mae_sec": float(mean_absolute_error(y_true, y_pred)),
        "rmse_sec": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n_eval": int(mask.sum()),
    }


def make_xgb_model(seed: int) -> XGBRegressor:
    """Create one XGBoost residual model."""
    return XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        learning_rate=XGB_LEARNING_RATE,
        max_depth=XGB_MAX_DEPTH,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        random_state=seed,
        objective=XGB_OBJECTIVE,
        n_jobs=-1,
    )


def load_feature_columns(event_df: pd.DataFrame) -> list[str]:
    """
    Load feature columns from the saved model package when available.

    If the model file is not available, use a fallback list based on the
    final feature-engineering script.
    """
    if XGB_MODEL_FILE.exists():
        model_package = joblib.load(XGB_MODEL_FILE)

        if isinstance(model_package, dict) and "feature_cols" in model_package:
            feature_cols = model_package["feature_cols"]
            feature_cols = [col for col in feature_cols if col in event_df.columns]

            if feature_cols:
                return feature_cols

    fallback_cols = [
        "N",
        "N_norm",
        "tB_sec",
        "dt_from_prev_B_sec",
        "dt_to_next_B_sec",
        "is_cv_int",
        "prev_cv_event_N",
        "next_cv_event_N",
        "count_gap_from_prev_cv",
        "count_gap_to_next_cv",
        "time_from_prev_cv_sec",
        "time_to_next_cv_sec",
        "prev_cv_join_sec",
        "time_since_prev_cv_join_sec",
        "latest_cv_N",
        "cv_seen_so_far",
        "N_base_at_t",
        "join_event_base",
    ]

    feature_cols = [col for col in fallback_cols if col in event_df.columns]

    if not feature_cols:
        raise ValueError("No usable ML feature columns found.")

    return feature_cols


def load_event_features() -> tuple[pd.DataFrame, list[str]]:
    """Load event-level feature data and return feature columns."""
    if not EVENT_FEATURES_ALLRATES_CSV.exists():
        raise FileNotFoundError(
            f"Missing feature file:\n{EVENT_FEATURES_ALLRATES_CSV}\n"
            "Run cv_ml_pipeline.py first."
        )

    df = pd.read_csv(EVENT_FEATURES_ALLRATES_CSV)

    required = [
        "run_id",
        "cv_rate_pct",
        "tB_sec",
        "t_event_true_sec",
        "resid_true_minus_base_sec",
    ]
    require_columns(df, required, "event_features_allrates.csv")

    df = safe_numeric(
        df,
        [
            "run_id",
            "cv_rate_pct",
            "tB_sec",
            "t_event_true_sec",
            "resid_true_minus_base_sec",
        ],
    )

    df = df.dropna(
        subset=[
            "run_id",
            "cv_rate_pct",
            "tB_sec",
            "t_event_true_sec",
            "resid_true_minus_base_sec",
        ]
    ).copy()

    df["run_id"] = df["run_id"].astype(int)
    df["cv_rate_pct"] = df["cv_rate_pct"].astype(int)

    feature_cols = load_feature_columns(df)

    df = safe_numeric(df, feature_cols)
    df = df.dropna(subset=feature_cols).copy()

    return df, feature_cols


def stratified_train_sample(
    train_df: pd.DataFrame,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    """
    Sample training data while preserving run and CV-rate representation.
    """
    if fraction >= 1.0:
        return train_df.copy()

    sampled_parts = []

    for (_, _), group in train_df.groupby(["run_id", "cv_rate_pct"], sort=False):
        n_group = len(group)
        n_sample = max(1, int(np.floor(fraction * n_group)))

        sampled = group.sample(
            n=n_sample,
            replace=False,
            random_state=seed,
        )

        sampled_parts.append(sampled)

    return pd.concat(sampled_parts, ignore_index=True)


def evaluate_model(
    model: XGBRegressor,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """
    Evaluate model on test data.

    Model predicts residual. Event-time prediction is:
        t_join_pred = tB_sec + predicted residual
    """
    x_test = test_df[feature_cols].copy()
    predicted_residual = model.predict(x_test)

    predicted_time = test_df["tB_sec"].to_numpy(dtype=float) + predicted_residual
    true_time = test_df["t_event_true_sec"].to_numpy(dtype=float)

    model_metrics = compute_metrics(true_time, predicted_time)
    baseline_metrics = compute_metrics(true_time, test_df["tB_sec"].to_numpy(dtype=float))

    return {
        "baseline_mae_sec": baseline_metrics["mae_sec"],
        "baseline_rmse_sec": baseline_metrics["rmse_sec"],
        "model_mae_sec": model_metrics["mae_sec"],
        "model_rmse_sec": model_metrics["rmse_sec"],
        "mae_improvement_sec": baseline_metrics["mae_sec"] - model_metrics["mae_sec"],
        "rmse_improvement_sec": baseline_metrics["rmse_sec"] - model_metrics["rmse_sec"],
        "n_eval": model_metrics["n_eval"],
    }


# ============================================================
# Robustness check 1: Data size robustness
# ============================================================

def run_data_size_robustness(
    event_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Retrain models using different training-data fractions.

    Evaluation is always performed on the same fixed test runs.
    """
    train_df = event_df[event_df["run_id"].isin(TRAIN_RUN_IDS)].copy()
    test_df = event_df[event_df["run_id"].isin(TEST_RUN_IDS)].copy()

    if train_df.empty:
        raise ValueError("No training rows found for data size robustness.")

    if test_df.empty:
        raise ValueError("No test rows found for data size robustness.")

    repeat_rows = []

    for fraction in DATA_SIZE_FRACTIONS:
        for repeat in range(DATA_SIZE_REPEATS):
            seed = RANDOM_SEED + 1000 + repeat

            sampled_train = stratified_train_sample(
                train_df=train_df,
                fraction=fraction,
                seed=seed,
            )

            model = make_xgb_model(seed=seed)

            model.fit(
                sampled_train[feature_cols],
                sampled_train["resid_true_minus_base_sec"].to_numpy(dtype=float),
            )

            metrics = evaluate_model(model, test_df, feature_cols)

            repeat_rows.append(
                {
                    "training_fraction": fraction,
                    "repeat": repeat + 1,
                    "n_train_rows": int(len(sampled_train)),
                    **metrics,
                }
            )

    repeat_df = pd.DataFrame(repeat_rows)

    summary = (
        repeat_df
        .groupby("training_fraction", as_index=False)
        .agg(
            n_repeats=("repeat", "nunique"),
            n_train_rows_mean=("n_train_rows", "mean"),
            baseline_mae_sec=("baseline_mae_sec", "mean"),
            baseline_rmse_sec=("baseline_rmse_sec", "mean"),
            model_mae_sec_mean=("model_mae_sec", "mean"),
            model_mae_sec_std=("model_mae_sec", "std"),
            model_rmse_sec_mean=("model_rmse_sec", "mean"),
            model_rmse_sec_std=("model_rmse_sec", "std"),
            mae_improvement_sec_mean=("mae_improvement_sec", "mean"),
            rmse_improvement_sec_mean=("rmse_improvement_sec", "mean"),
            n_eval=("n_eval", "mean"),
        )
    )

    summary = summary.round(4)

    return summary


# ============================================================
# Robustness check 2: Noise / perturbation robustness
# ============================================================

def get_or_train_full_model(
    event_df: pd.DataFrame,
    feature_cols: list[str],
) -> XGBRegressor:
    """
    Load the saved global model if available. Otherwise train one from training runs.
    """
    if XGB_MODEL_FILE.exists():
        package = joblib.load(XGB_MODEL_FILE)

        if isinstance(package, dict) and "model" in package:
            return package["model"]

        return package

    train_df = event_df[event_df["run_id"].isin(TRAIN_RUN_IDS)].copy()

    if train_df.empty:
        raise ValueError("No training rows available to train full model.")

    model = make_xgb_model(seed=RANDOM_SEED)
    model.fit(
        train_df[feature_cols],
        train_df["resid_true_minus_base_sec"].to_numpy(dtype=float),
    )

    return model


def perturb_features(
    x: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    noise_level: float,
    seed: int,
) -> pd.DataFrame:
    """
    Add Gaussian noise to selected continuous features.

    Noise standard deviation is:
        noise_level * training feature standard deviation
    """
    if noise_level <= 0:
        return x.copy()

    rng = np.random.default_rng(seed)
    x_noisy = x.copy()

    perturb_cols = [
        col for col in NOISE_FEATURE_CANDIDATES
        if col in feature_cols and col in x_noisy.columns
    ]

    for col in perturb_cols:
        feature_std = float(pd.to_numeric(train_df[col], errors="coerce").std())

        if not np.isfinite(feature_std) or feature_std <= 0:
            continue

        noise = rng.normal(
            loc=0.0,
            scale=noise_level * feature_std,
            size=len(x_noisy),
        )

        x_noisy[col] = pd.to_numeric(x_noisy[col], errors="coerce") + noise

    return x_noisy


def run_noise_robustness(
    event_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Add controlled noise to selected input features and evaluate model stability.
    """
    train_df = event_df[event_df["run_id"].isin(TRAIN_RUN_IDS)].copy()
    test_df = event_df[event_df["run_id"].isin(TEST_RUN_IDS)].copy()

    if train_df.empty:
        raise ValueError("No training rows found for noise robustness.")

    if test_df.empty:
        raise ValueError("No test rows found for noise robustness.")

    model = get_or_train_full_model(event_df, feature_cols)

    true_time = test_df["t_event_true_sec"].to_numpy(dtype=float)
    baseline_time = test_df["tB_sec"].to_numpy(dtype=float)

    baseline_metrics = compute_metrics(true_time, baseline_time)

    repeat_rows = []

    for noise_level in NOISE_LEVELS:
        repeats = 1 if noise_level == 0 else NOISE_REPEATS

        for repeat in range(repeats):
            seed = RANDOM_SEED + 2000 + repeat

            x_test_noisy = perturb_features(
                x=test_df[feature_cols],
                train_df=train_df,
                feature_cols=feature_cols,
                noise_level=noise_level,
                seed=seed,
            )

            predicted_residual = model.predict(x_test_noisy)

            # Use original tB_sec here so this check isolates feature perturbation sensitivity.
            predicted_time = baseline_time + predicted_residual

            model_metrics = compute_metrics(true_time, predicted_time)

            repeat_rows.append(
                {
                    "noise_level": noise_level,
                    "repeat": repeat + 1,
                    "baseline_mae_sec": baseline_metrics["mae_sec"],
                    "baseline_rmse_sec": baseline_metrics["rmse_sec"],
                    "model_mae_sec": model_metrics["mae_sec"],
                    "model_rmse_sec": model_metrics["rmse_sec"],
                    "mae_improvement_sec": baseline_metrics["mae_sec"] - model_metrics["mae_sec"],
                    "rmse_improvement_sec": baseline_metrics["rmse_sec"] - model_metrics["rmse_sec"],
                    "n_eval": model_metrics["n_eval"],
                }
            )

    repeat_df = pd.DataFrame(repeat_rows)

    summary = (
        repeat_df
        .groupby("noise_level", as_index=False)
        .agg(
            n_repeats=("repeat", "nunique"),
            baseline_mae_sec=("baseline_mae_sec", "mean"),
            baseline_rmse_sec=("baseline_rmse_sec", "mean"),
            model_mae_sec_mean=("model_mae_sec", "mean"),
            model_mae_sec_std=("model_mae_sec", "std"),
            model_rmse_sec_mean=("model_rmse_sec", "mean"),
            model_rmse_sec_std=("model_rmse_sec", "std"),
            mae_improvement_sec_mean=("mae_improvement_sec", "mean"),
            rmse_improvement_sec_mean=("rmse_improvement_sec", "mean"),
            n_eval=("n_eval", "mean"),
        )
    )

    summary = summary.round(4)

    return summary


# ============================================================
# Main
# ============================================================

def run_robustness_checks() -> None:
    """Run selected robustness checks and save two output tables."""
    ensure_project_directories()

    event_df, feature_cols = load_event_features()

    print("=" * 80)
    print("Running robustness check 1: Data size robustness")
    print("=" * 80)

    data_size_table = run_data_size_robustness(event_df, feature_cols)
    ROBUSTNESS_DATA_SIZE_CSV.parent.mkdir(parents=True, exist_ok=True)
    data_size_table.to_csv(ROBUSTNESS_DATA_SIZE_CSV, index=False)

    print("\nData size robustness table:")
    print(data_size_table.to_string(index=False))
    print(f"\nSaved: {ROBUSTNESS_DATA_SIZE_CSV}")

    print("\n" + "=" * 80)
    print("Running robustness check 2: Noise / perturbation robustness")
    print("=" * 80)

    noise_table = run_noise_robustness(event_df, feature_cols)
    ROBUSTNESS_NOISE_CSV.parent.mkdir(parents=True, exist_ok=True)
    noise_table.to_csv(ROBUSTNESS_NOISE_CSV, index=False)

    print("\nNoise robustness table:")
    print(noise_table.to_string(index=False))
    print(f"\nSaved: {ROBUSTNESS_NOISE_CSV}")

    print("\nRobustness checks complete.")


def main() -> None:
    """Script entry point."""
    run_robustness_checks()


if __name__ == "__main__":
    main()