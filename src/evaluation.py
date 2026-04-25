"""
Evaluate baseline B curve and final CV-corrected curve against GT.

This script reads saved files only:
    - baseline B curve file
    - GT queue-join files
    - segmented corrected prediction files

Metrics:
    1. MAE  : Mean absolute time error
    2. RMSE : Root mean squared time error
    3. ABC  : Area between cumulative curves
    4. MaxAE: Maximum absolute time error

Outputs:
    output/results/evaluation_by_run_and_rate.csv
    output/results/evaluation_summary_by_rate.csv
    output/results/evaluation_summary_by_split.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    RUN_IDS,
    TRAIN_RUN_IDS,
    TEST_RUN_IDS,
    CV_RATES_PCT,
    VB_CURVES_CSV,
    GT_JOIN_PATTERN,
    SEGMENTED_PRED_PATTERN,
    EVALUATION_BY_RUN_RATE_CSV,
    EVALUATION_SUMMARY_BY_RATE_CSV,
    EVALUATION_SUMMARY_BY_SPLIT_CSV,
    ensure_project_directories,
)


# ============================================================
# Helpers
# ============================================================

def format_path(path_pattern, **kwargs):
    """Format a Path pattern containing named fields."""
    return type(path_pattern)(str(path_pattern).format(**kwargs))


def require_columns(df: pd.DataFrame, required_columns: list[str] | set[str], label: str) -> None:
    """Raise an error if required columns are missing."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def safe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert selected columns to numeric if they exist."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def split_name_for_run(run_id: int) -> str:
    """Return train/test label based on configured run split."""
    if int(run_id) in TRAIN_RUN_IDS:
        return "train"

    if int(run_id) in TEST_RUN_IDS:
        return "test"

    return "unknown"


def compute_time_error_metrics(
    true_times: np.ndarray,
    predicted_times: np.ndarray,
) -> dict[str, float]:
    """
    Compute pointwise time-error metrics after aligning curves by cumulative order N.

    Parameters
    ----------
    true_times:
        GT event times.
    predicted_times:
        Predicted or baseline event times.

    Returns
    -------
    dict
        MAE, RMSE, MaxAE, n_eval.
    """
    true_times = np.asarray(true_times, dtype=float)
    predicted_times = np.asarray(predicted_times, dtype=float)

    mask = np.isfinite(true_times) & np.isfinite(predicted_times)

    if not np.any(mask):
        return {
            "mae_sec": np.nan,
            "rmse_sec": np.nan,
            "maxae_sec": np.nan,
            "n_eval": 0,
        }

    err = predicted_times[mask] - true_times[mask]
    abs_err = np.abs(err)

    return {
        "mae_sec": float(np.mean(abs_err)),
        "rmse_sec": float(np.sqrt(np.mean(err ** 2))),
        "maxae_sec": float(np.max(abs_err)),
        "n_eval": int(mask.sum()),
    }


def area_between_cumulative_curves(
    true_event_times: np.ndarray,
    predicted_event_times: np.ndarray,
) -> float:
    """
    Compute area between two cumulative count curves.

    The curves are treated as step functions:
        N(t) = number of events with event_time <= t

    The area is:
        ∫ |N_true(t) - N_pred(t)| dt

    Units:
        vehicle-seconds
    """
    true_times = np.asarray(true_event_times, dtype=float)
    pred_times = np.asarray(predicted_event_times, dtype=float)

    true_times = true_times[np.isfinite(true_times)]
    pred_times = pred_times[np.isfinite(pred_times)]

    if len(true_times) == 0 or len(pred_times) == 0:
        return np.nan

    true_times.sort()
    pred_times.sort()

    all_breaks = np.unique(np.concatenate([true_times, pred_times]))

    if len(all_breaks) < 2:
        return 0.0

    area = 0.0

    for i in range(len(all_breaks) - 1):
        left = all_breaks[i]
        right = all_breaks[i + 1]
        duration = right - left

        if duration <= 0:
            continue

        n_true = np.searchsorted(true_times, left, side="right")
        n_pred = np.searchsorted(pred_times, left, side="right")

        area += abs(n_true - n_pred) * duration

    return float(area)


def load_gt_curve_for_run(run_id: int) -> pd.DataFrame:
    """
    Load GT curve for one run.

    Returns columns:
        run_id, veh_uid, N_gt, t_queue_join_sec
    """
    gt_path = format_path(GT_JOIN_PATTERN, run_id=run_id)

    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT file: {gt_path}")

    gt = pd.read_csv(gt_path)
    require_columns(gt, ["veh_uid", "t_queue_join_sec"], f"GT file run {run_id:03d}")

    gt["veh_uid"] = gt["veh_uid"].astype(str).str.strip()
    gt["t_queue_join_sec"] = pd.to_numeric(gt["t_queue_join_sec"], errors="coerce")

    gt = gt.dropna(subset=["veh_uid", "t_queue_join_sec"]).copy()
    gt = gt.sort_values(["t_queue_join_sec", "veh_uid"]).reset_index(drop=True)

    if "N_gt" not in gt.columns:
        gt["N_gt"] = np.arange(1, len(gt) + 1, dtype=int)
    else:
        gt["N_gt"] = pd.to_numeric(gt["N_gt"], errors="coerce")
        if gt["N_gt"].isna().any():
            gt["N_gt"] = np.arange(1, len(gt) + 1, dtype=int)
        else:
            gt["N_gt"] = gt["N_gt"].astype(int)

    gt["run_id"] = int(run_id)

    return gt[["run_id", "veh_uid", "N_gt", "t_queue_join_sec"]].copy()


def load_baseline_curve_for_run(vb_all: pd.DataFrame, run_id: int) -> pd.DataFrame:
    """
    Load baseline B curve for one run.

    Returns columns:
        run_id, N_gt, tB_sec
    """
    vb = vb_all[vb_all["run_id"] == int(run_id)].copy()

    if vb.empty:
        return pd.DataFrame(columns=["run_id", "N_gt", "tB_sec"])

    vb = vb.dropna(subset=["N", "tB_sec"]).copy()
    vb["N"] = pd.to_numeric(vb["N"], errors="coerce")
    vb["tB_sec"] = pd.to_numeric(vb["tB_sec"], errors="coerce")
    vb = vb.dropna(subset=["N", "tB_sec"]).copy()

    vb["N_gt"] = vb["N"].astype(int)
    vb["run_id"] = int(run_id)

    return vb[["run_id", "N_gt", "tB_sec"]].copy()


def load_corrected_curve_for_run_rate(rate_pct: int, run_id: int) -> pd.DataFrame:
    """
    Load segmented corrected prediction file and filter to one run.

    Returns columns:
        run_id, cv_rate_pct, veh_uid, N_gt, t_join_pred_segmented_sec
    """
    corr_path = format_path(SEGMENTED_PRED_PATTERN, rate=rate_pct)

    if not corr_path.exists():
        raise FileNotFoundError(f"Missing corrected prediction file: {corr_path}")

    corr = pd.read_csv(corr_path)
    require_columns(
        corr,
        ["run_id", "cv_rate_pct", "veh_uid", "N_gt", "t_join_pred_segmented_sec"],
        f"corrected file rate {rate_pct}",
    )

    corr = safe_numeric(
        corr,
        ["run_id", "cv_rate_pct", "N_gt", "t_join_pred_segmented_sec"],
    )

    corr["veh_uid"] = corr["veh_uid"].astype(str).str.strip()

    corr = corr[
        (corr["run_id"] == int(run_id)) &
        (corr["cv_rate_pct"] == int(rate_pct))
    ].copy()

    corr = corr.dropna(
        subset=[
            "run_id",
            "cv_rate_pct",
            "veh_uid",
            "N_gt",
            "t_join_pred_segmented_sec",
        ]
    ).copy()

    corr["run_id"] = corr["run_id"].astype(int)
    corr["cv_rate_pct"] = corr["cv_rate_pct"].astype(int)
    corr["N_gt"] = corr["N_gt"].astype(int)

    return corr[
        [
            "run_id",
            "cv_rate_pct",
            "veh_uid",
            "N_gt",
            "t_join_pred_segmented_sec",
        ]
    ].copy()


# ============================================================
# Evaluation logic
# ============================================================

def evaluate_one_run_rate(
    vb_all: pd.DataFrame,
    run_id: int,
    rate_pct: int,
) -> dict:
    """
    Evaluate baseline and corrected curves for one run and CV rate.
    """
    gt = load_gt_curve_for_run(run_id)
    base = load_baseline_curve_for_run(vb_all, run_id)
    corr = load_corrected_curve_for_run_rate(rate_pct, run_id)

    if gt.empty:
        raise ValueError(f"No GT rows for run {run_id:03d}")

    if base.empty:
        raise ValueError(f"No baseline B rows for run {run_id:03d}")

    if corr.empty:
        raise ValueError(f"No corrected rows for run {run_id:03d}, rate {rate_pct}%")

    # --------------------
    # Baseline vs GT
    # --------------------
    base_eval = gt.merge(base, on=["run_id", "N_gt"], how="inner")

    baseline_metrics = compute_time_error_metrics(
        true_times=base_eval["t_queue_join_sec"].to_numpy(dtype=float),
        predicted_times=base_eval["tB_sec"].to_numpy(dtype=float),
    )

    baseline_abc = area_between_cumulative_curves(
        true_event_times=base_eval["t_queue_join_sec"].to_numpy(dtype=float),
        predicted_event_times=base_eval["tB_sec"].to_numpy(dtype=float),
    )

    # --------------------
    # Corrected vs GT
    # --------------------
    corr_eval = gt.merge(
        corr[["run_id", "N_gt", "t_join_pred_segmented_sec"]],
        on=["run_id", "N_gt"],
        how="inner",
    )

    corrected_metrics = compute_time_error_metrics(
        true_times=corr_eval["t_queue_join_sec"].to_numpy(dtype=float),
        predicted_times=corr_eval["t_join_pred_segmented_sec"].to_numpy(dtype=float),
    )

    corrected_abc = area_between_cumulative_curves(
        true_event_times=corr_eval["t_queue_join_sec"].to_numpy(dtype=float),
        predicted_event_times=corr_eval["t_join_pred_segmented_sec"].to_numpy(dtype=float),
    )

    row = {
        "run_id": int(run_id),
        "split": split_name_for_run(run_id),
        "cv_rate_pct": int(rate_pct),

        "n_gt": int(len(gt)),
        "n_baseline_eval": int(baseline_metrics["n_eval"]),
        "n_corrected_eval": int(corrected_metrics["n_eval"]),

        "mae_baseline_sec": baseline_metrics["mae_sec"],
        "rmse_baseline_sec": baseline_metrics["rmse_sec"],
        "maxae_baseline_sec": baseline_metrics["maxae_sec"],
        "abc_baseline_vehicle_sec": baseline_abc,

        "mae_corrected_sec": corrected_metrics["mae_sec"],
        "rmse_corrected_sec": corrected_metrics["rmse_sec"],
        "maxae_corrected_sec": corrected_metrics["maxae_sec"],
        "abc_corrected_vehicle_sec": corrected_abc,
    }

    # Improvements: positive value means corrected is better.
    for metric in ["mae", "rmse", "maxae"]:
        base_col = f"{metric}_baseline_sec"
        corr_col = f"{metric}_corrected_sec"
        imp_col = f"{metric}_improvement_sec"
        pct_col = f"{metric}_improvement_pct"

        row[imp_col] = row[base_col] - row[corr_col]

        if pd.notna(row[base_col]) and abs(row[base_col]) > 1e-9:
            row[pct_col] = 100.0 * row[imp_col] / row[base_col]
        else:
            row[pct_col] = np.nan

    row["abc_improvement_vehicle_sec"] = (
        row["abc_baseline_vehicle_sec"] - row["abc_corrected_vehicle_sec"]
    )

    if pd.notna(row["abc_baseline_vehicle_sec"]) and abs(row["abc_baseline_vehicle_sec"]) > 1e-9:
        row["abc_improvement_pct"] = (
            100.0 * row["abc_improvement_vehicle_sec"] / row["abc_baseline_vehicle_sec"]
        )
    else:
        row["abc_improvement_pct"] = np.nan

    return row


def summarize_by_rate(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize evaluation metrics by CV penetration rate.
    """
    metric_cols = [
        "mae_baseline_sec",
        "rmse_baseline_sec",
        "maxae_baseline_sec",
        "abc_baseline_vehicle_sec",
        "mae_corrected_sec",
        "rmse_corrected_sec",
        "maxae_corrected_sec",
        "abc_corrected_vehicle_sec",
        "mae_improvement_sec",
        "mae_improvement_pct",
        "rmse_improvement_sec",
        "rmse_improvement_pct",
        "maxae_improvement_sec",
        "maxae_improvement_pct",
        "abc_improvement_vehicle_sec",
        "abc_improvement_pct",
    ]

    summary = (
        evaluation_df
        .groupby("cv_rate_pct", as_index=False)
        .agg(
            n_runs=("run_id", "nunique"),
            n_gt_total=("n_gt", "sum"),
            **{col: (col, "mean") for col in metric_cols}
        )
    )

    return summary


def summarize_by_split(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize evaluation metrics by train/test split and CV penetration rate.
    """
    metric_cols = [
        "mae_baseline_sec",
        "rmse_baseline_sec",
        "maxae_baseline_sec",
        "abc_baseline_vehicle_sec",
        "mae_corrected_sec",
        "rmse_corrected_sec",
        "maxae_corrected_sec",
        "abc_corrected_vehicle_sec",
        "mae_improvement_sec",
        "mae_improvement_pct",
        "rmse_improvement_sec",
        "rmse_improvement_pct",
        "maxae_improvement_sec",
        "maxae_improvement_pct",
        "abc_improvement_vehicle_sec",
        "abc_improvement_pct",
    ]

    summary = (
        evaluation_df
        .groupby(["split", "cv_rate_pct"], as_index=False)
        .agg(
            n_runs=("run_id", "nunique"),
            n_gt_total=("n_gt", "sum"),
            **{col: (col, "mean") for col in metric_cols}
        )
    )

    return summary


def run_evaluation() -> None:
    """
    Run evaluation for all configured runs and CV penetration rates.
    """
    ensure_project_directories()

    if not VB_CURVES_CSV.exists():
        raise FileNotFoundError(f"Missing baseline B curve file: {VB_CURVES_CSV}")

    vb_all = pd.read_csv(VB_CURVES_CSV)
    require_columns(vb_all, ["run_id", "N", "tB_sec"], "vb_curves_all_runs.csv")

    vb_all = safe_numeric(vb_all, ["run_id", "N", "tB_sec"])
    vb_all = vb_all.dropna(subset=["run_id", "N", "tB_sec"]).copy()
    vb_all["run_id"] = vb_all["run_id"].astype(int)
    vb_all["N"] = vb_all["N"].astype(int)

    rows = []

    for rate_pct in CV_RATES_PCT:
        print("=" * 80)
        print(f"Evaluating CV rate {rate_pct}%")
        print("=" * 80)

        for run_id in RUN_IDS:
            row = evaluate_one_run_rate(vb_all, run_id, rate_pct)
            rows.append(row)

            print(
                f"[Run {run_id:03d}, {rate_pct:03d}%] "
                f"RMSE baseline={row['rmse_baseline_sec']:.3f}, "
                f"RMSE corrected={row['rmse_corrected_sec']:.3f}, "
                f"improvement={row['rmse_improvement_pct']:.2f}%"
            )

    evaluation_df = pd.DataFrame(rows)

    EVALUATION_BY_RUN_RATE_CSV.parent.mkdir(parents=True, exist_ok=True)
    evaluation_df.to_csv(EVALUATION_BY_RUN_RATE_CSV, index=False)

    summary_by_rate = summarize_by_rate(evaluation_df)
    summary_by_rate.to_csv(EVALUATION_SUMMARY_BY_RATE_CSV, index=False)

    summary_by_split = summarize_by_split(evaluation_df)
    summary_by_split.to_csv(EVALUATION_SUMMARY_BY_SPLIT_CSV, index=False)

    print("\nEvaluation complete.")
    print(f"Saved run/rate evaluation : {EVALUATION_BY_RUN_RATE_CSV}")
    print(f"Saved rate summary        : {EVALUATION_SUMMARY_BY_RATE_CSV}")
    print(f"Saved split summary       : {EVALUATION_SUMMARY_BY_SPLIT_CSV}")


def main() -> None:
    """Script entry point."""
    run_evaluation()


if __name__ == "__main__":
    main()