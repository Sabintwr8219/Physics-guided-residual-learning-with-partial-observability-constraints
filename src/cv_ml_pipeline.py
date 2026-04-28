"""
Connected vehicle allocation, feature engineering, and ML residual prediction.

This script:
1. Creates CV allocation files for all runs and all CV penetration rates.
2. Builds time-grid feature files for all CV penetration rates.
3. Builds event-level ML features.
4. Trains one global XGBoost residual model using training runs only.
5. Saves raw event-time predictions for all CV rates.

Outputs:
    data/processed_data/cv_allocation_runXXX_rateYYY.csv
    data/processed_data/cv_allocation_allruns_rateYYY.csv
    data/processed_data/timegrid_vehicle_features_allruns_rateYYY.csv
    data/processed_data/event_features_allrates.csv
    data/processed_data/event_predictions_raw_allrates.csv
    output/models/xgb_event_residual_model.joblib
    output/results/raw_model_metrics.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GroupKFold

from config import (
    RUN_IDS,
    TRAIN_RUN_IDS,
    TEST_RUN_IDS,
    CV_RATES_PCT,
    RANDOM_SEED,
    TIMEGRID_DT_SEC,
    ASOF_TOL_SEC,
    TRAJ_NB_FILTERED_CSV,
    AD_TIMES_CSV,
    VB_CURVES_CSV,
    GT_JOIN_PATTERN,
    CV_ALLOC_RUN_PATTERN,
    CV_ALLOC_ALLRUNS_PATTERN,
    TIMEGRID_FEATURES_PATTERN,
    EVENT_FEATURES_ALLRATES_CSV,
    EVENT_PREDICTIONS_RAW_ALLRATES_CSV,
    RAW_MODEL_METRICS_CSV,
    XGB_MODEL_FILE,
    XGB_N_ESTIMATORS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_SUBSAMPLE,
    XGB_COLSAMPLE_BYTREE,
    XGB_RANDOM_STATE,
    XGB_OBJECTIVE,
    M_TO_FT,
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_project_directories,
)


# ============================================================
# Local settings
# ============================================================

# Using one random score per vehicle makes CV samples nested across rates.
# Example: vehicles selected at 1% are also included at 2%, 5%, etc.
NESTED_CV_SAMPLING = True

TRAJ_CHUNKSIZE = 300_000

# ============================================================
# ML diagnostics and hyperparameter tuning settings
# ============================================================

RUN_FEATURE_CORRELATION_DIAGNOSTIC = False
RUN_FEATURE_IMPORTANCE_DIAGNOSTIC = True
RUN_HYPERPARAMETER_TUNING = True

XGB_FEATURE_IMPORTANCE_CSV = RESULTS_DIR / "xgb_feature_importance.csv"
XGB_FEATURE_IMPORTANCE_FIG = FIGURES_DIR / "xgb_feature_importance.png"
TOP_N_FEATURES_TO_PLOT = 10

XGB_TUNING_RESULTS_CSV = RESULTS_DIR / "xgb_hyperparameter_tuning_results.csv"

TUNING_N_ITER = 12
TUNING_CV_SPLITS = 3
TUNING_SCORING = "neg_root_mean_squared_error"
# ============================================================
# General helpers
# ============================================================

def format_path(path_pattern, **kwargs):
    """Format a Path pattern containing named fields."""
    return type(path_pattern)(str(path_pattern).format(**kwargs))


def require_columns(df: pd.DataFrame, needed: list[str] | set[str], label: str) -> None:
    """Raise an error if required columns are missing."""
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def safe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert selected columns to numeric if they exist."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def robust_bool_to_flag(series: pd.Series) -> pd.Series:
    """Convert boolean-like values to 0/1 integer flags."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

    text = series.astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y"]).astype(int)


def pick_gt_join_col(gt_df: pd.DataFrame) -> str:
    """Pick the available GT queue-join time column."""
    for col in ["t_queue_join_sec", "t_join_true", "t_join_sec"]:
        if col in gt_df.columns:
            return col

    raise ValueError(
        "GT file missing join-time column. Expected one of: "
        "t_queue_join_sec, t_join_true, t_join_sec"
    )


def pick_departure_col(ad_df: pd.DataFrame) -> str:
    """Pick the available stopbar departure-time column."""
    for col in ["t_dep_stopbar_sec", "t_dep_sec", "t_dep_stopbar", "tD_sec"]:
        if col in ad_df.columns:
            return col

    raise ValueError(
        "A/D file missing departure-time column. Expected one of: "
        "t_dep_stopbar_sec, t_dep_sec, t_dep_stopbar, tD_sec"
    )


def compute_metrics(y_true, y_pred) -> tuple[float, float]:
    """Return MAE and RMSE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if not np.any(mask):
        return np.nan, np.nan

    mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
    rmse = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))

    return mae, rmse

def save_feature_correlation_diagnostic(event_df: pd.DataFrame, feature_cols: list[str]) -> None:
    """
    Compute, save, and plot the feature correlation matrix.

    This diagnostic uses the same feature columns that are fed into the
    XGBoost residual model.
    """
    print("=" * 80)
    print("Saving feature correlation diagnostics")
    print("=" * 80)

    feature_df = event_df[feature_cols].copy()

    for col in feature_cols:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    feature_df = feature_df.dropna(axis=0, how="any")

    if feature_df.empty:
        raise ValueError("Feature dataframe is empty after numeric conversion.")

    corr = feature_df.corr(method="pearson")

    FEATURE_CORRELATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(FEATURE_CORRELATION_CSV)

    fig_size = max(8, 0.45 * len(corr.columns))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    image = ax.imshow(corr.values, vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))

    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)

    ax.set_title("Feature Correlation Matrix", fontsize=12, fontweight="bold")

    plt.tight_layout()

    FEATURE_CORRELATION_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FEATURE_CORRELATION_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] Correlation matrix CSV : {FEATURE_CORRELATION_CSV}")
    print(f"[Saved] Correlation matrix plot: {FEATURE_CORRELATION_FIG}")

    high_corr_pairs = []

    cols = list(corr.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]

            if pd.notna(value) and abs(value) >= 0.80:
                high_corr_pairs.append(
                    {
                        "feature_1": cols[i],
                        "feature_2": cols[j],
                        "correlation": float(value),
                    }
                )

    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(
            "correlation",
            key=lambda x: x.abs(),
            ascending=False,
        )

        print("\nHighly correlated feature pairs with |r| >= 0.80:")
        print(high_corr_df.to_string(index=False))
    else:
        print("\nNo feature pairs found with |r| >= 0.80.")

def save_xgb_feature_importance(
    model: XGBRegressor,
    feature_cols: list[str],
) -> None:
    """
    Save and plot XGBoost feature importance.

    This diagnostic shows which engineered features contributed most to the
    residual prediction model.
    """
    print("=" * 80)
    print("Saving XGBoost feature importance diagnostic")
    print("=" * 80)

    if not hasattr(model, "feature_importances_"):
        raise ValueError("The trained model does not expose feature_importances_.")

    importance = np.asarray(model.feature_importances_, dtype=float)

    if len(importance) != len(feature_cols):
        raise ValueError(
            "Feature importance length does not match number of feature columns."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": importance,
        }
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    XGB_FEATURE_IMPORTANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(XGB_FEATURE_IMPORTANCE_CSV, index=False)

    top_df = importance_df.head(TOP_N_FEATURES_TO_PLOT).copy()
    top_df = top_df.sort_values("importance", ascending=True)

    fig_height = max(4, 0.45 * len(top_df))
    fig, ax = plt.subplots(figsize=(9, fig_height))

    ax.barh(
        top_df["feature"],
        top_df["importance"],
    )

    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top XGBoost Feature Importances")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.7)

    plt.tight_layout()

    XGB_FEATURE_IMPORTANCE_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(XGB_FEATURE_IMPORTANCE_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] Feature importance CSV : {XGB_FEATURE_IMPORTANCE_CSV}")
    print(f"[Saved] Feature importance plot: {XGB_FEATURE_IMPORTANCE_FIG}")

    print("\nTop feature importances:")
    print(importance_df.head(TOP_N_FEATURES_TO_PLOT).to_string(index=False))
    
# ============================================================
# Loading shared inputs
# ============================================================

def load_ad_times() -> pd.DataFrame:
    """Load A/D event-time file."""
    ad = pd.read_csv(AD_TIMES_CSV)
    require_columns(ad, ["run_id", "veh_uid"], "ad_times_all_runs.csv")

    dep_col = pick_departure_col(ad)

    ad["run_id"] = pd.to_numeric(ad["run_id"], errors="coerce")
    ad["veh_uid"] = ad["veh_uid"].astype(str).str.strip()
    ad[dep_col] = pd.to_numeric(ad[dep_col], errors="coerce")

    ad = ad.dropna(subset=["run_id", "veh_uid", dep_col]).copy()
    ad["run_id"] = ad["run_id"].astype(int)

    ad = ad.rename(columns={dep_col: "t_exit_sec"})

    keep_cols = ["run_id", "veh_uid", "t_exit_sec"]
    if "vehID" in ad.columns:
        keep_cols.insert(1, "vehID")

    return ad[keep_cols].copy()


def load_vb_curves() -> pd.DataFrame:
    """Load baseline V/B curve file."""
    vb = pd.read_csv(VB_CURVES_CSV)
    require_columns(vb, ["run_id", "N", "tB_sec"], "vb_curves_all_runs.csv")

    vb = safe_numeric(vb, ["run_id", "N", "tB_sec"])
    vb = vb.dropna(subset=["run_id", "N", "tB_sec"]).copy()

    vb["run_id"] = vb["run_id"].astype(int)
    vb["N"] = vb["N"].astype(int)

    return vb


def load_gt_for_run(run_id: int) -> pd.DataFrame:
    """Load one GT queue-join file."""
    gt_path = format_path(GT_JOIN_PATTERN, run_id=run_id)

    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT file: {gt_path}")

    gt = pd.read_csv(gt_path)
    require_columns(gt, ["veh_uid", "joined_queue"], f"GT file for run {run_id:03d}")

    join_col = pick_gt_join_col(gt)

    gt["veh_uid"] = gt["veh_uid"].astype(str).str.strip()
    gt["joined_queue_flag"] = robust_bool_to_flag(gt["joined_queue"])
    gt[join_col] = pd.to_numeric(gt[join_col], errors="coerce")

    gt = gt.rename(columns={join_col: "t_gt_event_sec"})

    keep_cols = [
        "veh_uid",
        "joined_queue_flag",
        "t_gt_event_sec",
    ]

    if "N_gt" in gt.columns:
        keep_cols.append("N_gt")

    return gt[keep_cols].copy()


# ============================================================
# CV allocation
# ============================================================

def compute_stopbar_cross_time(group: pd.DataFrame) -> float:
    """Fallback stopbar crossing time from trajectory data."""
    g = group.sort_values("Total_Sim_Time_Sec")

    t = g["Total_Sim_Time_Sec"].to_numpy(dtype=float)
    y = g["s_rel_stop_ft"].to_numpy(dtype=float)

    idx = np.where(y >= 0)[0]

    if len(idx) > 0:
        return float(t[idx[0]])

    return float(t[-1])


def build_fallback_leaving_orders_from_trajectory(run_ids: list[int]) -> dict[int, pd.DataFrame]:
    """
    Build leaving order from filtered trajectories.

    This is only used if A/D times are missing for a run.
    """
    run_set = set(int(r) for r in run_ids)
    per_run_vehicle = {int(r): {} for r in run_ids}

    usecols = ["run_id", "veh_uid", "Total_Sim_Time_Sec", "s_rel_stop_m"]

    for chunk in pd.read_csv(TRAJ_NB_FILTERED_CSV, usecols=usecols, chunksize=TRAJ_CHUNKSIZE):
        chunk["run_id"] = pd.to_numeric(chunk["run_id"], errors="coerce")
        chunk["veh_uid"] = chunk["veh_uid"].astype(str).str.strip()
        chunk["Total_Sim_Time_Sec"] = pd.to_numeric(chunk["Total_Sim_Time_Sec"], errors="coerce")
        chunk["s_rel_stop_m"] = pd.to_numeric(chunk["s_rel_stop_m"], errors="coerce")

        chunk = chunk.dropna(
            subset=["run_id", "veh_uid", "Total_Sim_Time_Sec", "s_rel_stop_m"]
        ).copy()

        if chunk.empty:
            continue

        chunk["run_id"] = chunk["run_id"].astype(int)
        chunk = chunk[chunk["run_id"].isin(run_set)].copy()

        if chunk.empty:
            continue

        chunk["s_rel_stop_ft"] = chunk["s_rel_stop_m"] * M_TO_FT

        for (run_id, veh_uid), g in chunk.groupby(["run_id", "veh_uid"], sort=False):
            recs = per_run_vehicle[int(run_id)]
            g = g.sort_values("Total_Sim_Time_Sec")

            t_vals = g["Total_Sim_Time_Sec"].to_numpy(dtype=float)
            y_vals = g["s_rel_stop_ft"].to_numpy(dtype=float)

            last_time = float(np.max(t_vals))
            cross_idx = np.where(y_vals >= 0)[0]
            first_cross = float(t_vals[cross_idx[0]]) if len(cross_idx) else np.nan

            if veh_uid not in recs:
                recs[veh_uid] = {
                    "last_time": last_time,
                    "cross_time": first_cross,
                }
            else:
                rec = recs[veh_uid]
                rec["last_time"] = max(rec["last_time"], last_time)

                if np.isfinite(first_cross):
                    if not np.isfinite(rec["cross_time"]):
                        rec["cross_time"] = first_cross
                    else:
                        rec["cross_time"] = min(rec["cross_time"], first_cross)

    out = {}

    for run_id in run_ids:
        rows = []

        for veh_uid, rec in per_run_vehicle[int(run_id)].items():
            t_exit = rec["cross_time"] if np.isfinite(rec["cross_time"]) else rec["last_time"]
            rows.append(
                {
                    "veh_uid": str(veh_uid),
                    "t_exit_sec": float(t_exit),
                }
            )

        if rows:
            df = pd.DataFrame(rows).sort_values("t_exit_sec").reset_index(drop=True)
            df["N"] = np.arange(1, len(df) + 1, dtype=int)
            out[int(run_id)] = df[["veh_uid", "t_exit_sec", "N"]].copy()
        else:
            out[int(run_id)] = pd.DataFrame(columns=["veh_uid", "t_exit_sec", "N"])

    return out


def build_leaving_order_for_run(
    run_id: int,
    ad: pd.DataFrame,
    fallback_orders: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build vehicle leaving order N for one run.

    Prefer A/D stopbar departure times. Fall back to trajectory-derived crossing.
    """
    ad_run = ad[ad["run_id"] == int(run_id)].copy()

    if not ad_run.empty:
        leave = ad_run.dropna(subset=["veh_uid", "t_exit_sec"]).copy()
        leave = leave.sort_values("t_exit_sec").reset_index(drop=True)
        leave["N"] = np.arange(1, len(leave) + 1, dtype=int)

        keep_cols = ["veh_uid", "t_exit_sec", "N"]
        if "vehID" in leave.columns:
            keep_cols.insert(0, "vehID")

        return leave[keep_cols].copy()

    return fallback_orders.get(
        int(run_id),
        pd.DataFrame(columns=["veh_uid", "t_exit_sec", "N"]),
    ).copy()


def select_cv_flags(leave: pd.DataFrame, run_id: int, rate_pct: int) -> pd.Series:
    """
    Select CV vehicles reproducibly.

    If NESTED_CV_SAMPLING is True, a stable random score is assigned per vehicle
    and the lowest scores are selected. This keeps lower-penetration CVs inside
    higher-penetration sets.
    """
    veh_ids = leave["veh_uid"].astype(str).to_numpy()
    n_total = len(veh_ids)

    if n_total == 0:
        return pd.Series([], dtype=bool)

    n_cv = int(np.floor((float(rate_pct) / 100.0) * n_total))

    if rate_pct > 0:
        n_cv = max(1, n_cv)

    n_cv = min(n_cv, n_total)

    if n_cv <= 0:
        return pd.Series(False, index=leave.index)

    rng = np.random.default_rng(RANDOM_SEED + int(run_id))

    if NESTED_CV_SAMPLING:
        random_scores = rng.random(n_total)
        selected_idx = np.argsort(random_scores)[:n_cv]
    else:
        selected_idx = rng.choice(np.arange(n_total), size=n_cv, replace=False)

    flags = np.zeros(n_total, dtype=bool)
    flags[selected_idx] = True

    return pd.Series(flags, index=leave.index)


def create_cv_allocation_files() -> None:
    """
    Create CV allocation files for all runs and CV penetration rates.
    """
    print("=" * 80)
    print("Creating CV allocation files")
    print("=" * 80)

    ad = load_ad_times()
    vb = load_vb_curves()

    runs_with_ad = set(ad["run_id"].unique().tolist())
    runs_needing_fallback = [r for r in RUN_IDS if r not in runs_with_ad]

    fallback_orders = {}
    if runs_needing_fallback:
        fallback_orders = build_fallback_leaving_orders_from_trajectory(runs_needing_fallback)

    all_rate_outputs = {}

    for rate_pct in CV_RATES_PCT:
        rate_parts = []

        for run_id in RUN_IDS:
            leave = build_leaving_order_for_run(run_id, ad, fallback_orders)

            if leave.empty:
                print(f"[Run {run_id:03d}, {rate_pct}%] No leaving order. Skipping.")
                continue

            leave["veh_uid"] = leave["veh_uid"].astype(str).str.strip()
            leave["N"] = pd.to_numeric(leave["N"], errors="coerce").astype(int)
            leave["t_exit_sec"] = pd.to_numeric(leave["t_exit_sec"], errors="coerce")

            vb_run = vb[vb["run_id"] == int(run_id)][["N", "tB_sec"]].copy()
            leave = leave.merge(vb_run, on="N", how="left")

            gt = load_gt_for_run(run_id)
            gt = gt.merge(leave[["veh_uid", "t_exit_sec"]], on="veh_uid", how="left")

            # Final true event time: GT event for queued vehicles; exit time for non-queued vehicles.
            # In the final GT files, t_gt_event_sec already uses exit time for non-queued vehicles,
            # but this expression keeps the rule explicit.
            gt["t_event_true_sec"] = np.where(
                (gt["joined_queue_flag"] == 1) & gt["t_gt_event_sec"].notna(),
                gt["t_gt_event_sec"],
                gt["t_exit_sec"],
            )

            leave["is_cv"] = select_cv_flags(leave, run_id=run_id, rate_pct=rate_pct)

            cv_truth = gt[["veh_uid", "t_event_true_sec"]].copy()
            cv_truth = cv_truth.rename(columns={"t_event_true_sec": "CV_join_sec"})

            leave = leave.merge(cv_truth, on="veh_uid", how="left")
            leave.loc[leave["is_cv"] == False, "CV_join_sec"] = np.nan

            leave["run_id"] = int(run_id)
            leave["cv_rate_pct"] = int(rate_pct)

            save_cols = [
                "run_id",
                "cv_rate_pct",
                "veh_uid",
                "N",
                "t_exit_sec",
                "tB_sec",
                "is_cv",
                "CV_join_sec",
            ]

            if "vehID" in leave.columns:
                save_cols.insert(2, "vehID")

            leave = leave[save_cols].copy()

            out_run = format_path(CV_ALLOC_RUN_PATTERN, run_id=run_id, rate=rate_pct)
            out_run.parent.mkdir(parents=True, exist_ok=True)
            leave.to_csv(out_run, index=False)

            rate_parts.append(leave)

            print(
                f"[Saved] run {run_id:03d}, rate {rate_pct:03d}% | "
                f"vehicles={len(leave):,}, CVs={int(leave['is_cv'].sum()):,}"
            )

        if not rate_parts:
            raise ValueError(f"No CV allocation rows created for rate {rate_pct}%")

        alloc_all = pd.concat(rate_parts, ignore_index=True)

        out_all = format_path(CV_ALLOC_ALLRUNS_PATTERN, rate=rate_pct)
        out_all.parent.mkdir(parents=True, exist_ok=True)
        alloc_all.to_csv(out_all, index=False)

        all_rate_outputs[int(rate_pct)] = alloc_all
        print(f"[Saved combined] {out_all} | rows={len(alloc_all):,}")

    print("\nCV allocation complete.")


# ============================================================
# Time-grid features
# ============================================================

def step_count_on_grid(event_times: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Compute cumulative step count N(t) on a time grid."""
    event_times = np.asarray(event_times, dtype=float)
    event_times = event_times[np.isfinite(event_times)]
    event_times.sort()

    return np.searchsorted(event_times, t_grid, side="right").astype(int)


def event_flag_on_grid(event_times: np.ndarray, t_grid: np.ndarray, dt: float) -> np.ndarray:
    """Flag grid times where one or more events occur within the time bin."""
    event_times = np.asarray(event_times, dtype=float)
    event_times = event_times[np.isfinite(event_times)]

    if len(event_times) == 0:
        return np.zeros(len(t_grid), dtype=int)

    event_times.sort()
    half = dt / 2.0
    out = np.zeros(len(t_grid), dtype=int)

    left_idx = 0
    right_idx = 0

    for i, t in enumerate(t_grid):
        left = t - half
        right = t + half

        while left_idx < len(event_times) and event_times[left_idx] < left:
            left_idx += 1

        if right_idx < left_idx:
            right_idx = left_idx

        while right_idx < len(event_times) and event_times[right_idx] < right:
            right_idx += 1

        out[i] = int(right_idx > left_idx)

    return out


def latest_time_so_far(t_grid: np.ndarray, event_times: np.ndarray) -> np.ndarray:
    """For each grid time, return latest event time <= current time."""
    event_times = np.asarray(event_times, dtype=float)
    event_times = event_times[np.isfinite(event_times)]

    if len(event_times) == 0:
        return np.full(len(t_grid), np.nan)

    event_times.sort()

    idx = np.searchsorted(event_times, t_grid, side="right") - 1
    out = np.full(len(t_grid), np.nan)

    valid = idx >= 0
    out[valid] = event_times[idx[valid]]

    return out


def latest_value_so_far(
    t_grid: np.ndarray,
    event_times: np.ndarray,
    event_values: np.ndarray,
) -> np.ndarray:
    """For each grid time, return latest value whose event time <= current time."""
    event_times = np.asarray(event_times, dtype=float)
    event_values = np.asarray(event_values, dtype=float)

    mask = np.isfinite(event_times) & np.isfinite(event_values)
    event_times = event_times[mask]
    event_values = event_values[mask]

    if len(event_times) == 0:
        return np.full(len(t_grid), np.nan)

    order = np.argsort(event_times)
    event_times = event_times[order]
    event_values = event_values[order]

    idx = np.searchsorted(event_times, t_grid, side="right") - 1
    out = np.full(len(t_grid), np.nan)

    valid = idx >= 0
    out[valid] = event_values[idx[valid]]

    return out


def build_timegrid_features_for_all_rates() -> None:
    """
    Build time-grid feature files for each CV penetration rate.
    """
    print("=" * 80)
    print("Building time-grid feature files")
    print("=" * 80)

    ad = load_ad_times()
    vb = load_vb_curves()

    for rate_pct in CV_RATES_PCT:
        rate_rows = []

        for run_id in RUN_IDS:
            gt = load_gt_for_run(run_id)
            ad_run = ad[ad["run_id"] == int(run_id)][["veh_uid", "t_exit_sec"]].copy()

            gt = gt.merge(ad_run, on="veh_uid", how="left")
            if gt["t_exit_sec"].isna().any():
                missing = int(gt["t_exit_sec"].isna().sum())
                raise ValueError(f"Run {run_id:03d}: {missing} GT vehicles missing exit time.")

            gt["t_event_true_sec"] = np.where(
                (gt["joined_queue_flag"] == 1) & gt["t_gt_event_sec"].notna(),
                gt["t_gt_event_sec"],
                gt["t_exit_sec"],
            )

            gt = gt.dropna(subset=["t_event_true_sec"]).copy()
            t_true_events = gt["t_event_true_sec"].to_numpy(dtype=float)

            vb_run = vb[vb["run_id"] == int(run_id)].copy().sort_values("N")
            t_base_events = vb_run["tB_sec"].to_numpy(dtype=float)

            cv_path = format_path(CV_ALLOC_RUN_PATTERN, run_id=run_id, rate=rate_pct)
            if not cv_path.exists():
                raise FileNotFoundError(f"Missing CV allocation file: {cv_path}")

            cv = pd.read_csv(cv_path)
            require_columns(cv, ["veh_uid", "N", "is_cv", "CV_join_sec"], f"CV allocation run {run_id}")

            cv["veh_uid"] = cv["veh_uid"].astype(str).str.strip()
            cv["N"] = pd.to_numeric(cv["N"], errors="coerce")
            cv["is_cv_flag"] = robust_bool_to_flag(cv["is_cv"])
            cv["CV_join_sec"] = pd.to_numeric(cv["CV_join_sec"], errors="coerce")

            cv_pts = cv[(cv["is_cv_flag"] == 1) & cv["CV_join_sec"].notna()].copy()
            cv_pts = cv_pts.sort_values("CV_join_sec").reset_index(drop=True)

            cv_event_times = cv_pts["CV_join_sec"].to_numpy(dtype=float)
            cv_event_n = cv_pts["N"].to_numpy(dtype=float)

            t_min = min(np.nanmin(t_base_events), np.nanmin(t_true_events))
            t_max = max(np.nanmax(t_base_events), np.nanmax(t_true_events))

            t0 = np.floor(t_min / TIMEGRID_DT_SEC) * TIMEGRID_DT_SEC
            t1 = np.ceil(t_max / TIMEGRID_DT_SEC) * TIMEGRID_DT_SEC

            t_grid = np.round(
                np.arange(t0, t1 + 0.5 * TIMEGRID_DT_SEC, TIMEGRID_DT_SEC),
                6,
            )

            n_base_at_t = step_count_on_grid(t_base_events, t_grid)
            n_true_at_t = step_count_on_grid(t_true_events, t_grid)

            join_event_true = event_flag_on_grid(t_true_events, t_grid, TIMEGRID_DT_SEC)
            join_event_base = event_flag_on_grid(t_base_events, t_grid, TIMEGRID_DT_SEC)

            prev_cv_join_sec = latest_time_so_far(t_grid, cv_event_times)
            latest_cv_n = latest_value_so_far(t_grid, cv_event_times, cv_event_n)

            cv_seen_so_far = np.isfinite(prev_cv_join_sec).astype(int)
            time_since_prev_cv_join_sec = np.where(
                np.isfinite(prev_cv_join_sec),
                t_grid - prev_cv_join_sec,
                np.nan,
            )

            true_event_time_at_step = np.full(len(t_grid), np.nan)
            half = TIMEGRID_DT_SEC / 2.0

            for event_time in t_true_events:
                idx = int(np.argmin(np.abs(t_grid - event_time)))
                if abs(t_grid[idx] - event_time) <= half:
                    true_event_time_at_step[idx] = event_time

            run_grid = pd.DataFrame(
                {
                    "run_id": int(run_id),
                    "cv_rate_pct": int(rate_pct),
                    "time_sec": t_grid,
                    "N_base_at_t": n_base_at_t,
                    "N_true_at_t": n_true_at_t,
                    "resid_true_minus_base": n_true_at_t - n_base_at_t,
                    "join_event_true": join_event_true,
                    "join_event_base": join_event_base,
                    "true_event_time_at_step": true_event_time_at_step,
                    "prev_cv_join_sec": prev_cv_join_sec,
                    "time_since_prev_cv_join_sec": time_since_prev_cv_join_sec,
                    "latest_cv_N": latest_cv_n,
                    "cv_seen_so_far": cv_seen_so_far,
                }
            )

            rate_rows.append(run_grid)

        if not rate_rows:
            raise ValueError(f"No time-grid feature rows created for rate {rate_pct}%")

        features = pd.concat(rate_rows, ignore_index=True)

        out_path = format_path(TIMEGRID_FEATURES_PATTERN, rate=rate_pct)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(out_path, index=False)

        print(f"[Saved] {out_path} | rows={len(features):,}")

    print("\nTime-grid feature generation complete.")


# ============================================================
# Event-level ML feature engineering
# ============================================================

def load_timegrid_for_rate(rate_pct: int) -> pd.DataFrame:
    """Load one time-grid feature file."""
    path = format_path(TIMEGRID_FEATURES_PATTERN, rate=rate_pct)

    if not path.exists():
        raise FileNotFoundError(f"Missing time-grid feature file: {path}")

    tg = pd.read_csv(path)
    require_columns(tg, ["run_id", "time_sec"], f"timegrid features rate {rate_pct}")

    numeric_cols = [
        "run_id",
        "cv_rate_pct",
        "time_sec",
        "prev_cv_join_sec",
        "time_since_prev_cv_join_sec",
        "latest_cv_N",
        "cv_seen_so_far",
        "N_base_at_t",
        "join_event_base",
    ]

    tg = safe_numeric(tg, numeric_cols)
    tg = tg.dropna(subset=["run_id", "time_sec"]).copy()
    tg["run_id"] = tg["run_id"].astype(int)

    keep_cols = [
        "run_id",
        "time_sec",
        "prev_cv_join_sec",
        "time_since_prev_cv_join_sec",
        "latest_cv_N",
        "cv_seen_so_far",
        "N_base_at_t",
        "join_event_base",
    ]

    keep_cols = [col for col in keep_cols if col in tg.columns]

    return tg[keep_cols].copy()


def nearest_timegrid_merge(event_df: pd.DataFrame, tg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge time-grid features to event-level rows using nearest baseline event time.
    """
    merged_parts = []

    for run_id, group in event_df.groupby("run_id", sort=False):
        group = group.sort_values("tB_sec").copy()

        tg_run = tg_df[tg_df["run_id"] == int(run_id)].copy()

        if tg_run.empty:
            raise ValueError(f"No time-grid rows for run {run_id}")

        tg_run = tg_run.sort_values("time_sec").copy()

        merged = pd.merge_asof(
            group,
            tg_run,
            left_on="tB_sec",
            right_on="time_sec",
            by="run_id",
            direction="nearest",
            tolerance=ASOF_TOL_SEC,
            suffixes=("", "_tg"),
        )

        merged_parts.append(merged)

    return pd.concat(merged_parts, ignore_index=True)


def add_prev_next_cv_event_features(event_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add previous/next CV anchor context in cumulative-count order N.
    """
    out_parts = []

    for (run_id, rate_pct), group in event_df.groupby(["run_id", "cv_rate_pct"], sort=False):
        g = group.sort_values("N").copy()

        cv_rows = g[g["is_cv_flag"] == 1].copy()
        cv_rows = cv_rows.dropna(subset=["cv_event_time_sec"]).sort_values("N")

        cv_n = cv_rows["N"].to_numpy(dtype=float)
        cv_t = cv_rows["cv_event_time_sec"].to_numpy(dtype=float)
        all_n = g["N"].to_numpy(dtype=float)

        prev_idx = np.searchsorted(cv_n, all_n, side="right") - 1
        next_idx = np.searchsorted(cv_n, all_n, side="left")

        prev_cv_n = np.full(len(g), np.nan)
        next_cv_n = np.full(len(g), np.nan)
        prev_cv_t = np.full(len(g), np.nan)
        next_cv_t = np.full(len(g), np.nan)

        has_prev = prev_idx >= 0
        has_next = next_idx < len(cv_n)

        if len(cv_n) > 0:
            prev_cv_n[has_prev] = cv_n[prev_idx[has_prev]]
            prev_cv_t[has_prev] = cv_t[prev_idx[has_prev]]

            next_cv_n[has_next] = cv_n[next_idx[has_next]]
            next_cv_t[has_next] = cv_t[next_idx[has_next]]

        g["prev_cv_event_N"] = prev_cv_n
        g["next_cv_event_N"] = next_cv_n
        g["prev_cv_event_time_sec"] = prev_cv_t
        g["next_cv_event_time_sec"] = next_cv_t

        own_cv_mask = g["is_cv_flag"] == 1

        g.loc[own_cv_mask, "prev_cv_event_N"] = g.loc[own_cv_mask, "N"]
        g.loc[own_cv_mask, "next_cv_event_N"] = g.loc[own_cv_mask, "N"]
        g.loc[own_cv_mask, "prev_cv_event_time_sec"] = g.loc[own_cv_mask, "cv_event_time_sec"]
        g.loc[own_cv_mask, "next_cv_event_time_sec"] = g.loc[own_cv_mask, "cv_event_time_sec"]

        g["count_gap_from_prev_cv"] = g["N"] - g["prev_cv_event_N"]
        g["count_gap_to_next_cv"] = g["next_cv_event_N"] - g["N"]
        g["time_from_prev_cv_sec"] = g["tB_sec"] - g["prev_cv_event_time_sec"]
        g["time_to_next_cv_sec"] = g["next_cv_event_time_sec"] - g["tB_sec"]

        out_parts.append(g)

    return pd.concat(out_parts, ignore_index=True)


def build_event_features_allrates() -> tuple[pd.DataFrame, list[str]]:
    """
    Build one combined event-level feature file across all CV rates.
    """
    print("=" * 80)
    print("Building event-level ML features")
    print("=" * 80)

    ad = load_ad_times()
    vb = load_vb_curves()

    event_parts = []

    for rate_pct in CV_RATES_PCT:
        tg = load_timegrid_for_rate(rate_pct)
        rate_parts = []

        for run_id in RUN_IDS:
            gt = load_gt_for_run(run_id)
            ad_run = ad[ad["run_id"] == int(run_id)][["veh_uid", "t_exit_sec"]].copy()
            gt = gt.merge(ad_run, on="veh_uid", how="left")

            if gt["t_exit_sec"].isna().any():
                missing = int(gt["t_exit_sec"].isna().sum())
                raise ValueError(f"Run {run_id:03d}: {missing} GT vehicles missing exit time.")

            gt["t_event_true_sec"] = np.where(
                (gt["joined_queue_flag"] == 1) & gt["t_gt_event_sec"].notna(),
                gt["t_gt_event_sec"],
                gt["t_exit_sec"],
            )

            cv_path = format_path(CV_ALLOC_RUN_PATTERN, run_id=run_id, rate=rate_pct)
            if not cv_path.exists():
                raise FileNotFoundError(f"Missing CV allocation file: {cv_path}")

            cv = pd.read_csv(cv_path)
            require_columns(
                cv,
                ["run_id", "veh_uid", "N", "t_exit_sec", "tB_sec", "is_cv"],
                f"CV allocation run {run_id:03d}, rate {rate_pct}",
            )

            cv["run_id"] = pd.to_numeric(cv["run_id"], errors="coerce")
            cv["veh_uid"] = cv["veh_uid"].astype(str).str.strip()
            cv["N"] = pd.to_numeric(cv["N"], errors="coerce")
            cv["tB_sec"] = pd.to_numeric(cv["tB_sec"], errors="coerce")
            cv["is_cv_flag"] = robust_bool_to_flag(cv["is_cv"])

            cv = cv.dropna(subset=["run_id", "veh_uid", "N", "tB_sec"]).copy()
            cv["run_id"] = cv["run_id"].astype(int)
            cv["N"] = cv["N"].astype(int)

            df = cv.merge(
                gt[["veh_uid", "joined_queue_flag", "t_gt_event_sec", "t_exit_sec", "t_event_true_sec"]],
                on="veh_uid",
                how="left",
                suffixes=("", "_gt"),
            )

            if df["t_event_true_sec"].isna().any():
                missing = int(df["t_event_true_sec"].isna().sum())
                raise ValueError(f"Run {run_id:03d}, rate {rate_pct}: {missing} rows missing GT event time.")

            df["cv_event_time_sec"] = np.where(
                df["is_cv_flag"] == 1,
                df["t_event_true_sec"],
                np.nan,
            )

            df["run_id"] = int(run_id)
            df["cv_rate_pct"] = int(rate_pct)
            df["split"] = np.where(df["run_id"].isin(TRAIN_RUN_IDS), "train", "test")
            df["resid_true_minus_base_sec"] = df["t_event_true_sec"] - df["tB_sec"]

            df = df.sort_values("N").reset_index(drop=True)

            max_n = float(df["N"].max())
            df["N_norm"] = df["N"] / max_n if max_n > 0 else 0.0
            df["dt_from_prev_B_sec"] = df["tB_sec"].diff()
            df["dt_to_next_B_sec"] = df["tB_sec"].shift(-1) - df["tB_sec"]

            rate_parts.append(df)

        rate_event = pd.concat(rate_parts, ignore_index=True)
        rate_event = add_prev_next_cv_event_features(rate_event)
        rate_event = nearest_timegrid_merge(rate_event, tg)

        event_parts.append(rate_event)

    event_all = pd.concat(event_parts, ignore_index=True)

    fill_999_cols = [
        "time_since_prev_cv_join_sec",
        "time_from_prev_cv_sec",
        "time_to_next_cv_sec",
        "dt_from_prev_B_sec",
        "dt_to_next_B_sec",
    ]

    for col in fill_999_cols:
        if col in event_all.columns:
            event_all[col] = pd.to_numeric(event_all[col], errors="coerce").fillna(999.0)

    fill_zero_cols = [
        "prev_cv_join_sec",
        "latest_cv_N",
        "cv_seen_so_far",
        "prev_cv_event_N",
        "next_cv_event_N",
        "count_gap_from_prev_cv",
        "count_gap_to_next_cv",
        "N_base_at_t",
        "join_event_base",
    ]

    for col in fill_zero_cols:
        if col in event_all.columns:
            event_all[col] = pd.to_numeric(event_all[col], errors="coerce").fillna(0.0)

    event_all["is_cv_int"] = event_all["is_cv_flag"].astype(int)

    feature_cols = [
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
    ]

    optional_cols = [
        "prev_cv_join_sec",
        "time_since_prev_cv_join_sec",
        "latest_cv_N",
        "cv_seen_so_far",
        "N_base_at_t",
        "join_event_base",
    ]

    feature_cols += [col for col in optional_cols if col in event_all.columns]

    event_all = safe_numeric(event_all, feature_cols + ["resid_true_minus_base_sec"])

    EVENT_FEATURES_ALLRATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    event_all.to_csv(EVENT_FEATURES_ALLRATES_CSV, index=False)

    print(f"[Saved] Event features: {EVENT_FEATURES_ALLRATES_CSV}")
    print(f"Rows: {len(event_all):,}")
    print(f"Features: {feature_cols}")

    return event_all, feature_cols


# ============================================================
# Global XGBoost model training and raw prediction
# ============================================================
def tune_xgboost_hyperparameters(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """
    Tune XGBoost hyperparameters using training runs only.

    GroupKFold is used with run_id as the group variable so that rows from
    the same simulation run are not split across validation folds.
    """
    print("=" * 80)
    print("Running XGBoost hyperparameter tuning")
    print("=" * 80)

    x_train = train_df[feature_cols].copy()
    y_train = train_df["resid_true_minus_base_sec"].to_numpy(dtype=float)
    groups = train_df["run_id"].to_numpy(dtype=int)

    n_unique_runs = train_df["run_id"].nunique()
    n_splits = min(TUNING_CV_SPLITS, n_unique_runs)

    if n_splits < 2:
        raise ValueError("At least two training runs are needed for GroupKFold tuning.")

    cv = GroupKFold(n_splits=n_splits)

    base_model = XGBRegressor(
        objective=XGB_OBJECTIVE,
        random_state=XGB_RANDOM_STATE,
        n_jobs=-1,
    )

    param_distributions = {
        "n_estimators": [200, 300, 400],
        "learning_rate": [0.03, 0.05, 0.08],
        "max_depth": [2, 3, 4],
        "subsample": [0.70, 0.80, 1.00],
        "colsample_bytree": [0.80, 0.90, 1.00],
        "min_child_weight": [1, 3, 5],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=TUNING_N_ITER,
        scoring=TUNING_SCORING,
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbose=1,
        refit=False,
    )

    search.fit(x_train, y_train, groups=groups)

    results = pd.DataFrame(search.cv_results_)

    results["mean_test_rmse"] = -results["mean_test_score"]
    results["std_test_rmse"] = results["std_test_score"].abs()

    XGB_TUNING_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(XGB_TUNING_RESULTS_CSV, index=False)

    best_params = search.best_params_

    print(f"[Saved] Tuning results: {XGB_TUNING_RESULTS_CSV}")
    print("\nBest XGBoost parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    print(f"\nBest CV RMSE: {-search.best_score_:.4f}")

    return best_params



def train_global_xgboost_model(event_all: pd.DataFrame, feature_cols: list[str]) -> None:
    """
    Train one global XGBoost residual model using all training runs across all CV rates.
    """
    print("=" * 80)
    print("Training one global XGBoost residual model")
    print("=" * 80)

    model_df = event_all.dropna(subset=["resid_true_minus_base_sec"]).copy()

    train_df = model_df[model_df["run_id"].isin(TRAIN_RUN_IDS)].copy()
    test_df = model_df[model_df["run_id"].isin(TEST_RUN_IDS)].copy()

    if train_df.empty:
        raise ValueError("No training rows available.")

    x_train = train_df[feature_cols].copy()
    y_train = train_df["resid_true_minus_base_sec"].to_numpy(dtype=float)

    if RUN_HYPERPARAMETER_TUNING:
        best_params = tune_xgboost_hyperparameters(train_df, feature_cols)
    else:
        best_params = {
            "n_estimators": XGB_N_ESTIMATORS,
            "learning_rate": XGB_LEARNING_RATE,
            "max_depth": XGB_MAX_DEPTH,
            "subsample": XGB_SUBSAMPLE,
            "colsample_bytree": XGB_COLSAMPLE_BYTREE,
        }

    model = XGBRegressor(
        **best_params,
        random_state=XGB_RANDOM_STATE,
        objective=XGB_OBJECTIVE,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)
    
    if RUN_FEATURE_IMPORTANCE_DIAGNOSTIC:
        save_xgb_feature_importance(model, feature_cols)

    x_all = model_df[feature_cols].copy()
    model_df["pred_resid_join_sec"] = model.predict(x_all)
    model_df["t_join_pred_raw_sec"] = model_df["tB_sec"] + model_df["pred_resid_join_sec"]

    pred_cols = [
        "run_id",
        "cv_rate_pct",
        "split",
        "veh_uid",
        "N",
        "is_cv_flag",
        "joined_queue_flag",
        "tB_sec",
        "t_exit_sec",
        "t_gt_event_sec",
        "t_event_true_sec",
        "cv_event_time_sec",
        "resid_true_minus_base_sec",
        "pred_resid_join_sec",
        "t_join_pred_raw_sec",
    ] + feature_cols

    pred_cols = [col for col in pred_cols if col in model_df.columns]

    EVENT_PREDICTIONS_RAW_ALLRATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    model_df[pred_cols].to_csv(EVENT_PREDICTIONS_RAW_ALLRATES_CSV, index=False)

    XGB_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "train_run_ids": TRAIN_RUN_IDS,
            "test_run_ids": TEST_RUN_IDS,
            "cv_rates_pct": CV_RATES_PCT,
            "random_seed": RANDOM_SEED,
            "best_params": best_params,
            "hyperparameter_tuning_used": RUN_HYPERPARAMETER_TUNING,
        },
        XGB_MODEL_FILE,
    )

    metrics = []

    for rate_pct in CV_RATES_PCT:
        for split_name, split_runs in {
            "train": TRAIN_RUN_IDS,
            "test": TEST_RUN_IDS,
        }.items():
            d = model_df[
                (model_df["cv_rate_pct"] == int(rate_pct))
                & (model_df["run_id"].isin(split_runs))
            ].copy()

            d = d.dropna(subset=["t_event_true_sec", "tB_sec", "t_join_pred_raw_sec"])

            base_mae, base_rmse = compute_metrics(d["t_event_true_sec"], d["tB_sec"])
            raw_mae, raw_rmse = compute_metrics(d["t_event_true_sec"], d["t_join_pred_raw_sec"])

            metrics.append(
                {
                    "cv_rate_pct": int(rate_pct),
                    "split": split_name,
                    "n_rows": int(len(d)),
                    "mae_base_sec": base_mae,
                    "mae_raw_ml_sec": raw_mae,
                    "rmse_base_sec": base_rmse,
                    "rmse_raw_ml_sec": raw_rmse,
                    "mae_improvement_sec": base_mae - raw_mae if pd.notna(base_mae) and pd.notna(raw_mae) else np.nan,
                    "rmse_improvement_sec": base_rmse - raw_rmse if pd.notna(base_rmse) and pd.notna(raw_rmse) else np.nan,
                }
            )

    metrics_df = pd.DataFrame(metrics)

    RAW_MODEL_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(RAW_MODEL_METRICS_CSV, index=False)

    print(f"[Saved] Raw predictions : {EVENT_PREDICTIONS_RAW_ALLRATES_CSV}")
    print(f"[Saved] XGBoost model  : {XGB_MODEL_FILE}")
    print(f"[Saved] Raw metrics    : {RAW_MODEL_METRICS_CSV}")
    print(metrics_df.to_string(index=False))


# ============================================================
# Main pipeline
# ============================================================

def run_cv_ml_pipeline() -> None:
    """Run CV allocation, feature engineering, and raw ML prediction."""
    ensure_project_directories()

    create_cv_allocation_files()
    build_timegrid_features_for_all_rates()
    event_all, feature_cols = build_event_features_allrates()

    if RUN_FEATURE_CORRELATION_DIAGNOSTIC:
        save_feature_correlation_diagnostic(event_all, feature_cols)

    train_global_xgboost_model(event_all, feature_cols)
    print("\nCV + ML pipeline complete.")


def main() -> None:
    """Script entry point."""
    run_cv_ml_pipeline()


if __name__ == "__main__":
    main()