"""
Apply CV-constrained segment-wise correction to raw ML predictions.

This script follows the final CV-fragment correction logic:

1. Read raw event-level ML predictions.
2. Read saved GT queue-join files.
3. Use GT cumulative order as the reference curve scale.
4. Use CV vehicles as hard anchors:
       anchor time = GT t_queue_join_sec
       anchor count/order = N_gt
5. Add first and last GT vehicles as boundary anchors.
6. Partition the curve into anchor-to-anchor segments.
7. Rescale raw ML prediction shape within each segment so both endpoints
   match the anchor times exactly.
8. Save final segmented corrected prediction files.

Outputs:
    output/corrected_curves/event_predictions_segmented_allruns_rateXXX.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    RUN_IDS,
    CV_RATES_PCT,
    EVENT_PREDICTIONS_RAW_ALLRATES_CSV,
    GT_JOIN_PATTERN,
    CV_ALLOC_RUN_PATTERN,
    SEGMENTED_PRED_PATTERN,
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


def robust_bool_to_flag(series: pd.Series) -> pd.Series:
    """Convert boolean-like values to integer 0/1 flags."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

    text = series.astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y"]).astype(int)


def load_gt_for_run(run_id: int) -> pd.DataFrame:
    """
    Load GT queue-join file and ensure N_gt exists.

    The GT file should already be sorted by t_queue_join_sec from
    ground_truth_queue.py. If N_gt is missing, it is reconstructed.
    """
    gt_path = format_path(GT_JOIN_PATTERN, run_id=run_id)

    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT file: {gt_path}")

    gt = pd.read_csv(gt_path)
    require_columns(gt, ["veh_uid", "t_queue_join_sec"], f"GT file for run {run_id:03d}")

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

    keep_cols = [
        "veh_uid",
        "N_gt",
        "t_queue_join_sec",
    ]

    optional_cols = [
        "vehID",
        "joined_queue",
        "join_rule_family",
        "t_stopbar_cross_sec",
    ]

    keep_cols += [col for col in optional_cols if col in gt.columns]

    return gt[keep_cols].copy()


def load_cv_ids_for_run_rate(run_id: int, rate_pct: int) -> set[str]:
    """Load CV allocation file and return selected CV vehicle IDs."""
    cv_path = format_path(CV_ALLOC_RUN_PATTERN, run_id=run_id, rate=rate_pct)

    if not cv_path.exists():
        raise FileNotFoundError(f"Missing CV allocation file: {cv_path}")

    cv = pd.read_csv(cv_path)
    require_columns(cv, ["veh_uid", "is_cv"], f"CV allocation run {run_id:03d}, rate {rate_pct}")

    cv["veh_uid"] = cv["veh_uid"].astype(str).str.strip()
    cv["is_cv_flag"] = robust_bool_to_flag(cv["is_cv"])

    cv_ids = set(cv.loc[cv["is_cv_flag"] == 1, "veh_uid"].tolist())

    return cv_ids


def build_anchor_table(df_run: pd.DataFrame, cv_ids: set[str]) -> pd.DataFrame:
    """
    Build anchor table for one run and one CV rate.

    Anchors include:
    - first GT vehicle as left boundary
    - all CV vehicles with GT times
    - last GT vehicle as right boundary
    """
    if df_run.empty:
        raise ValueError("Cannot build anchors from an empty run dataframe.")

    base = df_run.dropna(subset=["N_gt", "t_queue_join_sec"]).copy()
    base = base.sort_values("N_gt").reset_index(drop=True)

    if base.empty:
        raise ValueError("No valid GT rows available for anchor construction.")

    cv_anchor = base[base["veh_uid"].isin(cv_ids)].copy()

    first_row = base.iloc[[0]].copy()
    last_row = base.iloc[[-1]].copy()

    first_row["anchor_type"] = "boundary_first"
    last_row["anchor_type"] = "boundary_last"

    if not cv_anchor.empty:
        cv_anchor["anchor_type"] = "cv"
        anchors = pd.concat([first_row, cv_anchor, last_row], ignore_index=True)
    else:
        anchors = pd.concat([first_row, last_row], ignore_index=True)

    anchors = anchors[
        [
            "veh_uid",
            "N_gt",
            "t_queue_join_sec",
            "anchor_type",
        ]
    ].copy()

    anchors = anchors.dropna(subset=["N_gt", "t_queue_join_sec"])
    anchors["N_gt"] = anchors["N_gt"].astype(int)
    anchors["t_queue_join_sec"] = pd.to_numeric(anchors["t_queue_join_sec"], errors="coerce")

    # If a boundary is also CV, preserve one row per N_gt and prefer CV label.
    priority = {
        "cv": 0,
        "boundary_first": 1,
        "boundary_last": 1,
    }
    anchors["anchor_priority"] = anchors["anchor_type"].map(priority).fillna(9)
    anchors = anchors.sort_values(["N_gt", "anchor_priority"]).drop_duplicates("N_gt", keep="first")
    anchors = anchors.sort_values("N_gt").reset_index(drop=True)
    anchors = anchors.drop(columns=["anchor_priority"])

    if len(anchors) < 2:
        raise ValueError("At least two anchors are required for segment correction.")

    return anchors


def segment_rescale_one_run(df_run: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """
    Apply anchor-to-anchor segment-wise temporal rescaling.

    The raw ML prediction provides the segment shape. For each segment between
    two consecutive anchors, the raw prediction is linearly transformed so that
    both segment endpoints match the anchor GT times exactly.
    """
    g = df_run.sort_values("N_gt").copy().reset_index(drop=True)

    g["t_join_pred_segmented_sec"] = np.nan
    g["is_cv_anchor"] = 0
    g["is_boundary_anchor"] = 0
    g["anchor_type"] = ""
    g["segment_id"] = -1

    anchor_lookup = anchors.set_index("N_gt")

    for n_gt, row in anchor_lookup.iterrows():
        mask = g["N_gt"] == int(n_gt)
        if not mask.any():
            continue

        anchor_type = str(row["anchor_type"])

        g.loc[mask, "anchor_type"] = anchor_type
        g.loc[mask, "is_cv_anchor"] = int(anchor_type == "cv")
        g.loc[mask, "is_boundary_anchor"] = int(anchor_type.startswith("boundary"))

    anchor_rows = anchors.sort_values("N_gt").reset_index(drop=True)

    for segment_id in range(len(anchor_rows) - 1):
        left_anchor = anchor_rows.iloc[segment_id]
        right_anchor = anchor_rows.iloc[segment_id + 1]

        n_left = int(left_anchor["N_gt"])
        n_right = int(right_anchor["N_gt"])

        t_left_anchor = float(left_anchor["t_queue_join_sec"])
        t_right_anchor = float(right_anchor["t_queue_join_sec"])

        if n_right < n_left:
            continue

        segment_mask = (g["N_gt"] >= n_left) & (g["N_gt"] <= n_right)
        segment = g.loc[segment_mask].copy()

        if segment.empty:
            continue

        raw_times = segment["t_join_pred_raw_sec"].to_numpy(dtype=float)
        n_values = segment["N_gt"].to_numpy(dtype=float)

        # Raw prediction at segment endpoints.
        left_raw_values = segment.loc[segment["N_gt"] == n_left, "t_join_pred_raw_sec"]
        right_raw_values = segment.loc[segment["N_gt"] == n_right, "t_join_pred_raw_sec"]

        raw_left = float(left_raw_values.iloc[0]) if len(left_raw_values) else np.nan
        raw_right = float(right_raw_values.iloc[0]) if len(right_raw_values) else np.nan

        if (
            np.isfinite(raw_left)
            and np.isfinite(raw_right)
            and abs(raw_right - raw_left) > 1e-9
        ):
            fraction = (raw_times - raw_left) / (raw_right - raw_left)
        else:
            if n_right == n_left:
                fraction = np.zeros(len(segment), dtype=float)
            else:
                fraction = (n_values - float(n_left)) / (float(n_right) - float(n_left))

        # Avoid extreme extrapolation inside the segment.
        fraction = np.clip(fraction, 0.0, 1.0)

        corrected = t_left_anchor + fraction * (t_right_anchor - t_left_anchor)

        idx = segment.index.to_numpy()
        g.loc[idx, "t_join_pred_segmented_sec"] = corrected
        g.loc[idx, "segment_id"] = int(segment_id)

        # Force exact anchor endpoints.
        g.loc[g["N_gt"] == n_left, "t_join_pred_segmented_sec"] = t_left_anchor
        g.loc[g["N_gt"] == n_right, "t_join_pred_segmented_sec"] = t_right_anchor

    # Safety fallback for any rows not assigned due to duplicated/missing anchors.
    missing_mask = g["t_join_pred_segmented_sec"].isna()
    if missing_mask.any():
        g.loc[missing_mask, "t_join_pred_segmented_sec"] = g.loc[missing_mask, "t_join_pred_raw_sec"]

    # Ensure monotonicity in GT cumulative order after segment correction.
    g = g.sort_values("N_gt").reset_index(drop=True)
    corrected = g["t_join_pred_segmented_sec"].to_numpy(dtype=float)
    corrected = np.maximum.accumulate(corrected)
    g["t_join_pred_segmented_sec"] = corrected

    # Force anchors one final time, then keep monotone consistency around them.
    for _, anchor in anchors.iterrows():
        n_anchor = int(anchor["N_gt"])
        t_anchor = float(anchor["t_queue_join_sec"])
        g.loc[g["N_gt"] == n_anchor, "t_join_pred_segmented_sec"] = t_anchor

    corrected = g["t_join_pred_segmented_sec"].to_numpy(dtype=float)
    corrected = np.maximum.accumulate(corrected)
    g["t_join_pred_segmented_sec"] = corrected

    return g


def correct_one_run_rate(raw_rate: pd.DataFrame, run_id: int, rate_pct: int) -> pd.DataFrame:
    """
    Apply segment-wise CV-anchor correction for one run and CV rate.
    """
    raw_run = raw_rate[raw_rate["run_id"] == int(run_id)].copy()

    if raw_run.empty:
        return pd.DataFrame()

    require_columns(
        raw_run,
        [
            "run_id",
            "cv_rate_pct",
            "split",
            "veh_uid",
            "N",
            "tB_sec",
            "t_event_true_sec",
            "t_join_pred_raw_sec",
        ],
        f"raw predictions run {run_id:03d}, rate {rate_pct}",
    )

    raw_run["veh_uid"] = raw_run["veh_uid"].astype(str).str.strip()
    raw_run = safe_numeric(
        raw_run,
        [
            "run_id",
            "cv_rate_pct",
            "N",
            "tB_sec",
            "t_event_true_sec",
            "t_join_pred_raw_sec",
        ],
    )

    raw_run = raw_run.dropna(
        subset=[
            "run_id",
            "cv_rate_pct",
            "veh_uid",
            "N",
            "tB_sec",
            "t_event_true_sec",
            "t_join_pred_raw_sec",
        ]
    ).copy()

    raw_run["run_id"] = raw_run["run_id"].astype(int)
    raw_run["cv_rate_pct"] = raw_run["cv_rate_pct"].astype(int)
    raw_run["N"] = raw_run["N"].astype(int)

    gt = load_gt_for_run(run_id)
    cv_ids = load_cv_ids_for_run_rate(run_id, rate_pct)

    # Merge GT cumulative order and GT event time to raw prediction rows.
    merged = raw_run.merge(
        gt[["veh_uid", "N_gt", "t_queue_join_sec"]],
        on="veh_uid",
        how="left",
    )

    if merged["N_gt"].isna().any():
        missing = int(merged["N_gt"].isna().sum())
        raise ValueError(
            f"Run {run_id:03d}, rate {rate_pct}%: "
            f"{missing} raw prediction rows missing GT cumulative order."
        )

    merged["N_gt"] = pd.to_numeric(merged["N_gt"], errors="coerce").astype(int)
    merged["t_queue_join_sec"] = pd.to_numeric(merged["t_queue_join_sec"], errors="coerce")

    anchors = build_anchor_table(merged, cv_ids)

    corrected = segment_rescale_one_run(merged, anchors)

    corrected["n_anchors_run_rate"] = len(anchors)
    corrected["n_cv_anchors_run_rate"] = int((anchors["anchor_type"] == "cv").sum())

    return corrected


# ============================================================
# Main correction pipeline
# ============================================================

def run_segmented_anchor_correction() -> None:
    """
    Run CV-constrained segment-wise correction for all CV rates.
    """
    ensure_project_directories()

    if not EVENT_PREDICTIONS_RAW_ALLRATES_CSV.exists():
        raise FileNotFoundError(
            f"Missing raw prediction file: {EVENT_PREDICTIONS_RAW_ALLRATES_CSV}"
        )

    raw_all = pd.read_csv(EVENT_PREDICTIONS_RAW_ALLRATES_CSV)

    require_columns(
        raw_all,
        [
            "run_id",
            "cv_rate_pct",
            "split",
            "veh_uid",
            "N",
            "tB_sec",
            "t_event_true_sec",
            "t_join_pred_raw_sec",
        ],
        "raw prediction file",
    )

    raw_all = safe_numeric(
        raw_all,
        [
            "run_id",
            "cv_rate_pct",
            "N",
            "tB_sec",
            "t_event_true_sec",
            "t_join_pred_raw_sec",
        ],
    )

    raw_all = raw_all.dropna(
        subset=[
            "run_id",
            "cv_rate_pct",
            "veh_uid",
            "N",
            "tB_sec",
            "t_event_true_sec",
            "t_join_pred_raw_sec",
        ]
    ).copy()

    raw_all["run_id"] = raw_all["run_id"].astype(int)
    raw_all["cv_rate_pct"] = raw_all["cv_rate_pct"].astype(int)
    raw_all["veh_uid"] = raw_all["veh_uid"].astype(str).str.strip()
    raw_all["N"] = raw_all["N"].astype(int)

    for rate_pct in CV_RATES_PCT:
        print("=" * 80)
        print(f"Applying segment-wise CV-anchor correction for CV rate {rate_pct}%")
        print("=" * 80)

        raw_rate = raw_all[raw_all["cv_rate_pct"] == int(rate_pct)].copy()

        if raw_rate.empty:
            print(f"[Warning] No raw prediction rows found for rate {rate_pct}%. Skipping.")
            continue

        corrected_parts = []

        for run_id in RUN_IDS:
            corrected_run = correct_one_run_rate(raw_rate, run_id, rate_pct)

            if corrected_run.empty:
                print(f"[Run {run_id:03d}] No corrected rows created.")
                continue

            corrected_parts.append(corrected_run)

            print(
                f"[Run {run_id:03d}] rows={len(corrected_run):,}, "
                f"anchors={int(corrected_run['n_anchors_run_rate'].iloc[0])}, "
                f"CV anchors={int(corrected_run['n_cv_anchors_run_rate'].iloc[0])}"
            )

        if not corrected_parts:
            raise ValueError(f"No corrected rows created for CV rate {rate_pct}%.")

        corrected_all = pd.concat(corrected_parts, ignore_index=True)

        save_cols = [
            "run_id",
            "cv_rate_pct",
            "split",
            "veh_uid",
            "N",
            "N_gt",
            "is_cv_flag",
            "is_cv_anchor",
            "is_boundary_anchor",
            "anchor_type",
            "segment_id",
            "n_anchors_run_rate",
            "n_cv_anchors_run_rate",
            "tB_sec",
            "t_event_true_sec",
            "t_queue_join_sec",
            "cv_event_time_sec",
            "t_join_pred_raw_sec",
            "t_join_pred_segmented_sec",
        ]

        save_cols += [
            col for col in [
                "pred_resid_join_sec",
                "resid_true_minus_base_sec",
                "joined_queue_flag",
            ]
            if col in corrected_all.columns
        ]

        save_cols = [col for col in save_cols if col in corrected_all.columns]

        out_path = format_path(SEGMENTED_PRED_PATTERN, rate=rate_pct)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        corrected_all[save_cols].to_csv(out_path, index=False)

        print(f"[Saved] {out_path} | rows={len(corrected_all):,}")

    print("\nCV-anchor segment correction complete.")


def main() -> None:
    """Script entry point."""
    run_segmented_anchor_correction()


if __name__ == "__main__":
    main()