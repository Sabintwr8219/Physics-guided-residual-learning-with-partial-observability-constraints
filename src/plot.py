"""
Plot selected figures from saved project outputs.

This script reads saved CSV files only. It does not recompute GT, baseline
curves, CV allocation, ML predictions, correction, or evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    TRAJ_NB_FILTERED_CSV,
    GT_CURVE_PLOT_READY_CSV,
    CUMULATIVE_CURVES_PLOT_READY_CSV,
    VB_CURVES_CSV,
    CV_ALLOC_RUN_PATTERN,
    SEGMENTED_PRED_PATTERN,
    FIGURES_DIR,
    RUN_IDS,
    CV_RATES_PCT,
    Y_UPSTREAM_DETECTOR_FT,
    Y_STOPBAR_FT,
    M_TO_FT,
    RESULTS_DIR,
    ensure_project_directories,
)


# ============================================================
# MAIN PLOT CONFIGURATION
# Change these values to control what plots are generated.
# ============================================================

RUN_ID = 11
CV_RATE_PCT = 10

# Plot style:
#   "bw"    = black-and-white journal style
#   "color" = color figures
PLOT_STYLE = "bw"

# Time display:
#   "full"   = full simulation period
#   "window" = only plot between T0 and T1
PLOT_TIME_MODE = "full"
T0 = 1200.0
T1 = 1400.0

# Save/show controls
SAVE_FIGURES = True
SHOW_FIGURES = True
FIGURE_DPI = 300

# Figure sizes
FIGSIZE_MAIN = (11, 7)
FIGSIZE_WIDE = (14, 6)

# Common formatting
SHOW_GRID = True
SHOW_LEGEND = True


# ============================================================
# WHICH PLOTS TO GENERATE
# Set True/False depending on what you want.
# ============================================================

PLOT_FILTERED_TRAJECTORIES = True
PLOT_GT_CURVE = True
PLOT_ADVB_CURVES = True
PLOT_CV_DOTS_ON_BASELINE = True
PLOT_FINAL_BASELINE_GT_CORRECTED = True
PLOT_FINAL_EVALUATION_TABLE = True


# ============================================================
# FILTERED TRAJECTORY PLOT CONFIGURATION
# ============================================================

# Vehicle selection:
#   "all"              = plot all vehicles
#   "every_nth_global" = plot every nth vehicle in the selected run
PLOT_VEHICLE_MODE = "all"
PLOT_EVERY_NTH = 5

# Show start/end markers on trajectories
SHOW_START_END_DOTS = True


# ============================================================
# A/D/V/B CUMULATIVE CURVE CONFIGURATION
# ============================================================

PLOT_A = True
PLOT_D = True
PLOT_V = True
PLOT_B = True


# ============================================================
# CV DOTS ON BASELINE B CONFIGURATION
# ============================================================

CV_DOT_SIZE = 28


# ============================================================
# FINAL MAIN PLOT CONFIGURATION
# Baseline B + GT + corrected curve
# ============================================================

PLOT_BASELINE_IN_FINAL = True
PLOT_GT_IN_FINAL = True
PLOT_CORRECTED_IN_FINAL = True


# ============================================================
# OUTPUT FIGURE NAMES
# ============================================================

TRAJECTORY_FIG_NAME = "filtered_trajectories_run{run_id:03d}.png"
GT_FIG_NAME = "gt_cumulative_curve_run{run_id:03d}.png"
ADVB_FIG_NAME = "advb_curves_run{run_id:03d}.png"
CV_DOT_FIG_NAME = "baseline_with_cv_points_run{run_id:03d}_rate{rate:03d}.png"
FINAL_FIG_NAME = "baseline_gt_corrected_run{run_id:03d}_rate{rate:03d}.png"
FINAL_EVALUATION_TABLE_CSV = RESULTS_DIR / "final_evaluation_table_test_only.csv"
FINAL_EVALUATION_TABLE_FIG_NAME = "final_evaluation_table_test_only.png"


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


def robust_bool_to_flag(series: pd.Series) -> pd.Series:
    """Convert boolean-like values to integer 0/1 flags."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

    text = series.astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y"]).astype(int)


def maybe_filter_window(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    If PLOT_TIME_MODE is 'window', filter rows to [PLOT_T0, PLOT_T1].
    """
    if PLOT_TIME_MODE.lower() != "window":
        return df

    out = df.copy()
    out[time_col] = pd.to_numeric(out[time_col], errors="coerce")
    out = out[(out[time_col] >= T0) & (out[time_col] <= T1)].copy()
    return out


def set_time_xlim(ax) -> None:
    """Apply x-axis window if requested."""
    if PLOT_TIME_MODE.lower() == "window":
        ax.set_xlim(T0, T1)


def get_line_styles() -> dict:
    """Return line styles depending on black-white or color plotting mode."""
    if PLOT_STYLE.lower() == "bw":
        return {
            "A": {"color": "black", "linestyle": "-", "linewidth": 2.0},
            "D": {"color": "black", "linestyle": "--", "linewidth": 2.0},
            "V": {"color": "black", "linestyle": ":", "linewidth": 2.2},
            "B": {"color": "black", "linestyle": "-.", "linewidth": 2.0},
            "baseline": {"color": "black", "linestyle": "-.", "linewidth": 1.8},
            "gt": {"color": "black", "linestyle": "-", "linewidth": 2.2},
            "corrected": {"color": "black", "linestyle": "--", "linewidth": 2.2},
            "cv": {"color": "black", "marker": "o"},
            "trajectory": {"color": "black", "linewidth": 0.7},
        }

    return {
        "A": {"linestyle": "-", "linewidth": 2.0},
        "D": {"linestyle": "--", "linewidth": 2.0},
        "V": {"linestyle": ":", "linewidth": 2.2},
        "B": {"linestyle": "-.", "linewidth": 2.0},
        "baseline": {"linestyle": "-.", "linewidth": 1.8},
        "gt": {"linestyle": "-", "linewidth": 2.2},
        "corrected": {"linestyle": "--", "linewidth": 2.2},
        "cv": {"color": "green", "marker": "o"},
        "trajectory": {"linewidth": 0.8},
    }


def apply_common_formatting(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply common axis formatting."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if SHOW_GRID:
        if PLOT_STYLE.lower() == "bw":
            ax.grid(True, which="major", linestyle=":", linewidth=0.7, color="0.75")
        else:
            ax.grid(True, alpha=0.3)

    if SHOW_LEGEND:
        ax.legend(loc="best", frameon=True)

    set_time_xlim(ax)


def save_or_show(fig, filename: str) -> None:
    """Save and/or show a matplotlib figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / filename

    if SAVE_FIGURES:
        fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"[Saved] {out_path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def build_step_arrays(times: np.ndarray, counts: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sorted step-plot arrays.

    If counts are not provided, cumulative count is 1..N after sorting times.
    """
    t = np.asarray(times, dtype=float)
    mask = np.isfinite(t)
    t = t[mask]

    if counts is None:
        t.sort()
        n = np.arange(1, len(t) + 1, dtype=int)
        return t, n

    n = np.asarray(counts, dtype=float)[mask]
    order = np.argsort(t)
    return t[order], n[order]


# ============================================================
# Data loading helpers
# ============================================================

def load_gt_curve(run_id: int) -> pd.DataFrame:
    """Load plot-ready GT curve for one run."""
    gt = pd.read_csv(GT_CURVE_PLOT_READY_CSV)
    require_columns(gt, ["run_id", "veh_uid", "N_gt", "t_queue_join_sec"], "GT plot-ready file")

    gt["run_id"] = pd.to_numeric(gt["run_id"], errors="coerce")
    gt["N_gt"] = pd.to_numeric(gt["N_gt"], errors="coerce")
    gt["t_queue_join_sec"] = pd.to_numeric(gt["t_queue_join_sec"], errors="coerce")
    gt["veh_uid"] = gt["veh_uid"].astype(str).str.strip()

    gt = gt[gt["run_id"] == int(run_id)].copy()
    gt = gt.dropna(subset=["N_gt", "t_queue_join_sec"]).copy()
    gt["N_gt"] = gt["N_gt"].astype(int)

    return gt.sort_values("N_gt").reset_index(drop=True)


def load_baseline_b_curve(run_id: int) -> pd.DataFrame:
    """Load baseline B curve for one run."""
    vb = pd.read_csv(VB_CURVES_CSV)
    require_columns(vb, ["run_id", "N", "tB_sec"], "vb_curves_all_runs.csv")

    vb["run_id"] = pd.to_numeric(vb["run_id"], errors="coerce")
    vb["N"] = pd.to_numeric(vb["N"], errors="coerce")
    vb["tB_sec"] = pd.to_numeric(vb["tB_sec"], errors="coerce")

    vb = vb[vb["run_id"] == int(run_id)].copy()
    vb = vb.dropna(subset=["N", "tB_sec"]).copy()
    vb["N"] = vb["N"].astype(int)

    return vb.sort_values("N").reset_index(drop=True)


def load_corrected_curve(run_id: int, rate_pct: int) -> pd.DataFrame:
    """Load final segmented corrected curve for one run and CV rate."""
    corr_path = format_path(SEGMENTED_PRED_PATTERN, rate=int(rate_pct))

    if not corr_path.exists():
        raise FileNotFoundError(f"Missing corrected prediction file: {corr_path}")

    corr = pd.read_csv(corr_path)
    require_columns(
        corr,
        ["run_id", "cv_rate_pct", "N_gt", "t_join_pred_segmented_sec"],
        f"corrected file rate {rate_pct}",
    )

    corr["run_id"] = pd.to_numeric(corr["run_id"], errors="coerce")
    corr["cv_rate_pct"] = pd.to_numeric(corr["cv_rate_pct"], errors="coerce")
    corr["N_gt"] = pd.to_numeric(corr["N_gt"], errors="coerce")
    corr["t_join_pred_segmented_sec"] = pd.to_numeric(corr["t_join_pred_segmented_sec"], errors="coerce")

    corr = corr[
        (corr["run_id"] == int(run_id)) &
        (corr["cv_rate_pct"] == int(rate_pct))
    ].copy()

    corr = corr.dropna(subset=["N_gt", "t_join_pred_segmented_sec"]).copy()
    corr["N_gt"] = corr["N_gt"].astype(int)

    return corr.sort_values("N_gt").reset_index(drop=True)


# ============================================================
# Plot 1: Filtered trajectories
# ============================================================

def select_vehicles_for_trajectory_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Select vehicles for trajectory plotting based on config."""
    if PLOT_VEHICLE_MODE == "all":
        return df

    veh_entry = (
        df.groupby("veh_uid", as_index=False)["Total_Sim_Time_Sec"]
        .min()
        .rename(columns={"Total_Sim_Time_Sec": "t_entry"})
        .sort_values("t_entry")
        .reset_index(drop=True)
    )

    if PLOT_VEHICLE_MODE == "every_nth_global":
        keep_vehicles = veh_entry.iloc[::PLOT_EVERY_NTH]["veh_uid"].tolist()
        return df[df["veh_uid"].isin(keep_vehicles)].copy()

    raise ValueError(
        "Unsupported PLOT_VEHICLE_MODE. Use 'all' or 'every_nth_global'. "
        "Cycle-wise selection requires saved cycle files, which are not part of the final main pipeline."
    )


def plot_filtered_trajectories(run_id: int) -> None:
    """Plot filtered northbound trajectories for one run."""
    df = pd.read_csv(TRAJ_NB_FILTERED_CSV)
    require_columns(df, ["run_id", "veh_uid", "Total_Sim_Time_Sec"], "traj_nb_filtered.csv")

    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
    df["veh_uid"] = df["veh_uid"].astype(str).str.strip()
    df["Total_Sim_Time_Sec"] = pd.to_numeric(df["Total_Sim_Time_Sec"], errors="coerce")

    if "s_rel_stop_ft" not in df.columns:
        require_columns(df, ["s_rel_stop_m"], "traj_nb_filtered.csv")
        df["s_rel_stop_ft"] = pd.to_numeric(df["s_rel_stop_m"], errors="coerce") * M_TO_FT
    else:
        df["s_rel_stop_ft"] = pd.to_numeric(df["s_rel_stop_ft"], errors="coerce")

    df = df[df["run_id"] == int(run_id)].copy()
    df = df.dropna(subset=["veh_uid", "Total_Sim_Time_Sec", "s_rel_stop_ft"]).copy()

    if df.empty:
        raise ValueError(f"No filtered trajectory rows found for run {run_id:03d}.")

    df = maybe_filter_window(df, "Total_Sim_Time_Sec")
    df = select_vehicles_for_trajectory_plot(df)

    styles = get_line_styles()

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for veh_uid, group in df.groupby("veh_uid", sort=False):
        g = group.sort_values("Total_Sim_Time_Sec")
        t = g["Total_Sim_Time_Sec"].to_numpy(dtype=float)
        y = g["s_rel_stop_ft"].to_numpy(dtype=float)

        if len(t) == 0:
            continue

        ax.plot(t, y, **styles["trajectory"])

        if SHOW_START_END_DOTS:
            if PLOT_STYLE.lower() == "bw":
                ax.scatter(t[0], y[0], s=10, color="black", marker="o")
                ax.scatter(t[-1], y[-1], s=10, color="black", marker="x")
            else:
                ax.scatter(t[0], y[0], s=10, color="green")
                ax.scatter(t[-1], y[-1], s=10, color="red")

    ax.axhline(
        Y_STOPBAR_FT,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Stopbar",
    )

    ax.axhline(
        Y_UPSTREAM_DETECTOR_FT,
        color="black",
        linestyle="-.",
        linewidth=1.2,
        label="Upstream detector",
    )

    title = f"Filtered Northbound Trajectories — Run {run_id:03d}"
    apply_common_formatting(ax, title, "Time (s)", "Stopbar-relative position (ft)")

    filename = TRAJECTORY_FIG_NAME.format(run_id=run_id)
    save_or_show(fig, filename)


# ============================================================
# Plot 2: GT cumulative curve
# ============================================================

def plot_gt_curve(run_id: int) -> None:
    """Plot GT cumulative queue-join curve for one run."""
    gt = load_gt_curve(run_id)
    gt = maybe_filter_window(gt, "t_queue_join_sec")

    if gt.empty:
        raise ValueError(f"No GT rows found for run {run_id:03d}.")

    styles = get_line_styles()

    fig, ax = plt.subplots(figsize=FIGSIZE_MAIN)

    ax.step(
        gt["t_queue_join_sec"].to_numpy(dtype=float),
        gt["N_gt"].to_numpy(dtype=float),
        where="post",
        label="GT curve",
        **styles["gt"],
    )

    title = f"GT Cumulative Queue-Join Curve — Run {run_id:03d}"
    apply_common_formatting(ax, title, "Time (s)", "Cumulative count")

    filename = GT_FIG_NAME.format(run_id=run_id)
    save_or_show(fig, filename)


# ============================================================
# Plot 3: A/D/V/B cumulative curves
# ============================================================

def plot_advb_curves(run_id: int) -> None:
    """Plot selected A/D/V/B cumulative curves for one run."""
    curves = pd.read_csv(CUMULATIVE_CURVES_PLOT_READY_CSV)
    require_columns(curves, ["run_id", "curve_type", "N", "time_sec"], "cumulative plot-ready file")

    curves["run_id"] = pd.to_numeric(curves["run_id"], errors="coerce")
    curves["N"] = pd.to_numeric(curves["N"], errors="coerce")
    curves["time_sec"] = pd.to_numeric(curves["time_sec"], errors="coerce")
    curves["curve_type"] = curves["curve_type"].astype(str).str.strip()

    curves = curves[curves["run_id"] == int(run_id)].copy()
    curves = curves.dropna(subset=["curve_type", "N", "time_sec"]).copy()
    curves["N"] = curves["N"].astype(int)

    if curves.empty:
        raise ValueError(f"No cumulative curve rows found for run {run_id:03d}.")

    curve_flags = {
        "A": PLOT_A,
        "D": PLOT_D,
        "V": PLOT_V,
        "B": PLOT_B,
    }

    labels = {
        "A": "Arrival curve, A(t)",
        "D": "Departure curve, D(t)",
        "V": "Virtual arrival curve, V(t)",
        "B": "Baseline back-of-queue curve, B(t)",
    }

    styles = get_line_styles()

    fig, ax = plt.subplots(figsize=FIGSIZE_MAIN)

    for curve_type in ["A", "D", "V", "B"]:
        if not curve_flags[curve_type]:
            continue

        d = curves[curves["curve_type"] == curve_type].copy()
        d = maybe_filter_window(d, "time_sec")
        d = d.sort_values("time_sec")

        if d.empty:
            continue

        ax.step(
            d["time_sec"].to_numpy(dtype=float),
            d["N"].to_numpy(dtype=float),
            where="post",
            label=labels[curve_type],
            **styles[curve_type],
        )

    title = f"Cumulative A/D/V/B Curves — Run {run_id:03d}"
    apply_common_formatting(ax, title, "Time (s)", "Cumulative vehicle count")

    filename = ADVB_FIG_NAME.format(run_id=run_id)
    save_or_show(fig, filename)


# ============================================================
# Plot 4: CV dots on baseline B curve
# ============================================================

def plot_cv_dots_on_baseline(run_id: int, rate_pct: int) -> None:
    """
    Plot baseline B curve with CV anchor points from GT cumulative order.

    Baseline:
        x = tB_sec
        y = N

    CV dots:
        x = t_queue_join_sec from GT
        y = N_gt from GT
    """
    baseline = load_baseline_b_curve(run_id)
    gt = load_gt_curve(run_id)

    cv_path = format_path(CV_ALLOC_RUN_PATTERN, run_id=run_id, rate=rate_pct)

    if not cv_path.exists():
        raise FileNotFoundError(f"Missing CV allocation file: {cv_path}")

    cv = pd.read_csv(cv_path)
    require_columns(cv, ["veh_uid", "is_cv"], f"CV allocation run {run_id:03d}, rate {rate_pct}")

    cv["veh_uid"] = cv["veh_uid"].astype(str).str.strip()
    cv["is_cv_flag"] = robust_bool_to_flag(cv["is_cv"])

    cv_ids = set(cv.loc[cv["is_cv_flag"] == 1, "veh_uid"].tolist())

    cv_gt = gt[gt["veh_uid"].isin(cv_ids)].copy()

    baseline = maybe_filter_window(baseline, "tB_sec")
    cv_gt = maybe_filter_window(cv_gt, "t_queue_join_sec")

    styles = get_line_styles()

    fig, ax = plt.subplots(figsize=FIGSIZE_MAIN)

    ax.plot(
        baseline["tB_sec"].to_numpy(dtype=float),
        baseline["N"].to_numpy(dtype=float),
        label="Baseline B curve",
        **styles["baseline"],
    )

    cv_style = styles["cv"].copy()
    marker = cv_style.pop("marker", "o")

    ax.scatter(
        cv_gt["t_queue_join_sec"].to_numpy(dtype=float),
        cv_gt["N_gt"].to_numpy(dtype=float),
        s=CV_DOT_SIZE,
        marker=marker,
        label=f"CV anchors ({rate_pct}%)",
        **cv_style,
    )

    title = f"Baseline B Curve with CV Anchor Points — Run {run_id:03d}, {rate_pct}% CV"
    apply_common_formatting(ax, title, "Time (s)", "Cumulative count")

    filename = CV_DOT_FIG_NAME.format(run_id=run_id, rate=rate_pct)
    save_or_show(fig, filename)


# ============================================================
# Plot 5: Final baseline + GT + corrected
# ============================================================

def plot_final_baseline_gt_corrected(run_id: int, rate_pct: int) -> None:
    """Plot baseline B, GT, and final corrected curve on one figure."""
    baseline = load_baseline_b_curve(run_id)
    gt = load_gt_curve(run_id)
    corrected = load_corrected_curve(run_id, rate_pct)

    baseline = maybe_filter_window(baseline, "tB_sec")
    gt = maybe_filter_window(gt, "t_queue_join_sec")
    corrected = maybe_filter_window(corrected, "t_join_pred_segmented_sec")

    styles = get_line_styles()

    fig, ax = plt.subplots(figsize=FIGSIZE_MAIN)

    if PLOT_BASELINE_IN_FINAL and not baseline.empty:
        ax.plot(
            baseline["tB_sec"].to_numpy(dtype=float),
            baseline["N"].to_numpy(dtype=float),
            label="Baseline B curve",
            **styles["baseline"],
        )

    if PLOT_GT_IN_FINAL and not gt.empty:
        ax.plot(
            gt["t_queue_join_sec"].to_numpy(dtype=float),
            gt["N_gt"].to_numpy(dtype=float),
            label="GT curve",
            **styles["gt"],
        )

    if PLOT_CORRECTED_IN_FINAL and not corrected.empty:
        ax.plot(
            corrected["t_join_pred_segmented_sec"].to_numpy(dtype=float),
            corrected["N_gt"].to_numpy(dtype=float),
            label=f"CV-corrected curve ({rate_pct}%)",
            **styles["corrected"],
        )

    if PLOT_TIME_MODE.lower() == "window":
        title = (
            f"Baseline vs GT vs Corrected — Run {run_id:03d}, "
            f"{rate_pct}% CV, {PLOT_T0:.0f}–{PLOT_T1:.0f}s"
        )
    else:
        title = f"Baseline vs GT vs Corrected — Run {run_id:03d}, {rate_pct}% CV"

    apply_common_formatting(ax, title, "Time (s)", "Cumulative count")

    filename = FINAL_FIG_NAME.format(run_id=run_id, rate=rate_pct)
    save_or_show(fig, filename)
    
# ============================================================
# Plot 6: Final evaluation table as figure
# ============================================================
def plot_final_evaluation_table() -> None:
    """
    Plot the final test-only evaluation table as a figure.

    Reads:
        output/results/final_evaluation_table_test_only.csv

    Saves:
        output/figures/final_evaluation_table_test_only.png
    """
    if not FINAL_EVALUATION_TABLE_CSV.exists():
        raise FileNotFoundError(
            f"Missing final evaluation table:\n{FINAL_EVALUATION_TABLE_CSV}\n"
            "Run evaluation.py first."
        )

    df = pd.read_csv(FINAL_EVALUATION_TABLE_CSV)

    # Keep the clean report columns. Adjust here if the table becomes too wide.
    preferred_cols = [
        "CV Penetration Rate (%)",
        "Baseline MAE (s)",
        "Corrected MAE (s)",
        "MAE Improvement (s)",
        "Baseline RMSE (s)",
        "Corrected RMSE (s)",
        "RMSE Improvement (s)",
        "Baseline ABC (veh-s)",
        "Corrected ABC (veh-s)",
        "ABC Improvement (veh-s)",
    ]

    cols = [c for c in preferred_cols if c in df.columns]
    table_df = df[cols].copy()

    # Round numeric values for display.
    for col in table_df.columns:
        if pd.api.types.is_numeric_dtype(table_df[col]):
            table_df[col] = table_df[col].round(3)

    fig_width = max(12, 1.35 * len(table_df.columns))
    fig_height = max(3.5, 0.45 * len(table_df) + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)

    ax.set_title(
        "Final Evaluation Summary on Test Runs",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    save_or_show(fig, FINAL_EVALUATION_TABLE_FIG_NAME)

# ============================================================
# Main
# ============================================================

def run_selected_plots() -> None:
    """Generate all selected plots based on configuration switches."""
    ensure_project_directories()

    if RUN_ID not in RUN_IDS:
        raise ValueError(f"RUN_ID={RUN_ID} is not in configured RUN_IDS={RUN_IDS}")

    if CV_RATE_PCT not in CV_RATES_PCT:
        raise ValueError(
            f"CV_RATE_PCT={CV_RATE_PCT} is not in configured CV_RATES_PCT={CV_RATES_PCT}"
        )

    print("=" * 80)
    print("Generating selected plots")
    print("=" * 80)
    print(f"Run ID       : {RUN_ID:03d}")
    print(f"CV rate      : {CV_RATE_PCT}%")
    print(f"Plot style   : {PLOT_STYLE}")
    print(f"Time mode    : {PLOT_TIME_MODE}")
    print(f"Figures dir  : {FIGURES_DIR}")
    print("=" * 80)

    if PLOT_FILTERED_TRAJECTORIES:
        plot_filtered_trajectories(RUN_ID)

    if PLOT_GT_CURVE:
        plot_gt_curve(RUN_ID)

    if PLOT_ADVB_CURVES:
        plot_advb_curves(RUN_ID)

    if PLOT_CV_DOTS_ON_BASELINE:
        plot_cv_dots_on_baseline(RUN_ID, CV_RATE_PCT)

    if PLOT_FINAL_BASELINE_GT_CORRECTED:
        plot_final_baseline_gt_corrected(RUN_ID, CV_RATE_PCT)
        
    if PLOT_FINAL_EVALUATION_TABLE:
        plot_final_evaluation_table()

    print("\nPlotting complete.")


def main() -> None:
    """Script entry point."""
    run_selected_plots()


if __name__ == "__main__":
    main()