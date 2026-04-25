"""
Project-wide configuration for the Hybrid CV-Constrained Cumulative Curve
Framework for Queue Reconstruction.

All major paths, run IDs, detector geometry, CV rates, model settings,
and plotting defaults are defined here so they can be changed from one place.
"""

from pathlib import Path


# ============================================================
# Repository paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
SAMPLE_DATA_DIR = DATA_DIR / "sample_data"

OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"
CORRECTED_CURVES_DIR = OUTPUT_DIR / "corrected_curves"

DOC_DIR = PROJECT_ROOT / "doc"


# ============================================================
# Raw VISSIM input settings
# ============================================================

VISSIM_RESULTS_DIR = RAW_DATA_DIR / "vissim_results"

VISSIM_BASE_NAME = "Project"

RUN_IDS = list(range(5, 15))          # 005–014
TRAIN_RUN_IDS = list(range(5, 11))    # 005–010
TEST_RUN_IDS = [r for r in RUN_IDS if r not in TRAIN_RUN_IDS]

CHUNKSIZE_FZP = 200_000
CHUNKSIZE_TRAJ = 1_000_000


# ============================================================
# Processed file paths
# ============================================================

MASTER_TRAJECTORY_CSV = PROCESSED_DATA_DIR / "master_trajectory.csv"
MASTER_PHASE_CSV = PROCESSED_DATA_DIR / "master_phase_time.csv"

VEHICLE_FILTER_SUMMARY_CSV = RESULTS_DIR / "veh_summary.csv"
TRAJ_NB_FILTERED_CSV = PROCESSED_DATA_DIR / "traj_nb_filtered.csv"

AD_TIMES_CSV = PROCESSED_DATA_DIR / "ad_times_all_runs.csv"
VB_CURVES_CSV = PROCESSED_DATA_DIR / "vb_curves_all_runs.csv"
BASELINE_B_JOIN_CSV = PROCESSED_DATA_DIR / "baseline_b_join_all_runs.csv"

GT_CURVE_PLOT_READY_CSV = PROCESSED_DATA_DIR / "gt_curve_plot_ready_allruns.csv"
CUMULATIVE_CURVES_PLOT_READY_CSV = PROCESSED_DATA_DIR / "cumulative_curves_plot_ready_allruns.csv"


# ============================================================
# Output filename patterns
# ============================================================

GT_JOIN_PATTERN = PROCESSED_DATA_DIR / "gt_queue_join_run{run_id:03d}.csv"
GT_QUEUE_LENGTH_PATTERN = PROCESSED_DATA_DIR / "gt_queue_length_run{run_id:03d}_0p1s.csv"

CV_ALLOC_RUN_PATTERN = PROCESSED_DATA_DIR / "cv_allocation_run{run_id:03d}_rate{rate:03d}.csv"
CV_ALLOC_ALLRUNS_PATTERN = PROCESSED_DATA_DIR / "cv_allocation_allruns_rate{rate:03d}.csv"

TIMEGRID_FEATURES_PATTERN = PROCESSED_DATA_DIR / "timegrid_vehicle_features_allruns_rate{rate:03d}.csv"

EVENT_FEATURES_ALLRATES_CSV = PROCESSED_DATA_DIR / "event_features_allrates.csv"
EVENT_PREDICTIONS_RAW_ALLRATES_CSV = PROCESSED_DATA_DIR / "event_predictions_raw_allrates.csv"

SEGMENTED_PRED_PATTERN = CORRECTED_CURVES_DIR / "event_predictions_segmented_allruns_rate{rate:03d}.csv"

RAW_MODEL_METRICS_CSV = RESULTS_DIR / "raw_model_metrics.csv"
EVALUATION_BY_RUN_RATE_CSV = RESULTS_DIR / "evaluation_by_run_and_rate.csv"
EVALUATION_SUMMARY_BY_RATE_CSV = RESULTS_DIR / "evaluation_summary_by_rate.csv"
EVALUATION_SUMMARY_BY_SPLIT_CSV = RESULTS_DIR / "evaluation_summary_by_split.csv"

XGB_MODEL_FILE = MODELS_DIR / "xgb_event_residual_model.joblib"


# ============================================================
# Corridor geometry and detector settings
# ============================================================

# WGS84 coordinates for the study corridor reference points
C1_LAT = 32.749832768069375
C1_LON = -97.09732266524595

C2_LAT = 32.75554890701484
C2_LON = -97.09727179559502

CRS_WGS84 = "EPSG:4326"
CRS_PROJECTED = "EPSG:32614"  # UTM Zone 14N

FT_TO_M = 0.3048
M_TO_FT = 3.280839895

UPSTREAM_DETECTOR_OFFSET_FT = 20.0
STOPBAR_OFFSET_FT = 20.0

UPSTREAM_DETECTOR_OFFSET_M = UPSTREAM_DETECTOR_OFFSET_FT * FT_TO_M
STOPBAR_OFFSET_M = STOPBAR_OFFSET_FT * FT_TO_M

# Detector spacing used in cumulative count theory
DETECTOR_SPACING_FT = 1800.0

Y_STOPBAR_FT = 0.0
Y_UPSTREAM_DETECTOR_FT = -1800.0

Y_STOPBAR_M = Y_STOPBAR_FT * FT_TO_M
Y_UPSTREAM_DETECTOR_M = Y_UPSTREAM_DETECTOR_FT * FT_TO_M


# ============================================================
# Trajectory filtering settings
# ============================================================

MAX_PERP_DIST_M = 25.0

BUFFER_BEFORE_M = 0.0
BUFFER_AFTER_M = 0.0

DIR_LOOKBACK_SEC = 1.0
STOPBAR_STORE_BAND_M = 120.0


# ============================================================
# Ground-truth queue detection settings
# ============================================================

CORRIDOR_MIN_FT = -1800.0
CORRIDOR_MAX_FT = 0.0

V_STOP_FPS = 5.0
STOP_PERSIST_SEC = 3.0

CREEP_DROP_FPS = 10.0
CREEP_LOOKBACK_SEC = 1.0
CREEP_PERSIST_SEC = 2.0

NEIGHBOR_TIME_TOL_SEC = 1.0
NEIGHBOR_MAX_GAP_FT = 80.0
NEIGHBOR_LOW_SPEED_FPS = 12.0
NEIGHBOR_DROP_FPS = 8.0

BACKTRACK_MAX_SEC = 4.0
ONSET_DROP_EPS_FPS = 2.0

JOIN_PERSIST_SEC = 1.0

GROW_TOL_FT = 1.0
DEFAULT_L_EFF_FT_PER_VEH = 25.0


# ============================================================
# Cumulative count theory settings
# ============================================================

VF_FPS = 40.584
VQ_FPS = 11.947

MPH_PER_FPS = 1.0 / 1.4666666667

VF_MPH = VF_FPS * MPH_PER_FPS
VQ_MPH = VQ_FPS * MPH_PER_FPS

T_FF_SEC = DETECTOR_SPACING_FT / VF_FPS
BETA = 1.0 - (VQ_FPS / VF_FPS)


# ============================================================
# Connected vehicle and ML settings
# ============================================================

CV_RATES_PCT = [1, 2, 5, 10, 20, 50, 100]

RANDOM_SEED = 42

TIMEGRID_DT_SEC = 0.10
ASOF_TOL_SEC = 0.11

# XGBoost model settings
XGB_N_ESTIMATORS = 300
XGB_LEARNING_RATE = 0.05
XGB_MAX_DEPTH = 3
XGB_SUBSAMPLE = 0.80
XGB_COLSAMPLE_BYTREE = 0.90
XGB_RANDOM_STATE = RANDOM_SEED
XGB_OBJECTIVE = "reg:squarederror"


# ============================================================
# Plot defaults
# ============================================================

PLOT_RUN_ID = 11
PLOT_CV_RATE_PCT = 10

PLOT_STYLE = "bw"          # options: "bw", "color"
PLOT_TIME_MODE = "full"    # options: "full", "window"

PLOT_T0 = 1200.0
PLOT_T1 = 1400.0

FIGSIZE_MAIN = (11, 7)
FIGSIZE_WIDE = (14, 6)

FIGURE_DPI = 300
SAVE_FIGURES = True
SHOW_FIGURES = True

SHOW_GRID = False
SHOW_LEGEND = True

# Vehicle trajectory plotting options
PLOT_VEHICLE_MODE = "all"  # options: "all", "every_nth_global", "every_nth_by_cycle"
PLOT_EVERY_NTH = 5
SHOW_START_END_DOTS = True


# ============================================================
# Utility
# ============================================================

def ensure_project_directories() -> None:
    """Create all standard project directories if they do not already exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLE_DATA_DIR,
        OUTPUT_DIR,
        FIGURES_DIR,
        RESULTS_DIR,
        MODELS_DIR,
        CORRECTED_CURVES_DIR,
        DOC_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)