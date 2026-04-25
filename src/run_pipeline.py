"""
Run the full project calculation pipeline.

This script executes the main processing steps in order:

1. preprocess_vissim.py
2. ground_truth_queue.py
3. cumulative_count_theory.py
4. cv_ml_pipeline.py
5. cv_anchor_correction.py
6. evaluation.py

Plotting is intentionally not included here. Run plot.py separately when
you want to generate selected figures.
"""

from __future__ import annotations

import time

from config import ensure_project_directories

import preprocess_vissim
import ground_truth_queue
import cumulative_count_theory
import cv_ml_pipeline
import cv_anchor_correction
import evaluation


# ============================================================
# Pipeline controls
# Set True/False depending on which stages you want to run.
# ============================================================

RUN_PREPROCESS_VISSIM = True
RUN_GROUND_TRUTH = True
RUN_CUMULATIVE_COUNT = True
RUN_CV_ML_PIPELINE = True
RUN_CV_ANCHOR_CORRECTION = True
RUN_EVALUATION = True


# ============================================================
# Helpers
# ============================================================

def run_stage(stage_name: str, stage_function) -> None:
    """Run one pipeline stage and print timing information."""
    print("\n" + "=" * 90)
    print(f"STARTING: {stage_name}")
    print("=" * 90)

    start_time = time.time()
    stage_function()
    elapsed = time.time() - start_time

    print("=" * 90)
    print(f"FINISHED: {stage_name} | elapsed = {elapsed / 60:.2f} min")
    print("=" * 90)


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Run selected project stages."""
    ensure_project_directories()

    if RUN_PREPROCESS_VISSIM:
        run_stage(
            "Preprocess VISSIM files and filter trajectories",
            preprocess_vissim.main,
        )

    if RUN_GROUND_TRUTH:
        run_stage(
            "Generate GT queue join and queue length files",
            ground_truth_queue.main,
        )

    if RUN_CUMULATIVE_COUNT:
        run_stage(
            "Generate A/D/V/B cumulative-count theory files",
            cumulative_count_theory.main,
        )

    if RUN_CV_ML_PIPELINE:
        run_stage(
            "Run CV allocation, feature engineering, and raw ML prediction",
            cv_ml_pipeline.main,
        )

    if RUN_CV_ANCHOR_CORRECTION:
        run_stage(
            "Apply CV-anchor segment-wise correction",
            cv_anchor_correction.main,
        )

    if RUN_EVALUATION:
        run_stage(
            "Evaluate baseline and corrected curves",
            evaluation.main,
        )

    print("\nFull selected pipeline completed.")


if __name__ == "__main__":
    main()