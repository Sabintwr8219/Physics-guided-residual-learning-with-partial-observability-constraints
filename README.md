# Physics-guided residual learning with partial observability constraints

## Project Overview

This project develops a hybrid connected-vehicle-constrained cumulative curve framework for queue reconstruction at a signalized intersection. The method starts from a baseline cumulative-count back-of-queue curve and improves it using connected vehicle queue-join information and machine learning-based residual prediction.

The data are extracted from a calibrated PTV VISSIM simulation model of a real-world intersection corridor. Vehicle trajectories are processed to generate detector-based cumulative arrival and departure curves, ground-truth queue-join events, connected vehicle anchor points, and corrected queue reconstruction curves.

The final objective is to compare the baseline cumulative-count method with the CV-constrained machine learning correction using evaluation metrics such as MAE, RMSE, area between curves, and maximum absolute error.


## Repository Structure

```text
Hybrid_CV_Queue_Reconstruction/
│
├── data/
│   ├── raw_data/
│   │   └── vissim_results/
│   ├── processed_data/
│   └── sample_data/
│
├── output/
│   ├── figures/
│   ├── results/
│   ├── models/
│   └── corrected_curves/
│
├── src/
│   ├── config.py
│   ├── preprocess_vissim.py
│   ├── ground_truth_queue.py
│   ├── cumulative_count_theory.py
│   ├── cv_ml_pipeline.py
│   ├── cv_anchor_correction.py
│   ├── evaluation.py
│   ├── plot.py
│   └── run_pipeline.py
|   └── robustness_check.py
│
├── requirements.txt
└── README.md
```

## Workflow

```text
Raw VISSIM FZP/LSA Files
        ↓
Preprocess VISSIM Data
        ↓
Filtered Northbound Trajectories
        ↓
Ground-Truth Queue Join Detection
        ↓
Cumulative Count Theory: A, D, V, and Baseline B Curves
        ↓
Connected Vehicle Allocation
        ↓
Feature Engineering and XGBoost Residual Prediction
        ↓
CV Anchor-Based Segment Correction
        ↓
Evaluation and Plotting
        ↓
Robustness Check
```

## Scripts

The `config.py` file contains the main project settings, including file paths, run IDs, detector geometry, CV penetration rates, train/test split, model parameters, and shared constants.

The `preprocess_vissim.py` file parses the raw VISSIM FZP and LSA files, creates master trajectory and signal files, and filters the northbound through-vehicle trajectories used in later steps.

The `ground_truth_queue.py` file generates the ground-truth queue-join events and queue-length files from the filtered trajectory data. It also saves plot-ready GT cumulative curve data.

The `cumulative_count_theory.py` file applies cumulative-count theory to build the A, D, V, and baseline B curves. It saves detector event times, baseline curve files, and plot-ready cumulative curve data.

The `cv_ml_pipeline.py` file creates CV allocation files for all penetration rates, builds time-grid and event-level features, trains one global XGBoost residual model, and saves raw ML predictions.

The `cv_anchor_correction.py` file applies the CV anchor-based segment correction. It uses CV points and boundary anchors to rescale the raw ML prediction curve and save the final corrected curve files.

The `evaluation.py` file evaluates the baseline and corrected curves against the ground-truth curve using MAE, RMSE, area between curves, and maximum absolute error.

The `robustness_check.py` file performs additional robustness testing after the main evaluation. It checks model stability under reduced training-data size and controlled feature noise, then saves two summary tables in `output/results/`.

The `plot.py` file generates selected figures directly from saved CSV files. Plot type, color style, zoom window, selected run, and selected CV penetration rate can be controlled from the setup section at the top of the file.

The `run_pipeline.py` file runs the full calculation workflow in the correct order. Plotting is kept separate and should be run using `plot.py`.

## Data

The raw data consist of vehicle trajectory and signal timing outputs extracted from PTV VISSIM after modeling a real-world signalized intersection corridor.

The raw VISSIM files should be placed in:

```text
data/raw_data/vissim_results/
```


## Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To run the full calculation pipeline, use:

```bash
python src/run_pipeline.py
```

To run each stage separately, use:

```bash
python src/preprocess_vissim.py
python src/ground_truth_queue.py
python src/cumulative_count_theory.py
python src/cv_ml_pipeline.py
python src/cv_anchor_correction.py
python src/evaluation.py
```

To generate selected plots, edit the plot configuration section at the top of `src/plot.py`, then run:

```bash
python src/plot.py
```

## Expected Outputs

After running the pipeline, the main processed files are saved in `data/processed_data/`. These include the filtered trajectory file, GT queue-join files, cumulative curve files, CV allocation files, feature files, and raw prediction files.

Final corrected curve files are saved in:

```text
output/corrected_curves/
```
Robustness check outputs are saved in:

```text
output/results/robustness_data_size.csv
output/results/robustness_noise.csv

## Notes and Limitations

The full VISSIM trajectory and signal files are not included in this repository because they may be large. The repository is designed so that raw data can be placed locally in the expected folder structure before running the pipeline.

The train/test split is performed strictly by simulation run to avoid vehicle-level leakage. Runs 005–010 are used for training, and runs 011–014 are used for testing.

The CV penetration rates evaluated in this project are 1%, 2%, 5%, 10%, 20%, 50%, and 100%.



