#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zhang/CNN-LSTM prediction-interval baseline aligned to the proposed protocol.

Why this file exists
====================

The original paper-style CNN-LSTM baseline is a clean-trained prediction-
interval anomaly detector. It does not naturally have a labeled attack-training
stage. For a fair comparison with the proposed protocol, this runner keeps the
CNN-LSTM + prediction-interval methodology, but aligns the data as follows.

Stage 1. Clean forecaster training/calibration
----------------------------------------------
Use exactly the same clean-data source as the proposed entro2 forecaster:

    ./dataset_clean/site5_5.9kw_2016_2019_clean.csv
    ./dataset_clean/site5_7.0kw_2016_2019_clean.csv
    ./dataset_clean/site5_226.8kw_2016_2019_clean.csv
    ./dataset_clean/site5_327.6kw_2016_2019_clean.csv

The first 80% of chronological dates are used to fit the CNN-LSTM forecaster.
The remaining 20% of clean dates are used only to calibrate the prediction-
interval half-width from clean residuals.

Stage 2. Attack training
------------------------
N/A. This baseline is treated as an unsupervised/clean-trained prediction-
interval detector. Labeled attack data are not used for model training or
threshold tuning in the official clean-only setting.

Stage 3. External evaluation
----------------------------
Evaluate directly on the same external SA files used by test_ver7:

    ./dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_5pct.csv
    ./dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_8pct.csv
    ./dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_10pct.csv

Labels are read from the CSV attack_label column and used only for metrics.

Implementation note
===================

This wrapper reuses zhang_multi_clean_attack_tuned.py because it already
implements the single-site CNN-LSTM-w/o-spatial variant, clean residual
interval calibration, external dataset6.0 evaluation, PA metrics, and day-level
metrics. Here we override only the data lists and output directory.
"""

from __future__ import annotations

from pathlib import Path

import zhang_multi_clean_attack_tuned as base


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_DIR


base.TRAIN_CLEAN_CSVS = [
    PROJECT_ROOT / "dataset_clean/site5_5.9kw_2016_2019_clean.csv",
    PROJECT_ROOT / "dataset_clean/site5_7.0kw_2016_2019_clean.csv",
    PROJECT_ROOT / "dataset_clean/site5_226.8kw_2016_2019_clean.csv",
    PROJECT_ROOT / "dataset_clean/site5_327.6kw_2016_2019_clean.csv",
]

# Official clean-only PI baseline: no labeled attack-training data are used.
# If an attack-tuned ablation is needed, use zhang_multi_clean_attack_tuned.py
# or create a separate explicitly named attack-tuned runner.
base.TRAIN_ATTACK_CSVS = []

base.EVAL_CSVS = [
    PROJECT_ROOT / "dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_5pct.csv",
    PROJECT_ROOT / "dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_8pct.csv",
    PROJECT_ROOT / "dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_10pct.csv",
]

base.OUT_DIR = PACKAGE_DIR / "results/zhang_entro_clean_only_aligned"
base.PRED_DIR = base.OUT_DIR / "predictions"
base.MODEL_DIR = base.OUT_DIR / "models"


if __name__ == "__main__":
    base.main()
