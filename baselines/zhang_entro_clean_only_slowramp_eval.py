#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN-LSTM PI single-site aligned slow-ramp evaluation.

Training protocol is unchanged from zhang_entro_clean_only_aligned.py:
    - clean train/calibration: same dataset_clean files as entro2
    - attack training: none
    - method: CNN-LSTM forecaster + clean residual prediction interval

Only the external evaluation files are changed:
    - dataset6.0_attack slowramp 8%
    - dataset6.0_attack slowramp 10%
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
base.TRAIN_ATTACK_CSVS = []
base.EVAL_CSVS = [
    PROJECT_ROOT / "dataset6.0_attack/site5_6.0kw_2021_2022_attack_slowramp_8pct.csv",
    PROJECT_ROOT / "dataset6.0_attack/site5_6.0kw_2021_2022_attack_slowramp_10pct.csv",
]
base.OUT_DIR = PACKAGE_DIR / "results/zhang_entro_clean_only_slowramp_eval"
base.PRED_DIR = base.OUT_DIR / "predictions"
base.MODEL_DIR = base.OUT_DIR / "models"


if __name__ == "__main__":
    base.main()
