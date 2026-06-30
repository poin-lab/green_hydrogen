#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tufail/compare2 supervised ensemble slow-ramp evaluation.

Training protocol is unchanged from compare2_test7_aligned.py:
    - train clean: dataset5.4_attack clean
    - train attack: dataset5.4_attack SA 5% and SA 10%
    - model: RF + MLP + CNN-LSTM + soft-voting ensemble

Only the external evaluation files are changed:
    - dataset6.0_attack slowramp 8%
    - dataset6.0_attack slowramp 10%
"""

from __future__ import annotations

from pathlib import Path

import compare2_test7_aligned as base


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_DIR

base.EVAL_CSVS = [
    PROJECT_ROOT / "dataset6.0_attack/site5_6.0kw_2021_2022_attack_slowramp_8pct.csv",
    PROJECT_ROOT / "dataset6.0_attack/site5_6.0kw_2021_2022_attack_slowramp_10pct.csv",
]
base.OUT_DIR = PACKAGE_DIR / "results/compare2_slowramp_eval"
base.PRED_DIR = base.OUT_DIR / "predictions"
base.MODEL_DIR = base.OUT_DIR / "models"


if __name__ == "__main__":
    base.main()
