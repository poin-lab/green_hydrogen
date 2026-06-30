#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proposed detector slow-ramp evaluation.

Training protocol is unchanged from test_ver7.py:
    - forecaster: ./model_output_robust/robust_forecaster_lgbm.pkl
    - detector train: dataset5.4_attack SA 5% and SA 10%

Only the external evaluation files are changed:
    - dataset6.0_attack slowramp 8%
    - dataset6.0_attack slowramp 10%
"""

from __future__ import annotations

import test_ver7 as base


base.EVAL_CSVS_SA = [
    "./dataset6.0_attack/site5_6.0kw_2021_2022_attack_slowramp_8pct.csv",
    "./dataset6.0_attack/site5_6.0kw_2021_2022_attack_slowramp_10pct.csv",
]
base.OUT_DIR = "./detector_model_fadre_v3_slowramp_eval"


if __name__ == "__main__":
    base.main()
