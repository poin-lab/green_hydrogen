#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated proposed-model pipeline: entro2 forecaster + test_ver7 detector.

Actual data used by the proposed model
======================================

Stage 1. Clean forecaster training (entro2.py)
----------------------------------------------
Source glob:
    ./dataset_clean/*2016_2019_clean.csv

Actual matched files:
    ./dataset_clean/site5_5.9kw_2016_2019_clean.csv
    ./dataset_clean/site5_7.0kw_2016_2019_clean.csv
    ./dataset_clean/site5_226.8kw_2016_2019_clean.csv
    ./dataset_clean/site5_327.6kw_2016_2019_clean.csv

How entro2.py splits them:
    - ./train_data/*.csv is the external-test glob, but there is currently no
      ./train_data directory/file in this project.
    - Therefore entro2.py uses an internal time split.
    - The last 60 days of the clean data become the internal forecaster test set.
    - The earlier clean data become train_base.
    - Inside train_base, the last 60 days are used as clean residual-calibration
      data for residual centering.
    - The remaining earlier train_base rows are used to fit the LightGBM and
      CatBoost normal-generation forecasters.

Stage 1 output used by the detector:
    ./model_output_robust/robust_forecaster_lgbm.pkl

Stage 2. Attack detector training/evaluation (test_ver7.py)
-----------------------------------------------------------
Forecaster loaded by test_ver7.py:
    ./model_output_robust/robust_forecaster_lgbm.pkl

Detector attack-training files:
    ./dataset5.4_attack/site5_5.4kw_2018_2020_attack_sa_5pct.csv
    ./dataset5.4_attack/site5_5.4kw_2018_2020_attack_sa_10pct.csv

Detector training split:
    - test_ver7.py extracts residual/window/GHI-zone features from the two
      attack-training files above.
    - Dates are sorted chronologically.
    - The first 80% of dates are used for detector fitting.
    - The remaining 20% of dates are used for detector validation/threshold
      checking.

Detector external-evaluation files:
    ./dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_5pct.csv
    ./dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_8pct.csv
    ./dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_10pct.csv

Stage 2 output:
    ./detector_model_fadre_v3/metrics_iter.json

This runner keeps the existing implementation files intact and executes them
in the same order used by the proposed model:

1. Train the clean-data normal-generation forecaster with entro2.py.
2. Train/evaluate the attack detector with test_ver7.py using that forecaster.

Run from anywhere:
    python proposed/proposed_entro2_test7_pipeline.py

Useful options:
    --skip-forecaster   Reuse ./model_output_robust/robust_forecaster_lgbm.pkl
    --skip-detector     Train only the forecaster
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

ENTRO2_PATH = SCRIPT_DIR / "entro2.py"
TEST_VER7_PATH = SCRIPT_DIR / "test_ver7.py"

FORECASTER_MODEL = PROJECT_ROOT / "model_output_robust/robust_forecaster_lgbm.pkl"
DETECTOR_METRICS = PROJECT_ROOT / "detector_model_fadre_v3/metrics_iter.json"


def _fmt_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def log(message: str) -> None:
    print(f"[pipeline] {message}", flush=True)


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def run_entro2() -> None:
    require_file(ENTRO2_PATH, "entro2.py")
    log("stage 1/2: training clean forecaster with entro2.py")
    started = time.time()
    entro2 = import_module_from_path("proposed_entro2_integrated", ENTRO2_PATH)
    entro2.main()
    require_file(FORECASTER_MODEL, "trained LGBM forecaster")
    log(f"stage 1/2 done in {_fmt_elapsed(time.time() - started)}")


def run_test_ver7() -> None:
    require_file(TEST_VER7_PATH, "test_ver7.py")
    require_file(FORECASTER_MODEL, "trained LGBM forecaster")
    log("stage 2/2: training/evaluating detector with test_ver7.py")
    started = time.time()
    test_ver7 = import_module_from_path("proposed_test_ver7_integrated", TEST_VER7_PATH)
    test_ver7.main()
    require_file(DETECTOR_METRICS, "detector metrics")
    log(f"stage 2/2 done in {_fmt_elapsed(time.time() - started)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the proposed entro2 + test_ver7 pipeline end to end."
    )
    parser.add_argument(
        "--skip-forecaster",
        action="store_true",
        help="Reuse the existing ./model_output_robust/robust_forecaster_lgbm.pkl.",
    )
    parser.add_argument(
        "--skip-detector",
        action="store_true",
        help="Only run entro2.py and stop before detector training/evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    os.chdir(PROJECT_ROOT)
    log(f"project root: {PROJECT_ROOT}")
    log("clean forecaster train source: ./dataset_clean/*2016_2019_clean.csv")
    log("detector attack train source: ./dataset5.4_attack/*2018_2020_attack_sa_5pct/10pct.csv")
    log("detector eval source: ./dataset6.0_attack/*2021_2022_attack_sa_5pct/8pct/10pct.csv")

    if args.skip_forecaster:
        require_file(FORECASTER_MODEL, "existing LGBM forecaster")
        log(f"stage 1/2 skipped: using {FORECASTER_MODEL}")
    else:
        run_entro2()

    if args.skip_detector:
        log("stage 2/2 skipped by request")
    else:
        run_test_ver7()

    log(f"pipeline finished in {_fmt_elapsed(time.time() - started)}")
    log(f"forecaster: {FORECASTER_MODEL}")
    if DETECTOR_METRICS.exists():
        log(f"detector metrics: {DETECTOR_METRICS}")


if __name__ == "__main__":
    main()
