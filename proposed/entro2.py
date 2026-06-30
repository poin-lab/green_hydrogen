#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust Solar Forecaster with Fixed GHI Dynamic Features + Residual Centering

목적:
  1) 태양광 발전량(power_ratio) 예측
  2) 공격을 따라가지 않는 비자기회귀 기반 residual 생성
  3) sweep 결과에서 선택된 GHI lag + short_roll_vol 동적 특징을 고정 사용
  4) clean calibration 구간으로 GHI-bin별 residual centering 적용
  5) raw residual과 calibrated/normalized residual을 모두 저장

핵심 설계:
- 발전량 lag(p_lag*) 제거 -> 공격 누설(leakage) 방지
- weather + time 기반 feature 중심
- 단순 dghi=ghi-ghi_lag1만 쓰지 않고, 여러 시간 스케일의 변화량을 고정 사용
- 예측값에 발전량 lag를 넣지 않고, clean residual 중심 보정만 후처리로 적용
"""

import os
import glob
import json
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


# =========================
# 사용자 설정
# =========================
DATA_DIR = "./dataset_clean"
TEST_DIR = "./train_data"

TRAIN_GLOB = os.path.join(DATA_DIR, "*2016_2019_clean.csv")
TEST_GLOB  = os.path.join(TEST_DIR, "*.csv")

OUT_DIR = "./model_output_robust"
PLOT_DIR = os.path.join(OUT_DIR, "plots")

LGBM_MODEL_PATH = os.path.join(OUT_DIR, "robust_forecaster_lgbm.pkl")
CAT_MODEL_PATH  = os.path.join(OUT_DIR, "robust_forecaster_catboost.cbm")
CAT_META_PATH   = os.path.join(OUT_DIR, "robust_forecaster_catboost_meta.pkl")

METRIC_PATH = os.path.join(OUT_DIR, "metrics_compare.txt")
RESID_STATS_PATH = os.path.join(OUT_DIR, "residual_stats.json")
PRED_TEST_CSV = os.path.join(OUT_DIR, "test_predictions.csv")
PRED_TRAIN_CSV = os.path.join(OUT_DIR, "train_predictions.csv")
FIXED_CONFIG_PATH = os.path.join(OUT_DIR, "fixed_forecaster_config.json")

TARGET = "power_ratio"
TS_COL = "timestamp"

VAL_LAST_DAYS = 60
MIN_GHI_FOR_DAY = 20.0
USE_SITE_ONEHOT = True

CLIP_PRED_TO_RANGE = True
PRED_MIN = 0.0
PRED_MAX = 1.2

# 변화율 분모 안정화
EPS_GHI = 1e-6

# =========================
# Fixed feature configuration from previous sweep result
# =========================
# Sweep 결과 1등:
# lag_set=[1,2,3,4,5,6,12,24], dynamic=short_roll_vol
# 이 설정을 고정하고, 이후 실험에서는 동일 forecaster를 사용한다.
FIXED_LAG_SET = [1, 2, 3, 4, 5, 6, 12, 24]

FIXED_DYNAMIC_CFG = {
    "name": "short_roll_vol",
    "diff_lags": [1, 2, 3, 6],
    "use_diff": True,
    "use_abs": True,
    "use_slope": True,
    "use_rate": False,
    "roll_windows": [3, 6],
}

# =========================
# Residual centering / baseline calibration
# =========================
# 주의: test/attack residual을 보고 보정하지 않는다.
# train clean 내부에서 분리한 calibration clean 구간으로만 GHI-bin별 residual center를 추정한다.
USE_RESIDUAL_CENTERING = True
CALIB_LAST_DAYS = 60
CALIB_MIN_ROWS_PER_BIN = 100

# power_ratio denominator 안정화
EPS_PRED = 1e-6

# GHI 구간은 논문에서 말하는 irradiance-conditioned residual calibration과 대응
# MIN_GHI_FOR_DAY 아래는 이미 filter_daytime에서 제거됨
GHI_CALIB_BINS = [MIN_GHI_FOR_DAY, 100.0, 300.0, 600.0, 900.0, np.inf]



# =========================
# Robust feature set
# =========================
BASE_CORE_FEATURES = [
    "ghi",
    "temp",
    "sin_hour",
    "cos_hour",
    "sin_doy",
    "cos_doy",
]

# 최종 모델에서는 optional feature도 기존처럼 포함한다.

OPTIONAL_FEATURES = [
    "wind_speed",
    "humidity",
    "cloud",
    "panel_age",
    "capacity_kw",
    "ghi_roll3",       # CSV에 이미 있으면 사용
    "ghi_roll6",       # CSV에 이미 있으면 사용
    "temp_lag1",
    "temp_lag3",
    "temp_lag6",
]


# =========================
# Lag 후보
# =========================
def make_lag_candidates():
    """
    5분 샘플링 기준:
    1  = 5분 전
    2  = 10분 전
    3  = 15분 전
    6  = 30분 전
    12 = 1시간 전
    24 = 2시간 전
    """
    candidates = []

    # 현재 GHI만 사용
    candidates.append([])

    # 최근 30분 구간을 촘촘하게 확장
    for m in range(1, 7):
        candidates.append(list(range(1, m + 1)))

    # sparse lag 후보
    candidates += [
        [1, 3],
        [1, 3, 6],
        [1, 3, 6, 12],
        [1, 3, 6, 12, 24],
    ]

    # short dense + long context
    candidates += [
        [1, 2, 3, 4, 5, 6, 12],
        [1, 2, 3, 4, 5, 6, 12, 24],
    ]

    uniq = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            uniq.append(c)
            seen.add(key)

    return uniq


LAG_CANDIDATES = [FIXED_LAG_SET]


# =========================
# GHI dynamic 후보
# =========================
def make_ghi_dynamic_candidates():
    """
    기존 코드의 dghi = ghi - ghi_lag1 한 개만 쓰는 구조를 확장한다.

    각 후보가 의미하는 것:
    - diff_lags: ghi - ghi_lag{k}
    - use_slope: (ghi - ghi_lag{k}) / k  -> 5분 단위 평균 기울기
    - use_rate : (ghi - ghi_lag{k}) / (abs(ghi_lag{k}) + eps) -> 상대 변화율
    - use_abs  : abs(ghi - ghi_lag{k}) -> 급변 크기
    - roll_windows: 최근 w포인트의 GHI 평균/표준편차/range/diff volatility
    """
    return [
        {
            "name": "none",
            "diff_lags": [],
            "use_diff": False,
            "use_abs": False,
            "use_slope": False,
            "use_rate": False,
            "roll_windows": [],
        },
        {
            "name": "d1_only",
            "diff_lags": [1],
            "use_diff": True,
            "use_abs": False,
            "use_slope": False,
            "use_rate": False,
            "roll_windows": [],
        },
        {
            "name": "short_diff_1_2_3",
            "diff_lags": [1, 2, 3],
            "use_diff": True,
            "use_abs": True,
            "use_slope": True,
            "use_rate": False,
            "roll_windows": [],
        },
        {
            "name": "dense_30min_diff",
            "diff_lags": [1, 2, 3, 4, 5, 6],
            "use_diff": True,
            "use_abs": True,
            "use_slope": True,
            "use_rate": False,
            "roll_windows": [],
        },
        {
            "name": "multiscale_diff",
            "diff_lags": [1, 3, 6, 12],
            "use_diff": True,
            "use_abs": True,
            "use_slope": True,
            "use_rate": False,
            "roll_windows": [],
        },
        {
            "name": "multiscale_diff_rate",
            "diff_lags": [1, 3, 6, 12, 24],
            "use_diff": True,
            "use_abs": True,
            "use_slope": True,
            "use_rate": True,
            "roll_windows": [],
        },
        {
            "name": "short_roll_vol",
            "diff_lags": [1, 2, 3, 6],
            "use_diff": True,
            "use_abs": True,
            "use_slope": True,
            "use_rate": False,
            "roll_windows": [3, 6],
        },
        {
            "name": "multiscale_roll_vol",
            "diff_lags": [1, 3, 6, 12, 24],
            "use_diff": True,
            "use_abs": True,
            "use_slope": True,
            "use_rate": True,
            "roll_windows": [3, 6, 12],
        },
    ]


GHI_DYNAMIC_CANDIDATES = [FIXED_DYNAMIC_CFG]

# 모든 후보에서 필요한 최대 lag/window를 미리 계산
MAX_GHI_LAG_FROM_LAG_SET = max([max(c) for c in LAG_CANDIDATES if len(c) > 0], default=0)
MAX_GHI_LAG_FROM_DYNAMIC = max(
    [max(cfg["diff_lags"]) for cfg in GHI_DYNAMIC_CANDIDATES if len(cfg["diff_lags"]) > 0],
    default=0,
)
MAX_GHI_ROLL_WINDOW = max(
    [max(cfg["roll_windows"]) for cfg in GHI_DYNAMIC_CANDIDATES if len(cfg["roll_windows"]) > 0],
    default=0,
)
MAX_REQUIRED_GHI_LAG = max(MAX_GHI_LAG_FROM_LAG_SET, MAX_GHI_LAG_FROM_DYNAMIC, MAX_GHI_ROLL_WINDOW)


# =========================
# 유틸
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse


def safe_mad(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def residual_stats(y_true, y_pred):
    """
    Forecaster 자체 성능뿐 아니라 탐지에 중요한 residual 중심/양의 tail도 함께 본다.
    resid = y_true - y_pred 이므로, inflation attack은 양의 residual shift로 나타난다.
    """
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    abs_resid = np.abs(resid)
    pos = resid[resid > 0.0]
    neg = resid[resid < 0.0]

    out = {
        "n": int(len(resid)),
        "resid_mean": float(np.mean(resid)),
        "resid_std": float(np.std(resid)),
        "resid_median": float(np.median(resid)),
        "abs_resid_mean": float(np.mean(abs_resid)),
        "abs_resid_median": float(np.median(abs_resid)),
        "abs_resid_p90": float(np.percentile(abs_resid, 90)),
        "abs_resid_p95": float(np.percentile(abs_resid, 95)),
        "abs_resid_p99": float(np.percentile(abs_resid, 99)),
        "pos_resid_p90": float(np.percentile(pos, 90)) if len(pos) else 0.0,
        "pos_resid_p95": float(np.percentile(pos, 95)) if len(pos) else 0.0,
        "pos_resid_p99": float(np.percentile(pos, 99)) if len(pos) else 0.0,
        "neg_resid_p10": float(np.percentile(neg, 10)) if len(neg) else 0.0,
        "neg_resid_p05": float(np.percentile(neg, 5)) if len(neg) else 0.0,
        "neg_resid_p01": float(np.percentile(neg, 1)) if len(neg) else 0.0,
        "over_pred_rate": float(np.mean(resid < 0.0)),
        "under_pred_rate": float(np.mean(resid > 0.0)),
        "resid_mad": safe_mad(resid),
    }
    return out


def load_many_csv(files, kind="train"):
    if len(files) == 0:
        return None

    dfs = []
    for f in files:
        df = pd.read_csv(f)

        if TS_COL not in df.columns:
            raise ValueError(f"[{kind}] missing '{TS_COL}' in {f}")

        df[TS_COL] = pd.to_datetime(df[TS_COL])

        if "site" not in df.columns:
            base = os.path.basename(f)
            site = os.path.splitext(base)[0]
            df["site"] = site

        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.sort_values(["site", TS_COL]).reset_index(drop=True)
    return all_df


def _group_rolling(series_grouped, window, func_name):
    """groupby rolling 결과 index 정리용 helper"""
    rolled = getattr(series_grouped.rolling(window, min_periods=1), func_name)()
    return rolled.reset_index(level=0, drop=True)


def add_engineered_time_and_ghi_features(df: pd.DataFrame):
    """
    lag/dynamic sweep에 필요한 GHI lag와 변화 특징을 스크립트 내부에서 생성한다.
    CSV에 ghi_lag1/3/6만 있더라도 여기서 lag2/4/5/12/24와 변화율 특징까지 생성한다.
    """
    df = df.copy()

    if "ghi" not in df.columns:
        raise ValueError("column 'ghi' is required")
    if "temp" not in df.columns:
        raise ValueError("column 'temp' is required")
    if TS_COL not in df.columns:
        raise ValueError(f"column '{TS_COL}' is required")

    df[TS_COL] = pd.to_datetime(df[TS_COL])

    if "site" not in df.columns:
        df["site"] = "site_0"

    df = df.sort_values(["site", TS_COL]).reset_index(drop=True)
    df["ghi"] = df["ghi"].astype(float)
    df["temp"] = df["temp"].astype(float)

    # time features
    hour = df[TS_COL].dt.hour + df[TS_COL].dt.minute / 60.0
    doy = df[TS_COL].dt.dayofyear.astype(float)

    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    g = df.groupby("site", group_keys=False)

    # GHI lag features
    for lag in range(1, MAX_REQUIRED_GHI_LAG + 1):
        df[f"ghi_lag{lag}"] = g["ghi"].shift(lag)

    # 여러 시간 스케일의 변화량 / 기울기 / 상대 변화율
    # dghi_1은 기존 dghi와 같은 의미. 호환성을 위해 dghi alias도 유지.
    for lag in range(1, MAX_REQUIRED_GHI_LAG + 1):
        lag_col = f"ghi_lag{lag}"
        diff_col = f"dghi_{lag}"
        abs_col = f"abs_dghi_{lag}"
        slope_col = f"slope_ghi_{lag}"
        rate_col = f"rate_ghi_{lag}"

        df[diff_col] = df["ghi"] - df[lag_col]
        df[abs_col] = df[diff_col].abs()
        df[slope_col] = df[diff_col] / float(lag)
        df[rate_col] = df[diff_col] / (df[lag_col].abs() + EPS_GHI)

    if MAX_REQUIRED_GHI_LAG >= 1:
        df["dghi"] = df["dghi_1"]
    else:
        df["dghi"] = 0.0

    # rolling weather dynamics: 구름/급변 구간에서 도움됨
    # rolling은 현재 시점 포함 + 과거 w-1개라 실시간 사용 가능.
    for w in range(2, MAX_GHI_ROLL_WINDOW + 1):
        roll = g["ghi"].rolling(w, min_periods=1)
        df[f"ghi_roll_mean_{w}"] = roll.mean().reset_index(level=0, drop=True)
        df[f"ghi_roll_std_{w}"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        df[f"ghi_roll_min_{w}"] = roll.min().reset_index(level=0, drop=True)
        df[f"ghi_roll_max_{w}"] = roll.max().reset_index(level=0, drop=True)
        df[f"ghi_roll_range_{w}"] = df[f"ghi_roll_max_{w}"] - df[f"ghi_roll_min_{w}"]

        # 최근 diff의 평균 절대값 = weather edge proxy
        df[f"abs_dghi_1_roll_mean_{w}"] = (
            df.groupby("site")["abs_dghi_1"]
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return df


def build_dynamic_feature_list(df: pd.DataFrame, dynamic_cfg: dict):
    feats = []
    diff_lags = dynamic_cfg.get("diff_lags", [])

    for lag in diff_lags:
        if dynamic_cfg.get("use_diff", False):
            feats.append(f"dghi_{lag}")
        if dynamic_cfg.get("use_abs", False):
            feats.append(f"abs_dghi_{lag}")
        if dynamic_cfg.get("use_slope", False):
            feats.append(f"slope_ghi_{lag}")
        if dynamic_cfg.get("use_rate", False):
            feats.append(f"rate_ghi_{lag}")

    for w in dynamic_cfg.get("roll_windows", []):
        feats += [
            f"ghi_roll_mean_{w}",
            f"ghi_roll_std_{w}",
            f"ghi_roll_range_{w}",
            f"abs_dghi_1_roll_mean_{w}",
        ]

    # 존재 확인 및 중복 제거
    out = []
    for c in feats:
        if c not in df.columns:
            raise ValueError(f"required dynamic feature missing: {c}")
        if c not in out:
            out.append(c)

    return out


def get_available_features(
    df: pd.DataFrame,
    lag_set=None,
    dynamic_cfg=None,
    include_optional=True,
):
    if lag_set is None:
        lag_set = []
    if dynamic_cfg is None:
        dynamic_cfg = GHI_DYNAMIC_CANDIDATES[0]

    feats = []

    for c in BASE_CORE_FEATURES:
        if c not in df.columns:
            raise ValueError(f"required feature missing: {c}")
        feats.append(c)

    # GHI 자체 lag
    for lag in lag_set:
        c = f"ghi_lag{lag}"
        if c not in df.columns:
            raise ValueError(f"required lag feature missing: {c}")
        feats.append(c)

    # GHI 변화 특징
    for c in build_dynamic_feature_list(df, dynamic_cfg):
        if c not in feats:
            feats.append(c)

    if include_optional:
        for c in OPTIONAL_FEATURES:
            if c in df.columns and c not in feats:
                feats.append(c)

    return feats


def filter_daytime(df: pd.DataFrame):
    if "ghi" not in df.columns:
        raise ValueError("column 'ghi' is required for daytime filtering")
    out = df[df["ghi"].astype(float) >= MIN_GHI_FOR_DAY].copy()
    return out


def drop_bad_rows(df: pd.DataFrame, feature_cols):
    need_cols = feature_cols + [TARGET, TS_COL]
    if USE_SITE_ONEHOT and "site" in df.columns:
        need_cols += ["site"]

    out = df.dropna(subset=need_cols).copy()
    return out


def prepare_X(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].copy()

    if USE_SITE_ONEHOT and "site" in df.columns:
        oh = pd.get_dummies(df["site"], prefix="site")
        X = pd.concat([X, oh], axis=1)

    return X


def prepare_xy(df: pd.DataFrame, feature_cols):
    if TARGET not in df.columns:
        raise ValueError(f"missing target column: {TARGET}")

    X = prepare_X(df, feature_cols)
    y = df[TARGET].astype(float).values
    return X, y


def align_columns(X_ref, X_other):
    return X_other.reindex(columns=X_ref.columns, fill_value=0.0)


def clip_pred(y_pred):
    if not CLIP_PRED_TO_RANGE:
        return y_pred
    return np.clip(y_pred, PRED_MIN, PRED_MAX)


def split_fit_calibration(df: pd.DataFrame):
    """
    train clean 안에서 model fitting 구간과 calibration 구간을 분리한다.
    calibration 구간은 residual centering만 추정하는 데 사용하고, test/attack은 절대 사용하지 않는다.
    """
    df = df.sort_values(["site", TS_COL] if "site" in df.columns else [TS_COL]).reset_index(drop=True)

    cutoff = df[TS_COL].max() - pd.Timedelta(days=CALIB_LAST_DAYS)
    fit_df = df[df[TS_COL] <= cutoff].copy()
    calib_df = df[df[TS_COL] > cutoff].copy()

    # 데이터가 짧으면 80/20 시간 split fallback
    if len(fit_df) == 0 or len(calib_df) == 0:
        fit_parts = []
        calib_parts = []
        group_cols = ["site"] if "site" in df.columns else [None]

        if group_cols == [None]:
            groups = [(None, df)]
        else:
            groups = list(df.groupby("site", group_keys=False))

        for _, g in groups:
            g = g.sort_values(TS_COL).reset_index(drop=True)
            n_fit = max(1, int(len(g) * 0.8))
            fit_parts.append(g.iloc[:n_fit].copy())
            calib_parts.append(g.iloc[n_fit:].copy())

        fit_df = pd.concat(fit_parts, ignore_index=True)
        calib_df = pd.concat(calib_parts, ignore_index=True)

    if len(fit_df) == 0 or len(calib_df) == 0:
        raise ValueError("fit/calibration split failed: empty split")

    return fit_df, calib_df


def _json_safe_bins(bins):
    out = []
    for x in bins:
        if np.isinf(x):
            out.append("inf")
        else:
            out.append(float(x))
    return out


def _ghi_bin_labels(df: pd.DataFrame, bins=None):
    if bins is None:
        bins = GHI_CALIB_BINS
    b = pd.cut(
        df["ghi"].astype(float),
        bins=bins,
        right=False,
        include_lowest=True,
    )
    return b.astype(str)


def build_residual_centering(df: pd.DataFrame, y_true, y_pred, model_name="model"):
    """
    clean calibration 구간에서 GHI-bin별 residual median을 추정한다.
    resid = y_true - y_pred.
    추론 때는 centered_resid = resid - b(c_t) 형태로 사용된다.
    예측값 관점에서는 y_pred_cal = y_pred + b(c_t)와 동일하다.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred

    global_bias = float(np.median(resid))
    labels = _ghi_bin_labels(df)

    tmp = pd.DataFrame({
        "ghi_bin": labels.values,
        "resid": resid,
    })

    bin_bias = {}
    bin_count = {}
    for bin_name, g in tmp.groupby("ghi_bin"):
        if bin_name == "nan":
            continue
        bin_count[bin_name] = int(len(g))
        if len(g) >= CALIB_MIN_ROWS_PER_BIN:
            bin_bias[bin_name] = float(np.median(g["resid"].values))
        else:
            # 표본이 적은 bin은 global center 사용
            bin_bias[bin_name] = global_bias

    return {
        "enabled": bool(USE_RESIDUAL_CENTERING),
        "method": "ghi_bin_median_residual_centering",
        "model_name": model_name,
        "bins": list(GHI_CALIB_BINS),
        "bins_json": _json_safe_bins(GHI_CALIB_BINS),
        "global_bias": global_bias,
        "bin_bias": bin_bias,
        "bin_count": bin_count,
        "calib_min_rows_per_bin": int(CALIB_MIN_ROWS_PER_BIN),
        "calib_last_days": int(CALIB_LAST_DAYS),
    }


def apply_residual_centering(df: pd.DataFrame, y_pred, calibrator):
    """
    y_pred_cal = y_pred + b(c_t). 반환값: calibrated prediction, applied bias vector.
    이후 residual은 y_true - y_pred_cal로 계산하면 centered residual이 된다.
    """
    y_pred = np.asarray(y_pred, dtype=float)

    if (not USE_RESIDUAL_CENTERING) or (calibrator is None) or (not calibrator.get("enabled", False)):
        return y_pred, np.zeros_like(y_pred, dtype=float)

    labels = _ghi_bin_labels(df, bins=calibrator["bins"])
    global_bias = float(calibrator["global_bias"])
    bin_bias = calibrator.get("bin_bias", {})

    bias_vec = np.array([
        float(bin_bias.get(str(lbl), global_bias)) for lbl in labels.values
    ], dtype=float)

    y_pred_cal = clip_pred(y_pred + bias_vec)
    return y_pred_cal, bias_vec


def normalized_residual(y_true, y_pred):
    return (np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) / (
        np.asarray(y_pred, dtype=float) + EPS_PRED
    )


def make_ts_df(df: pd.DataFrame):
    cols = [TS_COL]
    if "site" in df.columns:
        cols.append("site")
    if "ghi" in df.columns:
        cols.append("ghi")
    if "temp" in df.columns:
        cols.append("temp")
    return df[cols].copy()


def save_prediction_csv(
    ts_df,
    y_true,
    y_pred_lgbm_raw,
    y_pred_lgbm_cal,
    y_pred_cat_raw,
    y_pred_cat_cal,
    bias_lgbm,
    bias_cat,
    save_path,
):
    out = ts_df.copy()
    out["y_true"] = y_true

    out["y_pred_lgbm_raw"] = y_pred_lgbm_raw
    out["y_pred_lgbm_cal"] = y_pred_lgbm_cal
    out["lgbm_calib_bias"] = bias_lgbm
    out["resid_lgbm_raw"] = out["y_true"] - out["y_pred_lgbm_raw"]
    out["resid_lgbm_cal"] = out["y_true"] - out["y_pred_lgbm_cal"]
    out["resid_norm_lgbm_cal"] = normalized_residual(out["y_true"].values, out["y_pred_lgbm_cal"].values)

    out["y_pred_cat_raw"] = y_pred_cat_raw
    out["y_pred_cat_cal"] = y_pred_cat_cal
    out["cat_calib_bias"] = bias_cat
    out["resid_cat_raw"] = out["y_true"] - out["y_pred_cat_raw"]
    out["resid_cat_cal"] = out["y_true"] - out["y_pred_cat_cal"]
    out["resid_norm_cat_cal"] = normalized_residual(out["y_true"].values, out["y_pred_cat_cal"].values)

    out = out.sort_values(TS_COL).reset_index(drop=True)
    out.to_csv(save_path, index=False, encoding="utf-8-sig")

def plot_and_save(ts_df, y_true, y_pred, prefix):
    dfp = ts_df.copy()
    dfp["y_true"] = y_true
    dfp["y_pred"] = y_pred
    dfp["resid"] = dfp["y_true"] - dfp["y_pred"]
    dfp = dfp.sort_values(TS_COL).reset_index(drop=True)

    if len(dfp) > 40000:
        dfp = dfp.iloc[-40000:].copy()

    dfp["abs_resid"] = np.abs(dfp["resid"])
    dfp["resid_med_roll"] = dfp["resid"].rolling(12, min_periods=1).median()
    dfp["abs_resid_roll"] = dfp["abs_resid"].rolling(12, min_periods=1).median()
    dfp["edge_proxy"] = dfp["resid"].diff().abs().rolling(3, min_periods=1).mean()

    plt.figure(figsize=(14, 4))
    plt.plot(dfp[TS_COL], dfp["y_true"], label="actual", linewidth=0.8)
    plt.plot(dfp[TS_COL], dfp["y_pred"], label="pred", linewidth=0.8)
    plt.title(f"{prefix}: actual vs pred (power_ratio, daytime)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{prefix}_ts_actual_pred.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(14, 3.5))
    plt.plot(dfp[TS_COL], dfp["resid"], linewidth=0.7, label="residual")
    plt.axhline(0, linewidth=1)
    plt.title(f"{prefix}: residual time series")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{prefix}_ts_residual.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(14, 4))
    plt.plot(dfp[TS_COL], dfp["resid_med_roll"], label="rolling median(resid)", linewidth=0.9)
    plt.plot(dfp[TS_COL], dfp["abs_resid_roll"], label="rolling median(abs resid)", linewidth=0.9)
    plt.title(f"{prefix}: rolling residual summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{prefix}_rolling_residual_summary.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(14, 3.5))
    plt.plot(dfp[TS_COL], dfp["edge_proxy"], linewidth=0.8)
    plt.title(f"{prefix}: edge proxy |diff(resid)| rolling mean")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{prefix}_edge_proxy.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(dfp["resid"], bins=120)
    plt.title(f"{prefix}: residual histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{prefix}_hist_residual.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(dfp["y_true"], dfp["y_pred"], s=3, alpha=0.18)
    plt.xlabel("actual ratio")
    plt.ylabel("pred ratio")
    plt.title(f"{prefix}: actual vs pred scatter")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{prefix}_scatter_actual_pred.png"), dpi=200)
    plt.close()


# =========================
# main
# =========================
def main():
    ensure_dir(OUT_DIR)
    ensure_dir(PLOT_DIR)

    train_files = sorted(glob.glob(TRAIN_GLOB))
    test_files  = sorted(glob.glob(TEST_GLOB))

    print("[TRAIN FILES]")
    for f in train_files[:30]:
        print(" -", f)
    if len(train_files) > 30:
        print(f" ... ({len(train_files)} files total)")

    print("\n[TEST FILES]")
    if len(test_files) == 0:
        print(" - (none) => internal time split")
    else:
        for f in test_files[:30]:
            print(" -", f)
        if len(test_files) > 30:
            print(f" ... ({len(test_files)} files total)")

    train = load_many_csv(train_files, kind="train")
    if train is None:
        raise ValueError(f"No train files matched: {TRAIN_GLOB}")

    use_external_test = (len(test_files) > 0)
    test = load_many_csv(test_files, kind="test") if use_external_test else None

    # -------------------------
    # feature engineering
    # -------------------------
    train = add_engineered_time_and_ghi_features(train)
    if use_external_test:
        test = add_engineered_time_and_ghi_features(test)

    # -------------------------
    # daytime filtering
    # -------------------------
    train = filter_daytime(train)
    if use_external_test:
        test = filter_daytime(test)

    # -------------------------
    # train/test 구성
    # -------------------------
    if use_external_test:
        train_base = train.copy()
        test_base = test.copy()
    else:
        train = train.sort_values(TS_COL).reset_index(drop=True)
        cutoff = train[TS_COL].max() - pd.Timedelta(days=VAL_LAST_DAYS)

        train_base = train[train[TS_COL] <= cutoff].copy()
        test_base = train[train[TS_COL] > cutoff].copy()

    if len(train_base) == 0 or len(test_base) == 0:
        raise ValueError("train_base or test_base is empty after split/filtering")

    # -------------------------
    # Fixed GHI lag + dynamic feature configuration
    # 이전 sweep 결과 1등 조합을 고정 사용한다.
    # -------------------------
    best_lag_set = list(FIXED_LAG_SET)
    best_dynamic_cfg = dict(FIXED_DYNAMIC_CFG)

    fixed_config = {
        "fixed_lag_set": best_lag_set,
        "fixed_dynamic_cfg": best_dynamic_cfg,
        "min_ghi_for_day": MIN_GHI_FOR_DAY,
        "use_site_onehot": USE_SITE_ONEHOT,
        "use_residual_centering": USE_RESIDUAL_CENTERING,
        "calib_last_days": CALIB_LAST_DAYS,
        "ghi_calib_bins": _json_safe_bins(GHI_CALIB_BINS),
        "note": "Fixed from previous lag + dynamic sweep result; no sweep is run in this script.",
    }
    with open(FIXED_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(fixed_config, f, ensure_ascii=False, indent=2)
    print("\n[FIXED FORECASTER CONFIG]")
    print(json.dumps(fixed_config, ensure_ascii=False, indent=2))
    print("saved fixed config:", FIXED_CONFIG_PATH)

    # -------------------------
    # fixed 조합으로 최종 feature 결정
    # 최종 모델에서는 optional feature를 기존처럼 포함
    # -------------------------
    feature_cols = get_available_features(
        train_base,
        lag_set=best_lag_set,
        dynamic_cfg=best_dynamic_cfg,
        include_optional=True,
    )

    print("\n[SELECTED FEATURES]")
    print(f"BEST lag_set={best_lag_set}, dynamic={best_dynamic_cfg['name']}")
    for c in feature_cols:
        print(" -", c)

    # -------------------------
    # drop bad rows
    # -------------------------
    train_base = drop_bad_rows(train_base, feature_cols)
    test_base = drop_bad_rows(test_base, feature_cols)

    # -------------------------
    # 최종 fit/calibration/test 구성
    # - calibration은 train clean 내부에서만 분리
    # - test/attack residual은 절대 보정값 추정에 사용하지 않음
    # -------------------------
    fit_base, calib_base = split_fit_calibration(train_base)

    X_train, y_train = prepare_xy(fit_base, feature_cols)
    X_calib, y_calib = prepare_xy(calib_base, feature_cols)
    X_test, y_test = prepare_xy(test_base, feature_cols)
    X_train_all, y_train_all = prepare_xy(train_base, feature_cols)

    X_calib = align_columns(X_train, X_calib)
    X_test = align_columns(X_train, X_test)
    X_train_all = align_columns(X_train, X_train_all)

    train_ts = make_ts_df(train_base)
    test_ts = make_ts_df(test_base)
    calib_ts = make_ts_df(calib_base)

    print("\nShapes:")
    print("X_fit:", X_train.shape, "X_calib:", X_calib.shape, "X_test:", X_test.shape)
    print("y_fit:", y_train.shape, "y_calib:", y_calib.shape, "y_test:", y_test.shape)

    # =========================================================
    # 1) LightGBM
    # =========================================================
    print("\n" + "=" * 80)
    print("LightGBM training")
    print("=" * 80)

    lgbm = lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
    )

    lgbm.fit(
        X_train,
        y_train,
        eval_set=[(X_calib, y_calib)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=True)]
    )

    pred_tr_lgbm_raw = clip_pred(lgbm.predict(X_train_all))
    pred_cal_lgbm_raw = clip_pred(lgbm.predict(X_calib))
    pred_te_lgbm_raw = clip_pred(lgbm.predict(X_test))

    lgbm_calibrator = build_residual_centering(
        calib_base, y_calib, pred_cal_lgbm_raw, model_name="lgbm"
    )

    pred_tr_lgbm_cal, bias_tr_lgbm = apply_residual_centering(train_base, pred_tr_lgbm_raw, lgbm_calibrator)
    pred_cal_lgbm_cal, bias_cal_lgbm = apply_residual_centering(calib_base, pred_cal_lgbm_raw, lgbm_calibrator)
    pred_te_lgbm_cal, bias_te_lgbm = apply_residual_centering(test_base, pred_te_lgbm_raw, lgbm_calibrator)

    r2_tr_l, mae_tr_l, rmse_tr_l = metrics(y_train_all, pred_tr_lgbm_cal)
    r2_te_l, mae_te_l, rmse_te_l = metrics(y_test, pred_te_lgbm_cal)

    joblib.dump(
        {
            "model": lgbm,
            "features": list(X_train.columns),
            "base_features": feature_cols,
            "best_lag_set": best_lag_set,
            "best_dynamic_cfg": best_dynamic_cfg,
            "residual_centering": lgbm_calibrator,
            "fixed_config_path": FIXED_CONFIG_PATH,
            "use_site_onehot": USE_SITE_ONEHOT,
            "min_ghi_for_day": MIN_GHI_FOR_DAY,
            "clip_pred_to_range": CLIP_PRED_TO_RANGE,
            "pred_min": PRED_MIN,
            "pred_max": PRED_MAX,
        },
        LGBM_MODEL_PATH
    )
    print("saved lgbm:", LGBM_MODEL_PATH)

    # =========================================================
    # 2) CatBoost
    # =========================================================
    print("\n" + "=" * 80)
    print("CatBoost training")
    print("=" * 80)

    cat = CatBoostRegressor(
        iterations=8000,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        random_seed=42,
        od_type="Iter",
        od_wait=200,
        verbose=200,
    )

    cat.fit(
        X_train,
        y_train,
        eval_set=(X_calib, y_calib),
        use_best_model=True,
    )

    pred_tr_cat_raw = clip_pred(cat.predict(X_train_all))
    pred_cal_cat_raw = clip_pred(cat.predict(X_calib))
    pred_te_cat_raw = clip_pred(cat.predict(X_test))

    cat_calibrator = build_residual_centering(
        calib_base, y_calib, pred_cal_cat_raw, model_name="catboost"
    )

    pred_tr_cat_cal, bias_tr_cat = apply_residual_centering(train_base, pred_tr_cat_raw, cat_calibrator)
    pred_cal_cat_cal, bias_cal_cat = apply_residual_centering(calib_base, pred_cal_cat_raw, cat_calibrator)
    pred_te_cat_cal, bias_te_cat = apply_residual_centering(test_base, pred_te_cat_raw, cat_calibrator)

    r2_tr_c, mae_tr_c, rmse_tr_c = metrics(y_train_all, pred_tr_cat_cal)
    r2_te_c, mae_te_c, rmse_te_c = metrics(y_test, pred_te_cat_cal)

    cat.save_model(CAT_MODEL_PATH)
    joblib.dump(
        {
            "features": list(X_train.columns),
            "base_features": feature_cols,
            "best_lag_set": best_lag_set,
            "best_dynamic_cfg": best_dynamic_cfg,
            "residual_centering": cat_calibrator,
            "fixed_config_path": FIXED_CONFIG_PATH,
            "use_site_onehot": USE_SITE_ONEHOT,
            "min_ghi_for_day": MIN_GHI_FOR_DAY,
            "clip_pred_to_range": CLIP_PRED_TO_RANGE,
            "pred_min": PRED_MIN,
            "pred_max": PRED_MAX,
        },
        CAT_META_PATH,
    )
    print("saved catboost:", CAT_MODEL_PATH)
    print("saved cat meta:", CAT_META_PATH)

    # =========================================================
    # 결과 요약
    # =========================================================
    train_resid_lgbm_raw_stats = residual_stats(y_train_all, pred_tr_lgbm_raw)
    test_resid_lgbm_raw_stats  = residual_stats(y_test, pred_te_lgbm_raw)
    calib_resid_lgbm_raw_stats = residual_stats(y_calib, pred_cal_lgbm_raw)

    train_resid_lgbm_stats = residual_stats(y_train_all, pred_tr_lgbm_cal)
    test_resid_lgbm_stats  = residual_stats(y_test, pred_te_lgbm_cal)
    calib_resid_lgbm_stats = residual_stats(y_calib, pred_cal_lgbm_cal)

    train_resid_cat_raw_stats = residual_stats(y_train_all, pred_tr_cat_raw)
    test_resid_cat_raw_stats  = residual_stats(y_test, pred_te_cat_raw)
    calib_resid_cat_raw_stats = residual_stats(y_calib, pred_cal_cat_raw)

    train_resid_cat_stats = residual_stats(y_train_all, pred_tr_cat_cal)
    test_resid_cat_stats  = residual_stats(y_test, pred_te_cat_cal)
    calib_resid_cat_stats = residual_stats(y_calib, pred_cal_cat_cal)

    lines = []
    lines.append("=== Robust Solar Forecaster with Fixed GHI Dynamic Features + Residual Centering ===")
    lines.append(f"MIN_GHI_FOR_DAY={MIN_GHI_FOR_DAY}")
    lines.append(f"USE_SITE_ONEHOT={USE_SITE_ONEHOT}")
    lines.append(f"CLIP_PRED_TO_RANGE={CLIP_PRED_TO_RANGE}")
    lines.append(f"USE_RESIDUAL_CENTERING={USE_RESIDUAL_CENTERING}")
    lines.append(f"CALIB_LAST_DAYS={CALIB_LAST_DAYS}")
    lines.append(f"GHI_CALIB_BINS={_json_safe_bins(GHI_CALIB_BINS)}")
    lines.append(f"BEST_LAG_SET={best_lag_set}")
    lines.append(f"BEST_DYNAMIC_NAME={best_dynamic_cfg['name']}")
    lines.append(f"BEST_DYNAMIC_CFG={json.dumps(best_dynamic_cfg, ensure_ascii=False, sort_keys=True)}")
    lines.append(f"FIXED_CONFIG_PATH={FIXED_CONFIG_PATH}")
    lines.append("")
    lines.append("Selected Features:")
    for c in feature_cols:
        lines.append(f" - {c}")
    lines.append("")
    lines.append("=== LightGBM calibrated ===")
    lines.append(f"TRAIN: R2={r2_tr_l:.4f}  MAE={mae_tr_l*100:.3f}%  RMSE={rmse_tr_l*100:.3f}%")
    lines.append(f"TEST : R2={r2_te_l:.4f}  MAE={mae_te_l*100:.3f}%  RMSE={rmse_te_l*100:.3f}%")
    lines.append(
        f"CALIB raw mean={calib_resid_lgbm_raw_stats['resid_mean']*100:.3f}%  "
        f"cal mean={calib_resid_lgbm_stats['resid_mean']*100:.3f}%  "
        f"raw over={calib_resid_lgbm_raw_stats['over_pred_rate']:.3f}  "
        f"cal over={calib_resid_lgbm_stats['over_pred_rate']:.3f}"
    )
    lines.append(
        f"TEST raw mean={test_resid_lgbm_raw_stats['resid_mean']*100:.3f}%  "
        f"cal mean={test_resid_lgbm_stats['resid_mean']*100:.3f}%  "
        f"cal PosP95={test_resid_lgbm_stats['pos_resid_p95']*100:.3f}%  "
        f"cal AbsP95={test_resid_lgbm_stats['abs_resid_p95']*100:.3f}%"
    )
    lines.append("")
    lines.append("=== CatBoost calibrated ===")
    lines.append(f"TRAIN: R2={r2_tr_c:.4f}  MAE={mae_tr_c*100:.3f}%  RMSE={rmse_tr_c*100:.3f}%")
    lines.append(f"TEST : R2={r2_te_c:.4f}  MAE={mae_te_c*100:.3f}%  RMSE={rmse_te_c*100:.3f}%")
    lines.append(
        f"CALIB raw mean={calib_resid_cat_raw_stats['resid_mean']*100:.3f}%  "
        f"cal mean={calib_resid_cat_stats['resid_mean']*100:.3f}%  "
        f"raw over={calib_resid_cat_raw_stats['over_pred_rate']:.3f}  "
        f"cal over={calib_resid_cat_stats['over_pred_rate']:.3f}"
    )
    lines.append(
        f"TEST raw mean={test_resid_cat_raw_stats['resid_mean']*100:.3f}%  "
        f"cal mean={test_resid_cat_stats['resid_mean']*100:.3f}%  "
        f"cal PosP95={test_resid_cat_stats['pos_resid_p95']*100:.3f}%  "
        f"cal AbsP95={test_resid_cat_stats['abs_resid_p95']*100:.3f}%"
    )

    report = "\n".join(lines)
    print("\n" + report)

    with open(METRIC_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("saved metrics:", METRIC_PATH)

    # residual stats json
    resid_stats_all = {
        "config": {
            "min_ghi_for_day": MIN_GHI_FOR_DAY,
            "use_site_onehot": USE_SITE_ONEHOT,
            "clip_pred_to_range": CLIP_PRED_TO_RANGE,
            "pred_min": PRED_MIN,
            "pred_max": PRED_MAX,
            "feature_cols": feature_cols,
            "best_lag_set": best_lag_set,
            "best_dynamic_cfg": best_dynamic_cfg,
            "fixed_config_path": FIXED_CONFIG_PATH,
            "fixed_lag_set": FIXED_LAG_SET,
            "fixed_dynamic_cfg": FIXED_DYNAMIC_CFG,
            "use_residual_centering": USE_RESIDUAL_CENTERING,
            "calib_last_days": CALIB_LAST_DAYS,
            "calib_min_rows_per_bin": CALIB_MIN_ROWS_PER_BIN,
            "ghi_calib_bins": _json_safe_bins(GHI_CALIB_BINS),
        },
        "lgbm": {
            "calibrator": lgbm_calibrator,
            "train_raw": train_resid_lgbm_raw_stats,
            "test_raw": test_resid_lgbm_raw_stats,
            "calib_raw": calib_resid_lgbm_raw_stats,
            "train_calibrated": train_resid_lgbm_stats,
            "test_calibrated": test_resid_lgbm_stats,
            "calib_calibrated": calib_resid_lgbm_stats,
        },
        "catboost": {
            "calibrator": cat_calibrator,
            "train_raw": train_resid_cat_raw_stats,
            "test_raw": test_resid_cat_raw_stats,
            "calib_raw": calib_resid_cat_raw_stats,
            "train_calibrated": train_resid_cat_stats,
            "test_calibrated": test_resid_cat_stats,
            "calib_calibrated": calib_resid_cat_stats,
        },
    }
    with open(RESID_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(resid_stats_all, f, ensure_ascii=False, indent=2, default=str)
    print("saved residual stats:", RESID_STATS_PATH)

    # =========================================================
    # 예측 결과 저장
    # =========================================================
    save_prediction_csv(
        train_ts, y_train_all,
        pred_tr_lgbm_raw, pred_tr_lgbm_cal,
        pred_tr_cat_raw, pred_tr_cat_cal,
        bias_tr_lgbm, bias_tr_cat,
        PRED_TRAIN_CSV,
    )
    save_prediction_csv(
        test_ts, y_test,
        pred_te_lgbm_raw, pred_te_lgbm_cal,
        pred_te_cat_raw, pred_te_cat_cal,
        bias_te_lgbm, bias_te_cat,
        PRED_TEST_CSV,
    )
    print("saved train predictions:", PRED_TRAIN_CSV)
    print("saved test predictions :", PRED_TEST_CSV)

    # =========================================================
    # 플롯 저장
    # =========================================================
    plot_and_save(train_ts, y_train_all, pred_tr_lgbm_cal, prefix="lgbm_train_calibrated")
    plot_and_save(test_ts,  y_test,      pred_te_lgbm_cal, prefix="lgbm_test_calibrated")

    plot_and_save(train_ts, y_train_all, pred_tr_cat_cal, prefix="cat_train_calibrated")
    plot_and_save(test_ts,  y_test,      pred_te_cat_cal, prefix="cat_test_calibrated")

    print("plots saved:", PLOT_DIR)
    print("DONE.")


if __name__ == "__main__":
    main()
