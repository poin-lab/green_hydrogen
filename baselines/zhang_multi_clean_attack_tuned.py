#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zhang-style CNN-LSTM prediction-interval anomaly baseline with more clean training data.

Preserved methodology:
  - train a deterministic CNN-LSTM forecaster on clean PV data
  - calibrate a prediction interval from clean validation residuals
  - flag anomaly when observed power_ratio is outside the interval

Aligned experiment setting:
  - train/calibration clean data: dataset_clean multi-site clean + dataset5.4_attack clean
  - external evaluation: dataset6.0_attack SA 5%, 8%, 10%
  - labels: attack_label > 0.2
  - metrics: same binary point/day metrics style as test_ver7
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_DIR

TRAIN_CLEAN_CSVS = [
    PROJECT_ROOT / "dataset_clean/site5_5.9kw_2016_2019_clean.csv",
    PROJECT_ROOT / "dataset_clean/site5_7.0kw_2016_2019_clean.csv",
    PROJECT_ROOT / "dataset_clean/site5_226.8kw_2016_2019_clean.csv",
    PROJECT_ROOT / "dataset_clean/site5_327.6kw_2016_2019_clean.csv",
    PROJECT_ROOT / "dataset5.4_attack/site5_5.4kw_2018_2020_clean.csv",
]
TRAIN_ATTACK_CSVS = [
    PROJECT_ROOT / "dataset5.4_attack/site5_5.4kw_2018_2020_attack_sa_5pct.csv",
    PROJECT_ROOT / "dataset5.4_attack/site5_5.4kw_2018_2020_attack_sa_10pct.csv",
]
EVAL_CSVS = [
    PROJECT_ROOT / "dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_5pct.csv",
    PROJECT_ROOT / "dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_8pct.csv",
    PROJECT_ROOT / "dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_10pct.csv",
]

OUT_DIR = PACKAGE_DIR / "results/zhang_multi_clean_attack_tuned"
PRED_DIR = OUT_DIR / "predictions"
MODEL_DIR = OUT_DIR / "models"

RANDOM_STATE = 42
FEATURE_COLUMNS = ["power_ratio", "ghi", "temp"]
TARGET_COLUMN = "power_ratio"
MIN_GHI_VALID = 200.0
TRAIN_RATIO = 0.80


class Timer:
    def __init__(self) -> None:
        self.start = time.time()

    def log(self, msg: str) -> None:
        print(f"[+{time.time() - self.start:7.1f}s] {msg}", flush=True)


class CnnLstmForecaster(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(n_features, 64, kernel_size=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=1)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=100, batch_first=True)
        self.dropout = nn.Dropout(0.10)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)
        self.head = nn.Linear(50, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, features]
        z = x.transpose(1, 2)
        z = self.pool(self.relu(self.conv(z)))
        z = z.transpose(1, 2)
        z, _ = self.lstm1(z)
        z = self.dropout(z)
        z, _ = self.lstm2(z)
        return self.head(z[:, -1, :]).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zhang-style baseline aligned to test_ver7")
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--horizon-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument(
        "--threshold-source",
        choices=["clean_quantile", "attack_validation"],
        default="clean_quantile",
        help="clean_quantile preserves the original PI baseline; attack_validation tunes the residual threshold on labeled train attacks.",
    )
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means use all")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="0 means use all")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "site" not in df.columns:
        df["site"] = path.stem
    if "power_ratio" not in df.columns and {"power", "capacity_kw"}.issubset(df.columns):
        df["power_ratio"] = df["power"] / df["capacity_kw"]
    if "attack_label" not in df.columns:
        df["attack_label"] = 0
    df["is_attack"] = (df["attack_label"].astype(float) > 0.2).astype(int)
    df = df.sort_values(["site", "timestamp"]).reset_index(drop=True)
    return df


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["power_ratio", "ghi", "temp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[(df["power_ratio"] < 0) | (df["power_ratio"] > 1.5), "power_ratio"] = np.nan
    df.loc[(df["ghi"] < 0) | (df["ghi"] > 1600), "ghi"] = np.nan
    df.loc[(df["temp"] < -50) | (df["temp"] > 80), "temp"] = np.nan
    df["date"] = df["timestamp"].dt.date
    df = df[df["ghi"].astype(float) >= MIN_GHI_VALID]
    df = df.dropna(subset=FEATURE_COLUMNS + ["is_attack"]).reset_index(drop=True)
    return df


def split_clean_by_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["date"].unique())
    split_idx = int(len(dates) * TRAIN_RATIO)
    train_dates = set(dates[:split_idx])
    return (
        df[df["date"].isin(train_dates)].reset_index(drop=True),
        df[~df["date"].isin(train_dates)].reset_index(drop=True),
    )


def load_attack_train_frame() -> pd.DataFrame:
    frames = [clean_frame(load_csv(path)) for path in TRAIN_ATTACK_CSVS]
    return pd.concat(frames, ignore_index=True).sort_values(["site", "timestamp"]).reset_index(drop=True)


def load_clean_train_frame() -> pd.DataFrame:
    frames = []
    for path in TRAIN_CLEAN_CSVS:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(clean_frame(load_csv(path)))
    return pd.concat(frames, ignore_index=True).sort_values(["site", "timestamp"]).reset_index(drop=True)


def build_sequences(df: pd.DataFrame, seq_len: int, horizon_steps: int, max_samples: int = 0):
    x_parts: List[np.ndarray] = []
    y_parts: List[float] = []
    label_parts: List[int] = []
    meta_parts: List[dict] = []

    for _, g in df.groupby(["site", "date"], sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        target_idx_start = seq_len - 1 + horizon_steps
        if len(g) <= target_idx_start:
            continue
        features = g[FEATURE_COLUMNS].astype(float).to_numpy(dtype=np.float32)
        y = g[TARGET_COLUMN].astype(float).to_numpy(dtype=np.float32)
        labels = g["is_attack"].astype(int).to_numpy()
        for target_idx in range(target_idx_start, len(g)):
            end_idx = target_idx - horizon_steps
            start_idx = end_idx - seq_len + 1
            x_parts.append(features[start_idx : end_idx + 1])
            y_parts.append(float(y[target_idx]))
            label_parts.append(int(labels[target_idx]))
            meta_parts.append({
                "timestamp": g.loc[target_idx, "timestamp"],
                "date": g.loc[target_idx, "date"],
                "ghi": float(g.loc[target_idx, "ghi"]),
                "is_attack": int(labels[target_idx]),
            })

    if not x_parts:
        raise ValueError("No sequences generated")

    X = np.asarray(x_parts, dtype=np.float32)
    y = np.asarray(y_parts, dtype=np.float32)
    labels = np.asarray(label_parts, dtype=np.int64)
    meta = pd.DataFrame(meta_parts)

    if max_samples and len(y) > max_samples:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = np.sort(rng.choice(len(y), size=max_samples, replace=False))
        X, y, labels, meta = X[idx], y[idx], labels[idx], meta.iloc[idx].reset_index(drop=True)
    return X, y, labels, meta


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    n, w, f = X_train.shape
    return StandardScaler().fit(X_train.reshape(n * w, f))


def transform_sequences(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    n, w, f = X.shape
    return scaler.transform(X.reshape(n * w, f)).reshape(n, w, f).astype(np.float32)


def train_model(model, X_train, y_train, X_val, y_val, args, device, timer):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_x = torch.from_numpy(X_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    patience = 7
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).detach().cpu())
        timer.log(f"epoch {epoch:03d}/{args.epochs} train_mse={np.mean(losses):.6f} val_mse={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, X, batch_size, device) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            p = model(xb.to(device)).detach().cpu().numpy()
            preds.append(p)
    return np.clip(np.concatenate(preds), 0.0, 1.2).astype(np.float32)


def calibrate_interval(y_val: np.ndarray, pred_val: np.ndarray, confidence: float) -> float:
    abs_resid = np.abs(y_val - pred_val)
    threshold = float(np.quantile(abs_resid, confidence))
    return max(threshold, 1e-4)


def metric_from_scores(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_f1_residual_threshold(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, dict]:
    candidates = np.unique(np.concatenate([
        np.linspace(0.001, 0.40, 160),
        np.quantile(scores, np.linspace(0.01, 0.99, 160)),
    ]))
    best_t = float(np.quantile(scores, 0.70))
    best_m = metric_from_scores(labels, scores, best_t)
    for t in candidates:
        m = metric_from_scores(labels, scores, float(t))
        if (m["f1"], m["recall"], -m["fpr"]) > (best_m["f1"], best_m["recall"], -best_m["fpr"]):
            best_t = float(t)
            best_m = m
    return max(best_t, 1e-4), best_m


def metrics_from_interval(y_true, pred, labels, meta, half_width):
    anomaly = (np.abs(y_true - pred) > half_width).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, anomaly, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    day = meta[["date", "is_attack"]].copy()
    day["pred"] = anomaly
    by_day = day.groupby("date").agg(is_attack=("is_attack", "max"), pred=("pred", "max"))
    atk = by_day[by_day["is_attack"] == 1]
    cln = by_day[by_day["is_attack"] == 0]

    pa_pred = anomaly.copy()
    meta_tmp = meta[["date", "is_attack"]].copy()
    meta_tmp["pred"] = anomaly
    detected_attack_days = set(
        meta_tmp.groupby("date").filter(
            lambda g: int(g["is_attack"].max()) == 1 and int(g["pred"].max()) == 1
        )["date"].unique()
    )
    pa_mask = meta_tmp["date"].isin(detected_attack_days) & (meta_tmp["is_attack"].astype(int) == 1)
    pa_pred[pa_mask.to_numpy()] = 1
    pa_tn, pa_fp, pa_fn, pa_tp = confusion_matrix(labels, pa_pred, labels=[0, 1]).ravel()
    pa_precision = pa_tp / (pa_tp + pa_fp) if (pa_tp + pa_fp) else 0.0
    pa_recall = pa_tp / (pa_tp + pa_fn) if (pa_tp + pa_fn) else 0.0
    pa_f1 = 2 * pa_precision * pa_recall / (pa_precision + pa_recall) if (pa_precision + pa_recall) else 0.0
    pa_fpr = pa_fp / (pa_fp + pa_tn) if (pa_fp + pa_tn) else 0.0

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "fpr": float(fpr),
        "pa_tp": int(pa_tp), "pa_fp": int(pa_fp), "pa_fn": int(pa_fn), "pa_tn": int(pa_tn),
        "pa_precision": float(pa_precision), "pa_recall": float(pa_recall),
        "pa_f1": float(pa_f1), "pa_fpr": float(pa_fpr),
        "day_recall": float((atk["pred"] == 1).mean()) if len(atk) else 0.0,
        "day_fpr": float((cln["pred"] == 1).mean()) if len(cln) else 0.0,
        "n_atk_days": int(len(atk)),
        "n_cln_days": int(len(cln)),
        "half_width": float(half_width),
        "mae": float(np.mean(np.abs(y_true - pred))),
        "rmse": float(np.sqrt(np.mean((y_true - pred) ** 2))),
    }


def save_predictions(tag, y, pred, labels, meta, half_width):
    out = meta[["timestamp", "date", "ghi", "is_attack"]].copy()
    out["y_true"] = y
    out["y_pred"] = pred
    out["pi_lower"] = np.clip(pred - half_width, 0.0, 1.2)
    out["pi_upper"] = np.clip(pred + half_width, 0.0, 1.2)
    out["is_anomaly"] = (np.abs(y - pred) > half_width).astype(int)
    out.to_csv(PRED_DIR / f"{tag}_predictions.csv", index=False)


def main():
    args = parse_args()
    set_seed(RANDOM_STATE)
    ensure_dirs()
    timer = Timer()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    timer.log(f"device={device}")

    clean = load_clean_train_frame()
    train_df, calib_df = split_clean_by_date(clean)
    X_tr, y_tr, _, _ = build_sequences(train_df, args.seq_len, args.horizon_steps, args.max_train_samples)
    X_cal, y_cal, lab_cal, meta_cal = build_sequences(calib_df, args.seq_len, args.horizon_steps, args.max_eval_samples)
    timer.log(f"train sequences={len(y_tr):,}, calibration sequences={len(y_cal):,}")

    scaler = fit_scaler(X_tr)
    X_tr_s = transform_sequences(scaler, X_tr)
    X_cal_s = transform_sequences(scaler, X_cal)

    model = CnnLstmForecaster(n_features=len(FEATURE_COLUMNS))
    model = train_model(model, X_tr_s, y_tr, X_cal_s, y_cal, args, device, timer)

    pred_cal = predict(model, X_cal_s, args.batch_size, device)
    if args.threshold_source == "clean_quantile":
        half_width = calibrate_interval(y_cal, pred_cal, args.confidence)
        threshold_info = {
            "source": "clean_quantile",
            "confidence": float(args.confidence),
        }
        timer.log(f"calibrated half_width={half_width:.5f} at confidence={args.confidence:.2f}")
    else:
        attack_train_all = load_attack_train_frame()
        _, attack_val_df = split_clean_by_date(attack_train_all)
        X_atk_val, y_atk_val, lab_atk_val, _ = build_sequences(
            attack_val_df, args.seq_len, args.horizon_steps, args.max_eval_samples
        )
        X_atk_val_s = transform_sequences(scaler, X_atk_val)
        pred_atk_val = predict(model, X_atk_val_s, args.batch_size, device)
        scores = np.abs(y_atk_val - pred_atk_val)
        half_width, tuned_metric = best_f1_residual_threshold(lab_atk_val, scores)
        threshold_info = {
            "source": "attack_validation",
            "train_attack_csvs": [str(p) for p in TRAIN_ATTACK_CSVS],
            "validation_metric": {k: float(v) for k, v in tuned_metric.items()},
        }
        timer.log(
            f"attack-validation half_width={half_width:.5f} "
            f"val_f1={tuned_metric['f1']:.4f} val_fpr={tuned_metric['fpr']:.4f}"
        )

    metrics = {
        "config": {
            "method": "Zhang-style CNN-LSTM prediction interval, multi-clean training",
            "train_clean_csvs": [str(p) for p in TRAIN_CLEAN_CSVS],
            "train_attack_csvs": [str(p) for p in TRAIN_ATTACK_CSVS],
            "eval_csvs": [str(p) for p in EVAL_CSVS],
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "seq_len": int(args.seq_len),
            "horizon_steps": int(args.horizon_steps),
            "confidence": float(args.confidence),
            "min_ghi_valid": MIN_GHI_VALID,
            "train_ratio": TRAIN_RATIO,
            "epochs": int(args.epochs),
            "max_train_samples": int(args.max_train_samples),
            "max_eval_samples": int(args.max_eval_samples),
            "threshold_source": args.threshold_source,
            "threshold_info": threshold_info,
            "note": "max_*_samples > 0 means capped smoke/fast run, not official full result.",
        },
        "calibration_clean": metrics_from_interval(y_cal, pred_cal, lab_cal, meta_cal, half_width),
    }
    metrics_iter = {}

    for path in EVAL_CSVS:
        tag = path.stem
        df = clean_frame(load_csv(path))
        X_te, y_te, labels, meta = build_sequences(df, args.seq_len, args.horizon_steps, args.max_eval_samples)
        X_te_s = transform_sequences(scaler, X_te)
        pred_te = predict(model, X_te_s, args.batch_size, device)
        m = metrics_from_interval(y_te, pred_te, labels, meta, half_width)
        metrics[tag] = m
        metrics_iter[tag] = m
        save_predictions(tag, y_te, pred_te, labels, meta, half_width)
        timer.log(f"{tag}: F1={m['f1']:.4f} Recall={m['recall']:.4f} FPR={m['fpr']:.4f}")

    with open(OUT_DIR / "metrics_full.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "metrics_iter.json", "w", encoding="utf-8") as f:
        json.dump(metrics_iter, f, ensure_ascii=False, indent=2)
    pd.DataFrame([{"dataset": k, **v} for k, v in metrics_iter.items()]).to_csv(
        OUT_DIR / "metrics_summary.csv", index=False
    )
    torch.save({"model_state_dict": model.cpu().state_dict(), "feature_columns": FEATURE_COLUMNS}, MODEL_DIR / "cnn_lstm_forecaster.pt")

    print("\n=== Zhang test7-aligned summary ===")
    print(pd.read_csv(OUT_DIR / "metrics_summary.csv")[["dataset", "precision", "recall", "f1", "fpr", "day_recall", "day_fpr"]].to_string(index=False))


if __name__ == "__main__":
    main()
