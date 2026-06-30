#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tufail-style ensemble classifier aligned to the test_ver7 experiment setting.

What is preserved from compare2 / Tufail-style methodology:
  - supervised classifier baseline
  - RF, MLP, CNN-LSTM base learners
  - soft-voting ensembles
  - sliding-window input for CNN-LSTM

What is aligned to the proposed model experiment:
  - train clean + SA 5/10 from dataset5.4_attack
  - validation by chronological date split, 80/20, like test_ver7
  - external SA 5/8/10 evaluation from dataset6.0_attack
  - binary attack_label task
  - GHI >= 200 filtering

TensorFlow is not available in the current green_hy environment, so the
CNN-LSTM is implemented with PyTorch using the same layer pattern:
Conv1D(64,k=1) -> MaxPool1D(1) -> LSTM(100) -> LSTM(50) -> Dropout -> Dense.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_DIR

TRAIN_CLEAN_CSV = PROJECT_ROOT / "dataset5.4_attack/site5_5.4kw_2018_2020_clean.csv"
TRAIN_ATTACK_CSVS = [
    PROJECT_ROOT / "dataset5.4_attack/site5_5.4kw_2018_2020_attack_sa_5pct.csv",
    PROJECT_ROOT / "dataset5.4_attack/site5_5.4kw_2018_2020_attack_sa_10pct.csv",
]
EVAL_CSVS = [
    PROJECT_ROOT / "dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_5pct.csv",
    PROJECT_ROOT / "dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_8pct.csv",
    PROJECT_ROOT / "dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_10pct.csv",
]

OUT_DIR = PACKAGE_DIR / "results"
PRED_DIR = OUT_DIR / "predictions"
MODEL_DIR = OUT_DIR / "models"

RANDOM_STATE = 42
WINDOW = 12
TRAIN_RATIO = 0.80
MIN_GHI_VALID = 200.0

FEATURE_MODES = {
    "paper_no_time": ["ghi", "power_ratio", "temp"],
    "paper_with_time": ["ghi", "power_ratio", "temp", "sin_hour", "cos_hour"],
    "weather_only": ["ghi", "temp", "sin_hour", "cos_hour"],
    "engineered": [
        "ghi", "power_ratio", "temp", "sin_hour", "cos_hour",
        "ghi_diff1", "temp_diff1", "power_ratio_diff1",
        "power_ratio_ghi_norm",
        "ghi_roll3_mean", "ghi_roll6_mean", "ghi_roll12_mean", "ghi_roll12_std",
        "temp_roll3_mean", "temp_roll6_mean", "temp_roll12_mean", "temp_roll12_std",
        "power_ratio_roll3_mean", "power_ratio_roll6_mean",
        "power_ratio_roll12_mean", "power_ratio_roll12_std",
    ],
}


class Timer:
    def __init__(self) -> None:
        self.start = time.time()

    def log(self, message: str) -> None:
        print(f"[+{time.time() - self.start:7.1f}s] {message}", flush=True)


class CnnLstmClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv1d(n_features, 64, kernel_size=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=1)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=100, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, window, features]
        z = x.transpose(1, 2)
        z = self.pool(self.relu(self.conv(z)))
        z = z.transpose(1, 2)
        z, _ = self.lstm1(z)
        z, _ = self.lstm2(z)
        z = self.dropout(z[:, -1, :])
        return self.fc(z)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="compare2 method aligned to test_ver7 data")
    parser.add_argument(
        "--feature-mode",
        choices=[*FEATURE_MODES.keys(), "all"],
        default="paper_with_time",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-windows", type=int, default=0, help="0 means use all")
    parser.add_argument("--max-eval-windows", type=int, default=0, help="0 means use all")
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
    return df.sort_values(["site", "timestamp"]).reset_index(drop=True)


def add_common_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["ghi", "temp", "power_ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    df["date"] = df["timestamp"].dt.date
    df["is_attack"] = (df["attack_label"].astype(float) > 0.2).astype(int)
    df = df.sort_values(["site", "timestamp"]).reset_index(drop=True)
    grouped = df.groupby(["site", "date"], sort=False)
    for col in ["ghi", "temp", "power_ratio"]:
        df[f"{col}_diff1"] = grouped[col].diff().fillna(0.0)
        for window in [3, 6, 12]:
            rolled = grouped[col].rolling(window=window, min_periods=1)
            df[f"{col}_roll{window}_mean"] = rolled.mean().reset_index(level=[0, 1], drop=True)
            if window == 12:
                df[f"{col}_roll{window}_std"] = rolled.std().reset_index(level=[0, 1], drop=True).fillna(0.0)
    df["power_ratio_ghi_norm"] = df["power_ratio"] / np.maximum(df["ghi"] / 1000.0, 1e-3)
    df = df[df["ghi"].astype(float) >= MIN_GHI_VALID].copy()
    return df.sort_values(["site", "timestamp"]).reset_index(drop=True)


def load_train_frame() -> pd.DataFrame:
    frames = [add_common_features(load_csv(TRAIN_CLEAN_CSV))]
    for path in TRAIN_ATTACK_CSVS:
        frames.append(add_common_features(load_csv(path)))
    return pd.concat(frames, ignore_index=True).sort_values(["site", "timestamp"]).reset_index(drop=True)


def split_by_date_like_test7(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["date"].unique())
    split_idx = int(len(dates) * TRAIN_RATIO)
    train_dates = set(dates[:split_idx])
    return (
        df[df["date"].isin(train_dates)].reset_index(drop=True),
        df[~df["date"].isin(train_dates)].reset_index(drop=True),
    )


def build_windows(df: pd.DataFrame, feature_cols: List[str], max_windows: int = 0):
    xw_parts = []
    xr_parts = []
    y_parts = []
    meta_parts = []
    for _, g in df.groupby(["site", "date"], sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) < WINDOW:
            continue
        arr = g[feature_cols].astype(float).to_numpy()
        labels = g["is_attack"].astype(int).to_numpy()
        for i in range(WINDOW - 1, len(g)):
            xw_parts.append(arr[i - WINDOW + 1 : i + 1])
            xr_parts.append(arr[i])
            y_parts.append(labels[i])
            meta_parts.append({
                "timestamp": g.loc[i, "timestamp"],
                "date": g.loc[i, "date"],
                "ghi": float(g.loc[i, "ghi"]),
                "is_attack": int(labels[i]),
            })

    if not xw_parts:
        raise ValueError("No windows generated")

    Xw = np.asarray(xw_parts, dtype=np.float32)
    Xr = np.asarray(xr_parts, dtype=np.float32)
    y = np.asarray(y_parts, dtype=np.int64)
    meta = pd.DataFrame(meta_parts)

    if max_windows and len(y) > max_windows:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = np.sort(rng.choice(len(y), size=max_windows, replace=False))
        Xw, Xr, y, meta = Xw[idx], Xr[idx], y[idx], meta.iloc[idx].reset_index(drop=True)

    return Xw, Xr, y, meta


def fit_scaler(Xr_train, Xw_train):
    scaler = StandardScaler().fit(Xr_train)
    return scaler


def transform_with_scaler(scaler, Xr, Xw):
    Xr_s = scaler.transform(Xr).astype(np.float32)
    n, w, f = Xw.shape
    Xw_s = scaler.transform(Xw.reshape(-1, f)).reshape(n, w, f).astype(np.float32)
    return Xr_s, Xw_s


def train_cnn_lstm(Xw_train, y_train, Xw_val, y_val, args, device, timer):
    model = CnnLstmClassifier(n_features=Xw_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_ds = TensorDataset(torch.from_numpy(Xw_train), torch.from_numpy(y_train))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_x = torch.from_numpy(Xw_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    best_state = None
    best_val = float("inf")
    patience = 6
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).detach().cpu())
        timer.log(f"  cnn epoch {epoch:03d}/{args.epochs} train_ce={np.mean(losses):.5f} val_ce={val_loss:.5f}")
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


def predict_cnn(model, Xw, args, device):
    model.eval()
    ds = TensorDataset(torch.from_numpy(Xw))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(probs, axis=0)


def safe_proba(clf, X):
    p = clf.predict_proba(X)
    if p.shape[1] == 2:
        return p
    out = np.zeros((len(X), 2), dtype=float)
    for i, cls in enumerate(clf.classes_):
        out[:, int(cls)] = p[:, i]
    return out


def soft_vote(prob_list):
    return np.mean(np.stack(prob_list, axis=0), axis=0)


def metric_dict(y_true, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "fpr": float(fpr),
    }


def best_f1_threshold(y_true, score):
    candidates = np.unique(np.concatenate([
        np.linspace(0.01, 0.99, 99),
        np.quantile(score, np.linspace(0.01, 0.99, 99)),
    ]))
    best_t = 0.5
    best_m = metric_dict(y_true, score, best_t)
    for t in candidates:
        m = metric_dict(y_true, score, float(t))
        if (m["f1"], m["recall"], -m["fpr"]) > (best_m["f1"], best_m["recall"], -best_m["fpr"]):
            best_t = float(t)
            best_m = m
    return best_t


def add_day_metrics(metric, meta, score, threshold):
    tmp = meta[["date", "is_attack"]].copy()
    tmp["pred"] = (score >= threshold).astype(int)
    day = tmp.groupby("date").agg(is_attack=("is_attack", "max"), pred=("pred", "max"))
    atk = day[day["is_attack"] == 1]
    cln = day[day["is_attack"] == 0]
    metric["day_recall"] = float((atk["pred"] == 1).mean()) if len(atk) else 0.0
    metric["day_fpr"] = float((cln["pred"] == 1).mean()) if len(cln) else 0.0
    metric["n_atk_days"] = int(len(atk))
    metric["n_cln_days"] = int(len(cln))
    return metric


def add_pa_metrics(metric, y_true, meta, score, threshold):
    pred = (score >= threshold).astype(int)
    tmp = meta[["date", "is_attack"]].copy()
    tmp["pred"] = pred
    detected_attack_days = set(
        tmp.groupby("date").filter(
            lambda g: int(g["is_attack"].max()) == 1 and int(g["pred"].max()) == 1
        )["date"].unique()
    )
    pa_pred = pred.copy()
    pa_mask = tmp["date"].isin(detected_attack_days) & (tmp["is_attack"].astype(int) == 1)
    pa_pred[pa_mask.to_numpy()] = 1
    tn, fp, fn, tp = confusion_matrix(y_true, pa_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    metric.update({
        "pa_tp": int(tp), "pa_fp": int(fp), "pa_fn": int(fn), "pa_tn": int(tn),
        "pa_precision": float(precision), "pa_recall": float(recall),
        "pa_f1": float(f1), "pa_fpr": float(fpr),
    })
    return metric


def evaluate_scores(label, y, meta, probs, thresholds=None, tune=False):
    out = {}
    tuned = {}
    for name, p in probs.items():
        score = p[:, 1]
        th = best_f1_threshold(y, score) if tune else thresholds[name]
        m = metric_dict(y, score, th)
        m = add_day_metrics(m, meta, score, th)
        m = add_pa_metrics(m, y, meta, score, th)
        m["threshold"] = float(th)
        out[name] = m
        tuned[name] = float(th)
    return out, tuned


def run_mode(feature_mode: str, args, device, timer):
    feature_cols = FEATURE_MODES[feature_mode]
    timer.log(f"mode={feature_mode}, features={feature_cols}")
    train_all = load_train_frame()
    train_df, val_df = split_by_date_like_test7(train_all)
    train_df = train_df.dropna(subset=feature_cols + ["is_attack"])
    val_df = val_df.dropna(subset=feature_cols + ["is_attack"])

    Xw_tr, Xr_tr, y_tr, meta_tr = build_windows(train_df, feature_cols, args.max_train_windows)
    Xw_va, Xr_va, y_va, meta_va = build_windows(val_df, feature_cols, args.max_eval_windows)
    timer.log(f"windows train={len(y_tr):,} pos={int(y_tr.sum()):,} val={len(y_va):,} pos={int(y_va.sum()):,}")

    scaler = fit_scaler(Xr_tr, Xw_tr)
    Xr_tr_s, Xw_tr_s = transform_with_scaler(scaler, Xr_tr, Xw_tr)
    Xr_va_s, Xw_va_s = transform_with_scaler(scaler, Xr_va, Xw_va)

    timer.log("  training RF")
    rf = RandomForestClassifier(
        n_estimators=500, criterion="gini", max_depth=70,
        min_samples_split=2, min_samples_leaf=1, bootstrap=True,
        class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf.fit(Xr_tr_s, y_tr)

    timer.log("  training MLP")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4,
        batch_size=512, learning_rate_init=1e-3, max_iter=120,
        early_stopping=True, random_state=RANDOM_STATE,
    )
    mlp.fit(Xr_tr_s, y_tr)

    timer.log("  training CNN-LSTM")
    cnn = train_cnn_lstm(Xw_tr_s, y_tr, Xw_va_s, y_va, args, device, timer)

    val_probs = {
        "RF": safe_proba(rf, Xr_va_s),
        "MLP": safe_proba(mlp, Xr_va_s),
        "CNN_LSTM": predict_cnn(cnn, Xw_va_s, args, device),
    }
    val_probs.update({
        "RF+MLP": soft_vote([val_probs["RF"], val_probs["MLP"]]),
        "CNN_LSTM+RF": soft_vote([val_probs["CNN_LSTM"], val_probs["RF"]]),
        "CNN_LSTM+MLP": soft_vote([val_probs["CNN_LSTM"], val_probs["MLP"]]),
        "CNN_LSTM+RF+MLP": soft_vote([val_probs["CNN_LSTM"], val_probs["RF"], val_probs["MLP"]]),
    })
    val_metrics, thresholds = evaluate_scores("validation", y_va, meta_va, val_probs, tune=True)

    metrics_iter = {"validation": val_metrics["CNN_LSTM+RF+MLP"]}
    metrics_full = {
        "config": {
            "feature_mode": feature_mode,
            "feature_columns": feature_cols,
            "window": WINDOW,
            "min_ghi_valid": MIN_GHI_VALID,
            "train_ratio": TRAIN_RATIO,
            "epochs": int(args.epochs),
            "max_train_windows": int(args.max_train_windows),
            "max_eval_windows": int(args.max_eval_windows),
            "train_clean_csv": str(TRAIN_CLEAN_CSV),
            "train_attack_csvs": [str(p) for p in TRAIN_ATTACK_CSVS],
            "eval_csvs": [str(p) for p in EVAL_CSVS],
            "note": "max_*_windows > 0 means this is a capped smoke/fast run, not the full official result.",
        },
        "validation": val_metrics,
        "thresholds": thresholds,
    }
    for path in EVAL_CSVS:
        tag = path.stem
        eval_df = add_common_features(load_csv(path)).dropna(subset=feature_cols + ["is_attack"])
        Xw_te, Xr_te, y_te, meta_te = build_windows(eval_df, feature_cols, args.max_eval_windows)
        Xr_te_s, Xw_te_s = transform_with_scaler(scaler, Xr_te, Xw_te)
        probs = {
            "RF": safe_proba(rf, Xr_te_s),
            "MLP": safe_proba(mlp, Xr_te_s),
            "CNN_LSTM": predict_cnn(cnn, Xw_te_s, args, device),
        }
        probs.update({
            "RF+MLP": soft_vote([probs["RF"], probs["MLP"]]),
            "CNN_LSTM+RF": soft_vote([probs["CNN_LSTM"], probs["RF"]]),
            "CNN_LSTM+MLP": soft_vote([probs["CNN_LSTM"], probs["MLP"]]),
            "CNN_LSTM+RF+MLP": soft_vote([probs["CNN_LSTM"], probs["RF"], probs["MLP"]]),
        })
        eval_metrics, _ = evaluate_scores(tag, y_te, meta_te, probs, thresholds=thresholds, tune=False)
        metrics_full[tag] = eval_metrics
        metrics_iter[tag] = eval_metrics["CNN_LSTM+RF+MLP"]

        pred = meta_te[["timestamp", "date", "ghi", "is_attack"]].copy()
        pred["score"] = probs["CNN_LSTM+RF+MLP"][:, 1]
        pred["pred"] = (pred["score"] >= thresholds["CNN_LSTM+RF+MLP"]).astype(int)
        pred.to_csv(PRED_DIR / f"{feature_mode}_{tag}_predictions.csv", index=False)

    mode_dir = OUT_DIR / feature_mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    with open(mode_dir / "metrics_full.json", "w", encoding="utf-8") as f:
        json.dump(metrics_full, f, ensure_ascii=False, indent=2)
    with open(mode_dir / "metrics_iter.json", "w", encoding="utf-8") as f:
        json.dump(metrics_iter, f, ensure_ascii=False, indent=2)
    with open(mode_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    joblib.dump({"rf": rf, "mlp": mlp, "scaler": scaler, "features": feature_cols}, MODEL_DIR / f"{feature_mode}_sklearn.joblib")
    torch.save(cnn.cpu().state_dict(), MODEL_DIR / f"{feature_mode}_cnn_lstm.pt")
    return metrics_iter


def save_summary(all_results: Dict[str, dict]) -> None:
    rows = []
    for mode, metrics in all_results.items():
        for dataset, m in metrics.items():
            row = {"feature_mode": mode, "dataset": dataset}
            row.update(m)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "comparison_summary.csv", index=False)
    with open(OUT_DIR / "metrics_iter_all_modes.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\n=== compare2_test7_aligned summary (CNN_LSTM+RF+MLP) ===")
    print(df[["feature_mode", "dataset", "precision", "recall", "f1", "fpr", "day_recall", "day_fpr"]].to_string(index=False))


def main():
    args = parse_args()
    set_seed(RANDOM_STATE)
    ensure_dirs()
    timer = Timer()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    timer.log(f"device={device}")

    modes = list(FEATURE_MODES) if args.feature_mode == "all" else [args.feature_mode]
    all_results = {}
    for mode in modes:
        all_results[mode] = run_mode(mode, args, device, timer)
    save_summary(all_results)


if __name__ == "__main__":
    main()
