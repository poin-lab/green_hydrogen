#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature-matched supervised baselines using the proposed test_ver7 representation.

Purpose:
  - Keep the same train/validation/test data as the proposed detector.
  - Keep the same residual/window/GHI features extracted by test_ver7.
  - Change only the classifier family: RF, MLP, CNN-LSTM, soft voting.

This is not a method-faithful Zhang/Tufail reproduction. It is a fairness
ablation that answers: "What if the baselines receive the same input features?"
"""

from __future__ import annotations

import argparse
import importlib.util
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
TEST_VER7_PATH = PROJECT_ROOT / "proposed/test_ver7.py"

OUT_DIR = PACKAGE_DIR / "results"
MODEL_DIR = OUT_DIR / "models"
PRED_DIR = OUT_DIR / "predictions"

RANDOM_STATE = 42


class Timer:
    def __init__(self) -> None:
        self.start = time.time()

    def log(self, msg: str) -> None:
        print(f"[+{time.time() - self.start:7.1f}s] {msg}", flush=True)


class CnnLstmClassifier(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(n_features, 64, kernel_size=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=1)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=100, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Sequential(
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.pool(self.relu(self.conv(z)))
        z = z.transpose(1, 2)
        z, _ = self.lstm1(z)
        z, _ = self.lstm2(z)
        z = self.dropout(z[:, -1, :])
        return self.head(z)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature-matched baselines using proposed features")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--max-train-rows", type=int, default=0, help="0 means all")
    parser.add_argument("--max-eval-rows", type=int, default=0, help="0 means all")
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
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)


def load_test_ver7_module():
    spec = importlib.util.spec_from_file_location("test_ver7_feature_source", TEST_VER7_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {TEST_VER7_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.VERBOSE_PROGRESS = False
    return mod


def split_by_date(df: pd.DataFrame, ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["date"].unique())
    split_idx = int(len(dates) * ratio)
    train_dates = set(dates[:split_idx])
    return (
        df[df["date"].isin(train_dates)].reset_index(drop=True),
        df[~df["date"].isin(train_dates)].reset_index(drop=True),
    )


def sample_rows(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows and len(df) > max_rows:
        return df.sort_values(["date", "timestamp"]).head(max_rows).reset_index(drop=True)
    return df.reset_index(drop=True)


def build_train_eval_features(src, timer: Timer, args):
    timer.log("loading proposed forecaster")
    fc = src.ForecasterWrapper().load()

    pieces = []
    for path in src.TRAIN_ATTACK_CSVS:
        tag = Path(path).stem
        timer.log(f"extract train proposed features: {tag}")
        resid = src.load_residual_seq(path, fc, tag)
        feat = src.extract_feat_cached(resid, tag)
        pieces.append(feat)
    all_train = pd.concat(pieces, ignore_index=True)
    all_train = all_train[all_train["ghi_zone"] != "ignore"].reset_index(drop=True)
    train_df, val_df = split_by_date(all_train, src.TRAIN_RATIO)

    zfc, drop = src.select_zone_features(train_df)
    feature_cols = sorted(set(zfc["mid"]) | set(zfc["high"]))
    train_df = train_df.drop(columns=drop, errors="ignore")
    val_df = val_df.drop(columns=drop, errors="ignore")
    train_df = sample_rows(train_df, args.max_train_rows)
    val_df = sample_rows(val_df, args.max_eval_rows)
    timer.log(
        f"feature rows train={len(train_df):,} pos={int(train_df['is_attack'].sum()):,} "
        f"val={len(val_df):,} pos={int(val_df['is_attack'].sum()):,} features={len(feature_cols)}"
    )

    eval_frames = {}
    for path in src.EVAL_CSVS_SA:
        tag = Path(path).stem
        timer.log(f"extract eval proposed features: {tag}")
        resid = src.load_residual_seq(path, fc, f"eval_{tag}")
        feat = src.extract_feat_cached(resid, f"eval_{tag}")
        feat = feat[feat["ghi_zone"] != "ignore"].drop(columns=drop, errors="ignore").reset_index(drop=True)
        eval_frames[tag] = sample_rows(feat, args.max_eval_rows)
    return train_df, val_df, eval_frames, feature_cols


def matrix(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = df.reindex(columns=feature_cols, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["is_attack"].astype(int).to_numpy()
    return X.to_numpy(dtype=np.float32), y


def build_sequences(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, seq_len: int):
    x_parts = []
    y_parts = []
    idx_parts = []
    for _, idx in meta.groupby("date", sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        if len(idx) < seq_len:
            continue
        for j in range(seq_len - 1, len(idx)):
            window_idx = idx[j - seq_len + 1 : j + 1]
            target_idx = idx[j]
            x_parts.append(X[window_idx])
            y_parts.append(y[target_idx])
            idx_parts.append(target_idx)
    if not x_parts:
        raise ValueError("No CNN-LSTM sequences generated")
    return (
        np.asarray(x_parts, dtype=np.float32),
        np.asarray(y_parts, dtype=np.int64),
        np.asarray(idx_parts, dtype=np.int64),
    )


def train_cnn(Xw_train, y_train, Xw_val, y_val, args, device, timer):
    model = CnnLstmClassifier(n_features=Xw_train.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xw_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_x = torch.from_numpy(Xw_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    best_state = None
    best_val = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
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
        if stale >= 4:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_cnn(model, Xw, batch_size, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(Xw)), batch_size=batch_size, shuffle=False)
    out = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            out.append(torch.softmax(model(xb.to(device)), dim=1).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def safe_proba(clf, X):
    p = clf.predict_proba(X)
    if p.shape[1] == 2:
        return p
    out = np.zeros((len(X), 2), dtype=float)
    for i, cls in enumerate(clf.classes_):
        out[:, int(cls)] = p[:, i]
    return out


def metric_dict(y_true, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "accuracy": float(accuracy), "precision": float(precision),
        "recall": float(recall), "f1": float(f1), "fpr": float(fpr),
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


def point_adjust(y_true, pred):
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(int)
    out = pred.copy()
    in_attack = False
    start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_attack:
            in_attack = True
            start = i
        elif y_true[i] == 0 and in_attack:
            in_attack = False
            if np.any(pred[start:i] == 1):
                out[start:i] = 1
    if in_attack and np.any(pred[start:] == 1):
        out[start:] = 1
    return out


def add_pa_and_day_metrics(metric, y_true, meta, score, threshold):
    raw_pred = (score >= threshold).astype(int)
    pa_pred = point_adjust(y_true, raw_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, pa_pred, labels=[0, 1]).ravel()
    pa_precision = tp / (tp + fp) if (tp + fp) else 0.0
    pa_recall = tp / (tp + fn) if (tp + fn) else 0.0
    pa_f1 = 2 * pa_precision * pa_recall / (pa_precision + pa_recall) if (pa_precision + pa_recall) else 0.0
    pa_fpr = fp / (fp + tn) if (fp + tn) else 0.0
    metric.update({
        "pa_tp": int(tp), "pa_fp": int(fp), "pa_fn": int(fn), "pa_tn": int(tn),
        "pa_precision": float(pa_precision), "pa_recall": float(pa_recall),
        "pa_f1": float(pa_f1), "pa_fpr": float(pa_fpr),
    })

    tmp = meta[["date", "is_attack"]].copy()
    tmp["pred"] = raw_pred
    day = tmp.groupby("date").agg(is_attack=("is_attack", "max"), pred=("pred", "max"))
    atk = day[day["is_attack"] == 1]
    cln = day[day["is_attack"] == 0]
    metric["day_recall"] = float((atk["pred"] == 1).mean()) if len(atk) else 0.0
    metric["day_fpr"] = float((cln["pred"] == 1).mean()) if len(cln) else 0.0
    metric["n_atk_days"] = int(len(atk))
    metric["n_cln_days"] = int(len(cln))
    return metric


def evaluate_scores(y, meta, probs: Dict[str, np.ndarray], thresholds=None, tune=False):
    metrics = {}
    tuned = {}
    for name, p in probs.items():
        score = p[:, 1]
        threshold = best_f1_threshold(y, score) if tune else thresholds[name]
        m = metric_dict(y, score, threshold)
        m = add_pa_and_day_metrics(m, y, meta, score, threshold)
        m["threshold"] = float(threshold)
        metrics[name] = m
        tuned[name] = float(threshold)
    return metrics, tuned


def align_probs_for_cnn(y, meta, probs_tab, cnn_prob, seq_idx):
    y_aligned = y[seq_idx]
    meta_aligned = meta.iloc[seq_idx].reset_index(drop=True)
    out = {name: p[seq_idx] for name, p in probs_tab.items()}
    out["CNN_LSTM"] = cnn_prob
    out["RF+MLP"] = np.mean(np.stack([out["RF"], out["MLP"]]), axis=0)
    out["CNN_LSTM+RF"] = np.mean(np.stack([out["CNN_LSTM"], out["RF"]]), axis=0)
    out["CNN_LSTM+MLP"] = np.mean(np.stack([out["CNN_LSTM"], out["MLP"]]), axis=0)
    out["CNN_LSTM+RF+MLP"] = np.mean(np.stack([out["CNN_LSTM"], out["RF"], out["MLP"]]), axis=0)
    return y_aligned, meta_aligned, out


def main():
    args = parse_args()
    set_seed(RANDOM_STATE)
    ensure_dirs()
    timer = Timer()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    timer.log(f"device={device}")
    src = load_test_ver7_module()

    train_df, val_df, eval_frames, feature_cols = build_train_eval_features(src, timer, args)
    X_train, y_train = matrix(train_df, feature_cols)
    X_val, y_val = matrix(val_df, feature_cols)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    timer.log("training RF on proposed features")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=70, min_samples_split=2, min_samples_leaf=1,
        class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf.fit(X_train_s, y_train)

    timer.log("training MLP on proposed features")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4,
        batch_size=512, learning_rate_init=1e-3, max_iter=120,
        early_stopping=True, random_state=RANDOM_STATE,
    )
    mlp.fit(X_train_s, y_train)

    timer.log("building CNN-LSTM sequences on proposed features")
    train_meta = train_df[["timestamp", "date", "is_attack"]].reset_index(drop=True)
    val_meta = val_df[["timestamp", "date", "is_attack"]].reset_index(drop=True)
    Xw_train, yw_train, _ = build_sequences(X_train_s, y_train, train_meta, args.seq_len)
    Xw_val, yw_val, val_seq_idx = build_sequences(X_val_s, y_val, val_meta, args.seq_len)
    timer.log(f"cnn windows train={len(yw_train):,} pos={int(yw_train.sum()):,} val={len(yw_val):,} pos={int(yw_val.sum()):,}")

    timer.log("training CNN-LSTM on proposed features")
    cnn = train_cnn(Xw_train, yw_train, Xw_val, yw_val, args, device, timer)

    val_tab_probs = {"RF": safe_proba(rf, X_val_s), "MLP": safe_proba(mlp, X_val_s)}
    val_cnn_prob = predict_cnn(cnn, Xw_val, args.batch_size, device)
    y_val_aligned, meta_val_aligned, val_probs = align_probs_for_cnn(
        y_val, val_meta, val_tab_probs, val_cnn_prob, val_seq_idx
    )
    val_metrics, thresholds = evaluate_scores(y_val_aligned, meta_val_aligned, val_probs, tune=True)

    all_metrics = {"validation": val_metrics["CNN_LSTM+RF+MLP"]}
    full_metrics = {
        "config": {
            "feature_source": str(TEST_VER7_PATH),
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
            "seq_len": int(args.seq_len),
            "epochs": int(args.epochs),
            "max_train_rows": int(args.max_train_rows),
            "max_eval_rows": int(args.max_eval_rows),
            "train_attack_csvs": src.TRAIN_ATTACK_CSVS,
            "eval_csvs": src.EVAL_CSVS_SA,
            "note": "All classifiers use the same proposed residual/window/GHI feature representation.",
        },
        "validation_all_models": val_metrics,
        "thresholds": thresholds,
    }

    for tag, df in eval_frames.items():
        X_eval, y_eval = matrix(df, feature_cols)
        X_eval_s = scaler.transform(X_eval).astype(np.float32)
        eval_meta = df[["timestamp", "date", "is_attack"]].reset_index(drop=True)
        Xw_eval, _, eval_seq_idx = build_sequences(X_eval_s, y_eval, eval_meta, args.seq_len)
        tab_probs = {"RF": safe_proba(rf, X_eval_s), "MLP": safe_proba(mlp, X_eval_s)}
        cnn_prob = predict_cnn(cnn, Xw_eval, args.batch_size, device)
        y_eval_aligned, meta_eval_aligned, probs = align_probs_for_cnn(
            y_eval, eval_meta, tab_probs, cnn_prob, eval_seq_idx
        )
        eval_metrics, _ = evaluate_scores(y_eval_aligned, meta_eval_aligned, probs, thresholds=thresholds, tune=False)
        full_metrics[tag] = eval_metrics
        all_metrics[tag] = eval_metrics["CNN_LSTM+RF+MLP"]

        pred = meta_eval_aligned[["timestamp", "date", "is_attack"]].copy()
        pred["score"] = probs["CNN_LSTM+RF+MLP"][:, 1]
        pred["pred"] = (pred["score"] >= thresholds["CNN_LSTM+RF+MLP"]).astype(int)
        pred.to_csv(PRED_DIR / f"{tag}_predictions.csv", index=False)
        m = all_metrics[tag]
        timer.log(f"{tag}: F1={m['f1']:.4f} PA-F1={m['pa_f1']:.4f} FPR={m['fpr']:.4f}")

    with open(OUT_DIR / "metrics_full.json", "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "metrics_iter.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    rows = []
    for dataset, m in all_metrics.items():
        row = {"dataset": dataset}
        row.update(m)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "feature_matched_summary.csv", index=False)
    joblib.dump({"rf": rf, "mlp": mlp, "scaler": scaler, "feature_columns": feature_cols}, MODEL_DIR / "rf_mlp_scaler.joblib")
    torch.save(cnn.cpu().state_dict(), MODEL_DIR / "cnn_lstm.pt")

    print("\n=== Feature-matched proposed-feature baseline summary (CNN_LSTM+RF+MLP) ===")
    df = pd.DataFrame(rows)
    print(df[["dataset", "accuracy", "precision", "recall", "f1", "fpr", "pa_f1", "day_recall", "day_fpr"]].to_string(index=False))


if __name__ == "__main__":
    main()
