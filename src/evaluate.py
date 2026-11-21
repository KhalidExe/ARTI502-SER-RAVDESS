"""Metrics (accuracy, UAR, macro-F1, confusion), statistical tests, trivial baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import EMOTIONS, N_CLASSES


def confusion(y_true, y_pred, n=N_CLASSES) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, int), np.asarray(y_pred, int)), 1)
    return cm


def compute_metrics(y_true, y_pred, n=N_CLASSES) -> dict:
    """UAR = mean recall over classes present in y_true  (== sklearn balanced_accuracy_score).
    Macro-F1 = mean F1 over classes present in y_true or y_pred (== sklearn f1_score(average='macro'))."""
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    cm = confusion(y_true, y_pred, n)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    pred_n = cm.sum(0).astype(float)
    recall = np.divide(tp, support, out=np.zeros(n), where=support > 0)
    denom = 2 * tp + (pred_n - tp) + (support - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros(n), where=denom > 0)
    present = support > 0
    used = present | (pred_n > 0)
    return {
        "acc": float(tp.sum() / max(len(y_true), 1)),
        "uar": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[used].mean()) if used.any() else 0.0,
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "confusion": cm.tolist(),
        "n": int(len(y_true)),
    }


# ----------------------------------------------------------------------------
# Paired statistics over the same folds and seeds
# ----------------------------------------------------------------------------
def paired_test(a, b) -> dict:
    from scipy import stats

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    out = {"n": int(len(d)), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
           "mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)) if len(d) > 1 else float("nan")}
    if len(d) < 2:
        return {**out, "t": np.nan, "t_p": np.nan, "wilcoxon_W": np.nan, "wilcoxon_p": np.nan}
    t = stats.ttest_rel(a, b)
    out.update({"t": float(t.statistic), "t_p": float(t.pvalue)})
    if np.allclose(d, 0):
        out.update({"wilcoxon_W": np.nan, "wilcoxon_p": 1.0})
    else:
        w = stats.wilcoxon(a, b, zero_method="wilcox")
        out.update({"wilcoxon_W": float(w.statistic), "wilcoxon_p": float(w.pvalue)})
    return out


def compare_runs(runs: pd.DataFrame, sel_a: dict, sel_b: dict, metric="test_uar", level="fold") -> dict:
    """Pair two run sets on (fold, seed). level='fold': average seeds per fold first (n=6, primary);
    level='run': pair every (fold, seed) (n=18)."""
    def pick(sel):
        df = runs
        for k, v in sel.items():
            df = df[df[k] == v]
        return df[["fold", "seed", metric]]

    m = pick(sel_a).merge(pick(sel_b), on=["fold", "seed"], suffixes=("_a", "_b"))
    if level == "fold":
        m = m.groupby("fold")[[f"{metric}_a", f"{metric}_b"]].mean().reset_index()
    res = paired_test(m[f"{metric}_a"], m[f"{metric}_b"])
    res["level"] = level
    return res


# ----------------------------------------------------------------------------
# Trivial baselines (chance UAR = 1/8 = 12.5 %)
# ----------------------------------------------------------------------------
def trivial_baselines(meta: pd.DataFrame, splits: dict, seed: int = 0) -> pd.DataFrame:
    from .data import split_indices

    y = meta["label"].to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for r in splits["folds"]["rounds"]:
        idx = split_indices(meta, "actor", r["round"], splits)
        ytr, yte = y[idx["train"]], y[idx["test"]]
        maj = np.bincount(ytr, minlength=N_CLASSES).argmax()
        prior = np.bincount(ytr, minlength=N_CLASSES) / len(ytr)
        for name, pred in (("majority", np.full_like(yte, maj)),
                           ("uniform_random", rng.integers(0, N_CLASSES, len(yte))),
                           ("prior_random", rng.choice(N_CLASSES, len(yte), p=prior))):
            m = compute_metrics(yte, pred)
            rows.append({"baseline": name, "fold": r["round"], "test_acc": m["acc"],
                         "test_uar": m["uar"], "test_macro_f1": m["macro_f1"]})
    return pd.DataFrame(rows)


def pooled_confusion(pred_files) -> np.ndarray:
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for f in pred_files:
        p = pd.read_csv(f)
        cm += confusion(p["y_true"], p["y_pred"])
    return cm


def top_confusions(cm: np.ndarray, k: int = 5) -> list[dict]:
    """Most confused (true -> predicted) pairs, by row-normalised rate."""
    rates = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    out = []
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if i != j:
                out.append({"true": EMOTIONS[i], "pred": EMOTIONS[j], "count": int(cm[i, j]),
                            "rate": float(rates[i, j])})
    return sorted(out, key=lambda r: -r["rate"])[:k]
