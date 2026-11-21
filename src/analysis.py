"""Builds every results table and figure from results/ (used by notebooks/03_results_analysis.ipynb)."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from . import evaluate, utils
from .utils import EMOTIONS

RES = utils.RESULTS_DIR
TAB = RES / "tables"
FIG = RES / "figures"
MODEL_NAMES = {"m1": "M1 BiLSTM+Att", "m2": "M2 CNN-BiLSTM", "m3": "M3 Transformer", "m4": "M4 wav2vec2-base"}


def load_runs(results_dir=RES, tags=None) -> pd.DataFrame:
    p = utils.resolve(results_dir) / "runs.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"fold": str})
    if tags:
        df = df[df["tag"].isin(tags)]
    # keep the LAST row if a run was repeated
    return df.drop_duplicates("run_id", keep="last").reset_index(drop=True)


def _pm(s: pd.Series, scale=100.0, nd=2) -> str:
    if len(s) == 0:
        return ""
    sd = s.std(ddof=1) if len(s) > 1 else 0.0
    return f"{scale * s.mean():.{nd}f} ± {scale * sd:.{nd}f}"


def summary_table(runs: pd.DataFrame, name="main_results") -> pd.DataFrame:
    """mean ± std (in %) over all folds x seeds per (model, features, protocol, tag)."""
    rows = []
    for (m, f, p, t), g in runs.groupby(["model", "features", "protocol", "tag"]):
        rows.append({"model": MODEL_NAMES.get(m, m), "features": f, "protocol": p, "tag": t,
                     "n_runs": len(g), "folds": g["fold"].nunique(), "seeds": g["seed"].nunique(),
                     "UAR (%)": _pm(g["test_uar"]), "Accuracy (%)": _pm(g["test_acc"]),
                     "Macro-F1 (%)": _pm(g["test_macro_f1"]), "val UAR (%)": _pm(g["val_uar"]),
                     "params": int(g["n_params"].iloc[0]), "trainable": int(g["n_trainable"].iloc[0]),
                     "mean train time (s)": round(g["train_time_s"].mean(), 1),
                     "mean best epoch": round(g["best_epoch"].mean(), 1)})
    df = pd.DataFrame(rows)
    utils.ensure_dir(TAB)
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(df.to_markdown(index=False) if len(df) else "", encoding="utf-8")
    return df


def per_class_recall_table(runs: pd.DataFrame, name="per_class_recall") -> pd.DataFrame:
    cols = [f"recall_{e}" for e in EMOTIONS]
    rows = []
    for (m, f, p, t), g in runs.groupby(["model", "features", "protocol", "tag"]):
        r = {"model": MODEL_NAMES.get(m, m), "features": f, "protocol": p, "tag": t}
        r.update({e: _pm(g[c], nd=1) for e, c in zip(EMOTIONS, cols)})
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(utils.ensure_dir(TAB) / f"{name}.csv", index=False)
    return df


def pred_files(runs: pd.DataFrame, results_dir=RES):
    d = utils.resolve(results_dir) / "predictions"
    return [d / f"{r}.csv" for r in runs["run_id"] if (d / f"{r}.csv").exists()]


def plot_confusion(cm: np.ndarray, title: str, fname: str):
    import matplotlib.pyplot as plt

    rates = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            ax.text(j, i, f"{100 * rates[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if rates[i, j] > 0.5 else "black")
    ax.set_xticks(range(8), EMOTIONS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(8), EMOTIONS, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(utils.ensure_dir(FIG) / fname, dpi=200)
    return fig


def confusion_figures(runs: pd.DataFrame, results_dir=RES) -> dict:
    """Pooled (all folds x seeds) row-normalised confusion matrix per group + top confusions."""
    out = {}
    for (m, f, p, t), g in runs.groupby(["model", "features", "protocol", "tag"]):
        files = pred_files(g, results_dir)
        if not files:
            continue
        cm = evaluate.pooled_confusion(files)
        key = f"{m}_{f}_{p}_{t}"
        pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS).to_csv(utils.ensure_dir(TAB) / f"confusion_{key}.csv")
        plot_confusion(cm, f"{MODEL_NAMES.get(m, m)} | {f} | {p} | {t}\npooled over {len(files)} runs (row %)",
                       f"cm_{key}.png")
        out[key] = {"cm": cm, "top": evaluate.top_confusions(cm)}
    tops = [{"group": k, **r} for k, v in out.items() for r in v["top"]]
    pd.DataFrame(tops).to_csv(utils.ensure_dir(TAB) / "top_confusions.csv", index=False)
    return out


def plot_curves(runs: pd.DataFrame, results_dir=RES, fname_prefix="curves"):
    """Train/val loss and val UAR per epoch; one thin line per run, per (model, features, protocol, tag)."""
    import matplotlib.pyplot as plt

    d = utils.resolve(results_dir) / "curves"
    figs = {}
    for (m, f, p, t), g in runs.groupby(["model", "features", "protocol", "tag"]):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for rid in g["run_id"]:
            path = d / f"{rid}.csv"
            if not path.exists():
                continue
            c = pd.read_csv(path)
            axes[0].plot(c["epoch"], c["train_loss"], color="#4C72B0", alpha=0.35, lw=0.8)
            axes[0].plot(c["epoch"], c["val_loss"], color="#DD8452", alpha=0.35, lw=0.8)
            axes[1].plot(c["epoch"], c["val_uar"], color="#55A868", alpha=0.35, lw=0.8)
        axes[0].plot([], [], color="#4C72B0", label="train loss (augmented)")
        axes[0].plot([], [], color="#DD8452", label="val loss")
        axes[0].set_xlabel("epoch")
        axes[0].legend(fontsize=8)
        axes[1].axhline(0.125, ls="--", color="grey", lw=0.8, label="chance UAR")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("val UAR")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"{MODEL_NAMES.get(m, m)} | {f} | {p} | {t} ({len(g)} runs)", fontsize=9)
        fig.tight_layout()
        key = f"{m}_{f}_{p}_{t}"
        fig.savefig(utils.ensure_dir(FIG) / f"{fname_prefix}_{key}.png", dpi=200)
        figs[key] = fig
    return figs


def stats_tests(runs: pd.DataFrame, tag="main") -> pd.DataFrame:
    """Paired comparisons on the same folds/seeds (actor protocol). Primary: paired t-test on per-fold
    seed-averaged UAR (n=6); also Wilcoxon, and both at run level (n = folds x seeds)."""
    r = runs[(runs.protocol == "actor") & (runs.tag == tag)]
    groups = [(m, f) for (m, f) in r.groupby(["model", "features"]).groups.keys()]
    rows = []
    for (ma, fa), (mb, fb) in itertools.combinations(groups, 2):
        for level in ("fold", "run"):
            res = evaluate.compare_runs(r, {"model": ma, "features": fa}, {"model": mb, "features": fb}, level=level)
            rows.append({"A": f"{ma}/{fa}", "B": f"{mb}/{fb}", **res})
    df = pd.DataFrame(rows)
    df.to_csv(utils.ensure_dir(TAB) / f"stats_tests_{tag}.csv", index=False)
    return df


def leakage_table(runs: pd.DataFrame, tag="main") -> pd.DataFrame:
    """Q3: actor-wise vs random utterance-level split, M1-M3 (diagnostic only)."""
    rows = []
    for m in ("m1", "m2", "m3"):
        a = runs[(runs.model == m) & (runs.features == "logmel") & (runs.protocol == "actor") & (runs.tag == tag)]
        b = runs[(runs.model == m) & (runs.features == "logmel") & (runs.protocol == "random") & (runs.tag == tag)]
        if len(a) == 0 or len(b) == 0:
            continue
        rows.append({"model": MODEL_NAMES[m], "actor-wise UAR (%)": _pm(a["test_uar"]), "n_actor": len(a),
                     "random-split UAR (%)": _pm(b["test_uar"]), "n_random": len(b),
                     "inflation UAR (points)": round(100 * (b["test_uar"].mean() - a["test_uar"].mean()), 2),
                     "inflation acc (points)": round(100 * (b["test_acc"].mean() - a["test_acc"].mean()), 2)})
    df = pd.DataFrame(rows)
    df.to_csv(utils.ensure_dir(TAB) / f"leakage_{tag}.csv", index=False)
    return df


def ablation_table(runs: pd.DataFrame, tag="main") -> pd.DataFrame:
    r = runs[(runs.model == "m1") & (runs.protocol == "actor") & (runs.tag == tag)]
    rows = [{"features": f, "n": len(g), "UAR (%)": _pm(g["test_uar"]), "Accuracy (%)": _pm(g["test_acc"]),
             "Macro-F1 (%)": _pm(g["test_macro_f1"])} for f, g in r.groupby("features")]
    df = pd.DataFrame(rows)
    if set(r["features"]) >= {"logmel", "mfcc"}:
        t = evaluate.compare_runs(r, {"features": "logmel"}, {"features": "mfcc"}, level="fold")
        df.attrs["paired_test"] = t
        utils.save_json(t, TAB / f"ablation_test_{tag}.json")
    df.to_csv(utils.ensure_dir(TAB) / f"ablation_m1_features_{tag}.csv", index=False)
    return df


def hparam_plots() -> pd.DataFrame:
    """Template section 5.4 / D-10: ONE figure per searched hyper-parameter and model:
    validation UAR vs value (thin line per round, bold mean ± std). Plus a table of the selected values."""
    import matplotlib.pyplot as plt

    base = RES / "hparam_search"
    best_rows = []
    if not base.exists():
        return pd.DataFrame()
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        sp, rp = d / "summary.csv", d / "search.csv"
        if not sp.exists():
            continue
        s = pd.read_csv(sp, dtype={"value": str})
        raw = pd.read_csv(rp, dtype={"value": str, "overrides": str}, keep_default_na=False) if rp.exists() else None
        for fac in dict.fromkeys(s["factor"]):
            g = s[s.factor == fac].copy()
            g["v"] = pd.to_numeric(g["value"], errors="coerce")
            g = g.sort_values("v").reset_index(drop=True)
            labels = [str(v) for v in g["value"]]
            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            if raw is not None:  # per-round lines
                default_val = g.loc[g.is_default, "value"]
                for r, gr in raw.groupby("round"):
                    ys = []
                    for v, isd in zip(g["value"], g["is_default"]):
                        sel = gr[gr.factor == "default"] if isd else gr[(gr.factor == fac) & (gr.value == json.dumps(_num(v)))]
                        ys.append(100 * sel["val_uar"].mean() if len(sel) else np.nan)
                    ax.plot(range(len(g)), ys, color="grey", alpha=0.35, lw=0.8)
            ax.errorbar(range(len(g)), 100 * g["val_uar_mean"], yerr=100 * g["val_uar_std"].fillna(0),
                        marker="o", capsize=3, lw=2, color="#2E74B5", label="mean ± std over rounds")
            for i, isd in enumerate(g["is_default"]):
                if isd:
                    ax.axvline(i, ls=":", color="black", lw=0.8)
            ax.set_xticks(range(len(g)), labels)
            ax.set_xlabel(f"{fac}  (dotted = default)")
            ax.set_ylabel("validation UAR (%)")
            ax.set_title(f"{d.name}: {fac}", fontsize=9)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(utils.ensure_dir(FIG) / f"hparam_{d.name}_{fac.replace('.', '-')}.png", dpi=200)
            plt.close(fig)
        bj = d / "best.json"
        if bj.exists():
            b = utils.load_json(bj)
            best_rows.append({"model": d.name, **{k: json.dumps(v) for k, v in b["overrides"].items()}})
    df = pd.DataFrame(best_rows)
    df.to_csv(utils.ensure_dir(TAB) / "hparam_best.csv", index=False)
    return df


def _num(v):
    """summary.csv stores values as text; turn '0.001' / '128' back into numbers for json matching."""
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in str(v) and "e" not in str(v).lower() else f
    except (TypeError, ValueError):
        return v


def baseline_table(meta, splits) -> pd.DataFrame:
    b = evaluate.trivial_baselines(meta, splits)
    s = b.groupby("baseline")[["test_acc", "test_uar", "test_macro_f1"]].agg(["mean", "std"])
    s.to_csv(utils.ensure_dir(TAB) / "baselines.csv")
    return s
