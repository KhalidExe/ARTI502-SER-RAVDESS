"""Final analysis: every table and figure for the final report, from results/ (no training).

    python -m src.final_analysis --tag main

Writes results/tables/*_<tag>.*, results/figures/*_<tag>.png, hparam plots, and
results/tables/final_summary_<tag>.json (key numbers), then executes notebooks/03_results_analysis.ipynb
(with TAGS=<tag>) so the notebook in the repository shows the same outputs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis, data, utils

TAB = utils.RESULTS_DIR / "tables"
EDA_TABLES = ["counts_by_emotion.csv", "counts_by_actor.csv", "counts_by_gender.csv", "counts_emotion_x_intensity.csv",
              "counts_emotion_x_gender.csv", "duration_stats.csv", "descriptor_stats_overall.csv",
              "descriptor_stats_by_emotion.csv", "descriptor_association.csv", "audit_wav_headers.csv"]


def variance_table(runs: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Q1: are model differences larger than the variation across seeds and folds?
    fold_std = std of the 6 seed-averaged fold UARs; seed_std = mean over folds of the std across 3 seeds."""
    r = runs[(runs.protocol == "actor") & (runs.tag == tag)]
    rows = []
    for (m, f), g in r.groupby(["model", "features"]):
        fold_means = g.groupby("fold")["test_uar"].mean()
        seed_std = g.groupby("fold")["test_uar"].std(ddof=1).mean() if g["seed"].nunique() > 1 else np.nan
        rows.append({"model": analysis.MODEL_NAMES.get(m, m), "features": f, "n_runs": len(g),
                     "mean UAR (%)": round(100 * g["test_uar"].mean(), 2),
                     "std over all runs (pts)": round(100 * g["test_uar"].std(ddof=1), 2),
                     "std across folds, seed-averaged (pts)": round(100 * fold_means.std(ddof=1), 2),
                     "mean std across seeds within a fold (pts)": round(100 * seed_std, 2) if pd.notna(seed_std) else None,
                     "min fold UAR (%)": round(100 * fold_means.min(), 2), "max fold UAR (%)": round(100 * fold_means.max(), 2)})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / f"variance_{tag}.csv", index=False)
    return df


def _fmt(v):
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() and abs(f) >= 1 else f"{f:g}"
    except (TypeError, ValueError):
        return str(v)


def hparam_table() -> pd.DataFrame:
    """Template 5.4: per model and searched hyper-parameter, default vs validation-selected value."""
    rows = []
    for d in sorted((utils.RESULTS_DIR / "hparam_search").glob("*_*")):
        sp, bj = d / "summary.csv", d / "best.json"
        if not (sp.exists() and bj.exists()):
            continue
        s = pd.read_csv(sp, dtype={"value": str})
        b = utils.load_json(bj)
        for fac, sel in b["overrides"].items():
            g = s[s.factor == fac].copy()
            g["v"] = pd.to_numeric(g["value"], errors="coerce")
            best_row = g.loc[g["val_uar_mean"].idxmax()]
            def_row = g[g["is_default"].astype(str) == "True"]
            rows.append({"model": d.name, "hyper-parameter": fac, "values searched": ", ".join(_fmt(v) for v in g.sort_values("v")["value"]),
                         "default": _fmt(b["defaults"].get(fac)), "selected (val UAR)": _fmt(sel),
                         "val UAR default (%)": round(100 * def_row["val_uar_mean"].iloc[0], 2) if len(def_row) else None,
                         "val UAR selected (%)": round(100 * best_row["val_uar_mean"], 2),
                         "rounds x seeds": f"{len(b['rounds'])} x {len(b['seeds'])}"})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "hparam_selected.csv", index=False)
    return df


def extras(runs: pd.DataFrame, tag: str) -> dict:
    return {"variance": variance_table(runs, tag), "hparams": hparam_table()}


def key_numbers(runs, tag, summary, stats, leak, abl) -> dict:
    out = {"tag": tag, "n_runs": int(len(runs)),
           "groups": summary.to_dict(orient="records"),
           "leakage": leak.to_dict(orient="records") if len(leak) else [],
           "ablation": abl.to_dict(orient="records") if len(abl) else [],
           "stats_fold_level": stats[stats.level == "fold"].to_dict(orient="records") if len(stats) else [],
           "missing_eda_tables": [t for t in EDA_TABLES if not (TAB / t).exists()]}
    utils.save_json(out, TAB / f"final_summary_{tag}.json")
    return out


def run_notebook(tag: str) -> None:
    import nbformat
    from nbclient import NotebookClient

    path = utils.ROOT / "notebooks" / "03_results_analysis.ipynb"
    nb = nbformat.read(path, as_version=4)
    os.environ["TAGS"] = tag
    NotebookClient(nb, timeout=3600, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, path)


def main():
    import matplotlib

    matplotlib.use("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--no-notebook", action="store_true")
    a = ap.parse_args()
    tag = a.tag
    runs = analysis.load_runs(tags=[tag])
    if runs.empty:
        raise SystemExit(f"no runs with tag {tag}")
    print(runs.groupby(["model", "features", "protocol"]).size().to_string())
    summary = analysis.summary_table(runs, name=f"results_{tag}")
    analysis.per_class_recall_table(runs, name=f"per_class_recall_{tag}")
    analysis.confusion_figures(runs, name=f"top_confusions_{tag}")
    analysis.plot_curves(runs)
    stats = analysis.stats_tests(runs, tag=tag)
    leak = analysis.leakage_table(runs, tag=tag)
    abl = analysis.ablation_table(runs, tag=tag)
    analysis.hparam_plots()
    analysis.baseline_table(data.load_metadata(), data.load_splits())
    ex = extras(runs, tag)
    key_numbers(runs, tag, summary, stats, leak, abl)
    print(summary.to_string(index=False))
    print(ex["variance"].to_string(index=False))
    if not a.no_notebook:
        try:
            run_notebook(tag)
            print("notebook 03 executed")
        except Exception as e:  # the tables/figures above are already written
            print(f"WARNING: notebook 03 execution failed: {e!r}")


if __name__ == "__main__":
    main()
    utils.hard_exit(0)
