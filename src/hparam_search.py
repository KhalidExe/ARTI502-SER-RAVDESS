"""Hyper-parameter search on the VALIDATION folds only (the test fold is never evaluated).

One-factor-at-a-time grid around the approved defaults (configs/hparam_search.yaml):
every factor is varied over its grid while the others stay at the default; the default
config is trained once and shared. Score = mean validation UAR over the chosen rounds/seeds.

    python -m src.hparam_search --model m1
    python -m src.hparam_search --model m3 --rounds 0-5 --seeds 0
    python -m src.hparam_search --model m4 --rounds 0,3        # M4 is expensive

Outputs  results/hparam_search/<model>_<features>/
    search.csv   one row per (setting, round, seed)
    summary.csv  mean/std val UAR per (factor, value)
    best.json    {"overrides": {...}}  -> pass to run_experiments --hparams
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import yaml

from . import data, features as feat, train, utils, verify_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=utils.MODELS)
    ap.add_argument("--features", default=None)
    ap.add_argument("--rounds", default=None, help="default from configs/hparam_search.yaml")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--factors", default=None, help="comma list subset, e.g. train.lr,model.dropout")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()

    with open(utils.CONFIG_DIR / "hparam_search.yaml", "r", encoding="utf-8") as f:
        hs = yaml.safe_load(f)
    grid = hs[a.model]
    if a.factors:
        grid = {k: v for k, v in grid.items() if k in a.factors.split(",")}
    rounds = utils.parse_int_list(a.rounds or hs.get("rounds", "0-5"))
    seeds = utils.parse_int_list(a.seeds or hs.get("seeds", "0"))
    features = a.features or utils.MODEL_FEATURES[a.model][0]

    base = utils.load_config(a.model)
    meta = data.load_metadata()
    splits = data.load_splits()
    banks = {features: feat.load_bank(base, features, meta, use_variants=base["augment"]["enabled"])}
    out = utils.ensure_dir(utils.RESULTS_DIR / "hparam_search" / f"{a.model}_{features}")
    search_csv = out / "search.csv"

    # settings: default + every non-default grid value of every factor
    settings = [("default", None, {})]
    defaults = {}
    for factor, values in grid.items():
        d = utils.get_by_path(base, factor)
        defaults[factor] = d
        for v in values:
            if v != d:
                settings.append((factor, v, {factor: v}))

    done = set()
    if search_csv.exists():
        prev = pd.read_csv(search_csv, dtype={"value": str, "overrides": str}, keep_default_na=False)
        done = set(zip(prev["config_hash"], prev["round"].astype(int), prev["seed"].astype(int)))

    for factor, value, ov in settings:
        cfg = utils.load_config(a.model, overrides=ov)
        h = utils.config_hash(cfg)
        for r in rounds:
            for s in seeds:
                if (h, r, s) in done:
                    continue
                tag = f"hp{h}"
                res = train.run_single(cfg, features, "actor", r, s, meta, banks, splits, out_root=out, tag=tag,
                                       eval_test=False, max_epochs=a.epochs, save_checkpoint=False,
                                       runs_csv_name="search_runs.csv")
                # D-10: the search must never touch the test fold
                assert "test_uar" not in res and not (out / "predictions" / f"{res['run_id']}.csv").exists(), \
                    "hyper-parameter search touched the test fold"
                problems = verify_runs.verify_run(res["run_id"], results_dir=out, csv_name="search_runs.csv")
                if problems:
                    raise RuntimeError(f"{res['run_id']}: outputs incomplete: {problems}")
                row = {"factor": factor, "value": json.dumps(value), "overrides": json.dumps(ov), "config_hash": h,
                       "round": r, "seed": s, "val_uar": res["val_uar"], "val_acc": res["val_acc"],
                       "val_macro_f1": res["val_macro_f1"], "best_epoch": res["best_epoch"],
                       "epochs_run": res["epochs_run"], "train_time_s": res["train_time_s"], "n_params": res["n_params"]}
                pd.DataFrame([row]).to_csv(search_csv, mode="a", header=not search_csv.exists(), index=False)

    # ---- summary + selection (validation UAR only)
    df = pd.read_csv(search_csv, dtype={"value": str, "overrides": str}, keep_default_na=False)
    df = df[df["round"].isin(rounds) & df["seed"].isin(seeds)]
    rows, best = [], {}
    default_rows = df[df["factor"] == "default"]
    for factor, values in grid.items():
        for v in sorted(set(values) | {defaults[factor]}, key=lambda x: (str(type(x)), x)):
            sub = default_rows if v == defaults[factor] else df[(df["factor"] == factor) & (df["value"] == json.dumps(v))]
            if len(sub) == 0:
                continue
            rows.append({"factor": factor, "value": v, "is_default": v == defaults[factor], "n": len(sub),
                         "val_uar_mean": sub["val_uar"].mean(), "val_uar_std": sub["val_uar"].std(ddof=1),
                         "mean_best_epoch": sub["best_epoch"].mean(), "mean_time_s": sub["train_time_s"].mean()})
        fr = [r for r in rows if r["factor"] == factor]
        if fr:
            best[factor] = max(fr, key=lambda r: r["val_uar_mean"])["value"]
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary.csv", index=False)
    utils.save_json({"model": a.model, "features": features, "rounds": rounds, "seeds": seeds,
                     "selection": "one-factor-at-a-time on VALIDATION folds only (test never evaluated); per factor, the value "
                                  "with the highest mean validation UAR; values combined. Limitation: OFAT does not "
                                  "capture interactions between hyper-parameters.",
                     "defaults": defaults, "overrides": best}, out / "best.json")
    print(summ.to_string(index=False))
    print(f"\nbest (validation-selected): {best}\n-> {out / 'best.json'}")


if __name__ == "__main__":
    main()
    utils.hard_exit(0)
