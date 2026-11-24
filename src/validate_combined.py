"""D-20: combined-config validation check (M1, M2, M3) — VALIDATION folds only, the test fold is never touched.

OFAT (D-10) selected each hyper-parameter separately. Here the combined selected config (best.json) is trained
on the same 6 validation rounds x seed 0 as the search and compared with the default config's validation UAR
from the search:
  combined >= default  -> use the combined config
  combined <  default  -> use the best single-factor setting from the search (or the defaults if none beat them)
The choice is written to results/hparam_search/<model>_<features>/final.json, which run_experiments uses
instead of best.json. Table: results/tables/combined_check.csv.

    python -m src.validate_combined [--models m1,m2,m3]
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from . import data, features as feat, train, utils, verify_runs

HP = utils.RESULTS_DIR / "hparam_search"


def search_scores(d):
    df = pd.read_csv(d / "search.csv", dtype={"value": str, "overrides": str}, keep_default_na=False)
    df = df[df["seed"] == 0]
    out = {}
    for ov, g in df.groupby("overrides"):
        out[ov] = {"val_uar": g["val_uar"].mean(), "n": len(g), "rounds": sorted(g["round"].astype(int))}
    return out


def check_model(model, meta, splits):
    features = utils.MODEL_FEATURES[model][0]
    d = HP / f"{model}_{features}"
    best = utils.load_json(d / "best.json")
    cfg = utils.load_config(model, overrides=best["overrides"])
    banks = {features: feat.load_bank(cfg, features, meta, use_variants=cfg["augment"]["enabled"])}
    out = d / "combined"
    vals = []
    for r in range(6):
        res = train.run_single(cfg, features, "actor", r, 0, meta, banks, splits, out_root=out, tag="combined",
                               eval_test=False, save_checkpoint=False, runs_csv_name="search_runs.csv")
        assert "test_uar" not in res, "combined check touched the test fold"
        problems = verify_runs.verify_run(res["run_id"], results_dir=out, csv_name="search_runs.csv")
        if problems:
            raise RuntimeError(f"{res['run_id']}: outputs incomplete: {problems}")
        vals.append(res["val_uar"])
    combined = sum(vals) / len(vals)
    scores = search_scores(d)
    default = scores["{}"]["val_uar"]
    singles = {k: v["val_uar"] for k, v in scores.items() if k != "{}"}
    best_single_key = max(singles, key=singles.get)
    if combined >= default:
        choice, overrides = "combined", best["overrides"]
    elif singles[best_single_key] > default:
        choice, overrides = "best single-factor", json.loads(best_single_key)
    else:
        choice, overrides = "defaults", {}
    rec = {"model": model, "features": features, "rounds": list(range(6)), "seeds": [0],
           "combined_overrides": best["overrides"], "combined_val_uar": combined, "combined_per_round": vals,
           "default_val_uar": default, "best_single_factor": json.loads(best_single_key),
           "best_single_factor_val_uar": singles[best_single_key], "choice": choice, "overrides": overrides,
           "rule": "combined >= default -> combined; else best single-factor setting (validation UAR only)",
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    utils.save_json(rec, d / "final.json")
    print(f"{model}: combined {100 * combined:.2f} vs default {100 * default:.2f} -> {choice} {overrides}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="m1,m2,m3")
    a = ap.parse_args()
    meta, splits = data.load_metadata(), data.load_splits()
    recs = [check_model(m, meta, splits) for m in a.models.split(",")]
    rows = [{"model": r["model"], "combined config": json.dumps(r["combined_overrides"]),
             "val UAR combined (%)": round(100 * r["combined_val_uar"], 2),
             "val UAR default (%)": round(100 * r["default_val_uar"], 2),
             "best single factor": json.dumps(r["best_single_factor"]),
             "val UAR best single (%)": round(100 * r["best_single_factor_val_uar"], 2),
             "choice": r["choice"], "final overrides": json.dumps(r["overrides"])} for r in recs]
    pd.DataFrame(rows).to_csv(utils.ensure_dir(utils.RESULTS_DIR / "tables") / "combined_check.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
    utils.hard_exit(0)
