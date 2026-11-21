"""Batch runner.

Examples
    # preliminary (untuned) M1, 6 folds x 1 seed
    python -m src.run_experiments --model m1 --folds 0-5 --seeds 0 --tag prelim
    # main protocol with tuned hyper-parameters
    python -m src.run_experiments --model m1 --folds 0-5 --seeds 0,1,2 --tag main \
        --hparams results/hparam_search/m1_logmel/best.json
    # MFCC ablation (M1 only)
    python -m src.run_experiments --model m1 --features mfcc --folds 0-5 --seeds 0,1,2 --tag main
    # leakage diagnostic (M1-M3 only)
    python -m src.run_experiments --model m3 --protocol random --seeds 0,1,2 --tag main
    # smoke test (separate results folder, never mixed with real runs)
    python -m src.run_experiments --model m2 --folds 0 --seeds 0 --epochs 2 --tag smoke --results-dir results/smoke
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from . import data, features as feat, train, utils, verify_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=utils.MODELS)
    ap.add_argument("--features", default=None, help="logmel | mfcc | wave (default: model's primary input)")
    ap.add_argument("--protocol", default="actor", choices=["actor", "random"])
    ap.add_argument("--folds", default="0-5")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--epochs", type=int, default=None, help="override max_epochs (smoke tests only)")
    ap.add_argument("--hparams", default=None, help="best.json from hparam_search")
    ap.add_argument("--override", action="append", default=[], help="dotted.key=value, e.g. train.lr=5e-4")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-checkpoint", action="store_true")
    a = ap.parse_args()

    features = a.features or utils.MODEL_FEATURES[a.model][0]
    if features not in utils.MODEL_FEATURES[a.model]:
        raise SystemExit(f"{a.model} does not take '{features}' (allowed: {utils.MODEL_FEATURES[a.model]})")
    if a.protocol == "random" and a.model == "m4":
        raise SystemExit("Leakage diagnostic is defined for M1-M3 only (see the proposal).")
    if a.tag == "main" and a.epochs is not None:
        raise SystemExit("--epochs is for smoke tests; do not use it with --tag main.")

    cfg = utils.load_config(a.model, overrides=a.override, extra_files=[a.hparams] if a.hparams else None)
    meta = data.load_metadata()
    splits = data.load_splits()
    banks = {features: feat.load_bank(cfg, features, meta, use_variants=cfg["augment"]["enabled"])}
    folds = utils.parse_int_list(a.folds) if a.protocol == "actor" else ["R"]
    seeds = utils.parse_int_list(a.seeds)

    out_root = utils.resolve(a.results_dir)
    done = set()
    runs_csv = out_root / "runs.csv"
    if a.skip_existing and runs_csv.exists():
        done = set(pd.read_csv(runs_csv)["run_id"])

    t0, results = time.time(), []
    for fold in folds:
        for seed in seeds:
            run_id = f"{a.model}_{features}_{a.protocol}_f{fold}_s{seed}_{a.tag}"
            if run_id in done:
                print(f"skip {run_id} (already in runs.csv)")
                continue
            r = train.run_single(cfg, features, a.protocol, fold if a.protocol == "actor" else None, seed,
                                 meta, banks, splits, out_root=out_root, tag=a.tag, max_epochs=a.epochs,
                                 save_checkpoint=False if a.no_checkpoint else None)
            problems = verify_runs.verify_run(r["run_id"], results_dir=out_root)
            if problems:
                raise RuntimeError(f"{r['run_id']}: outputs incomplete: {problems}")
            print(f"   outputs verified: runs.csv row, curves, {verify_runs.N_TEST} test predictions")
            results.append(r)
    if results:
        df = pd.DataFrame(results)
        cols = ["fold", "seed", "best_epoch", "val_uar", "test_uar", "test_acc", "test_macro_f1", "train_time_s"]
        print("\n" + df[cols].to_string(index=False))
        for m in ("test_uar", "test_acc", "test_macro_f1"):
            print(f"{m}: {df[m].mean():.4f} ± {df[m].std(ddof=1) if len(df) > 1 else 0:.4f} (n={len(df)})")
    print(f"total time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
    utils.hard_exit(0)
