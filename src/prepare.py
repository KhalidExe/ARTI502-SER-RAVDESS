"""Data preparation CLI.

    python -m src.prepare download      # Zenodo 1188976 -> data/raw (Audio_Speech_Actors_01-24.zip)
    python -m src.prepare metadata      # data/metadata.csv + splits/label_map.json
    python -m src.prepare splits        # splits/folds.json + splits/random_split.json (+ assertions)
    python -m src.prepare features      # data/cache/* (waves, log-mel, MFCC; clean + augmented variants)
    python -m src.prepare all
"""
from __future__ import annotations

import argparse

from . import data, features, utils


def do_download(cfg):
    data.download_ravdess(cfg["data"]["zenodo_url"], cfg["data"]["raw_dir"])


def do_metadata(cfg):
    meta = data.build_metadata(cfg["data"]["raw_dir"])
    print(f"metadata: {len(meta)} files -> {data.METADATA_PATH}")
    print(meta.groupby("emotion").size().reindex(utils.EMOTIONS).to_string())
    for r in data.verify_data(meta):
        print(("PASS " if r["passed"] else "FAIL ") + r["check"], "|", r["detail"])
    return meta


def do_splits(cfg, force=False):
    meta = data.load_metadata()
    seed = int(cfg["seed_split"])
    if data.FOLDS_PATH.exists() and not force:
        print(f"{data.FOLDS_PATH} exists -> kept (folds are fixed once; use --force to recreate)")
    else:
        utils.save_json(data.make_actor_folds(seed), data.FOLDS_PATH)
    if data.RANDOM_SPLIT_PATH.exists() and not force:
        print(f"{data.RANDOM_SPLIT_PATH} exists -> kept")
    else:
        utils.save_json(data.make_random_split(meta, seed), data.RANDOM_SPLIT_PATH)
    splits = data.load_splits()
    print("\nfold | actors (F/M)")
    g = dict(zip(meta["actor"], meta["gender"]))
    for k, f in enumerate(splits["folds"]["folds"]):
        print(f"  {k}  | " + ", ".join(f"{a:02d}{g[a]}" for a in f))
    checks = data.verify_splits(meta, splits)
    for r in checks:
        print(("PASS " if r["passed"] else "FAIL ") + r["check"], "|", r["detail"])
    if not all(r["passed"] for r in checks):
        raise SystemExit("Split checks FAILED — do not train.")


def do_features(cfg, n_jobs):
    meta = data.load_metadata()
    features.build_feature_banks(cfg, meta, n_jobs=n_jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["download", "metadata", "splits", "features", "all"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = utils.load_config("m1")  # data/audio/feature/augment settings live in base.yaml
    if a.step in ("download", "all"):
        do_download(cfg)
    if a.step in ("metadata", "all"):
        do_metadata(cfg)
    if a.step in ("splits", "all"):
        do_splits(cfg, a.force)
    if a.step in ("features", "all"):
        do_features(cfg, a.n_jobs)


if __name__ == "__main__":
    main()
    utils.hard_exit(0)
