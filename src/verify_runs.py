"""Check that every finished run left complete outputs (independent of the process exit code).

A run counts as complete when
  * its row is in runs.csv (search_runs.csv for hyper-parameter search runs),
  * curves/<run_id>.csv exists and has at least one epoch,
  * runs/<run_id>.json exists, and for runs evaluated on the test fold:
  * predictions/<run_id>.csv has exactly one row per test utterance (240) and
    test_access_count in the run json equals 240.

    python -m src.verify_runs                       # every run in results/runs.csv
    python -m src.verify_runs --tag prelim
    python -m src.verify_runs --log results/logs/prelim_m1.log   # runs started by one batch
Only the standard library is used, so it also works outside the project environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_TEST = 240


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def results_dir_for(run_id: str, base: Path) -> tuple[Path, str]:
    """Hyper-parameter search runs live in hparam_search/<model>_<features>/ with search_runs.csv."""
    if re.search(r"_hp[0-9a-f]{10}$", run_id):
        model, features = run_id.split("_")[:2]
        return base / "hparam_search" / f"{model}_{features}", "search_runs.csv"
    return base, "runs.csv"


def verify_run(run_id: str, base: Path | None = None, results_dir: Path | None = None,
               csv_name: str | None = None) -> list[str]:
    """Return a list of problems (empty list = complete)."""
    base = base or ROOT / "results"
    if results_dir:
        rdir, csv_name = Path(results_dir), csv_name or "runs.csv"
    else:
        rdir, default_csv = results_dir_for(run_id, base)
        csv_name = csv_name or default_csv
    problems = []
    if run_id not in {r["run_id"] for r in _rows(rdir / csv_name)}:
        problems.append(f"no row in {csv_name}")
    curves = rdir / "curves" / f"{run_id}.csv"
    if not curves.exists() or len(_rows(curves)) == 0:
        problems.append("curves file missing or empty")
    rj = rdir / "runs" / f"{run_id}.json"
    if not rj.exists():
        problems.append("run json missing")
        return problems
    info = json.loads(rj.read_text(encoding="utf-8"))
    if info.get("test_access_count") is not None:  # evaluated on the test fold
        preds = rdir / "predictions" / f"{run_id}.csv"
        n = len(_rows(preds))
        if n != N_TEST:
            problems.append(f"predictions rows = {n} (expected {N_TEST})")
        if info["test_access_count"] != N_TEST:
            problems.append(f"test_access_count = {info['test_access_count']}")
    return problems


def run_ids_in_log(log_path: Path) -> list[str]:
    """Run ids a batch started ('== <run_id> | ...') or skipped as already done ('skip <run_id>')."""
    ids = []
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^== (\S+) \|", line) or re.match(r"^skip (\S+) \(already", line)
        if m and m.group(1) not in ids:
            ids.append(m.group(1))
    return ids


def verify_log(log_path, base: Path | None = None) -> dict:
    ids = run_ids_in_log(Path(log_path))
    bad = {r: p for r in ids if (p := verify_run(r, base))}
    return {"ok": not bad, "n_runs": len(ids), "incomplete": bad,
            "summary": f"[verified] {len(ids) - len(bad)}/{len(ids)} runs complete" + (f"; INCOMPLETE: {bad}" if bad else "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    base = Path(a.results_dir)
    base = base if base.is_absolute() else ROOT / base
    if a.log:
        res = verify_log(a.log, ROOT / "results")
        print(res["summary"])
        sys.exit(0 if res["ok"] else 1)
    rows = [r for r in _rows(base / "runs.csv") if a.tag is None or r["tag"] == a.tag]
    bad = {r["run_id"]: p for r in rows if (p := verify_run(r["run_id"], results_dir=base))}
    print(f"[verified] {len(rows) - len(bad)}/{len(rows)} runs complete in {base}")
    for k, v in bad.items():
        print("  INCOMPLETE", k, v)
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
