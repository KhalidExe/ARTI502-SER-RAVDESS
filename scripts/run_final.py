"""Unattended final batch: remaining hyper-parameter searches, M4 checks, main protocol, ablation,
leakage diagnostic and final analysis, in this order.

    python scripts/run_final.py            # run (resumes: jobs already marked done are skipped)
    python scripts/run_final.py --detach   # start in the background and return immediately

Every job runs as `python -m ...` with its log in results/logs/final_<job>.log. A job counts as done only if
the process reported completion AND src.verify_runs confirms complete outputs (and the expected number of
runs). A failed job is retried once; after a second failure it is marked failed and jobs that depend on it
are skipped; independent jobs continue. Status: results/logs/final_status.json; a line is appended to
SESSION_STATE_coding.md after every job.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import verify_runs  # noqa: E402

LOGS = ROOT / "results" / "logs"
STATUS = LOGS / "final_status.json"
STATE = ROOT.parent / "SESSION_STATE_coding.md"
HP = "results/hparam_search"


def main_runs(model, best, features=None, protocol="actor", seeds="0,1,2"):
    a = ["src.run_experiments", "--model", model, "--seeds", seeds, "--tag", "main", "--skip-existing",
         "--hparams", f"{HP}/{best}/best.json"]
    if features:
        a += ["--features", features]
    if protocol == "random":
        a += ["--protocol", "random"]
    else:
        a += ["--folds", "0-5"]
    return a


# (step, name, module args, dependencies, kind, expected number of runs)
JOBS = [
    (1, "hp_m1", ["src.hparam_search", "--model", "m1"], [], "hp", 42),
    (2, "hp_m4", ["src.hparam_search", "--model", "m4", "--rounds", "0,3"], [], "hp", 6),
    (3, "checks_m4", ["src.checks", "--with-m4", "--overfit"], [], "checks", 0),
    (4, "main_m1", main_runs("m1", "m1_logmel"), ["hp_m1"], "runs", 18),
    (4, "main_m2", main_runs("m2", "m2_logmel"), [], "runs", 18),
    (4, "main_m3", main_runs("m3", "m3_logmel"), [], "runs", 18),
    (4, "main_m4", main_runs("m4", "m4_wave"), ["hp_m4"], "runs", 18),
    (5, "abl_m1_mfcc", main_runs("m1", "m1_logmel", features="mfcc"), ["hp_m1"], "runs", 18),
    (5, "leak_m1", main_runs("m1", "m1_logmel", protocol="random"), ["hp_m1"], "runs", 3),
    (5, "leak_m2", main_runs("m2", "m2_logmel", protocol="random"), [], "runs", 3),
    (5, "leak_m3", main_runs("m3", "m3_logmel", protocol="random"), [], "runs", 3),
    (6, "final_analysis", ["src.final_analysis", "--tag", "main"], [], "analysis", 0),
]


def load_status() -> dict:
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def save_status(st: dict) -> None:
    STATUS.write_text(json.dumps(st, indent=2), encoding="utf-8")


def note(line: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M")
    with open(STATE, "a", encoding="utf-8") as f:
        f.write(f"- {stamp} [run_final] {line}\n")


def run_once(name: str, args: list[str]) -> tuple[int, Path, float]:
    log = LOGS / f"final_{name}.log"
    cmd = [sys.executable, "-u", "-m", *args]
    t0 = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
        f.write(f"\n[exit {p.returncode}] {time.time() - t0:.0f} s\n")
    return p.returncode, log, time.time() - t0


def hp_rows(name: str) -> int:
    model = name.split("_")[1]
    feats = "wave" if model == "m4" else "logmel"
    path = ROOT / HP / f"{model}_{feats}" / "search.csv"
    if not path.exists():
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        return len({(r["config_hash"], r["round"], r["seed"]) for r in csv.DictReader(f)})


def evaluate(name, kind, expected, rc, log) -> tuple[bool, str]:
    text = log.read_text(encoding="utf-8", errors="replace")
    last = text[text.rfind("### "):]          # only the latest attempt
    finished = ("[done] exit code 0" in last) or rc == 0
    if kind == "checks":
        ran = "[done] exit code" in last
        cj = ROOT / "results" / "checks" / "checks.json"
        if not ran or not cj.exists():
            return False, f"checks did not finish (exit {rc})"
        res = json.loads(cj.read_text(encoding="utf-8"))["checks"]
        fails = [r["check"] for r in res if not r["passed"]]
        return True, f"{len(res) - len(fails)}/{len(res)} checks passed" + (f"; FAILED: {fails}" if fails else "")
    if kind == "analysis":
        return finished, "analysis finished" if finished else f"analysis failed (exit {rc})"
    tmp = LOGS / f"_tmp_{name}.log"
    tmp.write_text(last, encoding="utf-8")
    v = verify_runs.verify_log(tmp)
    tmp.unlink(missing_ok=True)
    if kind == "hp":
        n = hp_rows(name)
        model = name.split("_")[1]
        best = ROOT / HP / f"{model}_{'wave' if model == 'm4' else 'logmel'}" / "best.json"
        ok = finished and v["ok"] and n >= expected and best.exists()
        return ok, f"{n}/{expected} search runs, best.json={'yes' if best.exists() else 'no'}; {v['summary']}"
    ok = finished and v["ok"] and v["n_runs"] >= expected
    return ok, v["summary"] + f" (expected {expected})"


def main():
    if "--detach" in sys.argv:
        flags = 0x00000008 | 0x00000200 | 0x08000000 if os.name == "nt" else 0  # DETACHED|NEW_GROUP|NO_WINDOW
        out = open(LOGS / "final_runner.log", "a", encoding="utf-8")
        p = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())], cwd=ROOT, stdout=out,
                             stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)
        print(f"started run_final in the background, pid {p.pid}; log results/logs/final_runner.log")
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    st = load_status()
    st["_runner"] = {"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_status(st)
    note(f"runner started (pid {os.getpid()})")
    for step, name, args, deps, kind, expected in JOBS:
        if st.get(name, {}).get("state") == "done":
            continue
        bad = [d for d in deps if st.get(d, {}).get("state") != "done"]
        if bad:
            st[name] = {"step": step, "state": "skipped", "detail": f"dependency not done: {bad}"}
            save_status(st)
            note(f"step {step} {name}: SKIPPED (dependency {bad} not done)")
            continue
        detail, ok, total = "", False, 0.0
        for attempt in (1, 2):
            st[name] = {"step": step, "state": "running", "attempt": attempt,
                        "started": time.strftime("%Y-%m-%d %H:%M:%S")}
            save_status(st)
            print(f"{time.strftime('%H:%M:%S')} start {name} (attempt {attempt})", flush=True)
            rc, log, secs = run_once(name, args)
            total += secs
            ok, detail = evaluate(name, kind, expected, rc, log)
            print(f"{time.strftime('%H:%M:%S')} {name}: {'OK' if ok else 'FAILED'} - {detail}", flush=True)
            if ok:
                break
        st[name] = {"step": step, "state": "done" if ok else "failed", "detail": detail,
                    "minutes": round(total / 60, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        save_status(st)
        note(f"step {step} {name}: {'done' if ok else 'FAILED twice'} in {total / 60:.0f} min - {detail}")
    st["_runner"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_status(st)
    note("runner finished")


if __name__ == "__main__":
    main()
