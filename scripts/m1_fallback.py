"""Pre-approved by Khalid (24 Nov 14:29). Runs in the background (VS Code "Run Python File").

Waits for the D-20 combined-config check. If M1's final choice is NOT the combined config (combined < default
on validation) while main_m1 already ran with the combined config:
  1. MOVES (never deletes) the 18 superseded M1 main runs to results/superseded/m1_combined/
     (runs/, predictions/, curves/, their runs.csv rows, the job log) and their checkpoints to
     checkpoints/superseded_m1_combined/,
  2. re-runs M1 main (6 folds x 3 seeds) with final.json while the queue stays paused,
  3. verifies the 18 new runs and sets results/logs/PAUSE to "resume".
If the combined config was kept, it does nothing.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
LOGS = RES / "logs"
STATE = ROOT.parent / "SESSION_STATE_coding.md"
FINAL = RES / "hparam_search" / "m1_logmel" / "final.json"
SUP = RES / "superseded" / "m1_combined"
IDS = [f"m1_logmel_actor_f{f}_s{s}_main" for f in range(6) for s in range(3)]


def note(line):
    with open(STATE, "a", encoding="utf-8") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} [m1_fallback] {line}\n")


def wait_for_check():
    t0 = time.time()
    while True:
        if FINAL.exists() and FINAL.stat().st_mtime > t0 - 3 * 3600:
            rec = json.loads(FINAL.read_text(encoding="utf-8"))
            if time.time() - FINAL.stat().st_mtime > 60:  # give run_combined_check time to set PAUSE
                return rec
        time.sleep(30)


def move_superseded():
    for sub in ("runs", "predictions", "curves"):
        (SUP / sub).mkdir(parents=True, exist_ok=True)
    moved = 0
    for rid in IDS:
        for sub, ext in (("runs", ".json"), ("predictions", ".csv"), ("curves", ".csv")):
            src = RES / sub / f"{rid}{ext}"
            if src.exists():
                shutil.move(str(src), str(SUP / sub / src.name))
                moved += 1
        ck = ROOT / "checkpoints" / "results" / f"{rid}.pt"
        if ck.exists():
            dst = ROOT / "checkpoints" / "superseded_m1_combined"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ck), str(dst / ck.name))
    with open(RES / "runs.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        cols = list(rows[0].keys())
    keep = [r for r in rows if r["run_id"] not in IDS]
    gone = [r for r in rows if r["run_id"] in IDS]
    with open(SUP / "runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(gone)
    with open(RES / "runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(keep)
    log = LOGS / "final_main_m1.log"
    if log.exists():
        shutil.move(str(log), str(SUP / "final_main_m1_combined.log"))
    return moved, len(gone)


def child():
    rec = wait_for_check()
    pause = (LOGS / "PAUSE").read_text(encoding="utf-8").strip()
    if rec["choice"] == "combined":
        note("M1 combined config kept -> nothing to do")
        return
    if pause != "pause":
        note(f"M1 choice {rec['choice']} but queue not paused (main_m1 had not finished?) -> nothing moved")
        return
    moved, rows = move_superseded()
    note(f"M1 choice {rec['choice']} {rec['overrides']}: moved {moved} files + {rows} runs.csv rows to results/superseded/m1_combined/; re-running M1 main")
    cmd = [sys.executable, "-u", "-m", "src.run_experiments", "--model", "m1", "--folds", "0-5", "--seeds", "0,1,2",
           "--tag", "main", "--skip-existing", "--hparams", "results/hparam_search/m1_logmel/best.json"]
    ok = False
    for attempt in (1, 2):
        with open(LOGS / "final_main_m1.log", "a", encoding="utf-8") as f:
            f.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} (m1_fallback attempt {attempt}) " + " ".join(cmd) + "\n")
            f.flush()
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                           env=dict(os.environ, IGNORE_PAUSE="1", PYTHONUNBUFFERED="1"))
        sys.path.insert(0, str(ROOT))
        from src import verify_runs
        bad = [r for r in IDS if verify_runs.verify_run(r, results_dir=RES)]
        if not bad:
            ok = True
            break
    if ok:
        (LOGS / "PAUSE").write_text("resume\n", encoding="utf-8")
        note("M1 main re-run complete (18/18 verified) -> PAUSE set to resume")
    else:
        note(f"M1 main re-run FAILED twice (incomplete: {bad}); queue stays PAUSED")


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
    else:
        flags = (0x00000008 | 0x00000200 | 0x08000000) if os.name == "nt" else 0
        out = open(LOGS / "m1_fallback_runner.log", "a", encoding="utf-8")
        p = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--child"], cwd=ROOT,
                             stdout=out, stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)
        print(f"m1_fallback watcher started in the background, pid {p.pid}")
