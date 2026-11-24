"""One-click (VS Code "Run Python File"): runs the D-20 combined-config check in the background.

It waits until the in-flight main_m1 job of run_final.py has finished (no GPU sharing), runs
`python -m src.validate_combined`, then sets results/logs/PAUSE to "resume" if no finished main run is affected.
If M1's final choice differs from the config main_m1 already used, the queue stays paused for a decision.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "logs"
STATE = ROOT.parent / "SESSION_STATE_coding.md"


def note(line):
    with open(STATE, "a", encoding="utf-8") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} [combined_check] {line}\n")


def child():
    status = LOGS / "final_status.json"
    while True:
        st = json.loads(status.read_text(encoding="utf-8")) if status.exists() else {}
        if st.get("main_m1", {}).get("state") != "running":
            break
        time.sleep(30)
    note("main_m1 no longer running; starting validate_combined")
    ok = False
    for attempt in (1, 2):
        with open(LOGS / "combined_check.log", "a", encoding="utf-8") as f:
            f.write(f"\n### attempt {attempt} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.flush()
            p = subprocess.run([sys.executable, "-u", "-m", "src.validate_combined"], cwd=ROOT, stdout=f,
                               stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        finals = [ROOT / "results" / "hparam_search" / f"{m}_logmel" / "final.json" for m in ("m1", "m2", "m3")]
        if all(x.exists() and time.time() - x.stat().st_mtime < 7200 for x in finals):
            ok = True
            break
    if not ok:
        note("validate_combined FAILED twice; queue stays PAUSED")
        return
    m1 = json.loads(finals[0].read_text(encoding="utf-8"))
    summary = "; ".join(f"{json.loads(x.read_text())['model']}: {json.loads(x.read_text())['choice']}" for x in finals)
    m1_done = json.loads((LOGS / "final_status.json").read_text()).get("main_m1", {}).get("state") == "done"
    if m1["choice"] != "combined" and m1_done:
        note(f"done ({summary}); M1 final config differs from the one main_m1 used -> queue stays PAUSED for a decision")
    else:
        (LOGS / "PAUSE").write_text("resume\n", encoding="utf-8")
        note(f"done ({summary}); PAUSE set to resume")


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
    else:
        flags = (0x00000008 | 0x00000200 | 0x08000000) if os.name == "nt" else 0
        out = open(LOGS / "combined_check_runner.log", "a", encoding="utf-8")
        p = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--child"], cwd=ROOT,
                             stdout=out, stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)
        print(f"combined check started in the background, pid {p.pid}; log results/logs/combined_check.log")
