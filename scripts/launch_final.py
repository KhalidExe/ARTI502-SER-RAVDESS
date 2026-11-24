"""One-click launcher (VS Code "Run Python File"): dry-tests the analysis code on the preliminary runs,
then starts scripts/run_final.py in the background and exits. Safe to run again: finished jobs are skipped."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "logs"
LOG.mkdir(parents=True, exist_ok=True)

status = LOG / "final_status.json"
if not (status.exists() and '"state": "running"' in status.read_text(encoding="utf-8")):
    r = subprocess.run([sys.executable, "-u", "-m", "src.final_analysis", "--tag", "prelim", "--no-notebook"],
                       cwd=ROOT, capture_output=True, text=True)
    (LOG / "analysis_dry_test.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    print("analysis dry test:", "see results/logs/analysis_dry_test.log")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_final.py"), "--detach"], cwd=ROOT)
else:
    print("run_final already running (results/logs/final_status.json) - not started again")
