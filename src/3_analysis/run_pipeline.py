"""
run_pipeline.py
Runs the project scripts in order.

Usage (from the folder containing the scripts):
    python run_pipeline.py                  # core pipeline
    python run_pipeline.py --stability      # also run cluster_stability.py
    python run_pipeline.py --from clustering.py   # resume from a given step
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # same ROOT the other scripts use
PROCESSED = ROOT / "dataset" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# Discovery/inspect scripts are skipped on purpose.
# cluster_scatter.py is skipped: cluster_profile.py already makes that plot.
STEPS = [
    "extract_features.py",
    "extract_final_features.py",
    "eda.py",
    "correlation_matrix.py",
    "elbow_method.py",
    "silhouette.py",
    "robust_compare.py",
    "pca_analysis.py",
    "pca_plot.py",
    "clustering.py",
    "cluster_profile.py",
]
OPTIONAL = ["cluster_stability.py"]

parser = argparse.ArgumentParser()
parser.add_argument("--stability", action="store_true", help="also run cluster_stability.py")
parser.add_argument("--from", dest="start", default=None, help="resume from this script")
args = parser.parse_args()

steps = STEPS + (OPTIONAL if args.stability else [])
if args.start:
    if args.start not in steps:
        sys.exit(f"--from: '{args.start}' is not in the step list: {steps}")
    steps = steps[steps.index(args.start):]

# Save plots instead of opening blocking windows
env = {**os.environ, "MPLBACKEND": "Agg"}

failed = None
start_all = time.time()
summary = []

for script in steps:
    path = HERE / script
    if not path.exists():
        print(f"[SKIP] {script} not found")
        summary.append((script, "skipped", 0.0))
        continue

    print(f"\n{'=' * 60}\n[RUN ] {script}\n{'=' * 60}")
    t0 = time.time()
    # cwd=PROCESSED so scripts that write/read "battles.csv" or
    # "final_battles.csv" use dataset/processed/ instead of the current folder
    result = subprocess.run([sys.executable, str(path)], cwd=PROCESSED, env=env)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"[ OK ] {script} ({elapsed:.1f}s)")
        summary.append((script, "ok", elapsed))
    else:
        print(f"[FAIL] {script} (exit code {result.returncode})")
        summary.append((script, "FAILED", elapsed))
        failed = script
        break  # later steps depend on earlier outputs

print(f"\n{'-' * 60}\nSummary")
for name, status, secs in summary:
    print(f"  {status:8s} {name:28s} {secs:6.1f}s")
print(f"Total time: {time.time() - start_all:.1f}s")

if failed:
    print(f"\nStopped at {failed}. Fix it, then resume with:\n"
          f"  python run_pipeline.py --from {failed}")
    sys.exit(1)
print("\nPipeline finished successfully.")