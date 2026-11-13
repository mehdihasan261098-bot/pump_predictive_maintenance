"""
Regenerate every generated artifact (datasets, ML models, reports) in order.

Datasets, models and the database are not stored in git; run this once after
cloning (about 3-4 minutes):

    python build_artifacts.py            # everything
    python build_artifacts.py --quick    # skip the slower validation reports

The dashboard also works without models (ML panels then show a hint).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Healthy baseline dataset + overview plot", "simulation.sensor_simulator", False),
    ("Fault datasets + fault signature plot", "simulation.fault_simulator", False),
    ("Pump performance report", "analysis.performance_report", False),
    ("Vibration comparison report", "analysis.vibration_report", False),
    ("ML feature dataset (51 runs)", "analysis.feature_extraction", False),
    ("Train + evaluate Isolation Forest / Random Forest", "detection.ml_detection", False),
    ("Rule-based diagnosis evaluation", "detection.rule_based", True),
    ("Health index / RUL demo", "predictive_maintenance.health_monitor", True),
    ("End-to-end fault scenario validation", "analysis.scenario_validation", True),
]


def main(quick: bool = False) -> int:
    for d in ("data/raw", "data/processed", "models/saved_models", "reports"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    for title, module, slow in STEPS:
        if quick and slow:
            print(f"[skip] {title}")
            continue
        t0 = time.perf_counter()
        print(f"[run ] {title} ... ", end="", flush=True)
        res = subprocess.run([sys.executable, "-m", module], cwd=ROOT, capture_output=True, text=True)
        if module == "detection.ml_detection":
            (ROOT / "reports/ml_training_log.txt").write_text(res.stdout + res.stderr, encoding="utf-8")
        if res.returncode != 0:
            print("FAILED")
            print(res.stdout[-3000:], res.stderr[-3000:])
            return res.returncode
        print(f"ok ({time.perf_counter() - t0:.0f} s)")
    print(f"All artifacts built in {time.perf_counter() - t_all:.0f} s.")
    return 0


if __name__ == "__main__":
    sys.exit(main("--quick" in sys.argv))
