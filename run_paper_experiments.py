#!/usr/bin/env python
"""
GeneNarrator experiment runner.

Examples:
  python run_paper_experiments.py                # train + evaluate
  python run_paper_experiments.py --full         # setup + train + evaluate
  python run_paper_experiments.py --evaluate-only
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def run_script(script_name: str) -> bool:
    script_path = SCRIPTS / script_name
    if not script_path.exists():
        print(f"[ERROR] Missing script: {script_path}")
        return False

    print(f"\n[RUN] {script_name}")
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[FAIL] {script_name} exited with code {result.returncode}")
        return False

    print(f"[OK] {script_name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GeneNarrator reproducible workflow")
    parser.add_argument("--full", action="store_true", help="Run setup + train + evaluate")
    parser.add_argument("--setup-data", action="store_true", help="Run data setup step")
    parser.add_argument("--train", action="store_true", help="Run model training")
    parser.add_argument("--evaluate-only", action="store_true", help="Run evaluation only")
    args = parser.parse_args()

    steps = []

    if args.full:
        steps.extend([
            "setup_data.py",
            "train_from_scratch.py",
            "generate_evaluation_results.py",
        ])
    elif args.evaluate_only:
        steps.append("generate_evaluation_results.py")
    else:
        if args.setup_data:
            steps.append("setup_data.py")
        if args.train:
            steps.append("train_from_scratch.py")
        if not steps:
            steps.extend([
                "train_from_scratch.py",
                "generate_evaluation_results.py",
            ])

    ok = True
    for step in steps:
        ok = run_script(step) and ok

    if ok:
        print("\n[Done] Workflow completed successfully.")
        return 0

    print("\n[Done] Workflow finished with errors.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
