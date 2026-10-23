"""[D-n] + [E] Build the report tables (CSV) and figures from the raw benchmark results.

Usage:
    python analyze.py                  # every algorithm with results, plus the [E] comparison
    python analyze.py --algo strassen  # one algorithm only
    python analyze.py --quick          # use the results of run_benchmark.py --quick
"""
import config  # first import: pins BLAS threads before NumPy loads

import argparse

from common import analysis


def main():
    parser = argparse.ArgumentParser(description="Build report tables and figures.")
    parser.add_argument("--algo", choices=list(config.ALGORITHMS.values()),
                        help="one algorithm (default: every algorithm that has results)")
    parser.add_argument("--quick", action="store_true", help="read results_quick/")
    parser.add_argument("--results-dir",
                        help="folder to read instead of results/ (e.g. results_mac_M5)")
    args = parser.parse_args()

    results_dir = (config.ROOT / args.results_dir if args.results_dir
                   else config.QUICK_RESULTS_DIR if args.quick else config.RESULTS_DIR)
    names = [args.algo] if args.algo else [
        name for name in config.ALGORITHMS.values()
        if (results_dir / name / "raw" / "seq.json").exists()]

    all_runs = [analysis.load(results_dir, name) for name in names]
    for runs in all_runs:
        print(f"== {runs.name}")
        for path in analysis.write_algorithm(runs, results_dir / runs.name):
            print(f"  {path.relative_to(config.ROOT)}")

    complete = [runs for runs in all_runs if runs.mp]
    if len(complete) > 1:
        print("== comparison [E]")
        for path in analysis.write_comparison(complete, results_dir / "comparison"):
            print(f"  {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
