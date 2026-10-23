"""[B-n] + [C-n] Time the sequential and multiprocessing versions on G0..G9.

Raw results go to results/<algorithm>/raw/seq.json and mp_p<p>.json.

Usage:
    python run_benchmark.py --algo strassen                  # sequential, then p = 1..6
    python run_benchmark.py --algo strassen --mode seq       # sequential only
    python run_benchmark.py --algo fox --mode mp --procs 5 6 # re-run selected p only
    python run_benchmark.py --algo all                       # final run, every algorithm
    python run_benchmark.py --algo cannon --quick            # 3-group run -> results_quick/
"""
import config  # first import: pins BLAS threads before NumPy loads

import argparse

from common import bench, sysinfo


def main():
    parser = argparse.ArgumentParser(description="Benchmark the matrix multiplication algorithms.")
    parser.add_argument("--algo", required=True, choices=[*config.ALGORITHMS.values(), "all"])
    parser.add_argument("--mode", choices=["seq", "mp", "all"], default="all",
                        help="sequential, multiprocessing, or both (default)")
    parser.add_argument("--procs", type=int, nargs="+", default=config.PROCESS_COUNTS,
                        help="process counts for --mode mp (default: config.PROCESS_COUNTS)")
    parser.add_argument("--quick", action="store_true",
                        help="G0..G(QUICK_NUM_GROUPS-1), pairs up to QUICK_MAX_SIZE, "
                             "written to results_quick/")
    parser.add_argument("--results-dir",
                        help="folder for the results instead of results/ (e.g. results_mac_M5)")
    args = parser.parse_args()

    names = list(config.ALGORITHMS.values()) if args.algo == "all" else [args.algo]
    groups = range(config.QUICK_NUM_GROUPS if args.quick else config.NUM_GROUPS)
    max_size = config.QUICK_MAX_SIZE if args.quick else config.MATRIX_SIZES[-1]
    results_dir = (config.ROOT / args.results_dir if args.results_dir
                   else config.QUICK_RESULTS_DIR if args.quick else config.RESULTS_DIR)
    process_counts = ([None] if args.mode in ("seq", "all") else []) + \
                     (args.procs if args.mode in ("mp", "all") else [])

    print(sysinfo.summary())
    for name in names:
        print(f"\n== {name}")
        for p in process_counts:
            bench.run(name, p, groups, max_size, results_dir / name / "raw")


if __name__ == "__main__":
    main()
