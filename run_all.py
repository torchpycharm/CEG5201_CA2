"""One-click benchmark of all four algorithms, with a cool-down between them.

For every algorithm in Table II order: verify.py, then run_benchmark.py. Before the
next algorithm starts, the script waits until the CPU is back to its starting speed,
so a laptop heated by one algorithm does not throttle the next one. Finally
analyze.py builds the tables and figures of every algorithm and their comparison.
Each step must succeed before the next starts.

CPU temperature cannot be read on macOS or Windows without admin tools, so the
cool-down times a fixed pure-Python probe (see COOLDOWN_* in config.py): a hot,
throttled CPU runs it more slowly. On macOS the computer is also kept awake.

Usage:
    python run_all.py                       # all groups (NUM_GROUPS)  -> results/
    python run_all.py --quick               # 3 groups G0-G2, all sizes -> results_quick/
    python run_all.py --algos cannon fox    # only some algorithms
    python run_all.py --results-dir results_mac_M5   # keep this machine's results apart
"""
import config  # first import: pins BLAS threads before NumPy loads

import argparse
import os
import platform
import subprocess
import sys
import time


def step(title, *args):
    """Run one project script with this Python; stop everything if it fails."""
    print(f"\n=== {title}", flush=True)
    start = time.perf_counter()
    subprocess.run([sys.executable, *args], cwd=config.ROOT, check=True)
    print(f"=== {title}: done in {(time.perf_counter() - start) / 60:.1f} min", flush=True)


def probe():
    """Seconds for a fixed pure-Python workload (best of 3); grows when the CPU throttles."""
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        total = 0
        for i in range(15_000_000):
            total += i * i
        best = min(best, time.perf_counter() - start)
    return best


def cool_down(baseline):
    """Wait until the probe is within COOLDOWN_TOLERANCE of `baseline` (or COOLDOWN_MAX_S)."""
    limit = baseline * (1 + config.COOLDOWN_TOLERANCE)
    waited = 0
    speed = probe()
    while speed > limit and waited < config.COOLDOWN_MAX_S:
        print(f"    probe {speed:.3f} s > {limit:.3f} s, waiting {config.COOLDOWN_CHECK_S} s",
              flush=True)
        time.sleep(config.COOLDOWN_CHECK_S)
        waited += config.COOLDOWN_CHECK_S
        speed = probe()
    print(f"=== cool-down: {waited} s waited, probe {speed:.3f} s (start {baseline:.3f} s)",
          flush=True)


def main():
    names = list(config.ALGORITHMS.values())
    parser = argparse.ArgumentParser(description="Benchmark all algorithms with cool-downs.")
    parser.add_argument("--quick", action="store_true", help="3 groups (G0-G2) instead of all")
    parser.add_argument("--algos", nargs="+", choices=names, default=names,
                        help="algorithms to run, in this order (default: all four)")
    parser.add_argument("--results-dir",
                        help="folder for the results instead of results/ (e.g. results_mac_M5)")
    args = parser.parse_args()
    quick = ["--quick"] if args.quick else []
    if args.results_dir:
        quick += ["--results-dir", args.results_dir]

    if platform.system() == "Darwin":
        # Prevent idle sleep for as long as this process is alive.
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])

    start = time.perf_counter()
    baseline = probe()
    print(f"Speed probe before the first algorithm: {baseline:.3f} s", flush=True)

    for index, name in enumerate(args.algos):
        if index:
            cool_down(baseline)
        step(f"{name}: verify", "verify.py", "--algo", name)
        step(f"{name}: benchmark", "run_benchmark.py", "--algo", name, *quick)

    step("tables and figures", "analyze.py", *quick)
    results = (args.results_dir if args.results_dir
               else (config.QUICK_RESULTS_DIR if args.quick else config.RESULTS_DIR).name)
    print(f"\nAll done in {(time.perf_counter() - start) / 3600:.1f} h. Results: {results}")


if __name__ == "__main__":
    main()
