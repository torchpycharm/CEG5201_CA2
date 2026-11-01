"""Single source of truth for every shared setting of the CA2 project.

Every script imports this module FIRST. Change settings here and nowhere else, so that
all members benchmark with identical data, process counts and timing rules.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# BLAS / OpenMP threads
# ---------------------------------------------------------------------------
# Must run before NumPy is imported anywhere. Worker processes inherit these
# variables, so no worker can oversubscribe the CPU with hidden BLAS threads.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

GROUP_ID = 4

MEMBERS = [
    # (Member ID, full name, matriculation number)   -> algorithm n
    (0, "ZHOU JUNMIN", "A0257966H"),                  # 1 Strassen
    (1, "RICARDO JACK LIWONGAN", "A0353688L"),        # 2 Coppersmith-Winograd
    (2, "ZHANG FANGYU", "A0353200Y"),                 # 3 Cannon
    (3, "DIVYALAKSHMI BALASUBRAMANIAM", "A0241265L"), # 4 Fox
]

# Table II: algorithm number n -> module in algorithms/ and display name.
ALGORITHMS = {1: "strassen", 2: "coppersmith_winograd", 3: "cannon", 4: "fox"}
ALGORITHM_TITLES = {1: "Strassen", 2: "Coppersmith-Winograd", 3: "Cannon", 4: "Fox"}


def algorithm_number(member_id):
    """n = (Group ID + Member ID) % 4 + 1  (Section II of the CA2 statement)."""
    return (GROUP_ID + member_id) % 4 + 1


# ---------------------------------------------------------------------------
# [A] Input matrices
# ---------------------------------------------------------------------------
SEED = 5201                     # same seed -> bit-identical G0..G9 on every machine
NUM_GROUPS = 10                 # G0 .. G9
MATRIX_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]   # pair i is M_i x M_i
VALUE_MIN, VALUE_MAX = 0, 255   # inclusive range of the random integer elements

# Format handed to multiply_seq / multiply_mp (converted before the clock starts):
#   "numpy" -> numpy.ndarray of int64        "list" -> list of lists of Python int
MATRIX_FORMAT = "numpy"

# ---------------------------------------------------------------------------
# [O] Benchmark protocol  (identical for every member; describe it in the report)
# ---------------------------------------------------------------------------
# p = 1 .. min(6, usable CPU cores of the benchmark machine)
PROCESS_COUNTS = list(range(1, min(6, os.cpu_count()) + 1))

# "spawn" is the only start method on Windows and the default on macOS.
START_METHOD = "spawn"

# C-n.1: is worker-pool creation inside the timed region? A fresh pool is created
# for every group. False: the clock starts once all p workers are up and running.
# True: the clock starts right before the pool is created.
TIME_POOL_CREATION = False

# ---------------------------------------------------------------------------
# Algorithm parameters  (each owner tunes only their own line)
# ---------------------------------------------------------------------------
STRASSEN_LEAF_SIZE = 64         # below this size, switch to the schoolbook product
CW_LEAF_SIZE = 64               # same, for Coppersmith-Winograd
CW_MP_DEPTH = 2                 # levels of recursion done by the main process: 7^d pool tasks
CANNON_GRID = 4                 # q: 4 x 4 grid of blocks (fixed by the revised statement)
FOX_GRID = 4                    # q: split each matrix into q x q blocks

# ---------------------------------------------------------------------------
# Cool-down between algorithms (run_all.py)
# ---------------------------------------------------------------------------
# CPU temperature cannot be read on macOS/Windows without admin tools, so run_all.py
# times a fixed pure-Python probe instead: the next algorithm starts once the probe is
# back within COOLDOWN_TOLERANCE of its value measured before the first algorithm.
COOLDOWN_TOLERANCE = 0.03       # 3 % slower than the starting probe still counts as cool
COOLDOWN_CHECK_S = 30           # seconds between two probes
COOLDOWN_MAX_S = 900            # start the next algorithm after 15 min at the latest

# ---------------------------------------------------------------------------
# Development shortcuts
# ---------------------------------------------------------------------------
QUICK_NUM_GROUPS = 3            # --quick: G0..G2 only (allowed by the revised statement) ...
QUICK_MAX_SIZE = 2048           # ... with all eight pair sizes
VERIFY_MAX_SIZE = 256           # verify.py checks the G0 pairs up to this size

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
FIGURE_FORMAT = "png"
FIGURE_DPI = 200
FIGURE_SIZE = (6.4, 3.6)        # inches

# ---------------------------------------------------------------------------
# Paths  (anchored at this file, so scripts work from any working directory)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
QUICK_RESULTS_DIR = ROOT / "results_quick"
DIST_DIR = ROOT / "dist"
