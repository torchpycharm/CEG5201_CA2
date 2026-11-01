CEG5201 CA2 - Benchmarking matrix multiplication with Python multiprocessing
===========================================================================
Group and members: see config.py (GROUP_ID, MEMBERS) and the first page of the report.

The program generates 10 groups (G0..G9) of 8 random integer matrix pairs (16 x 16 up
to 2048 x 2048, element values 0..255), multiplies them with four algorithms
(Strassen, Coppersmith-Winograd, Cannon, Fox), each in a sequential version and a
multiprocessing.Pool version, records the cumulative elapsed times, and builds the
tables and figures used in the report.


1. REQUIREMENTS
---------------
- Windows 10/11 or macOS (Intel or Apple Silicon), 64-bit.
- Python 3.14.7 (64-bit), the version all results were measured with:
  https://www.python.org/downloads/release/python-3147/
  (or with conda: conda create -n ceg5201 python=3.14.7)
- Python packages, installed in step 2 (exact versions are pinned in requirements.txt):
      numpy 2.5.3        https://pypi.org/project/numpy/
      pandas 3.0.6       https://pypi.org/project/pandas/
      matplotlib 3.11.2  https://pypi.org/project/matplotlib/
      psutil 7.2.2       https://pypi.org/project/psutil/
  Everything else used (multiprocessing, json, argparse, zipfile, ...) is part of the
  Python standard library.
- No data is downloaded from any URL: the input matrices are generated locally (3.2).
- About 110 MB of free disk space for the generated matrices.


2. Environment
------------------
Note: use python 3.14.7 for running
    pip install -r requirements.txt

    
3. RUNNING  (in this order)
---------------------------
3.1  python show_setup.py
     Prints the member -> algorithm assignment, the machine (CPU, cores, RAM, OS,
     Python and NumPy versions) and the benchmark protocol.

3.2  python generate_data.py
     Writes data/G0.npz .. data/G9.npz and prints a data fingerprint. The fixed seed
     gives the same fingerprint, i.e. bit-identical matrices, on every machine.

3.3  python verify.py --algo all
     Checks every sequential and multiprocessing implementation against NumPy on the
     G0 pairs up to 256 x 256. It must end with "All checks passed."

3.4  python run_benchmark.py --algo all
     The benchmark: for every algorithm, the sequential version and then the
     multiprocessing version with p = 1 .. min(6, CPU cores) processes, each on all
     ten groups. This takes several hours. It can also run one algorithm or one part
     at a time, for example:
         python run_benchmark.py --algo strassen
         python run_benchmark.py --algo strassen --mode seq
         python run_benchmark.py --algo strassen --mode mp --procs 5 6
     Algorithm names: strassen, coppersmith_winograd, cannon, fox.
     Raw results: results/<algorithm>/raw/seq.json and mp_p<p>.json

3.5  python analyze.py
     Builds the report tables and figures from the raw results.

Optional smoke test: add --quick to run_benchmark.py and analyze.py. It uses only G0,
G1 and G2 (all eight sizes) and writes to results_quick/ instead of results/.

One-click alternative to 3.3-3.5 (this is how the report results were produced):
     python run_all.py
     Runs, for each algorithm in turn, verify.py and then run_benchmark.py, and finally
     analyze.py for all of them. Between two algorithms it waits until the CPU is back
     to its starting speed (a short pure-Python speed probe; settings COOLDOWN_* in
     config.py), so heat from one algorithm does not slow down the next. On macOS it
     also keeps the computer awake. It stops if any step fails. Still run 3.2 first.
     Options:
         python run_all.py --quick                  # G0-G2 only -> results_quick/
         python run_all.py --algos cannon fox       # only these algorithms, in this order
         python run_all.py --results-dir results_windows   # write to another folder

Optional bottleneck analysis ([C-n.1]):
     python profile_seq.py --algo cannon --size 1024
     Profiles one sequential multiplication with cProfile, prints the functions with
     the largest share of the run time and saves the table to
     results/<algorithm>/profile/seq_<size>.txt.


4. OUTPUT FILES
---------------
n is the algorithm number: 1 Strassen, 2 Coppersmith-Winograd, 3 Cannon, 4 Fox.

  results/<algorithm>/tables/B-n.1.csv   sequential, G0: cumulative and per-pair times
  results/<algorithm>/tables/B-n.2.csv   sequential: time of every group, cumulative
  results/<algorithm>/tables/C-n.1.csv   multiprocessing, G0: cumulative and per-pair, per p
  results/<algorithm>/tables/C-n.2.csv   multiprocessing: time of every group, per p
  results/<algorithm>/tables/C-n_pool_startup.csv   worker-pool start-up time per p
  results/<algorithm>/tables/D-n.1.csv   speed-up S_p(k), k = 1..8 pairs of G0
  results/<algorithm>/tables/D-n.2.csv   speed-up from cumulative group times G0..G9
  results/<algorithm>/figures/D-n.1_speedup_vs_p.png   speed-up vs p (G0, k = 8)
  results/<algorithm>/figures/D-n.1_speedup_vs_k.png   speed-up vs k, one curve per p
  results/<algorithm>/figures/D-n.2_speedup_vs_p.png   speed-up vs p, all groups
  results/comparison/E_seq_time_vs_size.png            sequential time per pair
  results/comparison/E_speedup_vs_p.png                speed-up vs p, all algorithms
  results/comparison/E_speedup_vs_k.png                speed-up vs k at the largest p
  results/comparison/E_summary.csv                     key numbers per algorithm


5. HARD-CODED PARAMETERS
------------------------
All parameters are constants in config.py; no other file has settings and nothing
has to be changed to reproduce the results.

  GROUP_ID, MEMBERS        group number; (Member ID, name, matriculation number)
  SEED = 5201              seed of the random input matrices
  NUM_GROUPS = 10          groups G0..G9
  MATRIX_SIZES             [16, 32, 64, 128, 256, 512, 1024, 2048]
  VALUE_MIN, VALUE_MAX     0, 255 (inclusive)
  MATRIX_FORMAT            "numpy": the algorithms receive int64 NumPy arrays
  PROCESS_COUNTS           1 .. min(6, CPU cores), computed automatically
  START_METHOD             "spawn" (the only start method available on Windows)
  TIME_POOL_CREATION       False: worker-pool creation is outside the timed region
  STRASSEN_LEAF_SIZE, CW_LEAF_SIZE, CANNON_GRID, FOX_GRID
                           algorithm parameters
  QUICK_*, VERIFY_MAX_SIZE, FIGURE_*                         smoke test, check, figures

config.py also sets OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS,
VECLIB_MAXIMUM_THREADS and NUMEXPR_NUM_THREADS to 1 before NumPy is imported, so no
environment variable has to be set by hand.


6. TIMING PROTOCOL  (identical for all algorithms)
--------------------------------------------------
- A group is loaded from disk and converted to MATRIX_FORMAT before the clock starts.
- The clock (time.perf_counter) starts once per group and is read after every pair:
  these readings are the cumulative times T_i; the per-pair time is t_i = T_i - T_(i-1).
- Multiprocessing: a new Pool of p workers is created for every group and fully
  started before timing; its start-up time is stored separately. All matrix data sent
  to the workers is sent inside the timed region.
- Order: groups G0..G9, pairs from the smallest to the largest matrix; the sequential
  run first, then p = 1, 2, ..., 6.


7. SOURCE FILES
---------------
  config.py                all shared settings (see section 5)
  show_setup.py            prints assignment, machine and protocol
  generate_data.py         [A] creates the input matrices
  verify.py                correctness check against NumPy
  profile_seq.py           cProfile of a sequential implementation (bottleneck)
  run_benchmark.py         [B] and [C] timing runs
  analyze.py               [D] and [E] tables and figures
  run_all.py               one-click run of verify, benchmark and analysis with CPU cool-down
  make_submission.py       builds the submission zip
  common/data.py           generation, storage and loading of the matrices
  common/bench.py          timing harness and worker-pool creation
  common/sysinfo.py        description of the machine and protocol
  common/analysis.py       table and figure generation
  algorithms/strassen.py               Strassen (n = 1)
  algorithms/coppersmith_winograd.py   Coppersmith-Winograd (n = 2)
  algorithms/cannon.py                 Cannon (n = 3)
  algorithms/fox.py                    Fox (n = 4)
