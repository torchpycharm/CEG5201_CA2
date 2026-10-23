# CEG5201 CA2: team guide

## 1. Set up the same environment

Everyone should use **Python 3.14.7** and the package versions pinned in
`requirements.txt`; a different interpreter changes the measured times.

With conda (macOS or Windows):

```
conda create -n ceg5201 python=3.14.7
conda activate ceg5201
pip install -r requirements.txt
```

Or with python.org's 3.14.7 installer and a virtual environment:

```
python -m venv .venv                # Windows: py -3.14 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Check with `python --version` (must print 3.14.7), then run `python show_setup.py`:
it lists who owns which algorithm file and describes your machine.

## 2. Use the same matrix data

```
python generate_data.py
```

This writes `data/G0.npz ...` from the fixed seed `SEED` in `config.py` and prints a
**data fingerprint**. It must be `b7c06b8479fd4602`: the same seed and the pinned NumPy
version give bit-identical matrices on every machine, so `data/` (about 110 MB) is not committed to git and each person
regenerates it in a few seconds. If your fingerprint differs, check your NumPy version.

All shared settings (seed, number of groups, matrix sizes, process counts, timing
rules) live in `config.py` only. Do not change them without telling the team.

## 3. Seeds inside your algorithm file

You do not need a seed or any random numbers in your algorithm file. The benchmark
loads the shared matrices and passes them to your functions as arguments. If you want
test matrices of your own while developing, load the shared data.

```python
import config
from common import data

pairs = data.load_group(0, max_size=256)   # G0 pairs up to 256 x 256: [(A, B), ...]
A, B = pairs[-1]
```

Your own tuning parameter (e.g. `STRASSEN_LEAF_SIZE`) also goes in `config.py`;
read it with `import config` in your file.

## 4. Write and run your algorithm file

Implement the two functions in `algorithms/<your algorithm>.py`:

```python
def multiply_seq(A, B):          # sequential version, returns A x B
def multiply_mp(A, B, pool, p):  # same product using `pool`, a ready Pool of p workers
```

Rules from the revised CA2 statement and FAQ, in short:
- Additions, subtractions and multiplications are your own Python loops (Strassen and
  Coppersmith-Winograd: an i-j-k triple loop as base case). NumPy only to store and
  slice matrices: no `np.matmul`, `np.dot`, `@`, `np.add`, `np.multiply`, `A + B`.
- Cannon and Fox: a fixed 4 x 4 grid of blocks. Appendix A corrections are in your stub.
- Parallel work goes through the given `pool` only: no threads, no `shared_memory`.
- Functions sent to the pool must be top-level functions of your file.

Algorithm files are modules, so do not run them directly. Run them through the
project scripts, from the project folder (names: strassen, coppersmith_winograd,
cannon, fox),the follow commands are exampled by "strassen". 

```
python verify.py --algo strassen                  # correctness vs NumPy, must end "All checks passed."
python profile_seq.py --algo strassen --size 512  # where your sequential time goes (report, C-n.1)
python run_benchmark.py --algo strassen --quick   # 3 groups G0-G2, all sizes -> results_quick/
python analyze.py --algo strassen --quick         # its tables and figures
python run_benchmark.py --algo strassen           # all NUM_GROUPS groups -> results/strassen/
python analyze.py --algo strassen                 # their tables and figures
```

`--quick` is the 3-group benchmark the revised statement allows (G0, G1, G2 with all
eight sizes); it takes roughly a third of the full run, but still a while for 2048 x
2048. For a fast correctness check use `verify.py`. This file is using numpy to check if your code calculation correct.

Final results for the report are run by one person on the team's chosen laptop, for
all four algorithms, so commit your algorithm file once `verify.py` passes.

## 5. How timing works

All timing is done by `common/bench.py`, identically for the four algorithms; your
functions contain no timing code.

For every group, in this order:

| Step | Sequential | Multiprocessing | Timed? |
|---|---|---|---|
| 1 | load the group from disk, convert to int64 arrays | same | no |
| 2 | `gc.collect()` | same | no |
| 3 | - | create a Pool of p workers, wait until all p run (time saved to `C-n_pool_startup.csv`) | no |
| 4 | **start the clock** | **start the clock** | |
| 5 | `multiply_seq(A, B)` for each of the 8 pairs | `multiply_mp(A, B, pool, p)` for each of the 8 pairs | **yes** |
| 6 | read the clock after each pair -> T_0 ... T_7 | same | |
| 7 | - | close the Pool | no |
| 8 | save T_0 ... T_7 to `results/<algo>/raw/` | same | no |

Step 5 includes everything inside your two functions: slicing, NumPy -> list
conversion, the arithmetic, sending data to the workers and back, joining C.
`analyze.py` then derives t_i = T_i - T_(i-1), the group total T_7 and the cumulative
totals across groups. Run order: sequential over all groups, then p = 1, 2, ..., 6,
each over all groups.

## 6. Script options

Every script is run from the project folder; `python <script> -h` prints the same
options. `--algo` takes one of `strassen`, `coppersmith_winograd`, `cannon`, `fox`
(some scripts also accept `all`). Defaults come from `config.py`.

**`show_setup.py`** - no options. Prints the member -> algorithm table, the machine
(CPU, cores, RAM, OS, Python, NumPy) and the benchmark protocol.

**`generate_data.py`** - no options. Writes `data/G0.npz ...` (`NUM_GROUPS` groups) and
prints the data fingerprint.

**`verify.py`** - correctness check against NumPy (no timing).

| Option | Default | Meaning |
|---|---|---|
| `--algo NAME` | required | algorithm to check, or `all` |
| `--max-size M` | `VERIFY_MAX_SIZE` (256) | check the G0 pairs up to M x M (e.g. 512, slower) |
| `--procs P [P ...]` | `PROCESS_COUNTS` (1..6) | process counts for `multiply_mp`, e.g. `--procs 2 6` |

Prints one `OK` / `WRONG at sizes [...]` line per version; exits with code 1 on any error.

**`profile_seq.py`** - cProfile of one `multiply_seq` call (bottleneck for C-n.1).

| Option | Default | Meaning |
|---|---|---|
| `--algo NAME` | required | algorithm to profile |
| `--size M` | 512 | which G0 pair to multiply: 16, 32, ..., 2048 (2048 takes minutes) |
| `--top N` | 12 | number of functions listed |

Saves the table to `results/<algo>/profile/seq_<M>.txt`.

**`run_benchmark.py`** - the timing runs for Tables B-n and C-n.

| Option | Default | Meaning |
|---|---|---|
| `--algo NAME` | required | algorithm to time, or `all` (the final run) |
| `--mode seq\|mp\|all` | `all` | sequential only, multiprocessing only, or both (sequential first) |
| `--procs P [P ...]` | `PROCESS_COUNTS` (1..6) | process counts for `--mode mp`/`all`, e.g. re-run only `--procs 5 6` |
| `--quick` | off | G0-G2 only (`QUICK_NUM_GROUPS`, `QUICK_MAX_SIZE`), written to `results_quick/` |

Writes `raw/seq.json` and `raw/mp_p<p>.json`; each file is saved after every group, so
an interrupted run keeps the finished groups. Re-running a part overwrites only that part.

**`analyze.py`** - tables (CSV) and figures (PNG) from the raw results.

| Option | Default | Meaning |
|---|---|---|
| `--algo NAME` | every algorithm with results | one algorithm only (no comparison) |
| `--quick` | off | read `results_quick/` instead of `results/` |

can diy by yourself to draw figure


**`run_all.py`** - the final run on the common laptop: for each algorithm `verify.py`
then `run_benchmark.py`, waiting between algorithms until the CPU is back to its
starting speed (a short speed probe, settings `COOLDOWN_*` in `config.py`), then
`analyze.py` for all of them. Stops if a step fails.
python run_all.py                             # 全量：所有组，结果写到 results/
python run_all.py --quick                     # 3 组（G0–G2），结果写到 results_quick/
python run_all.py --quick --algos cannon coppersmith_winograd   # 只跑指定的算法

| Option | Default | Meaning |
|---|---|---|
| `--quick` | off | 3 groups G0-G2 -> `results_quick/` instead of all groups -> `results/` |
| `--algos NAME [NAME ...]` | all four | run only these, in this order |

**`make_submission.py`** - one argument: the exported report PDF, e.g.
`python make_submission.py report.pdf`. Writes `dist/CA2_<GroupID>_<matric numbers>.zip`.

Settings that are not command-line options (number of groups, sizes, seed, timing
rules, algorithm parameters) are changed in `config.py` only, after telling the team.

```
config.py           all shared settings (seed, sizes, process counts, algorithm parameters)
show_setup.py       who owns which algorithm + machine description
generate_data.py    creates the shared matrices in data/
verify.py           correctness check of an algorithm
profile_seq.py      cProfile of a sequential version
run_benchmark.py    timing runs -> results/<algorithm>/raw/
analyze.py          tables (CSV) and figures (PNG) -> results/<algorithm>/
run_all.py          one-click final run: all four algorithms with a CPU cool-down between them
make_submission.py  builds the Canvas zip
common/             shared code: data loading, timing harness, platform info, plots
algorithms/         one file per algorithm - the only place you write code
```
