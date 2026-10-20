"""[B-3] / [C-3]  Cannon's algorithm  (n = 3 in Table II).

Owner: ZHANG FANGYU (Member ID 2)

Reference: L. E. Cannon, "A Cellular Computer to Implement the Kalman Filter
Algorithm," PhD thesis, Montana State University, 1969.

How the algorithm works
-----------------------
A, B and C are cut into a 4 x 4 grid of blocks (q = config.CANNON_GRID = 4, as the
revised statement requires: 16 blocks per matrix). Every grid position (i, j) holds
one block of A and one block of B at a time:

1. Initial alignment (skew): block row i of A is rotated left by i positions and block
   column j of B is rotated up by j positions. Position (i, j) now holds A[i][k] and
   B[k][j] with k = (i + j) mod q, a pair that can be multiplied.
2. q stages: every position multiplies the pair it holds and adds the product to its
   block C[i][j]; then all A blocks move one position left and all B blocks one position
   up (circularly). At stage s position (i, j) holds the pair with k = (i + j + s) mod q,
   so over the q stages it meets every k exactly once: C[i][j] = sum_k A[i][k] B[k][j].

At every stage each block of A and of B sits at exactly one position, so no block is
ever needed by two positions at once (Fox's algorithm broadcasts blocks instead).

Implementation notes
--------------------
* All arithmetic is our own pure-Python loops: every block product is an i-j-k triple
  loop over lists of Python ints (_multiply_add). NumPy is only used to store the input
  and cut it into blocks; no NumPy arithmetic, no library product or sum.
* Block boundaries are t * M // q; every matrix size (16 ... 2048) is divisible by 4.
* Sequential version: the skew and the shifts are carried out literally, by rotating
  the grids of block references (no data is copied).

Bottleneck of the sequential version (python profile_seq.py --algo cannon)
-----------------------------------------------------------------------------
The block multiply-adds of the q stages (_multiply_add) take 99.7 % of the run time at
M = 512 and 99.9 % at M = 1024; skew, shifts, NumPy -> list conversion and joining are
O(M^2) work and together below 0.3 %. Within one stage the 16 block products are independent, and the q products
of one grid position only add into its own C block, so this is the work to spread
over processes.

Multiprocessing version
-----------------------
* Every grid position (i, j) becomes one pool task, a "virtual processor". The main
  process sends it, in stage order, the q block pairs that the skew and the shifts
  would bring to it; the task runs its q multiply-add stages and returns C[i][j]. The
  16 tasks are independent, so no barrier is needed between stages, and the pool hands
  them out one at a time (chunksize=1).
* Measured overheads are small: building tasks, pickling the blocks and joining C cost
  below 1 % of the run time. The remaining software loss is load imbalance: 16 equal
  tasks need 4 rounds on 4 or 5 workers and 3 rounds on 6, so with p = 5 the last round
  holds a single task while four workers wait (see results/cannon/profile/mp_diagnosis.txt).
"""
from itertools import chain

import numpy as np

import config


# ---------------------------------------------------------------------------
# Block helpers
# ---------------------------------------------------------------------------
def _edges(n, q):
    """Boundaries of the q block rows (= block columns) of an n x n matrix."""
    return [n * t // q for t in range(q + 1)]


def _block(M, edges, i, k):
    """Block (i, k) of the NumPy matrix M, as a view (no copy)."""
    return M[edges[i]:edges[i + 1], edges[k]:edges[k + 1]]


def _zeros(rows, cols):
    return [[0] * cols for _ in range(rows)]


def _multiply_add(C, A, Bt):
    """C += A x B in place, for list blocks; Bt is B transposed (Bt[j] = column j of B).

    Plain i-j-k triple loop: entry C[i][j] accumulates A[i][k] * B[k][j] over k.
    Storing B by columns lets the inner loop walk two Python lists side by side.
    """
    rows, cols, depth = len(A), len(Bt), len(Bt[0])
    for i in range(rows):
        a_row, c_row = A[i], C[i]
        for j in range(cols):
            b_col = Bt[j]
            total = c_row[j]
            for k in range(depth):
                total += a_row[k] * b_col[k]
            c_row[j] = total


def _join(grid):
    """Glue a q x q grid of list blocks back into one matrix (list of rows)."""
    return [list(chain.from_iterable(row_parts))
            for block_row in grid for row_parts in zip(*block_row)]


# ---------------------------------------------------------------------------
# [B-3] Sequential version
# ---------------------------------------------------------------------------
def multiply_seq(A, B):
    """[B-3] Sequential Cannon: skew once, then q stages of multiply-add and shift."""
    A, B = np.asarray(A), np.asarray(B)
    q = config.CANNON_GRID
    edges = _edges(len(A), q)

    # Initial alignment: position (i, j) receives A[i][(i+j)%q] and B[(i+j)%q][j].
    # B blocks are stored transposed (by columns) for the multiply-add kernel.
    A_grid = [[_block(A, edges, i, (i + j) % q).tolist() for j in range(q)] for i in range(q)]
    B_grid = [[_block(B, edges, (i + j) % q, j).T.tolist() for j in range(q)] for i in range(q)]
    C_grid = [[_zeros(edges[i + 1] - edges[i], edges[j + 1] - edges[j]) for j in range(q)]
              for i in range(q)]

    for _ in range(q):
        # Every position multiplies the pair it currently holds into its C block ...
        for i in range(q):
            for j in range(q):
                _multiply_add(C_grid[i][j], A_grid[i][j], B_grid[i][j])
        # ... then every A block moves one position left, every B block one position up.
        A_grid = [row[1:] + row[:1] for row in A_grid]
        B_grid = B_grid[1:] + B_grid[:1]

    return _join(C_grid)


# ---------------------------------------------------------------------------
# [C-3] Multiprocessing version
# ---------------------------------------------------------------------------
def _grid_position(a_blocks, b_blocks):
    """Worker task: all q stages of one grid position (i, j); returns block C[i][j].

    a_blocks[s] and b_blocks[s] are the NumPy blocks that the skew and s shifts bring
    to this position, i.e. A[i][k] and B[k][j] with k = (i + j + s) mod q.
    """
    C = _zeros(a_blocks[0].shape[0], b_blocks[0].shape[1])
    for a, b in zip(a_blocks, b_blocks):
        _multiply_add(C, a.tolist(), b.T.tolist())
    return C


def multiply_mp(A, B, pool, p):
    """[C-3] Cannon with its q x q grid positions run as independent tasks on `pool`."""
    A, B = np.asarray(A), np.asarray(B)
    q = config.CANNON_GRID
    edges = _edges(len(A), q)

    # One task per grid position, with its block pairs listed in stage order.
    tasks = []
    for i in range(q):
        for j in range(q):
            ks = [(i + j + s) % q for s in range(q)]
            tasks.append(([_block(A, edges, i, k) for k in ks],
                          [_block(B, edges, k, j) for k in ks]))

    blocks = pool.starmap(_grid_position, tasks, chunksize=1)   # row-major (i, j) order
    return _join([blocks[i * q:(i + 1) * q] for i in range(q)])
