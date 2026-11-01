"""[B-3] / [C-3]  Cannon's algorithm

Owner: ZHANG FANGYU (Member ID 2)

Reference: L. E. Cannon, "A Cellular Computer to Implement the Kalman Filter
Algorithm," PhD thesis, Montana State University, 1969.
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
