"""[B-1] / [C-1]  Strassen's algorithm  (n = 1 in Table II).

Owner: ZHOU JUNMIN (Member ID 0)

Reference: V. Strassen, "Gaussian elimination is not optimal," Numer. Math., vol. 13,
no. 4, pp. 354-356, 1969.

How the algorithm works
-----------------------
X and Y are cut into a 2 x 2 grid of blocks. A normal block multiplication needs
eight block products; Strassen uses seven products and some extra additions:

1. Prepare the blocks: split X into X11, X12, X21, X22 and Y into Y11, Y12, Y21,
   Y22. _operands forms the sums and differences needed by the seven products
   M1..M7, following Appendix A of the revised statement.
2. Multiply: calculate each product using the same method recursively. When its
   size is <= config.STRASSEN_LEAF_SIZE, use the i-j-k triple loop directly.
3. Rebuild Z: _combine calculates Z11 = M2 + M3 - M6 - M7, Z12 = M4 + M6,
   Z21 = M5 + M7 and Z22 = M1 - M3 - M4 - M5, then joins the four blocks.

Each recursive split saves one half-size multiplication but needs 18 block
additions and subtractions. Both versions use the same seven-product calculation.

Implementation notes
--------------------
* All arithmetic is our own pure-Python loops: block additions and subtractions
  visit each entry, and _base_multiply is an i-j-k triple loop over Python ints.
  NumPy is only used to store the input; no NumPy arithmetic or library product.
* NumPy inputs are converted with .tolist() inside the timed functions and are not
  changed. At each leaf, Y is stored by columns for the inner multiplication loop.
* config.STRASSEN_LEAF_SIZE = 64 controls when recursion stops. The assignment
  sizes (16 ... 2048) split evenly; odd sizes use the base loop directly.
* Sequential version: the seven products are calculated one after another, then
  combined before returning to the previous recursion level.

Bottleneck of the sequential version (python profile_seq.py --algo strassen)
-----------------------------------------------------------------------------
The leaf products (_base_multiply) contain the i-j-k triple loops and are the main
computation cost. Splitting, block additions, NumPy -> list conversion and joining
also take time. At one recursion level the seven products do not depend on each
other: only their input sums are needed first, and their results are combined later.
These products are the work to spread over processes. Use the profiling command
above to measure their time share on the team's chosen platform.

Multiprocessing version
-----------------------
* The main process prepares the top recursion levels until there are at least
  7 * p independent products, or the leaf size is reached. For p = 2..6 this gives
  7 or 49 tasks, depending on the matrix size. Each task runs the remaining
  recursion sequentially and returns its product. Tasks go to the supplied pool
  one at a time with apply_async; workers do not create their own pools. With
  p = 1, or an input already at the leaf size, there is just one product task.
* The main process collects each group in M1..M7 order and combines it as soon as
  its results are ready, while other tasks can still be running. The levels above
  are then combined to recover the complete matrix. No shared memory is used.
* Preparing and pickling blocks, transferring them between processes and joining
  the results add overhead. Task counts can also leave workers idle near the end:
  49 tasks do not divide evenly over 6 workers. These costs limit the speed-up and
  can make small matrices slower than the sequential version. The shared framework
  handles all timing, including the rule for excluding worker-pool startup.
"""
import config


# ---------------------------------------------------------------------------
# Block helpers
# ---------------------------------------------------------------------------
def _to_lists(matrix):
    """Convert a NumPy input to lists; list inputs are used directly."""
    return matrix.tolist() if hasattr(matrix, "tolist") else matrix


def _add(X, Y):
    """Element-wise X + Y."""
    return [[x + y for x, y in zip(x_row, y_row)]
            for x_row, y_row in zip(X, Y)]


def _sub(X, Y):
    """Element-wise X - Y."""
    return [[x - y for x, y in zip(x_row, y_row)]
            for x_row, y_row in zip(X, Y)]


def _split(matrix):
    """Return the four quarters in row-major order: 11, 12, 21, 22."""
    half = len(matrix) // 2
    top, bottom = matrix[:half], matrix[half:]
    return ([row[:half] for row in top], [row[half:] for row in top],
            [row[:half] for row in bottom], [row[half:] for row in bottom])


def _base_multiply(X, Y):
    """i-j-k triple loop, with Y stored by columns."""
    y_columns = list(zip(*Y))
    Z = []
    for x_row in X:                         # i: output row
        z_row = []
        for y_column in y_columns:         # j: output column
            total = 0
            for x, y in zip(x_row, y_column):  # k: shared dimension
                total += x * y
            z_row.append(total)
        Z.append(z_row)
    return Z


def _operands(X, Y):
    """The operand pairs of M1..M7, exactly as in Appendix A."""
    X11, X12, X21, X22 = _split(X)
    Y11, Y12, Y21, Y22 = _split(Y)
    return [
        (_add(X11, X21), _add(Y11, Y12)),  # M1
        (_add(X12, X22), _add(Y21, Y22)),  # M2
        (_sub(X11, X22), _add(Y11, Y22)),  # M3
        (X11, _sub(Y12, Y22)),            # M4
        (_add(X21, X22), Y11),            # M5
        (_add(X11, X12), Y22),            # M6
        (X22, _sub(Y21, Y11)),            # M7
    ]


def _combine(products):
    """Reconstruct Z from M1..M7 using the four Appendix A formulas."""
    M1, M2, M3, M4, M5, M6, M7 = products
    Z11 = _sub(_sub(_add(M2, M3), M6), M7)
    Z12 = _add(M4, M6)
    Z21 = _add(M5, M7)
    Z22 = _sub(_sub(_sub(M1, M3), M4), M5)
    return ([left + right for left, right in zip(Z11, Z12)]
            + [left + right for left, right in zip(Z21, Z22)])


def _splittable(n, leaf_size):
    """Assignment sizes halve evenly; other odd sizes use the base loop."""
    return n > leaf_size and n % 2 == 0


def _multiply(X, Y, leaf_size):
    """Sequential Strassen recursion, also called by pool workers."""
    if not _splittable(len(X), leaf_size):
        return _base_multiply(X, Y)
    return _combine([_multiply(left, right, leaf_size)
                     for left, right in _operands(X, Y)])


# ---------------------------------------------------------------------------
# [B-1] Sequential version
# ---------------------------------------------------------------------------
def multiply_seq(A, B):
    """[B-1] Sequential Strassen; return A x B as a new list of rows."""
    return _multiply(_to_lists(A), _to_lists(B), config.STRASSEN_LEAF_SIZE)


# ---------------------------------------------------------------------------
# [C-1] Multiprocessing version
# ---------------------------------------------------------------------------
def multiply_mp(A, B, pool, p):
    """[C-1] Strassen accelerated with `pool`, a ready multiprocessing.Pool of p workers."""
    if p < 1:
        raise ValueError("p must be a positive process count")
    X, Y = _to_lists(A), _to_lists(B)
    leaf_size = config.STRASSEN_LEAF_SIZE
    if p == 1 or not _splittable(len(X), leaf_size):
        return pool.apply_async(_multiply, (X, Y, leaf_size)).get()

    # Prepare the top levels in the main process before sending out products.
    pairs = [(X, Y)]
    depth, size = 0, len(X)
    while len(pairs) < 7 * p and _splittable(size, leaf_size):
        pairs = [pair for left, right in pairs for pair in _operands(left, right)]
        depth += 1
        size //= 2

    # Submit separate tasks so each free worker can take the next product.
    # Send the leaf size too, so Windows workers use the same cutoff.
    jobs = [pool.apply_async(_multiply, (left, right, leaf_size))
            for left, right in pairs]
    del pairs

    # Combine each group of seven as soon as its results are available.
    products = [_combine([job.get() for job in jobs[i:i + 7]])
                for i in range(0, len(jobs), 7)]
    for _ in range(depth - 1):
        products = [_combine(products[i:i + 7])
                    for i in range(0, len(products), 7)]
    return products[0]
