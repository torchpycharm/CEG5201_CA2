"""[B-1] / [C-1]  Strassen's algorithm  (n = 1 in Table II).

Owner: ZHOU JUNMIN (Member ID 0)

Reference: V. Strassen, "Gaussian elimination is not optimal," Numer. Math., vol. 13,
no. 4, pp. 354-356, 1969.
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
