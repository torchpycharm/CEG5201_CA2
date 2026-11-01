"""[B-2] / [C-2]  Coppersmith-Winograd algorithm  (n = 2 in Table II).

Owner: RICARDO JACK LIWONGAN (Member ID 1)

Reference: D. Coppersmith and S. Winograd, "Matrix multiplication via arithmetic
progressions," J. Symb. Comput., vol. 9, no. 3, pp. 251-280, 1990.
"""
import config

# ---------------------------------------------------------------------------
# Block helpers
# ---------------------------------------------------------------------------
def _to_lists(M):
    """NumPy array -> list of rows of Python ints (a list is returned unchanged)."""
    return M.tolist() if hasattr(M, "tolist") else M

def _add(X, Y):
    """Element-wise X + Y."""
    return [[x + y for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)]

def _sub(X, Y):
    """Element-wise X - Y."""
    return [[x - y for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)]

def _split(M):
    """Quarters (M11, M12, M21, M22) of an even-sized square matrix."""
    h = len(M) // 2
    top, bottom = M[:h], M[h:]
    return ([r[:h] for r in top], [r[h:] for r in top],
            [r[:h] for r in bottom], [r[h:] for r in bottom])

def _join(Z11, Z12, Z21, Z22):
    """Combine four quarters into one matrix."""
    return [a + b for a, b in zip(Z11, Z12)] + [a + b for a, b in zip(Z21, Z22)]

def _base_multiply(X, Y):
    """i-j-k triple loop: Z[i][j] += X[i][k] * Y[k][j]."""
    y_cols = [list(col) for col in zip(*Y)]
    Z = []
    for x_row in X:
        z_row = []
        for y_col in y_cols:
            s = 0
            for x, y in zip(x_row, y_col):
                s += x * y
            z_row.append(s)
        Z.append(z_row)
    return Z

def _operands(X, Y):
    """Split step: the 7 operand pairs (left, right) of M1..M7"""
    X11, X12, X21, X22 = _split(X)
    Y11, Y12, Y21, Y22 = _split(Y)
    S1 = _add(X21, X22)
    S2 = _sub(S1, X11)
    S3 = _sub(X11, X21)
    S4 = _sub(X12, S2)
    T1 = _sub(Y12, Y11)
    T2 = _sub(Y22, T1)
    T3 = _sub(Y22, Y12)
    T4 = _sub(T2, Y21)
    return [(X11, Y11), (X12, Y21), (S4, Y22), (X22, T4),
            (S1, T1), (S2, T2), (S3, T3)]

def _combine(M):
    """Combine step: the 4 quarters of Z from the 7 products."""
    M1, M2, M3, M4, M5, M6, M7 = M
    U2 = _add(M1, M6)
    U3 = _add(U2, M7)
    U4 = _add(U2, M5)
    Z11 = _add(M1, M2)
    Z12 = _add(U4, M3)
    Z21 = _sub(U3, M4)
    Z22 = _add(U3, M5)
    return _join(Z11, Z12, Z21, Z22)

def _multiply(X, Y):
    """Recursive Coppersmith-Winograd of 2 square matrices."""
    n = len(X)
    # Stop at leaf size and at odd sizes which cannot be halved.
    if n <= config.CW_LEAF_SIZE or n % 2:
        return _base_multiply(X, Y)
    # 7 half size products, each computed by the same method recursively then rebuild Z.
    return _combine([_multiply(L, R) for L, R in _operands(X, Y)])


# ---------------------------------------------------------------------------
# [B-2] Sequential version
# ---------------------------------------------------------------------------
def multiply_seq(A, B):
    """[B-2] Sequential Coppersmith-Winograd. Return the product A x B."""
    return _multiply(_to_lists(A), _to_lists(B))

# ---------------------------------------------------------------------------
# [C-2] Multiprocessing version
# ---------------------------------------------------------------------------
def _splittable(n):
    """True when an n x n product is still split by the recursion."""
    return n > config.CW_LEAF_SIZE and n % 2 == 0

def _parallel_depth(n):
    """Levels expanded by the main process: CW_MP_DEPTH, fewer for small matrices."""
    depth = 0
    while depth < config.CW_MP_DEPTH and _splittable(n):
        n //= 2
        depth += 1
    return depth

def _result(job):
    """Product of a task or a tail group of 7 smaller tasks."""
    # A tail leaf was submitted as 7 tasks so its 7 products are combined here.
    return _combine([j.get() for j in job]) if isinstance(job, list) else job.get()

def multiply_mp(A, B, pool, p):
    """[C-2] Coppersmith-Winograd accelerated with `pool`, a ready Pool of p workers."""
    X, Y = _to_lists(A), _to_lists(B)
    depth = _parallel_depth(len(X))

    # Expand the top depth levels in the main process 1 pair -> 7 -> 49.
    # pairs stay in M1..M7 order inside every group of 7.
    pairs = [(X, Y)]
    for _ in range(depth): #split, top down
        pairs = [op for L, R in pairs for op in _operands(L, R)]

    # Tail balancing: the last (7^d mod p) leaves are split once more into 7 smaller
    # tasks so the final round keeps all p workers busy (e.g. 49 = 8 x 6 + 1 on p = 6).
    n_tail = len(pairs) % p if depth and _splittable(len(pairs[0][0])) else 0
    head, tail = pairs[:len(pairs) - n_tail], pairs[len(pairs) - n_tail:]

    # Submit the large tasks first and the small tail tasks last. apply_async hands
    # out one task at a time so faster workers take more tasks.
    jobs = [pool.apply_async(_multiply, pair) for pair in head]
    jobs += [[pool.apply_async(_multiply, op) for op in _operands(L, R)] for L, R in tail]

    # Small matrices (n <= leaf size) are not split and run as a single task.
    if not depth:
        return _result(jobs[0])

    #Combine each group of 7 as soon as it is complete, then the levels above.
    products = [_combine([_result(job) for job in jobs[i:i+7]]) 
                for i in range(0, len(jobs), 7)]
    for _ in range(depth - 1):
        products = [_combine(products[i:i+7]) for i in range(0, len(products), 7)]

    return products[0]

