"""[B-2] / [C-2]  Coppersmith-Winograd algorithm  (n = 2 in Table II).

Owner: RICARDO JACK LIWONGAN (Member ID 1)

Reference: D. Coppersmith and S. Winograd, "Matrix multiplication via arithmetic
progressions," J. Symb. Comput., vol. 9, no. 3, pp. 251-280, 1990.

Appendix A actually describes the Strassen-Winograd variant (7 products, 15 block
additions), not the asymptotic algorithm of the 1990 paper.

How the algorithm works
-----------------------
X and Y are cut into 2 x 2 blocks of half size. Eight block additions build
    S1 = X21 + X22   S2 = S1 - X11  S3 = X11 - X21  S4 = X12 - S2
    T1 = Y12 - Y11   T2 = Y22 - T1  T3 = Y22 - Y12  T4 = T2 - Y21
then seven half size products, each computed by the same method recursively,
    M1 = X11 Y11  M2 = X12 Y21   M3 = S4 Y22   M4 = X22 T4
    M5 = S1 T1    M6 = S2 T2     M7 = S3 T3
and seven more additions give Z (U2 and U3 are reused, hence 15 additions in total):
    U2 = M1 + M6    U3  = U2 + M7   U4  = U2 + M5
    Z11 = M1 + M2   Z12 = U4 + M3   Z21 = U3 - M4   Z22 = U3 + M5      
Blocks of size <= CW_LEAF_SIZE are multiplied with the own i-j-k triple loop.

Implementation notes
--------------------
* All arithmetic is pure Python on lists of Python ints. NumPy is only used to store the input
  which is converted once with .tolist() inside the timed functions. Looping over NumPy elements
  one by one is about 4x slower than over lists.
* Base case: i-j-k triple loop. Y is transposed once per leaf so the k loop walks a
  row of X and a column of Y side by side.
* CW_LEAF_SIZE = 64 was chosen by measurement. Below that, the 15 additions of a level costs
  more than the one multiplication this algorithm saves.

Bottleneck of the sequential version
------------------------------------
The i-j-k lef products take around 89.6% of the time at M = 512 and around 88%
at M = 1024. The S/T/U block additions take about 6-7%. The 7 products of one level
do not depend on each other, they only need the S/T sums before them and only the U/Z
additions after them use their result. So the products are the work to spread over
processes while additions of the top levels stay in the main process.

Multiprocessing version
-----------------------
Pool workers cannot create their own pools, so the recursion tree is flattened:
1. The main process expands the top d = CW_MP_DEPTH levels (only the cheap S/T additions)
   giving 7^d independent leave products. d = 2 gives 49 tasks. d = 1 (7 tasks) leavees workers
   idle while d = 3 (343 tasks) costs more serial additions and data transfer per extra level
   than it gains in balance.
2. Tail balancing: 7^d equal tasks are rarely divide evenly over p workers (49 tasks on p = 6)
   takes 9 rounds, the last one with a single busy worker). The last 7^d mod p leaves are therefore
   split one level further into 7 smaller tasks each which spread evenly over the workers in the
   final round. The are submitted last so the larger tasks start first.
3. Every task is submitted with apply_async (one task at a time, so faster workers take more).
   Each worker runs the sequential recursion on its operands.
4. The main process collects the results in submission order and combines every group of 7
   as soon as it is complete, overlapping its U/Z additions with the tasks still running.
   The remaining levels are combine at the end.
The depth is reduced for small matrices so that no leaf is smaller than CW_LEAF_SIZE.
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

