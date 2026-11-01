import os
from time import perf_counter
from collections import defaultdict
import config

def timed_fox(col_b, row_a, a_row, b_col):
    start = perf_counter()
    index_c, block_c, compute_duration = fox(col_b, row_a, a_row, b_col)
    end = perf_counter()

    return {
        "pid": os.getpid(),
        "task": (row_a, col_b),
        "start": start,
        "end": end,
        "duration": end - start,
        "compute_duration": compute_duration,
        "output": (index_c, block_c),
    }

def multiply(A, B, C):
    """Matric multiplication of C = A x B using transposed B"""
    n = len(A)
    # Store each column of B as a row in Bt
    Bt = [[B[k][j] for k in range(n)] for j in range(n)]

    for i in range(n):
        a_row = A[i]
        c_row = C[i]
        for j in range(n):
            b_col = Bt[j]
            total = c_row[j]
            for k in range(n):
                total += a_row[k] * b_col[k]
            c_row[j] = total

def split_blocks(matrix, block_size):
    """Split a matrix into 16 blocks of size block_size x block_size"""
    n = len(matrix)
    return [
        [row[j:j + block_size] for row in matrix[i:i + block_size]]
        for i in range(0, n, block_size)
        for j in range(0, n, block_size)
    ]

def fox(col_b, row_a, a_row, b_col):
    """Compute one block of C = A x B using Fox's algorithm"""
    block_size = len(a_row[0])
    index_c = row_a * 4 + col_b
    block_c = [[0 for _ in range(block_size)] for _ in range(block_size)]

    compute_start = perf_counter()
    for stage in range(4):
        k = (col_b + stage) % 4
        multiply(a_row[k], b_col[k], block_c)

    compute_duration = perf_counter() - compute_start
    return index_c, block_c, compute_duration

def assemble_blocks(blocks_C, n, block_size):
    """Assemble the output blocks into a single matrix C"""
    C = [[0 for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            C[r][c] = blocks_C[(r // block_size) * 4 + (c // block_size)][r % block_size][c % block_size]
    return C


def multiply_seq(A, B):
    """[B-4] Sequential Fox. Return the product A x B."""
    # convert to Python lists 
    A = A.tolist() 
    B = B.tolist()

    grid = config.FOX_GRID
    n = len(A)
    block_size = n // grid

    # split each matrix into a 4-by-4 grid of blocks
    blocks_A, blocks_B = split_blocks(A, block_size), split_blocks(B, block_size)
    blocks_C = [[[0 for _ in range(block_size)] for _ in range(block_size)] for _ in range(grid * grid)]
    
    # get diagonal B blocks and shifted A blocks for each stage for C block multiplication
    for stage in range(grid):
        for col_b in range(grid):
            row_b = (col_b + stage) % grid
            index_b = row_b * grid + col_b
            for row_a in range(grid):
                index_a = row_a * grid + row_b
                index_c = row_a * grid + col_b
                multiply(blocks_A[index_a], blocks_B[index_b], blocks_C[index_c])

    return assemble_blocks(blocks_C, n, block_size)

def multiply_mp(A, B, pool, p):
    """[C-4] Fox accelerated with `pool`, a ready multiprocessing.Pool of p workers."""
    start_prep = perf_counter()
    # convert to Python lists 
    A = A.tolist() 
    B = B.tolist()
    
    tasks = ()
    grid = config.FOX_GRID
    n = len(A)
    block_size = n // grid

    # split each matrix into a 4-by-4 grid of blocks
    blocks_A, blocks_B, blocks_C = split_blocks(A, block_size), split_blocks(B, block_size), [None] * 16

    # create a list of tasks for the multiprocessing pool
    for row_a in range(grid):
        for col_b in range(grid):
            a_row = [blocks_A[row_a * grid + k] for k in range(grid)]
            b_col = [blocks_B[k * grid + col_b] for k in range(grid)]
            tasks += ((col_b, row_a, a_row, b_col),)

    # use pool to compute the blocks of C in parallel
    start_mp = perf_counter()
    preparation_time = start_mp - start_prep

    results = pool.starmap(timed_fox, tasks, chunksize=1)

    end_mp = perf_counter()

    start_assembly = perf_counter()
    for result in results:
        index_c, block_c = result["output"]
        blocks_C[index_c] = block_c
    C = assemble_blocks(blocks_C, n, block_size)
    assembly_time = perf_counter() - start_assembly

    print(f"\nSequential preparation: {preparation_time:.6f} s")
    print(f"Parallel execution (starmap): {end_mp - start_mp:.6f} s")
    print(f"Sequential assembly: {assembly_time:.6f} s")
    print("Worker compute times overlap; do not add them to starmap time.")

    workers = defaultdict(list)

    for result in results:
        workers[result["pid"]].append(result)

    print("\nWorker task distribution")
    print("-" * 65)

    for pid, records in sorted(workers.items()):
        total_compute = sum(r["compute_duration"] for r in records)

        last_end = max(r["end"] for r in records)

        print(
            f"PID {pid} | "
            f"Tasks: {len(records)} | "
            f"Compute: {total_compute:.3f}s | "
            f"Finish: {last_end - start_mp:.3f}s"
        )

        for r in records:
            print(
                f"    Block {r['task']} "
                f"| Compute: {r['compute_duration']:.6f}s "
                f"| Task duration: {r['duration']:.6f}s"
            )

    return C, results
