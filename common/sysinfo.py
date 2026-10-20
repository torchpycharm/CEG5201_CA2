"""[O] Description of the benchmark machine and protocol, stored with every run."""
import os
import platform
import subprocess

import numpy as np
import psutil

import config


def cpu_name():
    """Marketing name of the processor, e.g. "Apple M5" or "Intel(R) Core(TM) i7-12700H"."""
    system = platform.system()
    if system == "Darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
    if system == "Windows":
        import winreg
        key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
            return winreg.QueryValueEx(handle, "ProcessorNameString")[0].strip()
    return platform.processor()


def machine():
    """Hardware and software of the machine running the benchmark."""
    return {
        "cpu": cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def protocol():
    """The shared settings from config.py that determine what is measured and how."""
    return {
        "group_id": config.GROUP_ID,
        "seed": config.SEED,
        "num_groups": config.NUM_GROUPS,
        "matrix_sizes": config.MATRIX_SIZES,
        "matrix_format": config.MATRIX_FORMAT,
        "process_counts": config.PROCESS_COUNTS,
        "start_method": config.START_METHOD,
        "time_pool_creation": config.TIME_POOL_CREATION,
        "strassen_leaf_size": config.STRASSEN_LEAF_SIZE,
        "cw_leaf_size": config.CW_LEAF_SIZE,
        "cannon_grid": config.CANNON_GRID,
        "fox_grid": config.FOX_GRID,
    }


def summary():
    """Human-readable version of machine() and protocol() for console output."""
    m, s = machine(), protocol()
    return "\n".join([
        f"Machine : {m['cpu']} | {m['physical_cores']} physical / {m['logical_cores']} logical"
        f" cores | {m['ram_gb']} GB RAM",
        f"Software: {m['os']} | Python {m['python']} | NumPy {m['numpy']}",
        f"Protocol: p = {s['process_counts']} | start method '{s['start_method']}'"
        f" | pool creation timed: {s['time_pool_creation']}"
        f" | matrix format '{s['matrix_format']}'",
        f"Data    : seed {s['seed']} | {s['num_groups']} groups x {len(s['matrix_sizes'])}"
        f" pairs | sizes {s['matrix_sizes']}",
    ])
