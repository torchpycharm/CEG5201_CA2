"""[A] Generate the input groups G0..G9 into data/ (deterministic, seeded by config.SEED).

Usage:
    python generate_data.py
"""
import config 
from common import data


def main():
    config.DATA_DIR.mkdir(exist_ok=True)
    for g in range(config.NUM_GROUPS):
        data.save_group(g, data.generate_group(g))
        print(f"  G{g} -> {data.group_path(g).relative_to(config.ROOT)}")
    print(f"Data fingerprint: {data.fingerprint()}  (must match on every member's machine)")


if __name__ == "__main__":
    main()
