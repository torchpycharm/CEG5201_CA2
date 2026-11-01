"""Package the Canvas upload (Section IV of the CA2 statement).

Creates dist/CA2_<GroupID>_<matric numbers>.zip containing the report renamed to
CA2_Report_<GroupID>_<matric numbers>.pdf, every .py file, readme.txt and requirements.txt.

Usage:
    python make_submission.py path/to/exported_report.pdf
"""
import argparse
import zipfile
from pathlib import Path

import config

SOURCE_DIRS = [".", "common", "algorithms"]     # every *.py file in these folders
EXTRA_FILES = ["readme.txt", "requirements.txt"]


def main():
    parser = argparse.ArgumentParser(description="Build the submission zip.")
    parser.add_argument("report", type=Path, help="the report exported as PDF")
    args = parser.parse_args()

    # Matriculation numbers in Member ID order, as in the naming examples.
    tag = "_".join([str(config.GROUP_ID)] + [matric for _, _, matric in sorted(config.MEMBERS)])
    folder = f"CA2_{tag}"
    files = {args.report: f"CA2_Report_{tag}.pdf"}
    for directory in SOURCE_DIRS:
        for path in sorted((config.ROOT / directory).glob("*.py")):
            files[path] = path.relative_to(config.ROOT).as_posix()
    for name in EXTRA_FILES:
        files[config.ROOT / name] = name

    config.DIST_DIR.mkdir(exist_ok=True)
    zip_path = config.DIST_DIR / f"{folder}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, name in files.items():
            archive.write(source, f"{folder}/{name}")
            print(f"  + {folder}/{name}")
    print(f"Upload to Canvas: {zip_path}")


if __name__ == "__main__":
    main()
