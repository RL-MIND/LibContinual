#!/usr/bin/env python
"""Download the processed TinyImageNet dataset used by the IDER code.

The original IDER repository reads TinyImageNet from a processed NumPy archive
instead of the raw Stanford tiny-imagenet-200 image folders. The expected
layout after this script finishes is:

    datasets/TINYIMG/processed/x_train_01.npy ... x_train_20.npy
    datasets/TINYIMG/processed/y_train_01.npy ... y_train_20.npy
    datasets/TINYIMG/processed/x_val_01.npy   ... x_val_20.npy
    datasets/TINYIMG/processed/y_val_01.npy   ... y_val_20.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


IDER_TINYIMG_URL = (
    "https://unimore365-my.sharepoint.com/:u:/g/personal/"
    "263133_unimore_it/EVKugslStrtNpyLGbgrhjaABqRHcE3PB_r2OEaV7Jy94oQ?e=9K29aD"
)


def expected_files(root: Path) -> list[Path]:
    processed = root / "processed"
    files: list[Path] = []
    for split in ("train", "val"):
        for idx in range(1, 21):
            files.append(processed / f"x_{split}_{idx:02d}.npy")
            files.append(processed / f"y_{split}_{idx:02d}.npy")
    return files


def missing_files(root: Path) -> list[Path]:
    return [path for path in expected_files(root) if not path.is_file()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="./datasets/TINYIMG",
        help="Destination directory expected by config/zz_IDER/ider_tinyimagenet_buf4000.yaml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if all expected processed files already exist.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    missing = missing_files(root)
    if not missing and not args.force:
        print(f"Processed TinyImageNet already exists under: {root}")
        return 0

    try:
        from onedrivedownloader import download
    except ImportError:
        print(
            "Missing dependency: onedrivedownloader\n"
            "Install it in the training environment with:\n"
            "  pip install onedrivedownloader\n"
            "Then rerun this script.",
            file=sys.stderr,
        )
        return 1

    archive = root / "tiny-imagenet-processed.zip"
    print(f"Downloading processed TinyImageNet to: {root}")
    print(f"Source: {IDER_TINYIMG_URL}")
    download(
        IDER_TINYIMG_URL,
        filename=str(archive),
        unzip=True,
        unzip_path=str(root),
        clean=True,
    )

    missing = missing_files(root)
    if missing:
        print("Download finished, but the expected processed files are incomplete.", file=sys.stderr)
        print(f"Missing count: {len(missing)}", file=sys.stderr)
        for path in missing[:10]:
            print(f"  {path}", file=sys.stderr)
        return 2

    print(f"OK: processed TinyImageNet is ready under: {root / 'processed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
