"""
scripts/download_data.py
========================
Downloads the NASA C-MAPSS turbofan engine degradation dataset and places the
raw .txt files in data/raw/.

Strategy (tries in order):
  1. Direct NASA download  (https://ti.arc.nasa.gov/c/6/)
  2. Kaggle API            (requires ~/.kaggle/kaggle.json)
  3. Manual instructions   (prints URLs for the user)

Usage:
  python scripts/download_data.py [--data-dir data/raw] [--force]

License:
  NASA Open Data — free to use for non-commercial research.
  Citation: Saxena et al. (2008), Damage Propagation Modeling for Aircraft
  Engine Run-to-Failure Simulation, PHM 2008.
"""

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────

NASA_DIRECT_URL = "https://ti.arc.nasa.gov/c/6/"
KAGGLE_DATASET = "behrad3d/nasa-cmaps"  # well-known Kaggle mirror

# Expected files after extraction
EXPECTED_FILES = [
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt",
    "train_FD002.txt",
    "test_FD002.txt",
    "RUL_FD002.txt",
    "train_FD003.txt",
    "test_FD003.txt",
    "RUL_FD003.txt",
    "train_FD004.txt",
    "test_FD004.txt",
    "RUL_FD004.txt",
]

# Expected row counts per file (allows ±10 for whitespace/header variations)
EXPECTED_ROW_COUNTS = {
    "train_FD001.txt": 20631,
    "test_FD001.txt":  13096,
    "RUL_FD001.txt":   100,
    "train_FD002.txt": 53759,
    "test_FD002.txt":  33991,
    "RUL_FD002.txt":   259,
    "train_FD003.txt": 24720,
    "test_FD003.txt":  16596,
    "RUL_FD003.txt":   100,
    "train_FD004.txt": 61249,
    "test_FD004.txt":  41214,
    "RUL_FD004.txt":   248,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _project_root() -> Path:
    """Return the project root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a text file."""
    count = 0
    with open(path, "r") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def verify_files(data_dir: Path) -> bool:
    """
    Check that all expected files exist and have approximately the right
    number of rows. Returns True if everything looks good, False otherwise.
    """
    all_ok = True
    print("\nVerifying downloaded files …")
    for fname in EXPECTED_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  ✗  MISSING  {fname}")
            all_ok = False
            continue
        n = _count_lines(fpath)
        expected = EXPECTED_ROW_COUNTS.get(fname, 0)
        tolerance = max(10, int(expected * 0.01))  # 1 % tolerance
        if abs(n - expected) > tolerance:
            print(f"  ✗  BAD COUNT  {fname}: got {n} rows, expected ~{expected}")
            all_ok = False
        else:
            print(f"  ✓  {fname} ({n} rows)")
    return all_ok


def _download_with_progress(url: str, dest: Path, timeout: int = 60) -> None:
    """Stream-download a URL to dest with a tqdm progress bar."""
    resp = requests.get(url, stream=True, timeout=timeout,
                        headers={"User-Agent": "predictive-maintenance-project/1.0"})
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
            pbar.update(len(chunk))


def _extract_zip(zip_path: Path, data_dir: Path) -> None:
    """Extract the C-MAPSS zip and move .txt files to data_dir."""
    print(f"\nExtracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Extract to a temp subfolder then collect .txt files
        extract_dir = data_dir / "_tmp_extract"
        zf.extractall(extract_dir)

    # Move all .txt files (recursively) into data_dir
    moved = 0
    for txt_file in extract_dir.rglob("*.txt"):
        dest = data_dir / txt_file.name
        txt_file.rename(dest)
        moved += 1

    # Clean up temp directory
    import shutil
    shutil.rmtree(extract_dir, ignore_errors=True)
    print(f"  Moved {moved} .txt files to {data_dir}")


# ── Download strategies ────────────────────────────────────────────────────────

def try_nasa_direct(data_dir: Path) -> bool:
    """
    Attempt to download directly from the NASA ARC server.
    Returns True on success.
    """
    print("\n[Strategy 1] Trying NASA direct download …")
    zip_path = data_dir / "CMAPSSData.zip"
    try:
        _download_with_progress(NASA_DIRECT_URL, zip_path, timeout=120)
        _extract_zip(zip_path, data_dir)
        zip_path.unlink(missing_ok=True)  # clean up zip
        return True
    except Exception as exc:
        print(f"  ✗  NASA direct download failed: {exc}")
        zip_path.unlink(missing_ok=True)
        return False


def try_kaggle(data_dir: Path) -> bool:
    """
    Attempt to download via the Kaggle API.
    Requires ~/.kaggle/kaggle.json with a valid API token.
    Returns True on success.
    """
    print("\n[Strategy 2] Trying Kaggle API download …")
    try:
        import kaggle  # type: ignore
    except ImportError:
        print("  ✗  kaggle package not installed. Run: pip install kaggle")
        return False

    try:
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", KAGGLE_DATASET,
                "--path", str(data_dir),
                "--unzip",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  ✗  Kaggle download failed:\n{result.stderr}")
            return False
        print("  ✓  Kaggle download succeeded")
        return True
    except Exception as exc:
        print(f"  ✗  Kaggle download failed: {exc}")
        return False


def print_manual_instructions(data_dir: Path) -> None:
    """Print instructions for downloading the data manually."""
    print(
        "\n" + "=" * 72
        + "\n[Manual download instructions]\n"
        + "=" * 72
        + f"""
All automated download strategies failed. Please download the dataset manually:

  Option A — NASA Prognostics Data Repository:
    1. Visit: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan
    2. Click the download link for "Turbofan Engine Degradation Simulation Data Set"
    3. Save the zip file to: {data_dir}
    4. Unzip it — you should see train_FD001.txt, test_FD001.txt, RUL_FD001.txt, etc.

  Option B — Kaggle mirror:
    1. Install the Kaggle CLI: pip install kaggle
    2. Place your kaggle.json token at ~/.kaggle/kaggle.json
    3. Run: kaggle datasets download -d behrad3d/nasa-cmaps --path {data_dir} --unzip

After placing the files, re-run this script to verify them:
  python scripts/download_data.py --verify-only
"""
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NASA C-MAPSS turbofan engine dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory to save raw data files (default: <project_root>/data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip download; just verify files that are already present.",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir or (_project_root() / "data" / "raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")

    # ── Verify only ──
    if args.verify_only:
        ok = verify_files(data_dir)
        sys.exit(0 if ok else 1)

    # ── Skip if already present ──
    existing = [f for f in EXPECTED_FILES if (data_dir / f).exists()]
    if existing and not args.force:
        print(f"\n{len(existing)}/{len(EXPECTED_FILES)} expected files already present.")
        ok = verify_files(data_dir)
        if ok:
            print("\nAll files verified. Use --force to re-download.")
            sys.exit(0)
        else:
            print("\nSome files are missing or corrupted. Attempting download …")

    # ── Try download strategies in order ──
    success = try_nasa_direct(data_dir)

    if not success:
        success = try_kaggle(data_dir)

    if not success:
        print_manual_instructions(data_dir)
        sys.exit(1)

    # ── Verify ──
    ok = verify_files(data_dir)
    if ok:
        print(
            "\n✓ All files downloaded and verified successfully!\n"
            "You can now run the notebooks or train.py.\n"
        )
        sys.exit(0)
    else:
        print(
            "\n✗ Download completed but verification failed.\n"
            "Please check the files in data/raw/ manually.\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
