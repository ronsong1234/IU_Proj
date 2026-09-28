"""Build a per-image metadata table from pneumoniamnist.npz and store it in DuckDB.

Usage:
    python -m src.build_metadata
"""
import hashlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.download_data import DEST as NPZ_PATH

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "output" / "pneumoniamnist.duckdb"
TABLE = "image_metadata"

# PneumoniaMNIST label mapping (from the MedMNIST dataset info)
LABEL_NAMES = {0: "normal", 1: "pneumonia"}


def find_splits(npz) -> list[str]:
    """Discover splits from '<split>_images' / '<split>_labels' key pairs."""
    keys = set(npz.files)
    splits = sorted(k[: -len("_images")] for k in keys if k.endswith("_images"))
    for split in splits:
        if f"{split}_labels" not in keys:
            raise ValueError(f"'{split}_images' has no matching '{split}_labels'")
    unpaired = keys - {f"{s}_{kind}" for s in splits for kind in ("images", "labels")}
    if unpaired:
        raise ValueError(f"Unexpected arrays in npz: {sorted(unpaired)}")
    return splits


def split_metadata(split: str, images: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    if images.ndim not in (3, 4):
        raise ValueError(f"{split}_images: expected (N, H, W) or (N, H, W, C), got {images.shape}")
    labels = labels.reshape(len(labels), -1)
    if len(labels) != len(images) or labels.shape[1] != 1:
        raise ValueError(f"{split}: {images.shape} images vs {labels.shape} labels")

    n, height, width = images.shape[:3]
    channels = images.shape[3] if images.ndim == 4 else 1
    flat = images.reshape(n, -1).astype(np.float64)
    label = labels[:, 0].astype(int)

    return pd.DataFrame({
        "image_id": [f"{split}_{i:05d}" for i in range(n)],
        "split": split,
        "split_index": np.arange(n),
        "label": label,
        "label_name": [LABEL_NAMES.get(v, "unknown") for v in label],
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": str(images.dtype),
        "mean_intensity": flat.mean(axis=1),
        "std_intensity": flat.std(axis=1),
        "min_intensity": flat.min(axis=1).astype(int),
        "max_intensity": flat.max(axis=1).astype(int),
        "median_intensity": np.median(flat, axis=1),
        # Fraction of pixels at the dtype's extremes; high values flag clipped/blank images
        "frac_black": (images.reshape(n, -1) == 0).mean(axis=1),
        "frac_white": (images.reshape(n, -1) == np.iinfo(images.dtype).max).mean(axis=1)
        if np.issubdtype(images.dtype, np.integer) else np.nan,
        # Hash of raw pixel bytes, for exact-duplicate detection across splits
        "pixel_sha1": [hashlib.sha1(img.tobytes()).hexdigest() for img in images],
    })


def build_metadata(npz_path: Path = NPZ_PATH) -> pd.DataFrame:
    with np.load(npz_path, allow_pickle=False) as npz:
        frames = [
            split_metadata(s, npz[f"{s}_images"], npz[f"{s}_labels"])
            for s in find_splits(npz)
        ]
    df = pd.concat(frames, ignore_index=True)
    df["source_file"] = npz_path.name
    return df


def write_duckdb(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        con.execute(f"CREATE OR REPLACE TABLE {TABLE} AS SELECT * FROM df")
        con.execute(f"ALTER TABLE {TABLE} ADD PRIMARY KEY (image_id)")


if __name__ == "__main__":
    from src.validate import run_checks, write_results

    df = build_metadata()
    write_duckdb(df)
    print(f"Wrote {len(df)} rows to {DB_PATH} (table '{TABLE}')\n")

    results, low_var, dups = run_checks(df)
    write_results(results, low_var, dups)
    print(results.to_string(index=False))

    import sys
    if (results["status"] == "FAIL").any():
        sys.exit("Validation failed; see checks above. Not exporting results.")

    from src.export_results import export_all
    print("\nExported:", ", ".join(p.name for p in export_all()))
