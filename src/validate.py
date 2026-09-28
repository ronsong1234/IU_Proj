"""Data-quality checks for the PneumoniaMNIST metadata table.

Usage:
    python -m src.validate        (requires output/pneumoniamnist.duckdb; see src.build_metadata)
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.build_metadata import DB_PATH, NPZ_PATH, TABLE, find_splits

# Official PneumoniaMNIST split sizes (MedMNIST dataset info)
EXPECTED_SPLIT_COUNTS = {"train": 4708, "val": 524, "test": 624}
ALLOWED_LABELS = {0, 1}
EXPECTED_SHAPE = (28, 28)  # (height, width)
# Tukey lower fence on std_intensity: std < Q1 - LOW_VAR_IQR_K * IQR
LOW_VAR_IQR_K = 1.5

REQUIRED_COLUMNS = [
    "image_id", "split", "label", "width", "height",
    "mean_intensity", "std_intensity", "min_intensity", "max_intensity",
]


def _result(check: str, passed: bool, detail: str, warn_only: bool = False) -> dict:
    status = "PASS" if passed else ("WARN" if warn_only else "FAIL")
    return {"check": check, "status": status, "detail": detail}


def check_split_counts(meta: pd.DataFrame) -> dict:
    counts = meta["split"].value_counts().to_dict()
    passed = counts == EXPECTED_SPLIT_COUNTS
    detail = ", ".join(f"{s}={counts.get(s, 0)} (expected {n})" for s, n in EXPECTED_SPLIT_COUNTS.items())
    extra = set(counts) - set(EXPECTED_SPLIT_COUNTS)
    if extra:
        detail += f"; unexpected splits: {sorted(extra)}"
    return _result("split_counts", passed, detail)


def check_labels(meta: pd.DataFrame) -> dict:
    bad = meta.loc[~meta["label"].isin(ALLOWED_LABELS)]
    per_split = meta.groupby("split")["label"].nunique()
    missing_class = per_split[per_split < len(ALLOWED_LABELS)].index.tolist()
    detail = (f"{len(bad)} rows with labels outside {sorted(ALLOWED_LABELS)}; "
              f"observed {sorted(meta['label'].unique().tolist())}")
    if missing_class:
        detail += f"; splits missing a class: {missing_class}"
    return _result("label_values", bad.empty and not missing_class, detail)


def check_dimensions(npz_path: Path) -> dict:
    # Checked on the raw arrays so a malformed image can't hide behind the table
    problems = []
    with np.load(npz_path, allow_pickle=False) as npz:
        for split in find_splits(npz):
            shape = npz[f"{split}_images"].shape
            if shape[1:] not in (EXPECTED_SHAPE, EXPECTED_SHAPE + (1,)):
                problems.append(f"{split}: {shape}")
    detail = "; ".join(problems) if problems else f"all images {EXPECTED_SHAPE[1]}x{EXPECTED_SHAPE[0]}, 1 channel"
    return _result("image_dimensions", not problems, detail)


def check_missing_invalid(meta: pd.DataFrame, npz_path: Path) -> dict:
    problems = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in meta.columns]
    if missing_cols:
        problems.append(f"missing columns {missing_cols}")
    nulls = meta.isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        problems.append(f"nulls: {nulls.to_dict()}")
    if meta["image_id"].duplicated().any():
        problems.append(f"{meta['image_id'].duplicated().sum()} duplicate image_ids")

    stats = meta[["mean_intensity", "std_intensity", "min_intensity", "max_intensity"]]
    if not np.isfinite(stats.to_numpy(dtype=float)).all():
        problems.append("non-finite intensity statistics")
    if ((meta["min_intensity"] < 0) | (meta["max_intensity"] > 255)).any():
        problems.append("intensities outside 0-255")
    order_ok = (meta["min_intensity"] <= meta["mean_intensity"]) & (meta["mean_intensity"] <= meta["max_intensity"])
    if not order_ok.all():
        problems.append(f"{(~order_ok).sum()} rows where min <= mean <= max fails")
    if (meta["std_intensity"] < 0).any():
        problems.append("negative std")

    with np.load(npz_path, allow_pickle=False) as npz:
        for key in npz.files:
            arr = npz[key]
            if arr.dtype != np.uint8:
                problems.append(f"{key} dtype {arr.dtype} (expected uint8)")
            if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
                problems.append(f"{key} contains NaN/inf")

    detail = "; ".join(problems) if problems else "no nulls, non-finite, out-of-range or inconsistent values"
    return _result("missing_or_invalid", not problems, detail)


def find_low_variance(meta: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    q1, q3 = meta["std_intensity"].quantile([0.25, 0.75])
    threshold = q1 - LOW_VAR_IQR_K * (q3 - q1)
    flagged = (meta.loc[meta["std_intensity"] < threshold,
                        ["image_id", "split", "label", "mean_intensity", "std_intensity"]]
               .sort_values("std_intensity")
               .assign(threshold=threshold))
    detail = (f"{len(flagged)} images with std_intensity < {threshold:.2f} "
              f"(Q1 - {LOW_VAR_IQR_K} x IQR); lowest std {meta['std_intensity'].min():.2f}")
    # Reported for review rather than failed: low contrast isn't necessarily wrong
    return _result("low_variance", flagged.empty, detail, warn_only=True), flagged


def find_duplicates(meta: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    dup = meta[meta.duplicated("pixel_sha1", keep=False)]
    groups = (dup.groupby("pixel_sha1")
              .agg(n_images=("image_id", "size"),
                   image_ids=("image_id", lambda s: ",".join(sorted(s))),
                   splits=("split", lambda s: ",".join(sorted(set(s)))),
                   n_labels=("label", "nunique"))
              .reset_index()
              .sort_values(["n_images", "image_ids"], ascending=[False, True]))
    cross_split = (groups["splits"].str.contains(",")).sum()
    conflicts = (groups["n_labels"] > 1).sum()
    detail = (f"{len(groups)} groups of identical images covering {len(dup)} images (SHA-1 of pixels); "
              f"{cross_split} groups span splits; {conflicts} groups have conflicting labels")
    return _result("exact_duplicates", groups.empty, detail, warn_only=True), groups


def run_checks(meta: pd.DataFrame, npz_path: Path = NPZ_PATH):
    """Returns (results, low_variance_images, duplicate_groups)."""
    low_var_result, low_var = find_low_variance(meta)
    dup_result, dups = find_duplicates(meta)
    results = pd.DataFrame([
        check_split_counts(meta),
        check_labels(meta),
        check_dimensions(npz_path),
        check_missing_invalid(meta, npz_path),
        low_var_result,
        dup_result,
    ])
    return results, low_var, dups


def write_results(results, low_var, dups, db_path: Path = DB_PATH) -> None:
    with duckdb.connect(str(db_path)) as con:
        con.execute("CREATE OR REPLACE TABLE validation_results AS SELECT * FROM results")
        con.execute("CREATE OR REPLACE TABLE low_variance_images AS SELECT * FROM low_var")
        con.execute("CREATE OR REPLACE TABLE duplicate_images AS SELECT * FROM dups")


if __name__ == "__main__":
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        meta = con.sql(f"SELECT * FROM {TABLE}").df()
    results, low_var, dups = run_checks(meta)
    write_results(results, low_var, dups)
    print(results.to_string(index=False))

    import sys
    if (results["status"] == "FAIL").any():
        sys.exit("Validation failed; see checks above. Not exporting results.")
