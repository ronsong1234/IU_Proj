"""Export the key result tables from DuckDB to small CSVs in output/, so they can be
read on GitHub without running anything.

Usage:
    python -m src.export_results     (after src.build_metadata)
"""
from pathlib import Path

import duckdb

from src.build_metadata import DB_PATH, ROOT

OUT_DIR = ROOT / "output"
SQL_DIR = ROOT / "sql"

# output file -> query (a table name or a .sql file in sql/)
EXPORTS = {
    "validation_results.csv": "SELECT * FROM validation_results",
    "duplicate_images.csv": "SELECT * FROM duplicate_images",
    "split_summary.csv": "split_summary.sql",
    "intensity_by_class.csv": "intensity_by_class.sql",
    "intensity_effect_size.csv": "intensity_effect_size.sql",
    "unusual_images.csv": "unusual_images.sql",
}


def export_all(db_path: Path = DB_PATH, out_dir: Path = OUT_DIR) -> list[Path]:
    written = []
    with duckdb.connect(str(db_path), read_only=True) as con:
        for filename, query in EXPORTS.items():
            if query.endswith(".sql"):
                query = (SQL_DIR / query).read_text()
            path = out_dir / filename
            con.sql(query).df().to_csv(path, index=False)
            written.append(path)
    return written


if __name__ == "__main__":
    for path in export_all():
        print(path.relative_to(ROOT))
