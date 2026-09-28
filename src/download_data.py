"""Download PneumoniaMNIST (28x28) from the official MedMNIST+ Zenodo record.

Usage:
    python -m src.download_data
"""
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "pneumoniamnist.npz"

# Official MedMNIST+ record: https://doi.org/10.5281/zenodo.10519652
# Note: don't send a browser User-Agent (e.g. "Mozilla/5.0"); Zenodo answers
# script traffic that claims to be a browser with HTTP 403.
URL = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"


def download(dest: Path = DEST) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as resp:
        dest.write_bytes(resp.read())
    return dest


if __name__ == "__main__":
    path = download()
    print(path, path.stat().st_size / 1e6, "MB")
