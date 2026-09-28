"""Download PneumoniaMNIST (28x28) from the official MedMNIST+ Zenodo record.

Usage:
    python -m src.download_data
"""
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "pneumoniamnist.npz"

# Official MedMNIST+ record: https://doi.org/10.5281/zenodo.10519652
# Note: don't send a browser User-Agent (e.g. "Mozilla/5.0"); Zenodo answers
# script traffic that claims to be a browser with HTTP 403.
URL = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"
# MD5 of the official file (also the checksum the medmnist package lists for it).
# Guards against truncated downloads or an HTML error page saved as .npz.
MD5 = "28209eda62fecd6e6a2d98b1501bb15f"


def md5sum(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def download(dest: Path = DEST) -> Path:
    if dest.exists():
        if md5sum(dest) == MD5:
            return dest
        print(f"{dest} exists but its MD5 does not match; downloading again")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(URL, timeout=120) as resp:
        tmp.write_bytes(resp.read())
    if md5sum(tmp) != MD5:
        tmp.unlink()
        raise RuntimeError(f"Downloaded file failed the MD5 check (expected {MD5}); not saved")
    tmp.replace(dest)
    return dest


if __name__ == "__main__":
    path = download()
    print(path, path.stat().st_size / 1e6, "MB")
