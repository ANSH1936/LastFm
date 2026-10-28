"""Download HetRec 2011 Last.fm 2K dataset from GroupLens."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import DATA_RAW

URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"


def main() -> None:
    DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_RAW / "hetrec2011-lastfm-2k.zip"
    if (DATA_RAW / "user_artists.dat").is_file():
        print(f"Already present: {DATA_RAW}")
        return
    print(f"Downloading {URL} ...")
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)
    print("Extracting ...")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(DATA_RAW)
    print(f"Done. Data at {DATA_RAW}")


if __name__ == "__main__":
    main()
