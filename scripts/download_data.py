"""Download the Apple-filtered Customer Support on Twitter subset."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apple_agent.config import RAW_CSV

URL = "https://huggingface.co/datasets/OpenArchive/AppleConvos/resolve/main/train.csv"


def main() -> None:
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, RAW_CSV)
    print(f"Wrote {RAW_CSV} ({RAW_CSV.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
