"""Download the released weights listed in weights/manifest.json and check their SHA-256."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((ROOT / "weights/manifest.json").read_text())
    base = manifest["base_url"].rstrip("/")
    assert not base.startswith("REPLACE"), "set base_url in weights/manifest.json to the release location"
    for f in manifest["files"]:
        path = ROOT / f["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            print("downloading", f["path"], flush=True)
            urllib.request.urlretrieve(f"{base}/{f['path']}", path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == f["sha256"], f"checksum mismatch: {f['path']}"
    print("all files verified")


if __name__ == "__main__":
    main()
