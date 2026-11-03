"""Stream records from a top-level JSON array of objects without loading the file.

nuScenes trainval tables are too big to ``json.load`` in 7.6 GB of RAM (sample_data.json 1.28 GB,
ego_pose.json 0.6 GB, sample_annotation.json 0.56 GB). Stdlib only, so it runs in both envs.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

_DECODER = json.JSONDecoder()
_SKIP = " \t\r\n,"


def iter_json_array(path: str | Path, chunk_size: int = 1 << 22) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        buf = f.read(chunk_size).lstrip()
        if not buf.startswith("["):
            raise ValueError(f"{path}: expected a JSON array")
        pos, eof = 1, False
        while True:
            while pos < len(buf) and buf[pos] in _SKIP:
                pos += 1
            if pos < len(buf) and buf[pos] == "]":
                return
            try:
                obj, pos = _DECODER.raw_decode(buf, pos)  # records are objects, so a cut one never decodes
                yield obj
                continue
            except json.JSONDecodeError:
                if eof:
                    raise
            chunk = f.read(chunk_size)
            eof = not chunk
            buf, pos = buf[pos:] + chunk, 0
