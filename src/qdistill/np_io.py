"""Whole-file .npz reads."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np


def load_npz(path):
    return np.load(io.BytesIO(Path(path).read_bytes()))
