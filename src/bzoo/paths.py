"""Where things live.

Every path in the project is derived from the repository root, which is
found by walking up from this file.  Nothing takes a path from the current
working directory, so scripts behave the same wherever they are launched.
The root can be overridden with the ``BZOO_ROOT`` environment variable,
which is how the tests point at a temporary directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("BZOO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
INTERIM = DATA / "interim"
RESULTS = DATA / "results"
CONFIGS = ROOT / "configs"
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"


def ensure_dirs() -> None:
    for d in (DATA, RAW, CACHE, INTERIM, RESULTS, TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
