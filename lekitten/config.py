"""Load config.yaml (one settings file for both backends)."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def repo_path(p: str | Path) -> Path:
    """Resolve a path from config.yaml relative to the repo root."""
    p = Path(p)
    return p if p.is_absolute() else REPO / p
