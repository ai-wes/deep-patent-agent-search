"""Helpers to pre-populate ADK session state from deterministic analysis outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception:
        return []


def load_state_from_directory(root: Path) -> Dict[str, Any]:
    """Load JSON/CSV/MD artifacts into a dict keyed by filename stem.

    The loader is intentionally generic: it ingests every *.json, *.csv, *.md,
    and *.txt file under the provided root. That makes it easy to point at
    deterministic analysis output directories (e.g., analysis/gemini) and pass
    the result into the ADK session state when invoking a pipeline.
    """
    state: Dict[str, Any] = {}
    if not root.exists():
        return state

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stem = path.stem
        if path.suffix.lower() == ".json":
            data = _load_json(path)
        elif path.suffix.lower() == ".csv":
            data = _load_csv(path)
        elif path.suffix.lower() in {".md", ".txt"}:
            try:
                data = path.read_text(encoding="utf-8")
            except Exception:
                data = None
        else:
            continue

        if data is not None:
            state[stem] = data
    return state


def load_state_from_env(env_var: str = "ADK_STATE_DIR") -> Dict[str, Any]:
    """Convenience wrapper to load state from a directory specified in env."""
    import os

    dir_value = os.environ.get(env_var)
    if not dir_value:
        return {}
    return load_state_from_directory(Path(dir_value))

