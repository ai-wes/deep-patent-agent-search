"""
Shared helpers for domain-specific ADK agents.

The helpers keep domain modules small while preserving a consistent
deep-research execution pattern and lightweight persistence of outputs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from google.adk.agents import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps.app import App

log = logging.getLogger(__name__)

# Default artifact root mirrors the legacy crew output layout.
ARTIFACT_ROOT = Path("05_artifacts")


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_state_json(state_key: str, filename: str) -> Callable[[CallbackContext], None]:
    """Return a callback that writes a JSON-serializable state value to file."""

    def _callback(ctx: CallbackContext) -> None:
        data = ctx.state.get(state_key)
        if data is None:
            log.debug("No state for %s; skipping write to %s", state_key, filename)
            return
        path = ARTIFACT_ROOT / filename
        try:
            _ensure_dir(path)
            # Pydantic models may carry model_dump; fall back to raw object.
            if hasattr(data, "model_dump"):
                data = data.model_dump()
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info("Wrote %s to %s", state_key, path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to write %s to %s: %s", state_key, path, exc)

    return _callback


def write_state_text(state_key: str, filename: str) -> Callable[[CallbackContext], None]:
    """Return a callback that writes a text state value to file."""

    def _callback(ctx: CallbackContext) -> None:
        data = ctx.state.get(state_key)
        if data is None:
            log.debug("No text for %s; skipping write to %s", state_key, filename)
            return
        if hasattr(data, "markdown"):
            data = getattr(data, "markdown")
        path = ARTIFACT_ROOT / filename
        try:
            _ensure_dir(path)
            path.write_text(str(data), encoding="utf-8")
            log.info("Wrote %s to %s", state_key, path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to write %s to %s: %s", state_key, path, exc)

    return _callback


def build_app(pipeline: BaseAgent, name: str) -> App:
    """Create an App from a prepared pipeline agent."""
    return App(root_agent=pipeline, name=name)
