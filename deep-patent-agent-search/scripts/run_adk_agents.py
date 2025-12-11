#!/usr/bin/env python3
"""
Run ADK agents sequentially against a running api_server.

Usage:
  # start the server in another terminal:
  #   uv run adk api_server app --allow_origins="*" --port 8080
  #
  # then invoke:
  uv run python scripts/run_adk_agents.py --project-id demo

Options:
  --agents prior_art,risk,...   comma list (default: all domain agents)
  --host 127.0.0.1              api_server host (env ADK_API_HOST)
  --port 8080                   api_server port (env ADK_API_PORT)
  --app-name domain_agents      ADK app name (env ADK_APP_NAME)
  --sources-path <path>         override sources JSON (default projects/<project>/01_sources/sources.json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

try:
    import requests  # type: ignore
except ImportError:
    print("requests is required. Install with: uv pip install requests")
    sys.exit(1)


DEFAULT_AGENTS = [
    "prior_art",
    "risk",
    "regulatory_cmc",
    "commercial",
    "mechanistic",
    "safety",
    "unmet_need",
]


def load_sources(path: Path) -> str:
    if not path.exists():
        print(f"[warn] {path} missing; using empty JSON")
        return "{}"
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] failed reading {path}: {exc}; using empty JSON")
        return "{}"


def create_session(base: str, app: str, user: str, session: str) -> None:
    url_variants = [
        f"{base}/apps/{app}/users/{user}/sessions/{session}",
        f"{base}/api/apps/{app}/users/{user}/sessions/{session}",
    ]
    for url in url_variants:
        try:
            resp = requests.post(url, json={}, timeout=15)
            if resp.status_code in (200, 201, 204, 409):
                return
        except Exception:
            continue
    # Not fatal; /run_sse will still be attempted.


def run_agent(base: str, app: str, agent: str, user: str, session: str, sources_text: str) -> bool:
    payload: Dict[str, object] = {
        "appName": app,
        "userId": user,
        "sessionId": session,
        "newMessage": {"role": "user", "parts": [{"text": sources_text}]},
        "streaming": False,
        "input": sources_text,
    }

    # Only use run_sse endpoints – /run returns 500/404 in this build.
    endpoints = [
        f"{base}/run_sse",
        f"{base}/api/run_sse",
    ]
    for url in endpoints:
        try:
            resp = requests.post(url, json=payload, timeout=240)
            if resp.ok:
                print(f"[ok] {agent} ({resp.status_code})")
                return True
            else:
                print(f"[warn] {agent} {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:  # noqa: BLE001
            print(f"[err] {agent} exception: {exc}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ADK agents sequentially against a running api_server.")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--agents", help="Comma list of agents (default all domain agents)")
    ap.add_argument("--host", default=os.environ.get("ADK_API_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("ADK_API_PORT", "8080")))
    ap.add_argument(
        "--base-url",
        default=os.environ.get("ADK_API_BASE", None),
        help="Full base URL (e.g., https://my-service-abc.a.run.app). Overrides host/port when set.",
    )
    ap.add_argument("--app-name", default=os.environ.get("ADK_APP_NAME", "domain_agents"))
    ap.add_argument("--sources-path", help="Override sources JSON path")
    args = ap.parse_args()

    agents: List[str] = (
        [a.strip() for a in args.agents.split(",") if a.strip()]
        if args.agents
        else DEFAULT_AGENTS
    )

    if not agents:
        print("No agents specified.")
        return 1

    base = args.base_url if args.base_url else f"http://{args.host}:{args.port}"
    sources_path = (
        Path(args.sources_path)
        if args.sources_path
        else Path("projects") / args.project_id / "01_sources" / "sources.json"
    )
    sources_text = load_sources(sources_path)

    any_fail = False
    for agent in agents:
        session_id = f"{args.project_id}-{agent}"
        create_session(base, args.app_name, args.project_id, session_id)
        ok = run_agent(base, args.app_name, agent, args.project_id, session_id, sources_text)
        if not ok:
            any_fail = True
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
