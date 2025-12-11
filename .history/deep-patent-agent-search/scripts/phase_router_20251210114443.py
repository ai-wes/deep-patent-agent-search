#!/usr/bin/env python3
"""
Lightweight phase → ADK agent router.

Example:
  uv run python scripts/phase_router.py --project-id demo123 --phase P7b \
    --state-dir projects/demo123/05_artifacts

It sets ADK_AGENT_NAME and ADK_STATE_DIR, then launches the ADK API server.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Phase → agent mapping (can be overridden with --agent)
PHASE_TO_AGENT = {
    "p7a": "prior_art",
    "p7b": "risk",
    "p7c": "regulatory_cmc",
    "p7d": "commercial",
    "p7e": "mechanistic",
    # Convenience aliases
    "risk": "risk",
    "commercial": "commercial",
    "reg": "regulatory_cmc",
    "unmet": "unmet_need",
    "safety": "safety",
    "mech": "mechanistic",
}

# Agent -> ADK appName mapping (matches build_app names)
AGENT_TO_APPNAME = {
    "prior_art": "app",
    "commercial": "commercial_agent",
    "risk": "risk_agent",
    "regulatory_cmc": "regulatory_cmc_agent",
    "unmet_need": "unmet_need_agent",
    "safety": "safety_agent",
    "mechanistic": "mechanistic_agent",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run ADK backend with phase → agent mapping.")
    ap.add_argument("--phase", required=True, help="Phase code (e.g., P7a, P7b, P7c, P7d, P7e)")
    ap.add_argument("--project-id", required=True, help="Project id used to locate artifacts")
    ap.add_argument(
        "--state-dir",
        help="Optional path with deterministic outputs to pre-load (defaults to projects/<project-id>/05_artifacts if exists)",
    )
    ap.add_argument(
        "--agent",
        help="Override agent name (commercial|risk|regulatory_cmc|unmet_need|safety|mechanistic|prior_art)",
    )
    ap.add_argument(
        "--allow-origins",
        default="*",
        help='CORS allow_origins passed to ADK API server (default "*")',
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Start the ADK API server in background (non-blocking) so pipelines can continue.",
    )
    ap.add_argument(
        "--auto-run",
        action="store_true",
        help="Sequential mode: start server, POST /run with 01_sources/sources.json, then shut down.",
    )
    ap.add_argument(
        "--port",
        type=int,
        help="Port for adk api_server (defaults to ADK_API_PORT env or 8080).",
    )
    return ap.parse_args()


def resolve_state_dir(arg_state: str | None, project_id: str) -> str:
    """
    Resolve the state directory for the run.

    Priority:
      1) Explicit --state-dir
      2) <repo>/projects/<project_id>/05_artifacts (create if missing)
      3) <repo>/projects/<project_id> (create if missing)
    Always returns a path (created if necessary) so callers have a stable location.
    """
    if arg_state:
        return arg_state

    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = repo_root / "projects" / project_id / "05_artifacts"
    project_dir = artifacts_dir.parent

    # Ensure project directory exists
    project_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return str(artifacts_dir if artifacts_dir.exists() else project_dir)


def main() -> int:
    args = parse_args()
    phase_key = args.phase.lower()
    agent = (args.agent or PHASE_TO_AGENT.get(phase_key) or "prior_art").strip()

    state_dir = resolve_state_dir(args.state_dir, args.project_id)
    os.environ["ADK_STATE_DIR"] = state_dir
    print(f"[phase-router] ADK_STATE_DIR={state_dir}")

    os.environ["ADK_AGENT_NAME"] = agent
    print(f"[phase-router] ADK_AGENT_NAME={agent} (phase={args.phase})")

    port = args.port or int(os.environ.get("ADK_API_PORT", "8080"))
    cmd = [
        "uv",
        "run",
        "adk",
        "api_server",
        "app",
        f"--allow_origins={args.allow_origins}",
        f"--port={port}",
    ]
    print(f"[phase-router] launching: {' '.join(cmd)}")

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"adk_{agent}.log"

    if args.headless and not args.auto_run:
        with log_path.open("a", encoding="utf-8") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        print(f"[phase-router] headless mode: pid={proc.pid}, log={log_path}")
        return 0

    # Sequential auto-run: start server, wait ready, POST /run with sources.json, then stop
    with log_path.open("a", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)

        try:
            import time
            import requests  # type: ignore
        except Exception as exc:  # noqa: BLE001
            print(f"[phase-router] ERROR: requests not available ({exc})")
            return 1

        base_url = f"http://127.0.0.1:{port}"

        def _wait_ready(timeout: float = 30.0) -> bool:
            start = time.time()
            while time.time() - start < timeout:
                try:
                    r = requests.get(f"{base_url}/list-apps", timeout=1.5)
                    if r.ok:
                        return True
                except Exception:
                    pass
                time.sleep(0.6)
            return False

        if args.auto_run or not args.headless:
            if not _wait_ready():
                print(f"[phase-router] Server not ready on port {port}; see {log_path}")
                proc.terminate()
                return 1

            sources_path = Path("projects") / args.project_id / "01_sources" / "sources.json"
            if not sources_path.exists():
                print(f"[phase-router] WARNING: {sources_path} missing; sending empty input")
                sources_text = "{}"
            else:
                try:
                    sources_text = sources_path.read_text(encoding="utf-8")
                except Exception as exc:  # noqa: BLE001
                    print(f"[phase-router] WARNING: failed reading sources.json: {exc}; sending empty input")
                    sources_text = "{}"

            app_name = AGENT_TO_APPNAME.get(agent, "app")
            payload = {
                "appName": app_name,
                "userId": args.project_id,
                "sessionId": f"{args.project_id}-{agent}",
                "newMessage": {
                    "role": "user",
                    "parts": [{"text": sources_text}],
                },
                "streaming": False,
            }

            try:
                resp = requests.post(f"{base_url}/run", json=payload, timeout=180)
                if not resp.ok:
                    print(f"[phase-router] /run failed (status {resp.status_code}): {resp.text[:400]}")
                    proc.terminate()
                    return 1
                print(f"[phase-router] /run completed for agent={agent}, phase={args.phase}")
            except Exception as exc:  # noqa: BLE001
                print(f"[phase-router] ERROR calling /run: {exc}")
                proc.terminate()
                return 1
            finally:
                proc.terminate()
            return 0

        # Fallback: blocking server (no auto-run)
        return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
