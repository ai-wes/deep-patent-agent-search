"""Lightweight renderers to mirror legacy crew chart outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from google.adk.agents.callback_context import CallbackContext

from .base import ARTIFACT_ROOT, _ensure_dir

log = logging.getLogger(__name__)


def _safe_imports():
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
        import networkx as nx  # noqa: WPS433

        return plt, nx
    except Exception as exc:  # noqa: BLE001
        log.warning("Plotting dependencies missing, skipping rendering: %s", exc)
        return None, None


def patient_flow_png_callback(ctx: CallbackContext) -> None:
    """Render patient flow JSON to a simple left-to-right PNG."""
    flow = ctx.state.get("patient_flow")
    if not flow:
        log.debug("No patient_flow in state; skipping PNG render")
        return

    plt, nx = _safe_imports()
    if not plt:
        return

    try:
        nodes = flow.get("nodes", [])
        edges = flow.get("edges", [])
        graph = nx.DiGraph()
        for node in nodes:
            graph.add_node(
                node.get("id"),
                label=node.get("label", ""),
                phase=node.get("phase", "Other"),
            )
        for edge in edges:
            graph.add_edge(edge.get("source"), edge.get("target"), note=edge.get("note", ""))

        phase_order = {"Diagnosis": 0, "Treatment": 1, "Outcome": 2}
        pos = {}
        for node in graph.nodes:
            phase = graph.nodes[node].get("phase", "Other")
            pos[node] = (phase_order.get(phase, 1), -list(graph.nodes).index(node) * 0.6)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {
            "Diagnosis": "#2563EB",
            "Treatment": "#14B8A6",
            "Outcome": "#F97316",
            "Other": "#9CA3AF",
        }
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=[colors.get(graph.nodes[n].get("phase", "Other"), colors["Other"]) for n in graph.nodes],
            node_size=1200,
            edgecolors="#1F2937",
            ax=ax,
        )
        nx.draw_networkx_labels(
            graph,
            pos,
            labels={n: graph.nodes[n].get("label", n) for n in graph.nodes},
            font_size=8,
            ax=ax,
        )
        nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle="-|>", width=1.6, edge_color="#6B7280", ax=ax)
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels={(u, v): d.get("note", "") for u, v, d in graph.edges(data=True) if d.get("note")},
            font_size=7,
            rotate=False,
            ax=ax,
        )
        ax.axis("off")
        output_path = ARTIFACT_ROOT / "patient_flow_map.png"
        _ensure_dir(output_path)
        plt.tight_layout()
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        ctx.state["patient_flow_png"] = str(output_path)
        log.info("Rendered patient flow PNG to %s", output_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to render patient flow PNG: %s", exc)


def risk_matrix_callback(ctx: CallbackContext) -> None:
    """Render a simple risk matrix bar chart from risk_triage output."""
    triage = ctx.state.get("risk_triage") or {}
    items = triage.get("triage_items") if isinstance(triage, dict) else None
    if not items:
        log.debug("No risk_triage items; skipping risk matrix render")
        return

    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        log.warning("matplotlib unavailable; skipping risk matrix: %s", exc)
        return

    try:
        items_sorted = sorted(items, key=lambda x: x.get("severity_score", 0), reverse=True)[:25]
        labels = [f"{it.get('domain','')}: {it.get('id')}" for it in items_sorted]
        scores = [float(it.get("severity_score", 0)) for it in items_sorted]
        colors = ["#EF4444" if s > 0.65 else "#F59E0B" if s > 0.35 else "#10B981" for s in scores]

        fig, ax = plt.subplots(figsize=(10, max(4, len(scores) * 0.35)))
        ax.barh(labels, scores, color=colors)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel("Severity Score")
        ax.set_title("Risk Matrix (top 25)")
        for idx, score in enumerate(scores):
            ax.text(score + 0.02, idx, f"{score:.2f}", va="center", fontsize=8)

        output_path = ARTIFACT_ROOT / "risk_matrix.png"
        _ensure_dir(output_path)
        plt.tight_layout()
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        ctx.state["risk_matrix_png"] = str(output_path)
        log.info("Rendered risk matrix PNG to %s", output_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to render risk matrix: %s", exc)


def dump_state_snapshot_callback(filename: str = "session_state_snapshot.json"):
    """Factory to persist entire state for debugging."""

    def _callback(ctx: CallbackContext) -> None:
        try:
            path = ARTIFACT_ROOT / filename
            _ensure_dir(path)
            path.write_text(json.dumps(ctx.state, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info("Wrote session state snapshot to %s", path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to dump session state snapshot: %s", exc)

    return _callback

