"""Mechanistic plausibility axis agent (ADK version)."""

from __future__ import annotations

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search

from ..config import config
from .axis_schema import AxisRating, AxisBadges
from .base import build_app, write_state_json
from app.agent import trace_persistence_callback


mechanistic_reviewer = LlmAgent(
    name="mechanistic_reviewer",
    model=config.worker_model,
    description="Assesses mechanistic plausibility with evidence-backed rating.",
    instruction="""
    You are the Mechanistic Reviewer.
    - Use provided deterministic analysis outputs (mechanism, pathways, models) from session state `mechanistic_inputs`.
    - Use google_search only to fill minor gaps.
    - Return ONLY JSON matching AxisRating (see fields).
    Requirements:
    * section_id: "mechanistic_plausibility"
    * display_name: "Mechanistic Plausibility"
    * grade: A/B/C/D/F/U with optional grade_modifier (sparse/conflicting/liability/data_desert/none)
    * badges: coverage, confidence, evidence_direction (favorable|mixed|unclear|adverse)
    * 3-5 pros and 3-5 caveats, each concise
    * evidence: list of URLs or dataset IDs; cite deterministic outputs where possible
    """,
    tools=[google_search],
    output_schema=AxisRating,
    output_key="axis_rating_mechanistic_plausibility",
    after_agent_callback=[
        write_state_json(
            "axis_rating_mechanistic_plausibility",
            "axis_rating_mechanistic_plausibility.json",
        ),
        trace_persistence_callback,
    ],
)

mechanistic_pipeline = SequentialAgent(
    name="mechanistic_pipeline",
    description="Produces mechanistic plausibility rating.",
    sub_agents=[mechanistic_reviewer],
)

app = build_app(mechanistic_pipeline, name="mechanistic_agent")
