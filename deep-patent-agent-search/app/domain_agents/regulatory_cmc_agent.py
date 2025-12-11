"""Regulatory & CMC diligence pipeline (ADK version)."""

from __future__ import annotations

from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search

from ..config import config
from .base import build_app, write_state_json, write_state_text
from app.agent import trace_persistence_callback


class RegulatoryDesignation(BaseModel):
    agency: str
    region: str
    approval_type: str
    status: str
    effective_date: str
    decision_scope: str
    labeling_notes: str
    post_market_commitments: list[str]
    renewal_or_review_date: str
    evidence_type: str
    evidence_source: str
    notes: str


class RegulatoryAction(BaseModel):
    requirement: str
    description: str
    owner: str
    due_date: str
    status: str
    risk_if_missed: str
    evidence_source: str


class RegulatoryPayload(BaseModel):
    regulatory_designations: list[RegulatoryDesignation]
    regulatory_actions: list[RegulatoryAction]


class RegulatoryNarrative(BaseModel):
    markdown: str


reg_payload_agent = LlmAgent(
    name="reg_payload_agent",
    model=config.worker_model,
    description="Builds structured regulatory designations and action list.",
    instruction="""
    You are a regulatory intelligence analyst.
    - Use google_search when needed; rely on provided target metadata in session state.
    - Return ONLY JSON with two arrays: regulatory_designations, regulatory_actions.
    - Every entry must include evidence_source (URL) and concise notes.
    JSON schema mirrors RegulatoryPayload.
    """,
    tools=[google_search],
    output_schema=RegulatoryPayload,
    output_key="reg_payload",
    after_agent_callback=[
        write_state_json("reg_payload", "regulatory_payload.json"),
        trace_persistence_callback,
    ],
)

reg_narrative_agent = LlmAgent(
    name="reg_narrative_agent",
    model=config.critic_model,
    description="Summarizes regulatory posture and CMC checkpoints in markdown.",
    instruction="""
    Write markdown titled '# Regulatory & CMC Brief'.
    Sections: Current Designations, Active Requirements, CMC Risks, Escalation Guidance.
    Base the brief on reg_payload plus any target metadata in state.
    Inline-cite evidence_source URLs for every claim. No code fences.
    """,
    output_schema=RegulatoryNarrative,
    output_key="regulatory_brief",
    after_agent_callback=[
        write_state_text("regulatory_brief", "regulatory_cmc_brief.md"),
        trace_persistence_callback,
    ],
)

regulatory_pipeline = SequentialAgent(
    name="regulatory_pipeline",
    description="Generates structured regulatory payload then synthesizes a brief.",
    sub_agents=[reg_payload_agent, reg_narrative_agent],
)

app = build_app(regulatory_pipeline, name="regulatory_cmc_agent")
