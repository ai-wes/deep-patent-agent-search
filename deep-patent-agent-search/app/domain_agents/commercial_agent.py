"""Commercial deep-research pipeline (ADK version).

Ports the legacy CrewAI commercial crew into ADK with the same output schemas:
- pricing_analogs.json
- patient_flow_map.json
- commercial_summary.md
"""

from __future__ import annotations

import json
import re
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import google_search

from ..config import config
from .base import build_app, write_state_json, write_state_text
from .renderers import patient_flow_png_callback
from app.agent import trace_persistence_callback


# ---------- Structured outputs ----------
class PricingAnalogItemOutput(BaseModel):
    product: str = Field(..., description="Product name")
    company: str = Field(..., description="Manufacturer or sponsor company")
    indication: str = Field(..., description="Approved indication")
    launch_year: str = Field(..., description="Year of regulatory approval/launch")
    price_reference: str = Field(..., description="Type of pricing data (WAC, List, Net)")
    price_currency: str = Field(..., description="Currency code")
    price_value: str = Field(..., description="Price amount with units and frequency")
    price_basis: str = Field(..., description="Data source or pricing framework")
    rationale: str = Field(..., description="Comparability justification")
    gb_tar_alignment: str = Field(..., description="GB-TAR alignment analysis")
    value_drivers: list[str] = Field(..., description="Key factors driving value")
    source: str = Field(..., description="Source/link of the pricing data")


class PricingAnalogOutput(BaseModel):
    analogs: list[PricingAnalogItemOutput]


class PatientFlowNodeOutput(BaseModel):
    id: str
    label: str
    phase: str = Field(..., pattern="^(Diagnosis|Treatment|Outcome)$")
    notes: str
    source: str


class PatientFlowEdgeOutput(BaseModel):
    source: str
    target: str
    note: str


class PatientFlowOutput(BaseModel):
    nodes: list[PatientFlowNodeOutput]
    edges: list[PatientFlowEdgeOutput]


class CommercialSummary(BaseModel):
    markdown: str = Field(..., description="Markdown-formatted commercial brief")


# ---------- Agents ----------
def _strip_code_fence(text: str) -> str:
    if not text:
        return text
    fence = re.search(r"```json\\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _coerce_json(text: str) -> dict | None:
    text = _strip_code_fence(text)
    try:
        return json.loads(text)
    except Exception:
        return None


def _parse_and_store_pricing(callback_context: CallbackContext):
    state = callback_context.state
    raw = state.get("pricing_analogs")
    if raw is None:
        return
    data = None
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        data = _coerce_json(raw)
    if data is None:
        return
    state["pricing_analogs"] = data


pricing_after_callbacks = [
    _parse_and_store_pricing,
    write_state_json("pricing_analogs", "pricing_analogs.json"),
]


pricing_researcher = LlmAgent(
    name="pricing_researcher",
    model=config.worker_model,
    description="Compiles pricing analogs and value references for comparable assets.",
    instruction="""
    You are the Pricing Researcher.
    - Use google_search to fetch pricing disclosures, labels, press releases.
    - Only return JSON (no markdown) matching PricingAnalogOutput schema.
    - Every analog must carry a verifiable citation in `source` (URL/PMID/press release/query used).
    - Focus on products with similar indication, mechanism, or population; keep 5-12 strong analogs.
    JSON shape:
    {
      "analogs": [
        {
          "product": "...",
          "company": "...",
          "indication": "...",
          "launch_year": "...",
          "price_reference": "...",
          "price_currency": "...",
          "price_value": "...",
          "price_basis": "...",
          "rationale": "...",
          "gb_tar_alignment": "...",
          "value_drivers": ["..."],
          "source": "ABSOLUTE URL OR EXACT QUERY"
        }
      ]
    }
    Return ONLY the JSON object.
    """,
    tools=[google_search],
    output_schema=None,  # allow fencey JSON; we parse in callback
    output_key="pricing_analogs",
    after_agent_callback=pricing_after_callbacks,
)

patient_flow_modeler = LlmAgent(
    name="patient_flow_modeler",
    model=config.worker_model,
    description="Designs patient-flow map highlighting commercial checkpoints.",
    instruction="""
    Build a patient-flow for the specific indication:
    - 5-8 nodes spanning Diagnosis -> Treatment -> Outcome (phase must be exactly one of these).
    - Include payer checkpoints (prior auth, coverage milestones).
    - Cite guideline/payer sources in each node's `source`.
    Output ONLY JSON matching PatientFlowOutput:
    {
      "nodes":[{"id":"n1","label":"...","phase":"Diagnosis|Treatment|Outcome","notes":"...","source":"URL"}],
      "edges":[{"source":"n1","target":"n2","note":"..."}]
    }
    No markdown or prose.
    """,
    tools=[google_search],
    output_schema=PatientFlowOutput,
    output_key="patient_flow",
    after_agent_callback=[
        write_state_json("patient_flow", "patient_flow_map.json"),
        patient_flow_png_callback,
        trace_persistence_callback,
    ],
)

commercial_narrator = LlmAgent(
    name="commercial_narrator",
    model=config.critic_model,
    description="Synthesizes pricing and patient-flow insights into a GB-TAR aligned brief.",
    instruction="""
    Write markdown titled '# Commercial Readiness' using:
    - pricing_analogs (pricing_researcher output)
    - patient_flow (patient_flow_modeler output)
    - risk markers in session state (if present: gb_tar scores/domain_rag)
    Structure:
    # Commercial Readiness
    <2-3 sentence exec summary with citations>
    ## Pricing Analog Insights
    * bullet insights with citations to pricing analogs
    ## Patient Flow Implications
    * bullets linking checkpoints to payer/commercial requirements (cite nodes)
    ## Commercial implication and action
    * one concise actionable takeaway
    Every factual claim must include an inline citation (URL or analog/node reference). No code fences.
    """,
    output_schema=CommercialSummary,
    output_key="commercial_summary",
    after_agent_callback=[
        write_state_text("commercial_summary", "commercial_summary.md"),
        trace_persistence_callback,
    ],
)

# ---------- Pipeline & App ----------
commercial_pipeline = SequentialAgent(
    name="commercial_pipeline",
    description="Runs pricing research, patient flow modeling, then synthesizes a commercial brief.",
    sub_agents=[pricing_researcher, patient_flow_modeler, commercial_narrator],
)

app = build_app(commercial_pipeline, name="commercial_agent")
