"""Unmet need / commercial landscape axis agent (ADK version)."""

from __future__ import annotations

from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search

from ..config import config
from .axis_schema import AxisRating
from .base import build_app, write_state_json, write_state_text
from app.agent import trace_persistence_callback


# ---------- Structured outputs ----------
class EpidemiologyData(BaseModel):
    """Epidemiology and prevalence data."""
    indication: str = Field(..., description="Target indication")
    prevalence: str = Field(..., description="Prevalence estimate with source")
    incidence: str | None = Field(default=None, description="Incidence estimate with source")
    patient_segments: list[str] = Field(default_factory=list, description="Key patient segments")
    geographic_distribution: str | None = Field(default=None, description="Geographic distribution notes")
    sources: list[str] = Field(default_factory=list, description="Source URLs/IDs")


class MarketSizeData(BaseModel):
    """Market size and opportunity data."""
    current_market_size: str = Field(..., description="Current market size estimate with currency/year")
    projected_market_size: str | None = Field(default=None, description="Projected market size with timeframe")
    cagr: str | None = Field(default=None, description="Compound annual growth rate if available")
    key_drivers: list[str] = Field(default_factory=list, description="Market growth drivers")
    sources: list[str] = Field(default_factory=list, description="Source URLs/IDs")


class CompetitiveLandscapeItem(BaseModel):
    """Individual competitive asset information."""
    asset_name: str = Field(..., description="Product/candidate name")
    company: str = Field(..., description="Company/sponsor")
    stage: str = Field(..., description="Development stage (preclinical/phase1/phase2/phase3/approved)")
    mechanism: str = Field(..., description="Mechanism of action")
    differentiation: str = Field(..., description="Key differentiation vs standard of care")
    source: str = Field(..., description="Source URL/ID")


class CompetitiveLandscape(BaseModel):
    """Competitive landscape analysis."""
    standard_of_care: list[str] = Field(default_factory=list, description="Current standard of care treatments")
    competitive_assets: list[CompetitiveLandscapeItem] = Field(default_factory=list, description="Competitive pipeline")
    unmet_gaps: list[str] = Field(default_factory=list, description="Identified unmet needs/gaps")
    sources: list[str] = Field(default_factory=list, description="Source URLs/IDs")


class PayerDynamics(BaseModel):
    """Payer and reimbursement dynamics."""
    coverage_status: str = Field(..., description="Current coverage status for similar assets")
    reimbursement_barriers: list[str] = Field(default_factory=list, description="Key reimbursement barriers")
    value_frameworks: list[str] = Field(default_factory=list, description="Relevant value frameworks (e.g., ICER, NICE)")
    payer_priorities: list[str] = Field(default_factory=list, description="Payer priorities and concerns")
    sources: list[str] = Field(default_factory=list, description="Source URLs/IDs")


class UnmetNeedResearchOutput(BaseModel):
    """Comprehensive research data on unmet need and commercial landscape."""
    epidemiology: EpidemiologyData
    market_size: MarketSizeData
    competitive_landscape: CompetitiveLandscape
    payer_dynamics: PayerDynamics
    additional_insights: list[str] = Field(default_factory=list, description="Additional relevant insights")
    research_metadata: dict[str, str] = Field(default_factory=dict, description="Research metadata (date, queries used, etc.)")


class StrategicInsight(BaseModel):
    """Individual strategic insight."""
    category: str = Field(..., description="Category (e.g., market_opportunity, competitive_positioning, payer_strategy)")
    insight: str = Field(..., description="Strategic insight description")
    rationale: str = Field(..., description="Rationale and evidence")
    implications: str = Field(..., description="Strategic implications")
    confidence: str = Field(..., description="Confidence level (high|medium|low)")
    sources: list[str] = Field(default_factory=list, description="Supporting source URLs/IDs")


class StrategicRecommendation(BaseModel):
    """Strategic recommendation."""
    recommendation: str = Field(..., description="Specific recommendation")
    priority: str = Field(..., description="Priority level (critical|high|medium|low)")
    rationale: str = Field(..., description="Why this recommendation matters")
    action_items: list[str] = Field(default_factory=list, description="Specific action items")


class UnmetNeedStrategyOutput(BaseModel):
    """Deep strategic analysis of unmet need and commercial landscape."""
    executive_summary: str = Field(..., description="2-3 sentence executive summary")
    key_insights: list[StrategicInsight] = Field(default_factory=list, description="Strategic insights")
    recommendations: list[StrategicRecommendation] = Field(default_factory=list, description="Strategic recommendations")
    risk_factors: list[str] = Field(default_factory=list, description="Key risk factors to monitor")
    opportunity_score: str = Field(..., description="Overall opportunity assessment (high|medium|low)")
    strategic_notes: str = Field(..., description="Additional strategic considerations")


class StrategicBrief(BaseModel):
    """Markdown-formatted strategic brief."""
    markdown: str = Field(..., description="Markdown-formatted strategic analysis brief")


unmet_need_reviewer = LlmAgent(
    name="unmet_need_reviewer",
    model=config.worker_model,
    description="Assesses unmet need, market size, and competitive intensity.",
    instruction="""
    You are the Commercial Landscape & Unmet Need reviewer.
    - Use epidemiology/market data supplied in session state `commercial_inputs` and google_search for benchmarks.
    - Quantify unmet need, SoC gaps, payer dynamics, pricing analogs; cite sources.
    - Return ONLY JSON AxisRating with:
      section_id: "commercial_landscape"
      display_name: "Commercial Landscape / Unmet Need"
      grade A-F/U (+ optional modifier)
      badges {coverage, confidence, evidence_direction}
      pros and caveats (3-5 each) with evidence URLs/IDs
    """,
    tools=[google_search],
    output_schema=AxisRating,
    output_key="axis_rating_commercial_landscape",
    after_agent_callback=[
        write_state_json(
            "axis_rating_commercial_landscape",
            "axis_rating_commercial_landscape.json",
        ),
        trace_persistence_callback,
    ],
)

unmet_need_pipeline = SequentialAgent(
    name="unmet_need_pipeline",
    description="Produces commercial landscape/unmet-need rating.",
    sub_agents=[unmet_need_reviewer],
)

app = build_app(unmet_need_pipeline, name="unmet_need_agent")
