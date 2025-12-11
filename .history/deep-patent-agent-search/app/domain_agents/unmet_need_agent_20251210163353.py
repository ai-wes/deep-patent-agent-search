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


# ---------- Agents ----------
unmet_need_researcher = LlmAgent(
    name="unmet_need_researcher",
    model=config.worker_model,
    description="Gathers comprehensive data on unmet need, epidemiology, market size, competitive landscape, and payer dynamics.",
    instruction="""
    You are the Unmet Need Data Researcher.
    - Use google_search extensively to gather comprehensive data on:
      1. Epidemiology: prevalence, incidence, patient segments, geographic distribution
      2. Market Size: current and projected market size, CAGR, growth drivers
      3. Competitive Landscape: standard of care, competitive pipeline, differentiation points, unmet gaps
      4. Payer Dynamics: coverage status, reimbursement barriers, value frameworks, payer priorities
    - Use session state `commercial_inputs` if available for initial context.
    - For each data point, include verifiable sources (URLs, PMIDs, press releases, etc.).
    - Be thorough and comprehensive - gather 10-20+ data points across all categories.
    - Return ONLY JSON matching UnmetNeedResearchOutput schema.
    - Ensure all sources are properly cited in the respective source fields.
    """,
    tools=[google_search],
    output_schema=UnmetNeedResearchOutput,
    output_key="unmet_need_research",
    after_agent_callback=[
        write_state_json("unmet_need_research", "unmet_need_research.json"),
        trace_persistence_callback,
    ],
)

unmet_need_strategist = LlmAgent(
    name="unmet_need_strategist",
    model=config.critic_model,
    description="Performs deep strategic analysis of unmet need research to identify opportunities, risks, and strategic recommendations.",
    instruction="""
    You are the Unmet Need Strategic Analyst.
    - Analyze the comprehensive research data from `unmet_need_research` in session state.
    - Perform deep strategic thinking to:
      1. Identify key market opportunities and positioning strategies
      2. Assess competitive positioning and differentiation opportunities
      3. Evaluate payer dynamics and reimbursement strategies
      4. Surface critical risk factors and mitigation approaches
      5. Generate actionable strategic recommendations
    - Consider cross-domain implications (clinical, regulatory, commercial, IP).
    - Think critically about market timing, competitive windows, and value proposition.
    - Provide 5-8 strategic insights across different categories.
    - Generate 3-5 prioritized strategic recommendations with specific action items.
    - Assess overall opportunity score (high/medium/low) with clear rationale.
    - Return ONLY JSON matching UnmetNeedStrategyOutput schema.
    - Every insight and recommendation must be grounded in the research data with citations.
    """,
    tools=[google_search],
    output_schema=UnmetNeedStrategyOutput,
    output_key="unmet_need_strategy",
    after_agent_callback=[
        write_state_json("unmet_need_strategy", "unmet_need_strategy.json"),
        trace_persistence_callback,
    ],
)

unmet_need_strategic_brief = LlmAgent(
    name="unmet_need_strategic_brief",
    model=config.critic_model,
    description="Synthesizes research and strategy into a comprehensive markdown strategic brief.",
    instruction="""
    Write a comprehensive markdown strategic brief titled '# Unmet Need & Commercial Landscape Strategic Analysis'.
    Use:
    - unmet_need_research (research data from researcher)
    - unmet_need_strategy (strategic analysis from strategist)
    Structure:
    # Unmet Need & Commercial Landscape Strategic Analysis
    ## Executive Summary
    <2-3 sentence summary>
    ## Market Opportunity
    * Key insights from epidemiology and market size data
    * Opportunity assessment with citations
    ## Competitive Landscape
    * Standard of care analysis
    * Competitive positioning
    * Unmet gaps and differentiation opportunities
    ## Payer & Reimbursement Strategy
    * Coverage and reimbursement dynamics
    * Value framework alignment
    * Payer strategy recommendations
    ## Strategic Recommendations
    * Prioritized recommendations with action items
    ## Risk Factors
    * Key risks to monitor
    Every factual claim must include inline citations (URL or source reference). No code fences.
    """,
    output_schema=StrategicBrief,
    output_key="unmet_need_strategic_brief",
    after_agent_callback=[
        write_state_text("unmet_need_strategic_brief", "unmet_need_strategic_brief.md"),
        trace_persistence_callback,
    ],
)

unmet_need_reviewer = LlmAgent(
    name="unmet_need_reviewer",
    model=config.worker_model,
    description="Assesses unmet need, market size, and competitive intensity based on research and strategic analysis.",
    instruction="""
    You are the Commercial Landscape & Unmet Need reviewer.
    - Use comprehensive research from `unmet_need_research` and strategic analysis from `unmet_need_strategy` in session state.
    - Also use epidemiology/market data from `commercial_inputs` if available.
    - Use google_search for additional benchmarks if needed.
    - Synthesize all information to quantify unmet need, SoC gaps, payer dynamics, pricing analogs.
    - Return ONLY JSON AxisRating with:
      section_id: "commercial_landscape"
      display_name: "Commercial Landscape / Unmet Need"
      grade A-F/U (+ optional modifier)
      badges {coverage, confidence, evidence_direction}
      summary: 2-4 sentence narrative synthesizing key findings
      decision_hook: One-sentence decision guidance
      pros and caveats (3-5 each) with evidence URLs/IDs
      evidence: List of key source URLs/IDs
    - Base your assessment on the comprehensive research and strategic analysis provided.
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
