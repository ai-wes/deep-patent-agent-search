"""Safety & liability axis agent (ADK version)."""

from __future__ import annotations

from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search

from ..config import config
from .axis_schema import AxisRating
from .base import build_app, write_state_json, write_state_text
from app.agent import trace_persistence_callback


# ---------- Structured outputs ----------
class ToxicityData(BaseModel):
    """Toxicity and safety pharmacology data."""
    on_target_toxicity: list[str] = Field(default_factory=list, description="On-target toxicity concerns")
    off_target_toxicity: list[str] = Field(default_factory=list, description="Off-target toxicity concerns")
    structural_toxicity_indicators: list[str] = Field(default_factory=list, description="Structural alerts and toxicity indicators")
    herg_liability: str | None = Field(default=None, description="hERG channel liability assessment")
    genotoxicity: str | None = Field(default=None, description="Genotoxicity assessment")
    mutagenicity: str | None = Field(default=None, description="Mutagenicity assessment")
    carcinogenicity: str | None = Field(default=None, description="Carcinogenicity assessment")
    sources: list[str] = Field(default_factory=list, description="Source URLs/IDs")


class DMPKData(BaseModel):
    """Drug metabolism and pharmacokinetics data."""
    absorption: str | None = Field(default=None, description="Absorption characteristics")
    distribution: str | None = Field(default=None, description="Distribution profile")
    metabolism: str | None = Field(default=None, description="Metabolism pathways and enzymes")
    excretion: str | None = Field(default=None, description="Excretion profile")
    drug_drug_interactions: list[str] = Field(default_factory=list, description="Potential drug-drug interactions")
    cyp_inhibition: list[str] = Field(default_factory=list, description="CYP enzyme inhibition profile")
    cyp_induction: list[str] = Field(default_factory=list, description="CYP enzyme induction profile")
    transporters: list[str] = Field(default_factory=list, description="Transporter interactions")
    sources: list[str] = Field(default_factory=list, description="Source URLs/IDs")


class ClinicalSafetyData(BaseModel):
    """Clinical safety signals and adverse events."""
    preclinical_safety_signals: list[str] = Field(default_factory=list, description="Preclinical safety signals")
    clinical_adverse_events: list[str] = Field(default_factory=list, description="Clinical adverse events from similar compounds")
    safety_margin: str | None = Field(default=None, description="Therapeutic index and safety margin")
    contraindications: list[str] = Field(default_factory=list, description="Known or predicted contraindications")
    special_populations: list[str] = Field(default_factory=list, description="Safety concerns in special populations")
    black_box_warnings: list[str] = Field(default_factory=list, description="Potential black box warning risks")
    sources: list[str] = Field(default_factory=list, description="Source URLs/IDs")


class LiabilityProfile(BaseModel):
    """Safety liability and risk profile."""
    high_risk_liabilities: list[str] = Field(default_factory=list, description="High-risk safety liabilities")
    medium_risk_liabilities: list[str] = Field(default_factory=list, description="Medium-risk safety liabilities")
    low_risk_liabilities: list[str] = Field(default_factory=list, description="Low-risk safety liabilities")
    liability_mitigation: list[str] = Field(default_factory=list, description="Potential mitigation strategies")
    regulatory_concerns: list[str] = Field(default_factory=list, description="Regulatory safety concerns")
    sources: list[str] = Field(default_factory=list, description="Source URLs/IDs")


class SafetyResearchOutput(BaseModel):
    """Comprehensive safety research data."""
    toxicity: ToxicityData
    dmpk: DMPKData
    clinical_safety: ClinicalSafetyData
    liability_profile: LiabilityProfile
    additional_insights: list[str] = Field(default_factory=list, description="Additional relevant safety insights")
    research_metadata: dict[str, str] = Field(default_factory=dict, description="Research metadata (date, queries used, etc.)")


class SafetyInsight(BaseModel):
    """Individual safety strategic insight."""
    category: str = Field(..., description="Category (e.g., toxicity_risk, dmpk_concern, clinical_safety, liability)")
    insight: str = Field(..., description="Safety insight description")
    rationale: str = Field(..., description="Rationale and evidence")
    implications: str = Field(..., description="Development and regulatory implications")
    confidence: str = Field(..., description="Confidence level (high|medium|low)")
    sources: list[str] = Field(default_factory=list, description="Supporting source URLs/IDs")


class SafetyRecommendation(BaseModel):
    """Safety-related recommendation."""
    recommendation: str = Field(..., description="Specific safety recommendation")
    priority: str = Field(..., description="Priority level (critical|high|medium|low)")
    rationale: str = Field(..., description="Why this recommendation matters")
    action_items: list[str] = Field(default_factory=list, description="Specific action items")


class SafetyStrategyOutput(BaseModel):
    """Deep strategic analysis of safety and liability."""
    executive_summary: str = Field(..., description="2-3 sentence executive summary")
    key_insights: list[SafetyInsight] = Field(default_factory=list, description="Safety strategic insights")
    recommendations: list[SafetyRecommendation] = Field(default_factory=list, description="Safety recommendations")
    risk_factors: list[str] = Field(default_factory=list, description="Key safety risk factors to monitor")
    safety_score: str = Field(..., description="Overall safety assessment (favorable|moderate|concerning|high_risk)")
    regulatory_readiness: str = Field(..., description="Regulatory safety readiness (ready|needs_work|not_ready)")
    strategic_notes: str = Field(..., description="Additional strategic safety considerations")


class SafetyBrief(BaseModel):
    """Markdown-formatted safety strategic brief."""
    markdown: str = Field(..., description="Markdown-formatted safety analysis brief")


# ---------- Agents ----------
safety_researcher = LlmAgent(
    name="safety_researcher",
    model=config.worker_model,
    description="Gathers comprehensive data on toxicity, DMPK, clinical safety signals, and liability profiles.",
    instruction="""
    You are the Safety Data Researcher.
    - Use google_search extensively to gather comprehensive safety data on:
      1. Toxicity: on-target and off-target toxicity, structural alerts, hERG liability, genotoxicity, mutagenicity, carcinogenicity
      2. DMPK: absorption, distribution, metabolism, excretion, drug-drug interactions, CYP inhibition/induction, transporter interactions
      3. Clinical Safety: preclinical safety signals, clinical adverse events from similar compounds, safety margins, contraindications, special populations, black box warning risks
      4. Liability Profile: high/medium/low risk liabilities, mitigation strategies, regulatory concerns
    - Use session state `safety_inputs` if available for deterministic tox/DMPK outputs.
    - Search for literature on similar compounds, mechanisms, and structural analogs.
    - For each data point, include verifiable sources (URLs, PMIDs, FDA labels, regulatory documents, etc.).
    - Be thorough and comprehensive - gather 15-25+ data points across all categories.
    - Return ONLY JSON matching SafetyResearchOutput schema.
    - Ensure all sources are properly cited in the respective source fields.
    """,
    tools=[google_search],
    output_schema=SafetyResearchOutput,
    output_key="safety_research",
    after_agent_callback=[
        write_state_json("safety_research", "safety_research.json"),
        trace_persistence_callback,
    ],
)

safety_strategist = LlmAgent(
    name="safety_strategist",
    model=config.critic_model,
    description="Performs deep strategic analysis of safety research to identify risks, mitigation strategies, and regulatory implications.",
    instruction="""
    You are the Safety Strategic Analyst.
    - Analyze the comprehensive safety research data from `safety_research` in session state.
    - Perform deep strategic thinking to:
      1. Identify critical safety risks and their development implications
      2. Assess DMPK concerns and their impact on dosing, drug-drug interactions, and special populations
      3. Evaluate clinical safety signals and their regulatory implications
      4. Surface liability concerns and potential mitigation strategies
      5. Assess regulatory readiness and identify gaps
      6. Generate actionable safety recommendations
    - Consider cross-domain implications (clinical development, regulatory, commercial, IP).
    - Think critically about:
      - Development timeline impacts (e.g., need for additional studies)
      - Regulatory approval risks (e.g., black box warnings, REMS requirements)
      - Commercial implications (e.g., label restrictions, market access)
      - Patient safety and therapeutic index
    - Provide 6-10 strategic insights across different safety categories.
    - Generate 4-6 prioritized safety recommendations with specific action items.
    - Assess overall safety score (favorable/moderate/concerning/high_risk) with clear rationale.
    - Assess regulatory readiness (ready/needs_work/not_ready) with specific gaps identified.
    - Return ONLY JSON matching SafetyStrategyOutput schema.
    - Every insight and recommendation must be grounded in the research data with citations.
    """,
    tools=[google_search],
    output_schema=SafetyStrategyOutput,
    output_key="safety_strategy",
    after_agent_callback=[
        write_state_json("safety_strategy", "safety_strategy.json"),
        trace_persistence_callback,
    ],
)

safety_strategic_brief = LlmAgent(
    name="safety_strategic_brief",
    model=config.critic_model,
    description="Synthesizes safety research and strategy into a comprehensive markdown strategic brief.",
    instruction="""
    Write a comprehensive markdown safety brief titled '# Safety & Liability Strategic Analysis'.
    Use:
    - safety_research (comprehensive safety research data from researcher)
    - safety_strategy (strategic safety analysis from strategist)
    Structure:
    # Safety & Liability Strategic Analysis
    ## Executive Summary
    <2-3 sentence summary of overall safety profile and key concerns>
    ## Toxicity Assessment
    * On-target and off-target toxicity concerns
    * Structural alerts and liability indicators
    * Genotoxicity, mutagenicity, carcinogenicity assessment
    * Citations to sources
    ## DMPK Profile
    * Absorption, distribution, metabolism, excretion characteristics
    * Drug-drug interaction risks
    * CYP and transporter interactions
    * Special population considerations
    ## Clinical Safety Signals
    * Preclinical safety signals
    * Clinical adverse events from similar compounds
    * Safety margins and therapeutic index
    * Contraindications and special population risks
    ## Liability Profile
    * High/medium/low risk liabilities
    * Regulatory concerns
    * Mitigation strategies
    ## Strategic Recommendations
    * Prioritized safety recommendations with action items
    ## Regulatory Readiness
    * Safety data gaps
    * Regulatory approval risks
    * Required studies or monitoring
    Every factual claim must include inline citations (URL or source reference). No code fences.
    """,
    output_schema=SafetyBrief,
    output_key="safety_strategic_brief",
    after_agent_callback=[
        write_state_text("safety_strategic_brief", "safety_strategic_brief.md"),
        trace_persistence_callback,
    ],
)

safety_reviewer = LlmAgent(
    name="safety_reviewer",
    model=config.worker_model,
    description="Evaluates safety & liability evidence based on comprehensive research and strategic analysis.",
    instruction="""
    You are the Safety/Liability reviewer.
    - Use comprehensive safety research from `safety_research` and strategic analysis from `safety_strategy` in session state.
    - Also use deterministic tox/DMPK outputs from `safety_inputs` if available.
    - Use google_search for additional benchmarks or literature if needed.
    - Synthesize all information to evaluate on/off-target tox, clinical safety signals, and liabilities.
    - Return ONLY JSON AxisRating with:
      section_id: "safety_liability"
      display_name: "Safety & Liability"
      grade A-F/U (+ optional grade_modifier like "data_desert", "conflicting", "liability")
      badges {coverage, confidence, evidence_direction}
      summary: 2-4 sentence narrative synthesizing key safety findings
      decision_hook: One-sentence decision guidance on safety risks
      pros: 3-5 positive safety aspects with evidence URLs/IDs
      caveats: 3-5 safety concerns/risks with evidence URLs/IDs
      evidence: List of key source URLs/IDs
    - Base your assessment on the comprehensive research and strategic analysis provided.
    - Highlight critical safety concerns that could impact development or regulatory approval.
    """,
    tools=[google_search],
    output_schema=AxisRating,
    output_key="axis_rating_safety_liability",
    after_agent_callback=[
        write_state_json(
            "axis_rating_safety_liability",
            "axis_rating_safety_liability.json",
        ),
        trace_persistence_callback,
    ],
)

safety_pipeline = SequentialAgent(
    name="safety_pipeline",
    description="Produces safety & liability rating.",
    sub_agents=[safety_reviewer],
)

app = build_app(safety_pipeline, name="safety_agent")
