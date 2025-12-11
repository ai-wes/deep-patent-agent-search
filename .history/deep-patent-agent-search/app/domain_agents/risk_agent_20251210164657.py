"""Risk & IP/FTO assessment pipeline (ADK version)."""

from __future__ import annotations

from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search

from ..config import config
from .base import build_app, write_state_json, write_state_text
from app.agent import trace_persistence_callback
from .renderers import risk_matrix_callback


# ---------- Structured outputs ----------
class DomainRiskData(BaseModel):
    """Risk data for a specific GB-TAR domain."""
    domain: str = Field(..., description="GB-TAR domain name")
    identified_risks: list[str] = Field(default_factory=list, description="Identified risks in this domain")
    risk_factors: list[str] = Field(default_factory=list, description="Key risk factors")
    historical_precedents: list[str] = Field(default_factory=list, description="Historical precedents or analogs")
    mitigation_approaches: list[str] = Field(default_factory=list, description="Potential mitigation approaches")
    sources: list[str] = Field(default_factory=list, description="Source URLs/IDs")


class IPFTOData(BaseModel):
    """IP/FTO (Freedom to Operate) risk data."""
    patent_landscape: list[str] = Field(default_factory=list, description="Key patents in the landscape")
    fto_concerns: list[str] = Field(default_factory=list, description="Freedom to operate concerns")
    patent_expirations: list[str] = Field(default_factory=list, description="Relevant patent expiration dates")
    competitive_ip: list[str] = Field(default_factory=list, description="Competitive IP positions")
    licensing_opportunities: list[str] = Field(default_factory=list, description="Potential licensing opportunities")
    sources: list[str] = Field(default_factory=list, description="Source URLs/IDs")


class RegulatoryRiskData(BaseModel):
    """Regulatory and CMC risk data."""
    regulatory_precedents: list[str] = Field(default_factory=list, description="Regulatory precedents for similar assets")
    regulatory_concerns: list[str] = Field(default_factory=list, description="Regulatory approval concerns")
    cmc_risks: list[str] = Field(default_factory=list, description="CMC/Manufacturing risks")
    supply_chain_risks: list[str] = Field(default_factory=list, description="Supply chain risks")
    regulatory_pathway: str | None = Field(default=None, description="Expected regulatory pathway")
    sources: list[str] = Field(default_factory=list, description="Source URLs/IDs")


class CommercialRiskData(BaseModel):
    """Commercial and market risk data."""
    market_risks: list[str] = Field(default_factory=list, description="Market and commercial risks")
    competitive_risks: list[str] = Field(default_factory=list, description="Competitive risks")
    payer_risks: list[str] = Field(default_factory=list, description="Payer and reimbursement risks")
    pricing_risks: list[str] = Field(default_factory=list, description="Pricing and market access risks")
    sources: list[str] = Field(default_factory=list, description="Source URLs/IDs")


class RiskResearchOutput(BaseModel):
    """Comprehensive risk research data across all GB-TAR domains."""
    domain_risks: list[DomainRiskData] = Field(default_factory=list, description="Risk data by GB-TAR domain")
    ip_fto: IPFTOData
    regulatory_risks: RegulatoryRiskData
    commercial_risks: CommercialRiskData
    emerging_risks: list[str] = Field(default_factory=list, description="Emerging or novel risks")
    risk_analogs: list[str] = Field(default_factory=list, description="Similar assets with comparable risk profiles")
    additional_insights: list[str] = Field(default_factory=list, description="Additional relevant risk insights")
    research_metadata: dict[str, str] = Field(default_factory=dict, description="Research metadata (date, queries used, etc.)")


class RiskInsight(BaseModel):
    """Individual risk strategic insight."""
    category: str = Field(..., description="Category (e.g., domain_risk, ip_fto, regulatory, commercial, execution)")
    insight: str = Field(..., description="Risk insight description")
    rationale: str = Field(..., description="Rationale and evidence")
    implications: str = Field(..., description="Development and business implications")
    confidence: str = Field(..., description="Confidence level (high|medium|low)")
    sources: list[str] = Field(default_factory=list, description="Supporting source URLs/IDs")


class RiskRecommendation(BaseModel):
    """Risk-related recommendation."""
    recommendation: str = Field(..., description="Specific risk recommendation")
    priority: str = Field(..., description="Priority level (critical|high|medium|low)")
    rationale: str = Field(..., description="Why this recommendation matters")
    action_items: list[str] = Field(default_factory=list, description="Specific action items")


class RiskStrategyOutput(BaseModel):
    """Deep strategic analysis of risks across all domains."""
    executive_summary: str = Field(..., description="2-3 sentence executive summary")
    key_insights: list[RiskInsight] = Field(default_factory=list, description="Risk strategic insights")
    recommendations: list[RiskRecommendation] = Field(default_factory=list, description="Risk recommendations")
    critical_risks: list[str] = Field(default_factory=list, description="Critical risks requiring immediate attention")
    overall_risk_score: str = Field(..., description="Overall risk assessment (low|moderate|high|critical)")
    portfolio_impact: str = Field(..., description="Impact on portfolio strategy")
    strategic_notes: str = Field(..., description="Additional strategic risk considerations")


class RiskBrief(BaseModel):
    """Markdown-formatted risk strategic brief."""
    markdown: str = Field(..., description="Markdown-formatted risk analysis brief")


class RiskConfigOutput(BaseModel):
    domain_weights: dict[str, float]
    rag_thresholds: dict[str, float]
    escalation_rules: list[dict[str, str | float]]


class RiskTriageItemOutput(BaseModel):
    id: int
    domain: str = Field(..., description="GB-TAR domain")
    severity_score: float = Field(..., ge=0.0, le=1.0)
    rag_label: str = Field(..., pattern="^(GREEN|YELLOW|RED)$")
    escalation_level: str
    escalation_cadence: str
    notes: str


class RiskTriageOutput(BaseModel):
    triage_items: list[RiskTriageItemOutput]


class IPWatchItemOutput(BaseModel):
    type: str
    value: str
    cadence: str
    rationale: str


class IPWatchOutput(BaseModel):
    watch_items: list[IPWatchItemOutput]


class RiskGapItemOutput(BaseModel):
    category: str
    risk: str
    trigger: str
    impact: str
    mitigation: str
    gate: str
    status: str = "Open"


class RiskGapOutput(BaseModel):
    gap_items: list[RiskGapItemOutput]


class RegulatorySnapshot(BaseModel):
    markdown: str


# ---------- Agents ----------
risk_config_agent = LlmAgent(
    name="risk_config_agent",
    model=config.worker_model,
    description="Calibrates GB-TAR domain weights and RAG thresholds.",
    instruction="""
    You are the Risk Configurator.
    - Analyze provided risk inventory and GB-TAR defaults in session state (risk_inventory, default_weights, default_rag).
    - Return ONLY JSON matching RiskConfigOutput.
    - Domain weights must include: Science, Clinical, Safety, CMC/Manufacturing, Regulatory, IP/FTO, Commercial, Team/Execution and sum to 1.0.
    - rag_thresholds must be ascending: green <= yellow <= red <= 1.0.
    - escalation_rules list objects with label, severity_min, cadence.
    """,
    tools=[google_search],
    output_schema=RiskConfigOutput,
    output_key="risk_config",
    after_agent_callback=[
        write_state_json("risk_config", "risk_config_output.json"),
        trace_persistence_callback,
    ],
)

risk_triage_agent = LlmAgent(
    name="risk_triage_agent",
    model=config.worker_model,
    description="Scores individual risks and assigns RAG/escalation.",
    instruction="""
    You are the Risk Triage Specialist.
    - Input risks are in session state `risk_items` (list of dicts) plus `risk_config`.
    - For EACH item, assign domain, severity_score (0-1), rag_label (GREEN/YELLOW/RED aligned to thresholds), escalation_level, escalation_cadence, notes with citations.
    - Return ONLY JSON array matching RiskTriageItemOutput (triage_items list).
    """,
    tools=[google_search],
    output_schema=RiskTriageOutput,
    output_key="risk_triage",
    after_agent_callback=[
        write_state_json("risk_triage", "risk_triage_output.json"),
        risk_matrix_callback,
        trace_persistence_callback,
    ],
)

ip_watch_agent = LlmAgent(
    name="ip_watch_agent",
    model=config.worker_model,
    description="Recommends top IP monitoring actions based on signals and FTO context.",
    instruction="""
    You are the IP Monitoring Advisor.
    - Use patent signals, fto_brief, and triage context in session state.
    - Return ONLY JSON with up to 5 watch_items (type, value, cadence, rationale with citation).
    """,
    tools=[google_search],
    output_schema=IPWatchOutput,
    output_key="ip_watch",
    after_agent_callback=[
        write_state_json("ip_watch", "ip_watch_output.json"),
        trace_persistence_callback,
    ],
)

risk_gap_agent = LlmAgent(
    name="risk_gap_agent",
    model=config.worker_model,
    description="Surfaces emerging or missing risks.",
    instruction="""
    Identify newly emerging or missing risks given triage results and domain coverage.
    Return ONLY JSON gap_items with category, risk, trigger, impact, mitigation, gate, status.
    """,
    tools=[google_search],
    output_schema=RiskGapOutput,
    output_key="risk_gaps",
    after_agent_callback=[
        write_state_json("risk_gaps", "risk_gaps.json"),
        trace_persistence_callback,
    ],
)

reg_snapshot_agent = LlmAgent(
    name="reg_snapshot_agent",
    model=config.critic_model,
    description="Synthesizes a markdown snapshot of regulatory/CMC posture.",
    instruction="""
    Write markdown starting with '# Regulatory & CMC Snapshot'.
    Sections to include: Regulatory Outlook, CMC & Supply, Escalation Guidance.
    Base analysis on risk_config, risk_triage, ip_watch, and provided target metadata in session state.
    Inline citations required for every factual claim. No code fences.
    """,
    output_schema=RegulatorySnapshot,
    output_key="reg_snapshot",
    after_agent_callback=[
        write_state_text("reg_snapshot", "regulatory_snapshot.md"),
        trace_persistence_callback,
    ],
)

# ---------- Pipeline & App ----------
risk_pipeline = SequentialAgent(
    name="risk_pipeline",
    description="Calibrates risk model, triages risks, recommends IP monitoring, identifies gaps, and drafts regulatory snapshot.",
    sub_agents=[
        risk_config_agent,
        risk_triage_agent,
        ip_watch_agent,
        risk_gap_agent,
        reg_snapshot_agent,
    ],
)

app = build_app(risk_pipeline, name="risk_agent")
