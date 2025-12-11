"""
Domain-specific ADK agent pipelines.

Each module exposes an `app` variable that can be mounted directly by the
Agent Engine. Pipelines share the deep-research flow used in `app/agent.py`
but are tuned with domain prompts and structured outputs.
"""

__all__ = [
    "base",
    "commercial_agent",
    "risk_agent",
    "regulatory_cmc_agent",
    "unmet_need_agent",
    "safety_agent",
    "mechanistic_agent",
    "axis_schema",
    "renderers",
]
