"""
Minimal root_agent shim for ADK loaders.

ADK's `adk web` / `adk api_server` expect `domain_agents.agent.root_agent`
to exist. We already expose individual App objects per domain (commercial,
risk, etc.), but for compatibility we expose a single root_agent here.

Default: use the commercial pipeline's root_agent. Override with
ADK_AGENT_NAME to pick a different domain when adk loads this module
directly (mostly for interactive/debug use; the scripted runner still
uses appName selection).
"""
from __future__ import annotations

import os

from app.domain_agents.commercial_agent import commercial_pipeline
from app.domain_agents.risk_agent import risk_pipeline
from app.domain_agents.regulatory_cmc_agent import regulatory_pipeline
from app.domain_agents.unmet_need_agent import unmet_need_pipeline
from app.domain_agents.safety_agent import safety_pipeline
from app.domain_agents.mechanistic_agent import mechanistic_pipeline

_default = commercial_pipeline
_map = {
    "commercial": commercial_pipeline,
    "risk": risk_pipeline,
    "regulatory_cmc": regulatory_pipeline,
    "unmet_need": unmet_need_pipeline,
    "safety": safety_pipeline,
    "mechanistic": mechanistic_pipeline,
}

root_agent = _map.get(os.environ.get("ADK_AGENT_NAME", "").strip().lower(), _default)
