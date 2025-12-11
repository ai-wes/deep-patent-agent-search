# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: disable-error-code="attr-defined,arg-type"
import logging
import os
from typing import Any, Optional

import vertexai
from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService
from google.cloud import logging as google_cloud_logging
from vertexai.agent_engines.templates.adk import AdkApp

from app.agent import app as adk_app
# Domain-specific ADK apps (converted from legacy crews)
from app.domain_agents import (
    commercial_agent,
    risk_agent,
    regulatory_cmc_agent,
    unmet_need_agent,
    safety_agent,
    mechanistic_agent,
)
from app.app_utils.telemetry import setup_telemetry
from app.app_utils.trace_persistence import (
    TracePersistenceService,
    create_trace_persistence_service_from_env,
    create_event_trace_from_adk_event,
    SessionStateSnapshot,
    AgentEventTrace
)
from app.app_utils.typing import Feedback
from app.app_utils.state_loader import load_state_from_env


class AgentEngineApp(AdkApp):
    def set_up(self) -> None:
        """Initialize the agent engine app with logging and telemetry."""
        vertexai.init()
        setup_telemetry()
        super().set_up()
        logging.basicConfig(level=logging.INFO)
        logging_client = google_cloud_logging.Client()
        self.logger = logging_client.logger(__name__)

        # Initialize trace persistence service
        self.trace_service = create_trace_persistence_service_from_env()

        if gemini_location:
            os.environ["GOOGLE_CLOUD_LOCATION"] = gemini_location

    def register_feedback(self, feedback: dict[str, Any]) -> None:
        """Collect and log feedback."""
        feedback_obj = Feedback.model_validate(feedback)
        self.logger.log_struct(feedback_obj.model_dump(), severity="INFO")

    def save_agent_event_trace(self, event: Any, session_id: str, event_type: str = "agent_event", additional_metadata: Optional[dict[str, Any]] = None) -> str:
        """Save an agent event trace for persistence and reproducibility.

        Args:
            event: The ADK Event object
            session_id: The session ID
            event_type: Type of event
            additional_metadata: Additional metadata to include

        Returns:
            The trace ID
        """
        try:
            event_trace = create_event_trace_from_adk_event(
                event, session_id, event_type, additional_metadata
            )
            trace_id = self.trace_service.save_event_trace(event_trace)
            logger.info(f"Saved agent event trace {trace_id} for session {session_id}")
            return trace_id
        except Exception as e:
            logger.error(f"Failed to save agent event trace: {str(e)}", exc_info=True)
            raise

    def save_session_snapshot(self, session_id: str, state: dict[str, Any], sources: Optional[dict[str, Any]] = None, url_to_short_id: Optional[dict[str, str]] = None) -> str:
        """Save a complete session state snapshot for reproducibility.

        Args:
            session_id: The session ID
            state: The current session state
            sources: Optional sources dictionary
            url_to_short_id: Optional URL to short ID mapping

        Returns:
            The snapshot ID
        """
        try:
            snapshot = SessionStateSnapshot(
                session_id=session_id,
                state=state,
                sources=sources or {},
                url_to_short_id=url_to_short_id or {}
            )
            snapshot_id = self.trace_service.save_session_snapshot(snapshot)
            logger.info(f"Saved session snapshot {snapshot_id} for session {session_id}")
            return snapshot_id
        except Exception as e:
            logger.error(f"Failed to save session snapshot: {str(e)}", exc_info=True)
            raise

    def get_trace_service(self) -> TracePersistenceService:
        """Get the trace persistence service.

        Returns:
            The trace persistence service
        """
        return self.trace_service

    def register_operations(self) -> dict[str, list[str]]:
        """Registers the operations of the Agent."""
        operations = super().register_operations()
        operations[""] = operations.get("", []) + ["register_feedback"]
        return operations


gemini_location = os.environ.get("GOOGLE_CLOUD_LOCATION")
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")

# --------------------------------------------------------------------------- #
# App registry & selection
# --------------------------------------------------------------------------- #
APP_REGISTRY = {
    "prior_art": adk_app,  # default (legacy ADK sample)
    "commercial": commercial_agent.app,
    "risk": risk_agent.app,
    "regulatory_cmc": regulatory_cmc_agent.app,
    "unmet_need": unmet_need_agent.app,
    "safety": safety_agent.app,
    "mechanistic": mechanistic_agent.app,
}

selected_app_name = os.environ.get("ADK_AGENT_NAME", "prior_art").strip().lower()
selected_app = APP_REGISTRY.get(selected_app_name, adk_app)
if selected_app is adk_app and selected_app_name not in APP_REGISTRY:
    logging.warning(
        "ADK_AGENT_NAME=%s not recognized; falling back to prior_art", selected_app_name
    )

agent_engine = AgentEngineApp(
    app=selected_app,
    artifact_service_builder=lambda: GcsArtifactService(bucket_name=logs_bucket_name)
    if logs_bucket_name
    else InMemoryArtifactService(),
)

# Optionally preload deterministic outputs into the app's default state.
default_state = load_state_from_env()
if default_state:
    try:
        agent_engine.app.default_state = default_state
        logging.info("Loaded default session state from ADK_STATE_DIR with %d keys", len(default_state))
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not attach default_state to app: %s", exc)
