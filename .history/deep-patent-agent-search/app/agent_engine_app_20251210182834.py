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
from google.adk.apps.app import App  # <--- ADD THIS IMPORT
import vertexai
from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService
from google.cloud import logging as google_cloud_logging
from vertexai.agent_engines.templates.adk import AdkApp

from app.agent import app as adk_app
from app.app_utils.telemetry import setup_telemetry
from app.app_utils.trace_persistence import (
    TracePersistenceService,
    create_trace_persistence_service_from_env,
    create_event_trace_from_adk_event,
    SessionStateSnapshot,
    AgentEventTrace
)
from app.app_utils.app_typing import Feedback


class AgentEngineApp(AdkApp):
    def __init__(self, app: App, *args, **kwargs):
        # 1. CRITICAL: Save 'app' explicitly so it persists after pickling
        self.app = app
        super().__init__(app=app, *args, **kwargs)


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


    # --- ADD THIS METHOD ---
    def query(self, input: str, **kwargs) -> dict[str, Any]:
        """
        Executes a query against the underlying ADK agent using InMemoryRunner.
        The Reasoning Engine calls this method synchronously.
        """
        self.logger.log_text(f"Received query input: {input}")
        
        # 2. Use a runner to execute the ADK App
        runner = InMemoryRunner(app=self.app)
        
        # 3. Define async execution logic
        async def _run_agent():
            # Create user message content
            user_msg = types.Content(role="user", parts=[types.Part(text=input)])
            
            # Use a session ID (new one per request, or passed in via kwargs)
            session_id = kwargs.get("session_id", "default-session")
            
            # Run the agent
            events = []
            async for event in runner.run_async(
                user_id="vertex-user",
                session_id=session_id,
                new_message=user_msg
            ):
                events.append(event)
            return events

        # 4. Run async loop synchronously
        try:
            events = asyncio.run(_run_agent())
            
            # 5. Extract the final answer from events
            final_text = "No response text generated."
            # Iterate backwards to find the last meaningful text response
            for event in reversed(events):
                # Check if event has content and parts
                if hasattr(event, 'content') and event.content and event.content.parts:
                    # Simple extraction: join all text parts
                    text_parts = [p.text for p in event.content.parts if p.text]
                    if text_parts:
                        final_text = "\n".join(text_parts)
                        break
            
            return {"response": final_text}

        except Exception as e:
            self.logger.log_text(f"Error during query execution: {e}", severity="ERROR")
            return {"error": str(e)}



    def register_operations(self) -> dict[str, list[str]]:
        """Registers the operations of the Agent."""
        operations = super().register_operations()
        # --- UPDATE THIS LINE TO INCLUDE 'query' ---
        # This tells the Reasoning Engine that 'query' is a public endpoint
        existing_ops = operations.get("", [])
        operations[""] = existing_ops + ["register_feedback", "query"] 
        return operations

gemini_location = os.environ.get("GOOGLE_CLOUD_LOCATION")
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")
agent_engine = AgentEngineApp(
    app=adk_app,
    artifact_service_builder=lambda: GcsArtifactService(bucket_name=logs_bucket_name)
    if logs_bucket_name
    else InMemoryArtifactService(),
)
