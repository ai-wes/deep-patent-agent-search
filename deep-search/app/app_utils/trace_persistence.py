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

import json
import logging
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from google.adk.events import Event
from google.cloud import storage
from pydantic import BaseModel, Field

from .typing import Feedback

# Configure logging
logger = logging.getLogger(__name__)

class AgentEventTrace(BaseModel):
    """Represents a complete trace of an agent event for persistence."""

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_name: str
    event_type: str
    content: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    grounding_metadata: Optional[Dict[str, Any]] = None
    actions: Optional[Dict[str, Any]] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SessionStateSnapshot(BaseModel):
    """Represents a snapshot of the complete session state."""

    session_id: str
    snapshot_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    state: Dict[str, Any]
    sources: Dict[str, Any] = Field(default_factory=dict)
    url_to_short_id: Dict[str, str] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class TracePersistenceConfig(BaseModel):
    """Configuration for trace persistence."""

    storage_type: str = "local"  # "local" or "gcs"
    local_storage_path: str = "agent_traces"
    gcs_bucket_name: Optional[str] = None
    gcs_prefix: str = "traces"
    max_local_traces: int = 1000
    enable_compression: bool = False

@dataclass
class TracePersistenceService:
    """Service for persisting agent events and traces."""

    config: TracePersistenceConfig

    def __post_init__(self):
        """Initialize the persistence service."""
        if self.config.storage_type == "gcs" and not self.config.gcs_bucket_name:
            raise ValueError("GCS bucket name is required for GCS storage")

        if self.config.storage_type == "local":
            self._ensure_local_storage_dir()

    def _ensure_local_storage_dir(self) -> None:
        """Ensure the local storage directory exists."""
        storage_path = Path(self.config.local_storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using local trace storage at: {storage_path.absolute()}")

    def _get_trace_file_path(self, trace_id: str, file_type: str = "json") -> Path:
        """Get the file path for a trace."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{trace_id}.{file_type}"
        return Path(self.config.local_storage_path) / filename

    def _get_gcs_blob_path(self, trace_id: str, file_type: str = "json") -> str:
        """Get the GCS blob path for a trace."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{self.config.gcs_prefix}/{timestamp}_{trace_id}.{file_type}"

    def save_event_trace(self, event_trace: AgentEventTrace) -> str:
        """Save an agent event trace to persistent storage.

        Args:
            event_trace: The event trace to save

        Returns:
            The trace ID
        """
        try:
            trace_data = event_trace.model_dump()

            if self.config.storage_type == "local":
                file_path = self._get_trace_file_path(event_trace.trace_id)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(trace_data, f, indent=2, ensure_ascii=False)

                logger.info(f"Saved event trace to: {file_path}")
                return event_trace.trace_id

            elif self.config.storage_type == "gcs":
                client = storage.Client()
                bucket = client.bucket(self.config.gcs_bucket_name)
                blob_path = self._get_gcs_blob_path(event_trace.trace_id)
                blob = bucket.blob(blob_path)

                blob.upload_from_string(
                    json.dumps(trace_data, ensure_ascii=False),
                    content_type="application/json"
                )

                logger.info(f"Saved event trace to GCS: gs://{self.config.gcs_bucket_name}/{blob_path}")
                return event_trace.trace_id

        except Exception as e:
            logger.error(f"Failed to save event trace: {str(e)}", exc_info=True)
            raise

    def save_session_snapshot(self, session_snapshot: SessionStateSnapshot) -> str:
        """Save a complete session state snapshot.

        Args:
            session_snapshot: The session snapshot to save

        Returns:
            The snapshot ID
        """
        try:
            snapshot_data = session_snapshot.model_dump()

            if self.config.storage_type == "local":
                file_path = self._get_trace_file_path(session_snapshot.snapshot_id, "snapshot")
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(snapshot_data, f, indent=2, ensure_ascii=False)

                logger.info(f"Saved session snapshot to: {file_path}")
                return session_snapshot.snapshot_id

            elif self.config.storage_type == "gcs":
                client = storage.Client()
                bucket = client.bucket(self.config.gcs_bucket_name)
                blob_path = self._get_gcs_blob_path(session_snapshot.snapshot_id, "snapshot")
                blob = bucket.blob(blob_path)

                blob.upload_from_string(
                    json.dumps(snapshot_data, ensure_ascii=False),
                    content_type="application/json"
                )

                logger.info(f"Saved session snapshot to GCS: gs://{self.config.gcs_bucket_name}/{blob_path}")
                return session_snapshot.snapshot_id

        except Exception as e:
            logger.error(f"Failed to save session snapshot: {str(e)}", exc_info=True)
            raise

    def load_event_trace(self, trace_id: str) -> Optional[AgentEventTrace]:
        """Load an event trace from storage.

        Args:
            trace_id: The trace ID to load

        Returns:
            The loaded event trace, or None if not found
        """
        try:
            if self.config.storage_type == "local":
                # Search for the trace file
                storage_path = Path(self.config.local_storage_path)
                for trace_file in storage_path.glob("*.json"):
                    if trace_id in trace_file.name:
                        with open(trace_file, "r", encoding="utf-8") as f:
                            trace_data = json.load(f)
                        return AgentEventTrace(**trace_data)

            elif self.config.storage_type == "gcs":
                client = storage.Client()
                bucket = client.bucket(self.config.gcs_bucket_name)

                # List blobs with the trace ID
                blobs = bucket.list_blobs(prefix=self.config.gcs_prefix)
                for blob in blobs:
                    if trace_id in blob.name and blob.name.endswith(".json"):
                        trace_data = json.loads(blob.download_as_text())
                        return AgentEventTrace(**trace_data)

        except Exception as e:
            logger.error(f"Failed to load event trace {trace_id}: {str(e)}", exc_info=True)

        return None

    def load_session_snapshot(self, snapshot_id: str) -> Optional[SessionStateSnapshot]:
        """Load a session snapshot from storage.

        Args:
            snapshot_id: The snapshot ID to load

        Returns:
            The loaded session snapshot, or None if not found
        """
        try:
            if self.config.storage_type == "local":
                # Search for the snapshot file
                storage_path = Path(self.config.local_storage_path)
                for snapshot_file in storage_path.glob("*.snapshot"):
                    if snapshot_id in snapshot_file.name:
                        with open(snapshot_file, "r", encoding="utf-8") as f:
                            snapshot_data = json.load(f)
                        return SessionStateSnapshot(**snapshot_data)

            elif self.config.storage_type == "gcs":
                client = storage.Client()
                bucket = client.bucket(self.config.gcs_bucket_name)

                # List blobs with the snapshot ID
                blobs = bucket.list_blobs(prefix=self.config.gcs_prefix)
                for blob in blobs:
                    if snapshot_id in blob.name and blob.name.endswith(".snapshot"):
                        snapshot_data = json.loads(blob.download_as_text())
                        return SessionStateSnapshot(**snapshot_data)

        except Exception as e:
            logger.error(f"Failed to load session snapshot {snapshot_id}: {str(e)}", exc_info=True)

        return None

    def list_traces(self, session_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List available traces, optionally filtered by session ID.

        Args:
            session_id: Optional session ID to filter by
            limit: Maximum number of traces to return

        Returns:
            List of trace metadata
        """
        traces = []

        try:
            if self.config.storage_type == "local":
                storage_path = Path(self.config.local_storage_path)
                trace_files = sorted(storage_path.glob("*.json"), reverse=True)

                for trace_file in trace_files[:limit]:
                    try:
                        with open(trace_file, "r", encoding="utf-8") as f:
                            trace_data = json.load(f)

                        if session_id is None or trace_data.get("session_id") == session_id:
                            traces.append({
                                "trace_id": trace_data.get("trace_id"),
                                "session_id": trace_data.get("session_id"),
                                "timestamp": trace_data.get("timestamp"),
                                "agent_name": trace_data.get("agent_name"),
                                "event_type": trace_data.get("event_type"),
                                "file_path": str(trace_file)
                            })
                    except Exception as e:
                        logger.warning(f"Failed to read trace file {trace_file}: {str(e)}")

            elif self.config.storage_type == "gcs":
                client = storage.Client()
                bucket = client.bucket(self.config.gcs_bucket_name)
                blobs = bucket.list_blobs(prefix=self.config.gcs_prefix)

                for blob in blobs:
                    if len(traces) >= limit:
                        break

                    if blob.name.endswith(".json"):
                        try:
                            trace_data = json.loads(blob.download_as_text())

                            if session_id is None or trace_data.get("session_id") == session_id:
                                traces.append({
                                    "trace_id": trace_data.get("trace_id"),
                                    "session_id": trace_data.get("session_id"),
                                    "timestamp": trace_data.get("timestamp"),
                                    "agent_name": trace_data.get("agent_name"),
                                    "event_type": trace_data.get("event_type"),
                                    "gcs_path": f"gs://{self.config.gcs_bucket_name}/{blob.name}"
                                })
                        except Exception as e:
                            logger.warning(f"Failed to read GCS blob {blob.name}: {str(e)}")

        except Exception as e:
            logger.error(f"Failed to list traces: {str(e)}", exc_info=True)

        return traces

    def cleanup_old_traces(self, max_age_days: int = 30) -> int:
        """Clean up old trace files.

        Args:
            max_age_days: Maximum age in days to keep traces

        Returns:
            Number of traces deleted
        """
        if self.config.storage_type != "local":
            logger.warning("Cleanup only supported for local storage")
            return 0

        deleted_count = 0
        try:
            storage_path = Path(self.config.local_storage_path)
            now = datetime.utcnow()
            cutoff = now.timestamp() - (max_age_days * 24 * 60 * 60)

            for trace_file in storage_path.glob("*.json"):
                try:
                    file_mtime = trace_file.stat().st_mtime
                    if file_mtime < cutoff:
                        trace_file.unlink()
                        deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete old trace file {trace_file}: {str(e)}")

            logger.info(f"Cleaned up {deleted_count} old trace files")

        except Exception as e:
            logger.error(f"Failed to cleanup old traces: {str(e)}", exc_info=True)

        return deleted_count

def create_event_trace_from_adk_event(
    event: Event,
    session_id: str,
    event_type: str = "agent_event",
    additional_metadata: Optional[Dict[str, Any]] = None
) -> AgentEventTrace:
    """Create an AgentEventTrace from an ADK Event.

    Args:
        event: The ADK Event object
        session_id: The session ID
        event_type: Type of event
        additional_metadata: Additional metadata to include

    Returns:
        AgentEventTrace object
    """
    metadata = additional_metadata or {}

    # Extract grounding metadata if available
    grounding_metadata = None
    if hasattr(event, 'grounding_metadata') and event.grounding_metadata:
        grounding_metadata = {
            "grounding_chunks": [
                {
                    "web": {
                        "uri": chunk.web.uri,
                        "title": chunk.web.title,
                        "domain": chunk.web.domain
                    } if hasattr(chunk, 'web') and chunk.web else None
                }
                for chunk in event.grounding_metadata.grounding_chunks
            ] if hasattr(event.grounding_metadata, 'grounding_chunks') else None,
            "grounding_supports": [
                {
                    "confidence_scores": support.confidence_scores,
                    "grounding_chunk_indices": support.grounding_chunk_indices,
                    "segment": {
                        "text": support.segment.text
                    } if hasattr(support.segment, 'text') else None
                }
                for support in event.grounding_metadata.grounding_supports
            ] if hasattr(event.grounding_metadata, 'grounding_supports') else None
        }

    # Extract actions if available
    actions = None
    if hasattr(event, 'actions') and event.actions:
        actions = {
            "escalate": event.actions.escalate,
            "transfer_to_parent": event.actions.transfer_to_parent,
            "transfer_to_peers": event.actions.transfer_to_peers
        }

    return AgentEventTrace(
        session_id=session_id,
        agent_name=event.author,
        event_type=event_type,
        content={
            "text": event.content if hasattr(event, 'content') else "",
            "raw_content": str(event) if hasattr(event, '__str__') else ""
        },
        metadata=metadata,
        grounding_metadata=grounding_metadata,
        actions=actions
    )

def create_trace_persistence_service_from_env() -> TracePersistenceService:
    """Create a trace persistence service from environment variables.

    Returns:
        Configured TracePersistenceService
    """
    storage_type = os.environ.get("TRACE_STORAGE_TYPE", "local")
    gcs_bucket = os.environ.get("TRACE_GCS_BUCKET_NAME")
    local_path = os.environ.get("TRACE_LOCAL_PATH", "agent_traces")

    config = TracePersistenceConfig(
        storage_type=storage_type,
        local_storage_path=local_path,
        gcs_bucket_name=gcs_bucket
    )

    return TracePersistenceService(config)
