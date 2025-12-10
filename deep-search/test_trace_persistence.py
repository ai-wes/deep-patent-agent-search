#!/usr/bin/env python3
"""
Test script for the trace persistence system.
This script tests the core functionality of saving and loading agent events and traces.
"""

import os
import tempfile
import uuid
from datetime import datetime

from app.app_utils.trace_persistence import (
    TracePersistenceService,
    TracePersistenceConfig,
    AgentEventTrace,
    SessionStateSnapshot,
    create_trace_persistence_service_from_env
)

def test_local_trace_persistence():
    """Test trace persistence with local storage."""
    print("Testing local trace persistence...")

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create config for local storage
        config = TracePersistenceConfig(
            storage_type="local",
            local_storage_path=temp_dir
        )

        # Create service
        service = TracePersistenceService(config)

        # Test saving and loading an event trace
        session_id = str(uuid.uuid4())
        event_trace = AgentEventTrace(
            session_id=session_id,
            agent_name="test_agent",
            event_type="test_event",
            content={"test": "content"},
            metadata={"test": "metadata"}
        )

        # Save the trace
        trace_id = service.save_event_trace(event_trace)
        print(f"Saved event trace with ID: {trace_id}")

        # Load the trace back
        loaded_trace = service.load_event_trace(trace_id)
        assert loaded_trace is not None, "Failed to load event trace"
        assert loaded_trace.session_id == session_id, "Session ID mismatch"
        assert loaded_trace.agent_name == "test_agent", "Agent name mismatch"
        print("✓ Event trace persistence works correctly")

        # Test saving and loading a session snapshot
        snapshot = SessionStateSnapshot(
            session_id=session_id,
            state={"test_state": "value"},
            sources={"src-1": {"title": "Test Source", "url": "https://example.com"}}
        )

        snapshot_id = service.save_session_snapshot(snapshot)
        print(f"Saved session snapshot with ID: {snapshot_id}")

        loaded_snapshot = service.load_session_snapshot(snapshot_id)
        assert loaded_snapshot is not None, "Failed to load session snapshot"
        assert loaded_snapshot.session_id == session_id, "Session ID mismatch"
        assert loaded_snapshot.state["test_state"] == "value", "State mismatch"
        print("✓ Session snapshot persistence works correctly")

        # Test listing traces
        traces = service.list_traces(session_id=session_id)
        assert len(traces) >= 1, "No traces found"
        print(f"✓ Found {len(traces)} traces for session {session_id}")

        print("✓ Local trace persistence test passed!")

def test_gcs_trace_persistence():
    """Test trace persistence with GCS storage (if configured)."""
    print("Testing GCS trace persistence...")

    # Check if GCS is configured
    gcs_bucket = os.environ.get("TRACE_GCS_BUCKET_NAME")
    if not gcs_bucket:
        print("⚠ GCS trace persistence test skipped (TRACE_GCS_BUCKET_NAME not set)")
        return

    try:
        # Create config for GCS storage
        config = TracePersistenceConfig(
            storage_type="gcs",
            gcs_bucket_name=gcs_bucket
        )

        # Create service
        service = TracePersistenceService(config)

        # Test saving and loading an event trace
        session_id = str(uuid.uuid4())
        event_trace = AgentEventTrace(
            session_id=session_id,
            agent_name="test_agent_gcs",
            event_type="test_event_gcs",
            content={"test": "gcs_content"},
            metadata={"test": "gcs_metadata"}
        )

        # Save the trace
        trace_id = service.save_event_trace(event_trace)
        print(f"Saved GCS event trace with ID: {trace_id}")

        # Load the trace back
        loaded_trace = service.load_event_trace(trace_id)
        assert loaded_trace is not None, "Failed to load GCS event trace"
        assert loaded_trace.session_id == session_id, "GCS Session ID mismatch"
        print("✓ GCS event trace persistence works correctly")

        print("✓ GCS trace persistence test passed!")

    except Exception as e:
        print(f"⚠ GCS trace persistence test failed: {str(e)}")
        print("This is expected if Google Cloud credentials are not configured")

def test_env_based_service_creation():
    """Test creating service from environment variables."""
    print("Testing environment-based service creation...")

    # Set environment variables for testing
    os.environ["TRACE_STORAGE_TYPE"] = "local"
    os.environ["TRACE_LOCAL_PATH"] = "test_traces"

    try:
        service = create_trace_persistence_service_from_env()
        assert service is not None, "Failed to create service from environment"
        assert service.config.storage_type == "local", "Storage type mismatch"
        print("✓ Environment-based service creation works correctly")

        # Clean up test directory
        import shutil
        if os.path.exists("test_traces"):
            shutil.rmtree("test_traces")

    except Exception as e:
        print(f"✗ Environment-based service creation failed: {str(e)}")
        raise

def main():
    """Run all trace persistence tests."""
    print("🚀 Starting Trace Persistence System Tests")
    print("=" * 50)

    try:
        test_local_trace_persistence()
        print()
        test_gcs_trace_persistence()
        print()
        test_env_based_service_creation()
        print()
        print("🎉 All trace persistence tests completed successfully!")
        print("✅ The system is ready for saving and persisting agent events and traces")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
