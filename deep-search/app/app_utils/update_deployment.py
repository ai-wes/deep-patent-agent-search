from vertexai.preview import reasoning_engines
from src.app.vertex.vertex_patent_agent import app, AgentEngineApp, gemini_location, logs_bucket_name
from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService

# 1. Initialize the corrected local agent object
# (Ensure this new class definition includes the 'query' method you just added)
agent_engine_update = AgentEngineApp(
    app=app,
    artifact_service_builder=lambda: GcsArtifactService(bucket_name=logs_bucket_name)
    if logs_bucket_name
    else InMemoryArtifactService(),
)

# 2. Reference the existing remote engine
# Replace with your actual Resource ID from the error logs
EXISTING_ENGINE_ID = "1173746254836924416" 
remote_engine = reasoning_engines.ReasoningEngine(EXISTING_ENGINE_ID)

print(f"Updating Reasoning Engine: {EXISTING_ENGINE_ID}...")

# 3. Push the update
# This re-packages your local code and updates the remote instance
remote_engine.update(
    reasoning_engine=agent_engine_update,
    requirements=[
        "google-cloud-aiplatform",
        "google-cloud-logging",
        # Add other dependencies from your requirements.txt here
    ]
)

print("Update complete. The 'query' method should now be available.")
