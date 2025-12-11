from vertexai.preview import reasoning_engines
from app.agent_engine_app import AgentEngineApp, logs_bucket_name
from app.agent import app  # Import app directly from where it is defined
from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService

import vertexai
from vertexai.preview import reasoning_engines

PROJECT_ID = "glassbox-marketplace-prod"
LOCATION = "us-central1"
EXISTING_ENGINE_ID = "941810874027343872"
# ADD THIS LINE:
STAGING_BUCKET = "gs://glassbox-bio-agents" # Replace with your actual bucket!

# --- INITIALIZE VERTEX AI WITH BUCKET ---
vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=STAGING_BUCKET  # <--- THIS WAS MISSING
)




# 1. Initialize your fixed class locally
agent_engine_update = AgentEngineApp(
    app=app,
    artifact_service_builder=lambda: GcsArtifactService(bucket_name=logs_bucket_name)
    if logs_bucket_name
    else InMemoryArtifactService(),
)

# 2. Reference your EXISTING engine
remote_engine = reasoning_engines.ReasoningEngine(EXISTING_ENGINE_ID)

print(f"Updating Reasoning Engine {EXISTING_ENGINE_ID}...")
print("This will take a few minutes, but it keeps the same ID.")

# 3. Push the update
operation = remote_engine.update(
    reasoning_engine=agent_engine_update,
    requirements=[
        "google-cloud-aiplatform",
        "google-cloud-logging",
        "google-auth",
        "requests"
        # Add other reqs from your requirements.txt
    ]
)

print("Update finished. Retrying your test now...")

# 4. Immediate Test (Optional)
# You can run your python test logic here immediately after it returns.
