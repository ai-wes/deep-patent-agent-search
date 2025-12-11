from google.api_core import client_options
from google.cloud import aiplatform_v1beta1

# CONFIGURATION
PROJECT_ID = "glassbox-marketplace-prod"
LOCATION = "us-central1"
ENGINE_ID = "6865170283926388736" # The stuck engine

# 1. Define the regional API endpoint
# This is critical to avoid the "valid location ID is global" error
api_endpoint = f"{LOCATION}-aiplatform.googleapis.com"
client_opts = client_options.ClientOptions(api_endpoint=api_endpoint)

# 2. Create the client with these options
client = aiplatform_v1beta1.ReasoningEngineServiceClient(client_options=client_opts)

# 3. Construct the full resource name
name = f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{ENGINE_ID}"

print(f"Force deleting: {name}")

# 4. Execute Force Delete
try:
    request = aiplatform_v1beta1.DeleteReasoningEngineRequest(
        name=name,
        force=True  # Force delete to remove child sessions
    )
    operation = client.delete_reasoning_engine(request=request)
    print("Delete operation started...")
    operation.result() # Wait for completion
    print("SUCCESS: Engine deleted.")

except Exception as e:
    print(f"FAILED: {e}")
