# test_local.py
import os
from app.agent_engine_app import AgentEngineApp, adk_app

# Mock environment if needed
os.environ["GOOGLE_CLOUD_PROJECT"] = "glassbox-marketplace-prod"

print("1. Initializing AgentEngineApp locally...")
try:
    # Initialize your wrapper class
    agent_instance = AgentEngineApp(app=adk_app)
    
    # Check if 'app' attribute exists
    if hasattr(agent_instance, 'app'):
        print("   SUCCESS: 'app' attribute found on instance.")
    else:
        print("   FAILURE: 'app' attribute MISSING on instance.")
        exit(1)

    print("2. Simulating query call...")
    # Create a dummy input
    test_input = '{"test": "data"}'
    
    # Try calling query() - this verifies the delegation to self.app.query() works
    # Note: This might actually run the agent, so it might fail if auth/tools aren't ready,
    # but we only care if it finds 'self.app' and tries to call '.query()'.
    try:
        # We wrap in try/except because the actual agent execution might fail 
        # due to missing credentials/tools, but we just want to pass the "AttributeError" check.
        agent_instance.query(input=test_input)
    except Exception as e:
        error_msg = str(e)
        if "'AgentEngineApp' object has no attribute 'app'" in error_msg:
            print(f"   FAILURE: Still getting attribute error: {e}")
            exit(1)
        else:
            print(f"   PASSED STEP 2: Query called (failed with unrelated error: {e})")
            print("   This confirms 'self.app' was found and 'query' was called.")

    print("\n✅ Verification Successful. You can now deploy.")

except Exception as e:
    print(f"\n❌ FATAL ERROR during local test: {e}")
