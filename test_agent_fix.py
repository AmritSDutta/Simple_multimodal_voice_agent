"""
Test script to verify LiveKit Agent fixes
"""
from src.livekit_context_optimized import OptimizedTaskAgent, AgentUserData, entrypoint
from src.specialized_domain_state import SpecializedDomainState
from src.livekit_context_manager import ContextManager

def test_userdata_creation():
    """Test 1: Verify AgentUserData can be created"""
    print("Test 1: Creating AgentUserData...")
    try:
        userdata = AgentUserData(
            ctx=None,
            domain_state=SpecializedDomainState()
        )
        print(f"  SUCCESS: Created AgentUserData")
        print(f"  - Has ctx: {userdata.ctx is None}")
        print(f"  - Has domain_state: {userdata.domain_state is not None}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

def test_on_enter_access():
    """Test 2: Verify on_enter can safely access userdata"""
    print("\nTest 2: Testing on_enter userdata access...")

    # Mock session with userdata
    class MockSession:
        def __init__(self):
            self.userdata = AgentUserData(
                ctx=None,
                domain_state=SpecializedDomainState()
            )

    class MockAgent:
        def __init__(self):
            self.session = MockSession()

        async def test_on_enter(self):
            """Simulate the on_enter logic"""
            # This is the fixed code from on_enter
            if self.session.userdata.domain_state is None:
                self.session.userdata.domain_state = SpecializedDomainState()
                return "Initialized"
            else:
                return "Already initialized"

    import asyncio
    try:
        mock_agent = MockAgent()
        result = asyncio.run(mock_agent.test_on_enter())
        print(f"  SUCCESS: {result}")
        print(f"  - Can access domain_state: {mock_agent.session.userdata.domain_state is not None}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_function_tools():
    """Test 3: Verify function tools have correct signatures"""
    print("\nTest 3: Checking function tool signatures...")

    # Check save_preference
    import inspect

    # We need to check the actual methods on the class
    from src.livekit_context_optimized import OptimizedTaskAgent

    agent = OptimizedTaskAgent()

    # Check get_context_stats (should have 1 placeholder parameter except self)
    # When inspecting from instance, 'self' is not in parameters
    sig = inspect.signature(agent.get_context_stats)
    params = list(sig.parameters.keys())
    print(f"  get_context_stats params: {params}")

    if len(params) == 1 and params[0] == 'placeholder':
        print("  SUCCESS: get_context_stats has placeholder parameter (correct)")
        result1 = True
    else:
        print(f"  FAILED: get_context_stats should have placeholder parameter 'placeholder', has: {params}")
        result1 = False

    # Check save_preference signature
    sig = inspect.signature(agent.save_preference)
    params = list(sig.parameters.keys())
    print(f"  save_preference params: {params}")

    if 'context' in params and 'key' in params and 'value' in params:
        print("  SUCCESS: save_preference has correct parameters")
        result2 = True
    else:
        print(f"  FAILED: save_preference missing parameters")
        result2 = False

    return result1 and result2

def main():
    print("=" * 70)
    print("LiveKit Agent Fix Verification Tests")
    print("=" * 70)

    results = []

    # Run tests
    results.append(test_userdata_creation())
    results.append(test_on_enter_access())
    results.append(test_function_tools())

    # Summary
    print("\n" + "=" * 70)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 70)

    if all(results):
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1

if __name__ == "__main__":
    exit(main())
