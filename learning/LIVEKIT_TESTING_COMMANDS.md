# LiveKit Context Optimization - Testing Commands

## 📋 Prerequisites

### 1. Install Dependencies

```bash
# Install all project dependencies
uv sync

# Or with pip
pip install -e .

# Verify LiveKit agents is installed
python -c "import livekit.agents; print(livekit.agents.__version__)"
```

### 2. Set Environment Variables

```bash
# Option 1: Set in terminal (Linux/Mac)
export GROQ_API_KEY="your-groq-api-key"
export SARVAM_API_KEY="your-sarvam-api-key"

# Option 2: Set in PowerShell (Windows)
$env:GROQ_API_KEY="your-groq-api-key"
$env:SARVAM_API_KEY="your-sarvam-api-key"

# Option 3: Create .env file
cat > .env << EOF
GROQ_API_KEY=your-groq-api-key
SARVAM_API_KEY=your-sarvam-api-key
CONTEXT_SUMMARIZE_THRESHOLD_ITEMS=20
CONTEXT_KEEP_LAST_N_TURNS=10
CONTEXT_SUMMARIZE_EVERY_N_TURNS=15
EOF
```

---

## 🧪 Run Tests

### Run All Context Management Tests

```bash
# Run all tests
pytest tests/test_livekit_context_management.py tests/integration/test_long_conversation.py -v

# Run with coverage
pytest tests/test_livekit_context_management.py tests/integration/test_long_conversation.py -v --cov=src/livekit_context_manager --cov=src/specialized_domain_state

# Run with detailed output
pytest tests/test_livekit_context_management.py tests/integration/test_long_conversation.py -vv -s
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/test_livekit_context_management.py -v

# Integration tests only
pytest tests/integration/test_long_conversation.py -v

# Context filtering tests
pytest tests/test_livekit_context_management.py::TestContextFiltering -v

# Summarization tests
pytest tests/test_livekit_context_management.py::TestSummarization -v

# External state tests
pytest tests/test_livekit_context_management.py::TestExternalState -v

# Long conversation simulation
pytest tests/integration/test_long_conversation.py::test_context_management_over_30_turns -v
```

### Quick Test (Fail Fast)

```bash
# Stop on first failure, short output
pytest tests/test_livekit_context_management.py -x

# Run only tests not marked as slow
pytest tests/test_livekit_context_management.py -v -m "not slow"
```

---

## 🚀 Run the Optimized Agent

### Development Mode (Recommended)

```bash
# Option 1: Development + Console (BEST - two terminals)
# Terminal 1: Start the agent
python -m src.livekit_context_optimized dev

# Terminal 2: Connect console for voice testing
python -m src.livekit_context_optimized console

# Option 2: Use the launcher script (single terminal)
python run_livekit_agent.py

# Option 3: With custom context settings
CONTEXT_SUMMARIZE_THRESHOLD_ITEMS=15 \
CONTEXT_KEEP_LAST_N_TURNS=8 \
CONTEXT_SUMMARIZE_EVERY_N_TURNS=10 \
python -m src.livekit_context_optimized dev
```

### Standard Workflow

1. **Terminal 1** - Start agent server:
   ```bash
   python -m src.livekit_context_optimized dev
   ```

2. **Terminal 2** - Connect console client:
   ```bash
   python -m src.livekit_context_optimized console
   ```

The `dev` command starts the LiveKit agent in development mode with hot-reload, while `console` starts an interactive voice client for testing.

### Production Mode

```bash
# Build for production
python -m build

# Or use the compiled version directly
python -c "from src.livekit_context_optimized import entrypoint; ..."
```

---

## 🔍 Manual Testing with LiveKit

### Option 1: Using LiveKit CLI

```bash
# Install LiveKit CLI
pip install livekit-cli

# Start a local LiveKit server (in another terminal)
livekit-server --dev

# Connect your agent to the room
python src/livekit_context_optimized.py

# In another terminal, connect a client
livekit-cli join-room --room-name test-room --url ws://localhost:7880
```

### Option 2: Using Python Test Client

Create a test client `test_client.py`:

```python
import asyncio
from livekit import rtc
from livekit.api import LiveKitAPI, RoomServiceClient

async def test_agent():
    # Connect to LiveKit server
    room = rtc.Room()
    await room.connect("ws://localhost:7880", "test-room")

    print("✅ Connected to room")

    # Send test audio (simulated)
    for i in range(25):
        print(f"Turn {i+1}: Sending test message...")
        # Simulate user input
        await asyncio.sleep(2)

    await room.disconnect()
    print("✅ Test complete")

if __name__ == "__main__":
    asyncio.run(test_agent())
```

Run it:
```bash
python test_client.py
```

---

## 📊 Monitor Context During Testing

### Enable Debug Logging

```bash
# Set environment variable
export LOGLEVEL=DEBUG

# Or in Python code
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Watch Logs in Real-Time

```bash
# Linux/Mac
python src/livekit_context_optimized.py 2>&1 | tee agent.log

# Windows PowerShell
python src/livekit_context_optimized.py 2>&1 | Tee-Object -FilePath agent.log

# Follow log file
tail -f agent.log  # Linux/Mac
Get-Content agent.log -Wait  # Windows
```

### Filter Specific Log Messages

```bash
# Show only context-related logs
python src/livekit_context_optimized.py 2>&1 | grep -i "context\|summariz"

# Show only tool calls
python src/livekit_context_optimized.py 2>&1 | grep -i "tool\|function"

# Show only errors
python src/livekit_context_optimized.py 2>&1 | grep -i "error\|failed"
```

---

## 🧪 Test Scenarios

### Test 1: Basic Context Compaction

```bash
# Set low threshold for quick testing
export CONTEXT_SUMMARIZE_THRESHOLD_ITEMS=10
export CONTEXT_SUMMARIZE_EVERY_N_TURNS=5

# Run agent
python -m src.livekit_context_optimized dev

# Expected output:
# 🔄 Turn 5: Checking context...
# 📊 Context stats: 12 items
# Context has 12 items, summarizing...
# ✅ Summarization complete: 12 -> 8 items
```

### Test 2: Long Conversation (50+ Turns)

```bash
# Set normal thresholds
export CONTEXT_SUMMARIZE_THRESHOLD_ITEMS=20
export CONTEXT_SUMMARIZE_EVERY_N_TURNS=15

# Run agent
python -m src.livekit_context_optimized dev

# Have a long conversation (50+ turns)
# Expected: Context stays around 15-20 items
```

### Test 3: External State

```bash
# Run agent
python -m src.livekit_context_optimized dev

# Use the save_preference tool
# Say: "Save my preference for language as English"
# Then: "What is my language preference?"
# Expected: Preference retrieved from external state, not chat context
```

### Test 4: Context Statistics

```bash
# Run agent
python -m src.livekit_context_optimized dev

# Say: "Get context stats"
# Expected output:
# Context Statistics:
# - Total items: 15
# - Type breakdown: {...}
# - Domain state: User Preferences: language=en-IN
# - Session duration: 123.4s
```

---

## 🐛 Debugging Commands

### Check Python Environment

```bash
# Check Python version
python --version

# Check installed packages
pip list | grep livekit
pip list | grep groq
pip list | grep sarvam

# Verify imports
python -c "from src.livekit_context_manager import ContextManager; print('✅ ContextManager imported')"
python -c "from src.specialized_domain_state import SpecializedDomainState; print('✅ SpecializedDomainState imported')"
python -c "from livekit.agents import Agent; print('✅ LiveKit Agent imported')"
```

### Test Context Manager Directly

```python
# test_context.py
import asyncio
from unittest.mock import MagicMock
from livekit.plugins import groq
from src.livekit_context_manager import ContextManager

async def test():
    # Create mock agent
    agent = MagicMock()
    agent.chat_ctx = MagicMock()
    agent.chat_ctx.items = list(range(25))  # 25 items
    agent.session = MagicMock()
    agent.session.llm = groq.LLM(model="llama-3.1-8b-instant")
    agent._chat_ctx = agent.chat_ctx

    # Test summarization
    print(f"Before: {len(agent.chat_ctx.items)} items")

    await ContextManager.summarize_if_needed(
        agent=agent,
        threshold_items=20
    )

    print(f"After: {len(agent.chat_ctx.items)} items")

asyncio.run(test())
```

Run it:
```bash
python test_context.py
```

### Test External State

```python
# test_state.py
from src.specialized_domain_state import SpecializedDomainState

# Create state
state = SpecializedDomainState()

# Test preferences
state.set_user_preference("language", "en-IN")
print(f"Language: {state.get_user_preference('language')}")

# Test task history
state.add_task_result("analysis", {"rows": 100})
print(f"Task history: {state.get_task_history()}")

# Test summary
print(f"Summary: {state.to_summary()}")
```

Run it:
```bash
python test_state.py
```

---

## 📈 Performance Testing

### Load Test with Concurrent Users

```bash
# Install locust
pip install locust

# Create load test file (locustfile.py)
# Then run:
locust -f locustfile.py --host=http://localhost:7880

# Open browser to http://localhost:8089
# Set number of users and spawn rate
```

### Measure Token Usage

```bash
# Run with token tracking
python src/livekit_context_optimized.py 2>&1 | grep -i "token\|items"

# Count context items over time
python src/livekit_context_optimized.py 2>&1 | grep "Context stats" | awk '{print $NF}'
```

### Profile Memory Usage

```bash
# Run with memory profiler
pip install memory_profiler
python -m memory_profiler src/livekit_context_optimized.py

# Or use Python's built-in profiler
python -m cProfile -o profile.stats src/livekit_context_optimized.py
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

---

## ✅ Verification Checklist

After running tests, verify:

- [ ] All 20 unit/integration tests pass
- [ ] Agent starts without errors
- [ ] Context stats logged every N turns
- [ ] Summarization triggers at threshold
- [ ] External state works (save/get preferences)
- [ ] Long conversations (50+ turns) work
- [ ] No context window errors
- [ ] Responses remain coherent after summarization
- [ ] Logs show expected behavior

---

## 🆘 Troubleshooting

### Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'src'
# Solution: Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Or run from project root with python -m
python -m src.livekit_context_optimized
```

### API Key Errors

```bash
# Error: API key not found
# Solution: Verify environment variables are set
echo $GROQ_API_KEY
echo $SARVAM_API_KEY

# Or check .env file
cat .env
```

### LiveKit Connection Errors

```bash
# Error: Cannot connect to LiveKit server
# Solution: Start LiveKit server or check URL
# Start local server:
livekit-server --dev

# Or use LiveKit Cloud
export LIVEKIT_URL=wss://your-cluster.livekit.cloud
export LIVEKIT_API_KEY=your-api-key
export LIVEKIT_API_SECRET=your-api-secret
```

---

## 📚 Quick Reference

```bash
# Run tests
pytest tests/test_livekit_context_management.py tests/integration/test_long_conversation.py -v

# Run agent (development mode - two terminals)
# Terminal 1:
python -m src.livekit_context_optimized dev

# Terminal 2:
python -m src.livekit_context_optimized console

# Monitor logs
python -m src.livekit_context_optimized dev 2>&1 | tee agent.log

# Check environment
python -c "from src.livekit_context_manager import ContextManager; print('✅ Ready')"
```

---

**Happy Testing! 🚀**
