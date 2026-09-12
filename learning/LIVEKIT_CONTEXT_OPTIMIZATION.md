# LiveKit Context Optimization Implementation Guide

## Overview

This implementation provides **LiveKit-native context management** for long-running voice agents. It prevents context window overflow through three complementary strategies:

1. **Interruption-based truncation** (automatic, zero-cost)
2. **Periodic summarization** (proactive, LLM-powered)
3. **Message filtering** (noise reduction, 30-50% token savings)

## Quick Start

### 1. Run the Optimized Agent

```bash
# Set environment variables
export GROQ_API_KEY="your-groq-api-key"
export SARVAM_API_KEY="your-sarvam-api-key"

# Optional: Configure context thresholds
export CONTEXT_SUMMARIZE_THRESHOLD_ITEMS=20
export CONTEXT_KEEP_LAST_N_TURNS=10
export CONTEXT_SUMMARIZE_EVERY_N_TURNS=15

# Run the agent (IMPORTANT: use -m flag for proper imports)
# Terminal 1: Start the agent in development mode
python -m src.livekit_context_optimized dev

# Terminal 2: Connect console for voice testing
python -m src.livekit_context_optimized console

# OR use the launcher script
python run_livekit_agent.py
```

⚠️ **Important:** Don't run with `python src/livekit_context_optimized.py` directly - this causes `ModuleNotFoundError`. Use the commands above instead.

**Usage:**
- `dev` - Starts the LiveKit agent in development mode with hot-reload
- `console` - Connects an interactive voice client for testing

### 2. Connect with LiveKit Client

```bash
# In another terminal
python -m livekit.plugins.example_client
```

### 3. Test Context Management

Have a long conversation (20+ turns) and observe:
- Context stats logged every N turns
- Automatic summarization when threshold exceeded
- External state storage for preferences

## Architecture

### Core Components

```
src/
├── livekit_context_manager.py       # Context filtering & summarization utilities
├── specialized_domain_state.py       # External state management
└── livekit_context_optimized.py      # Optimized agent implementation

tests/
├── test_livekit_context_management.py     # Unit tests
└── integration/test_long_conversation.py   # Integration tests
```

### Data Flow

```
User Speech → VAD → STT → LLM → TTS → Audio Output
                      ↓
                 ChatContext (grows with each turn)
                      ↓
            ContextManager.check_every_n_turns()
                      ↓
         If threshold exceeded → Summarization
                      ↓
              Old conversation compressed
                      ↓
         Recent N turns + Summary retained
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTEXT_SUMMARIZE_THRESHOLD_ITEMS` | `20` | Trigger summarization when context exceeds this many items |
| `CONTEXT_KEEP_LAST_N_TURNS` | `10` | Number of recent turns to keep uncompressed |
| `CONTEXT_SUMMARIZE_EVERY_N_TURNS` | `15` | Check summarization every N turns |

### Tuning Guidelines

**For shorter conversations (< 50 turns):**
```bash
CONTEXT_SUMMARIZE_THRESHOLD_ITEMS=30
CONTEXT_SUMMARIZE_EVERY_N_TURNS=20
```

**For longer conversations (> 100 turns):**
```bash
CONTEXT_SUMMARIZE_THRESHOLD_ITEMS=15
CONTEXT_SUMMARIZE_EVERY_N_TURNS=10
```

**For faster models (reduce latency):**
```bash
CONTEXT_SUMMARIZE_THRESHOLD_ITEMS=25
CONTEXT_KEEP_LAST_N_TURNS=5
```

## API Reference

### ContextManager

Static utility class for context management.

#### `get_compacted_context()`

Create a filtered, truncated version of chat context.

```python
compacted_ctx = await ContextManager.get_compacted_context(
    agent=self,
    keep_last_n_turns=10,    # Keep last 10 turns
    exclude_system=True      # Exclude system messages
)
```

**Returns:** `ChatContext` with noise removed and size limited

**Use case:** Call before expensive operations (tool calls, LLM generation)

---

#### `summarize_if_needed()`

Proactively summarize context if approaching token limits.

```python
await ContextManager.summarize_if_needed(
    agent=self,
    threshold_items=20,        # Trigger at 20 items
    llm_for_summary=None,      # Use agent's LLM
    keep_last_turns=3          # Keep last 3 turns uncompressed
)
```

**Returns:** `ChatContext` (summarized or unchanged)

**Use case:** Call periodically in `on_user_turn_ended()`

---

#### `get_context_stats()`

Get statistics about current chat context.

```python
stats = ContextManager.get_context_stats(self)
print(f"Total items: {stats['total_items']}")
print(f"Has system messages: {stats['has_system_messages']}")
```

**Returns:** `dict` with context statistics

**Use case:** Debugging and monitoring

### SpecializedDomainState

Store persistent data outside chat context.

#### Methods

```python
# Preferences
state.set_user_preference("language", "en-IN")
value = state.get_user_preference("language")

# Task history
state.add_task_result("analysis", {"rows": 100})
history = state.get_task_history("analysis")

# Domain facts
state.set_domain_fact("user_level", "advanced")
fact = state.get_domain_fact("user_level")

# User profile
state.update_user_profile({"name": "Alice"})

# Session info
duration = state.get_session_duration()
idle_time = state.get_idle_time()

# Summary
summary = state.to_summary()
```

**Use case:** Store data that shouldn't consume context tokens

## Implementation Patterns

### Pattern 1: Basic Context-Aware Agent

```python
from livekit.agents import Agent
from src.livekit_context_manager import ContextManager

class MyAgent(Agent):
    async def on_user_turn_ended(self):
        # Check every 10 turns
        if hasattr(self, '_turn_counter'):
            self._turn_counter += 1
        else:
            self._turn_counter = 1

        if self._turn_counter % 10 == 0:
            await ContextManager.summarize_if_needed(
                agent=self,
                threshold_items=20
            )
```

### Pattern 2: Tool with Context Compaction

```python
@function_tool()
async def expensive_operation(self, context: RunContext, input: str) -> str:
    # Compact context before expensive operation
    compacted_ctx = await ContextManager.get_compacted_context(
        agent=self,
        keep_last_n_turns=10,
        exclude_system=True
    )

    self._chat_ctx = compacted_ctx

    # Perform operation
    result = await perform_expensive_calculation(input)
    return result
```

### Pattern 3: External State for Preferences

```python
@function_tool()
async def save_preference(
    self,
    context: RunContext,
    key: str,
    value: str
) -> str:
    # Store in external state (not chat context)
    context.session.userdata.domain_state.set_user_preference(key, value)
    return f"Saved: {key} = {value}"

@function_tool()
async def get_preference(
    self,
    context: RunContext,
    key: str
) -> str:
    # Retrieve from external state
    value = context.session.userdata.domain_state.get_user_preference(key)
    return value if value else "Not set"
```

## Monitoring & Debugging

### Enable Detailed Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Context manager logger
logger = logging.getLogger("livekit-context-manager")
logger.setLevel(logging.DEBUG)
```

### Log Output

```
📊 Context settings:
  - Summarize every N turns: 15
  - Threshold items: 20
  - Keep last N turns: 10

👤 User joined session
📊 Context stats: 5 items

🔄 Turn 15: Checking context...
📊 Context stats: 22 items
Context has 22 items, summarizing...
✅ Summarization complete: 22 -> 8 items

🔍 Performed web search: Python async patterns
💾 Saved preference: language = en-IN
```

### Check Context Stats in Real-Time

```python
@function_tool()
async def get_context_stats(self, context: RunContext) -> str:
    """Get current context statistics"""
    stats = ContextManager.get_context_stats(self)

    return (
        f"Context Statistics:\n"
        f"- Total items: {stats['total_items']}\n"
        f"- Type breakdown: {stats['type_breakdown']}\n"
        f"- Session duration: {context.session.userdata.domain_state.get_session_duration():.1f}s"
    )
```

## Testing

### Run Unit Tests

```bash
# Test context filtering
pytest tests/test_livekit_context_management.py::TestContextFiltering -v

# Test summarization
pytest tests/test_livekit_context_management.py::TestSummarization -v

# Test external state
pytest tests/test_livekit_context_management.py::TestExternalState -v

# Run all unit tests
pytest tests/test_livekit_context_management.py -v
```

### Run Integration Tests

```bash
# Test long conversation simulation
pytest tests/integration/test_long_conversation.py -v

# Test external state doesn't bloat context
pytest tests/integration/test_long_conversation.py::test_external_state_doesnt_bloat_context -v
```

### Manual Testing Checklist

- [ ] Start agent and connect client
- [ ] Have 20+ turn conversation
- [ ] Verify summarization triggers (check logs)
- [ ] Save preferences using `save_preference` tool
- [ ] Retrieve preferences using `get_preference` tool
- [ ] Check context stats using `get_context_stats` tool
- [ ] Verify no context window errors after 30+ turns

## Troubleshooting

### Issue: Summarization Not Triggering

**Symptoms:** Context grows beyond threshold without summarization

**Solutions:**
1. Check `CONTEXT_SUMMARIZE_THRESHOLD_ITEMS` value
2. Verify `on_user_turn_ended()` is called each turn
3. Check logs for "Checking context..." messages
4. Ensure `summarize_if_needed()` is called

```python
# Add debug logging
logger.info(f"Turn counter: {self._turn_counter}")
logger.info(f"Context items: {len(self.chat_ctx.items)}")
```

### Issue: Summarization Latency

**Symptoms:** Responses are slow after summarization

**Solutions:**
1. Use faster model for summarization
2. Increase `CONTEXT_SUMMARIZE_THRESHOLD_ITEMS`
3. Decrease `CONTEXT_SUMMARIZE_EVERY_N_TURNS`
4. Use smaller `keep_last_turns` value

```python
# Use faster model for summarization
fast_llm = groq.LLM(model="llama-3.1-8b-instant")
await ContextManager.summarize_if_needed(
    agent=self,
    threshold_items=20,
    llm_for_summary=fast_llm  # Faster model
)
```

### Issue: Information Loss After Summarization

**Symptoms:** Agent forgets important facts

**Solutions:**
1. Store critical facts in external state
2. Increase `keep_last_turns` value
3. Use domain state for persistent information

```python
# Store important facts externally
@function_tool()
async def remember_fact(self, context: RunContext, fact: str) -> str:
    context.session.userdata.domain_state.set_domain_fact(
        f"fact_{time.time()}",
        fact
    )
    return "Fact saved to memory"
```

### Issue: External State Not Persisting

**Symptoms:** Preferences lost between turns

**Solutions:**
1. Ensure domain state is initialized in `on_enter()`
2. Check that `userdata` is properly attached to session
3. Verify state is accessed via `context.session.userdata.domain_state`

```python
async def on_enter(self):
    # Initialize domain state
    if not hasattr(self.session.userdata, 'domain_state'):
        self.session.userdata.domain_state = SpecializedDomainState()
```

## Performance Optimization

### Reduce Summarization Frequency

```python
# Only check every 20 turns instead of 10
self._SUMMARIZE_EVERY_N_TURNS = 20
```

### Use Faster Model for Summarization

```python
# Create fast summarization LLM
fast_llm = groq.LLM(model="llama-3.1-8b-instant")

await ContextManager.summarize_if_needed(
    agent=self,
    threshold_items=20,
    llm_for_summary=fast_llm  # Use faster model
)
```

### Increase Aggressive Filtering

```python
# Filter more aggressively
compacted_ctx = await ContextManager.get_compacted_context(
    agent=self,
    keep_last_n_turns=5,  # Keep only 5 turns
    exclude_system=True
)
```

### Store More in External State

```python
# Move persistent data to external state
@function_tool()
async def set_user_profile(self, context: RunContext, data: dict) -> str:
    context.session.userdata.domain_state.update_user_profile(data)
    return "Profile saved to external state"
```

## Migration Guide

### From Basic Agent to Context-Optimized Agent

**Step 1: Import ContextManager**

```python
# Add imports
from src.livekit_context_manager import ContextManager
from src.specialized_domain_state import SpecializedDomainState
```

**Step 2: Initialize Domain State**

```python
async def on_enter(self):
    # Add domain state initialization
    if not hasattr(self.session.userdata, 'domain_state'):
        self.session.userdata.domain_state = SpecializedDomainState()
```

**Step 3: Add Periodic Checks**

```python
async def on_user_turn_ended(self):
    # Add turn counter
    if not hasattr(self, '_turn_counter'):
        self._turn_counter = 0
    self._turn_counter += 1

    # Add periodic summarization check
    if self._turn_counter % 15 == 0:
        await ContextManager.summarize_if_needed(
            agent=self,
            threshold_items=20
        )
```

**Step 4: Update Tools to Use External State**

```python
# Before (stores in chat context)
@function_tool()
async def save_preference(self, key: str, value: str) -> str:
    # This adds to chat context, consuming tokens
    return f"Saved: {key} = {value}"

# After (stores in external state)
@function_tool()
async def save_preference(self, context: RunContext, key: str, value: str) -> str:
    # This doesn't add to chat context
    context.session.userdata.domain_state.set_user_preference(key, value)
    return f"Saved: {key} = {value}"
```

## Best Practices

### ✅ DO

- **Check context periodically** (every 10-20 turns)
- **Use external state** for persistent data
- **Compact before expensive operations**
- **Monitor context stats** during development
- **Test with long conversations** (50+ turns)
- **Keep recent turns uncompressed** (3-5 turns)
- **Use faster model** for summarization

### ❌ DON'T

- **Don't summarize every turn** (adds latency)
- **Don't store preferences in chat context** (wastes tokens)
- **Don't set threshold too low** (causes frequent summarization)
- **Don't set keep_last_turns too high** (defeats purpose)
- **Don't ignore external state** (missed optimization opportunity)
- **Don't forget to handle summarization errors** (graceful degradation)

## Expected Results

### Before Implementation

```
Turn 1: Context items: 2
Turn 5: Context items: 10
Turn 10: Context items: 20
Turn 15: Context items: 30
Turn 20: Context items: 40 ⚠️ Approaching limit
Turn 25: Context items: 50 ❌ Token limit exceeded
```

### After Implementation

```
Turn 1: Context items: 2
Turn 5: Context items: 10
Turn 10: Context items: 20
Turn 15: Context items: 22
Turn 15: ✅ Summarization complete: 22 -> 8 items
Turn 20: Context items: 18
Turn 25: Context items: 20
Turn 25: ✅ Summarization complete: 20 -> 8 items
Turn 30: Context items: 18
...
Turn 100: Context items: 20 ✅ Stable
```

## Additional Resources

- [LiveKit Agents Documentation](https://docs.livekit.io/agents)
- [ChatContext API Reference](https://github.com/livekit/agents-python)
- [LangGraph Integration](../flow_agent/) (for reference only)
- [Unit Tests](../../tests/test_livekit_context_management.py)
- [Integration Tests](../../tests/integration/test_long_conversation.py)

## Support

For issues or questions:
1. Check logs for error messages
2. Review troubleshooting section above
3. Run unit tests to verify setup
4. Check environment variables are set correctly
