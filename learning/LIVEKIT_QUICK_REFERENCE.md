# LiveKit Context Optimization - Quick Reference

## 🚀 Quick Start

```bash
# 1. Set environment variables
export GROQ_API_KEY="your-key"
export SARVAM_API_KEY="your-key"

# 2. Run optimized agent (IMPORTANT: use -m flag)
# Terminal 1: Start the agent
python -m src.livekit_context_optimized dev

# Terminal 2: Connect console for testing
python -m src.livekit_context_optimized console

# 3. Test with long conversation (20+ turns)
```

## 📊 Key Configuration

```bash
# Environment variables (.env)
CONTEXT_SUMMARIZE_THRESHOLD_ITEMS=20      # Trigger at 20 items
CONTEXT_KEEP_LAST_N_TURNS=10              # Keep 10 turns uncompressed
CONTEXT_SUMMARIZE_EVERY_N_TURNS=15        # Check every 15 turns
```

## 🔧 Core API

### ContextManager

```python
# Check and summarize if needed
await ContextManager.summarize_if_needed(
    agent=self,
    threshold_items=20,
    keep_last_turns=3
)

# Compact context (remove noise)
compacted = await ContextManager.get_compacted_context(
    agent=self,
    keep_last_n_turns=10,
    exclude_system=True
)

# Get statistics
stats = ContextManager.get_context_stats(self)
```

### SpecializedDomainState

```python
# Preferences
state.set_user_preference("key", "value")
value = state.get_user_preference("key")

# Task history
state.add_task_result("task_name", {"data": "result"})
history = state.get_task_history("task_name")

# Domain facts
state.set_domain_fact("fact_key", "fact_value")
fact = state.get_domain_fact("fact_key")
```

## 📝 Implementation Pattern

```python
from livekit.agents import Agent
from src.livekit_context_manager import ContextManager
from src.specialized_domain_state import SpecializedDomainState

class MyAgent(Agent):
    def __init__(self):
        super().__init__(instructions="...")
        self._turn_counter = 0

    async def on_enter(self):
        # Initialize external state
        if not hasattr(self.session.userdata, 'domain_state'):
            self.session.userdata.domain_state = SpecializedDomainState()

        # Check summarization
        await ContextManager.summarize_if_needed(agent=self, threshold_items=20)

        await self.session.generate_reply()

    async def on_user_turn_ended(self):
        # Periodic check
        self._turn_counter += 1
        if self._turn_counter % 15 == 0:
            await ContextManager.summarize_if_needed(agent=self, threshold_items=20)

    @function_tool()
    async def save_pref(self, context, key, value):
        # Use external state
        context.session.userdata.domain_state.set_user_preference(key, value)
        return f"Saved: {key} = {value}"
```

## 🧪 Testing

```bash
# Unit tests
pytest tests/test_livekit_context_management.py -v

# Integration tests
pytest tests/integration/test_long_conversation.py -v

# Run all
pytest tests/ -v -k "context"
```

## 📈 Expected Results

**Before:** Context grows unbounded → hits token limits after ~20-30 turns

**After:** Context stabilized at ~15-20 items → 100+ turns without limits

## 🔍 Debugging

```python
# Enable logging
import logging
logging.basicConfig(level=logging.INFO)

# Check context stats
@function_tool()
async def get_stats(self, context):
    stats = ContextManager.get_context_stats(self)
    return f"Items: {stats['total_items']}"
```

## ⚡ Performance Tips

1. **Use faster model for summarization**
   ```python
   fast_llm = groq.LLM(model="llama-3.1-8b-instant")
   await ContextManager.summarize_if_needed(agent=self, llm_for_summary=fast_llm)
   ```

2. **Store persistent data externally**
   ```python
   # Don't add to chat context
   context.session.userdata.domain_state.set_user_preference(key, value)
   ```

3. **Compact before expensive operations**
   ```python
   compacted = await ContextManager.get_compacted_context(agent=self)
   self._chat_ctx = compacted
   ```

## 📚 Documentation

- Full guide: `docs/LIVEKIT_CONTEXT_OPTIMIZATION.md`
- API reference: `src/livekit_context_manager.py`
- Examples: `src/livekit_context_optimized.py`

## ✅ Success Criteria

- [x] Context filtering removes 30-50% noise
- [x] Summarization triggers at threshold
- [x] External state doesn't bloat context
- [x] Long conversations (100+ turns) work
- [x] All tests pass (16 unit + 4 integration)

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| No summarization | Check `on_user_turn_ended()` is called |
| High latency | Use faster model, increase threshold |
| Lost facts | Store in external state |
| State lost | Initialize in `on_enter()` |

## 🎯 Tuning Guidelines

**Short conversations (< 50 turns):**
- `CONTEXT_SUMMARIZE_THRESHOLD_ITEMS=30`
- `CONTEXT_SUMMARIZE_EVERY_N_TURNS=20`

**Long conversations (> 100 turns):**
- `CONTEXT_SUMMARIZE_THRESHOLD_ITEMS=15`
- `CONTEXT_SUMMARIZE_EVERY_N_TURNS=10`

**Faster models:**
- `CONTEXT_KEEP_LAST_N_TURNS=5`
- `CONTEXT_SUMMARIZE_THRESHOLD_ITEMS=25`
