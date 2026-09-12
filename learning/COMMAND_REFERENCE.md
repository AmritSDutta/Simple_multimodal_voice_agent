# 🚀 LiveKit Agent - Command Reference

## Standard Development Workflow

### Step 1: Set Environment Variables

```bash
# Linux/Mac
export GROQ_API_KEY="your-groq-api-key"
export SARVAM_API_KEY="your-sarvam-api-key"

# Windows PowerShell
$env:GROQ_API_KEY="your-groq-api-key"
$env:SARVAM_API_KEY="your-sarvam-api-key"

# Windows CMD
set GROQ_API_KEY=your-groq-api-key
set SARVAM_API_KEY=your-sarvam-api-key
```

### Step 2: Start the Agent (Two Terminals)

**Terminal 1** - Start the agent in development mode:
```bash
python -m src.livekit_context_optimized dev
```

**Terminal 2** - Connect console for voice testing:
```bash
python -m src.livekit_context_optimized console
```

### What These Commands Do

- **`dev`** - Starts the LiveKit agent server in development mode with hot-reload
- **`console`** - Connects an interactive voice client for testing the agent

---

## Quick Commands

### Run Tests
```bash
# All context management tests
pytest tests/test_livekit_context_management.py tests/integration/test_long_conversation.py -v

# Unit tests only
pytest tests/test_livekit_context_management.py -v

# Integration tests only
pytest tests/integration/test_long_conversation.py -v
```

### Run Agent with Custom Settings
```bash
CONTEXT_SUMMARIZE_THRESHOLD_ITEMS=15 \
CONTEXT_KEEP_LAST_N_TURNS=8 \
CONTEXT_SUMMARIZE_EVERY_N_TURNS=10 \
python -m src.livekit_context_optimized dev
```

### Monitor Logs
```bash
# Save logs to file
python -m src.livekit_context_optimized dev 2>&1 | tee agent.log

# Follow log file
tail -f agent.log  # Linux/Mac
Get-Content agent.log -Wait  # Windows PowerShell
```

---

## Common Workflows

### 1. Quick Test (5 minutes)
```bash
# Terminal 1
export CONTEXT_SUMMARIZE_THRESHOLD_ITEMS=10
export CONTEXT_SUMMARIZE_EVERY_N_TURNS=5
python -m src.livekit_context_optimized dev

# Terminal 2
python -m src.livekit_context_optimized console
```

### 2. Long Conversation Test (15+ minutes)
```bash
# Terminal 1
export CONTEXT_SUMMARIZE_THRESHOLD_ITEMS=20
export CONTEXT_SUMMARIZE_EVERY_N_TURNS=15
python -m src.livekit_context_optimized dev

# Terminal 2
python -m src.livekit_context_optimized console
```

### 3. Debug Mode
```bash
# Terminal 1
LOGLEVEL=DEBUG python -m src.livekit_context_optimized dev

# Terminal 2
python -m src.livekit_context_optimized console
```

---

## ❌ Wrong Commands (Don't Use)

```bash
# WRONG - Causes import error
python src/livekit_context_optimized.py

# WRONG - Causes import error
python src/livekit_context_optimized.py dev

# WRONG - Causes import error
python src/livekit_context_optimized.py console
```

---

## ✅ Correct Commands

```bash
# CORRECT - Run as module
python -m src.livekit_context_optimized dev

# CORRECT - Console client
python -m src.livekit_context_optimized console

# CORRECT - Launcher script
python run_livekit_agent.py
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | Required | Groq API key for LLM/STT |
| `SARVAM_API_KEY` | Required | SarvamAI API key for TTS |
| `CONTEXT_SUMMARIZE_THRESHOLD_ITEMS` | `10` | Trigger summarization at N items |
| `CONTEXT_KEEP_LAST_N_TURNS` | `3` | Keep last N turns uncompressed |
| `CONTEXT_SUMMARIZE_EVERY_N_TURNS` | `5` | Check every N turns |
| `LOGLEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

---

## Troubleshooting

### ModuleNotFoundError

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Use `python -m src.livekit_context_optimized dev` instead of `python src/livekit_context_optimized.py`

### API Connection Errors

**Problem:** Cannot connect to API

**Solution:** Verify API keys are set:
```bash
echo $GROQ_API_KEY
echo $SARVAM_API_KEY
```

### Tool Schema Errors

**Problem:** `invalid JSON schema for tool`

**Solution:** This is fixed in the latest version. Ensure you're using the updated code.

---

## Summary Card

```bash
# 🚀 Standard workflow (two terminals)

# Terminal 1:
export GROQ_API_KEY="your-key"
export SARVAM_API_KEY="your-key"
python -m src.livekit_context_optimized dev

# Terminal 2:
python -m src.livekit_context_optimized console

# ✅ That's it!
```

---

**For more details, see:** `docs/LIVEKIT_TESTING_COMMANDS.md`
