# 🚀 Quick Start - Run LiveKit Agent

## ⚠️ Important: Don't Run Directly!

**WRONG:** ❌ `python src/livekit_context_optimized.py`
**RIGHT:** ✅ `python -m src.livekit_context_optimized`

## ✅ Correct Commands

### Option 1: Development Mode (Recommended)
```bash
# Terminal 1: Start the agent in development mode
python -m src.livekit_context_optimized dev

# Terminal 2: Start the console (for testing)
python -m src.livekit_context_optimized console
```

### Option 2: Use Launcher Script
```bash
python run_livekit_agent.py
```

### Option 3: With Custom Settings
```bash
CONTEXT_SUMMARIZE_THRESHOLD_ITEMS=15 \
CONTEXT_KEEP_LAST_N_TURNS=8 \
CONTEXT_SUMMARIZE_EVERY_N_TURNS=10 \
python -m src.livekit_context_optimized dev
```

## 📋 Usage Workflow

1. **Terminal 1** - Start the agent:
   ```bash
   python -m src.livekit_context_optimized dev
   ```

2. **Terminal 2** - Connect console (for voice testing):
   ```bash
   python -m src.livekit_context_optimized console
   ```

The `dev` command starts the LiveKit agent server, while `console` connects a test client for voice interaction.

## 🧪 Run Tests

```bash
# Run all tests
pytest tests/test_livekit_context_management.py tests/integration/test_long_conversation.py -v

# Quick test
pytest tests/test_livekit_context_management.py -x
```

## 📋 Pre-Flight Checklist

```bash
# 1. Set API keys
export GROQ_API_KEY="your-key"
export SARVAM_API_KEY="your-key"

# 2. Verify Python can find modules
python -c "from src.livekit_context_manager import ContextManager; print('✅ Imports OK')"

# 3. Run the agent
python -m src.livekit_context_optimized
```

## 🔧 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'src'`

**Cause:** Running with `python src/livekit_context_optimized.py`

**Solution:** Use `python -m src.livekit_context_optimized` instead

---

**For more details, see:** `docs/LIVEKIT_TESTING_COMMANDS.md`
