# Continuous Voice UI - Quick Start Guide

## What's New

**File:** `ui/app_continuous_voice.py`
- ✅ **Continuous recording** with VAD (no "record → stop" needed)
- ✅ **Automatic speech detection** (Silero VAD)
- ✅ **Real-time confidence meter** visualization
- ✅ **Push-to-talk UX** (speak naturally, AI responds)
- ✅ **UI-only implementation** (no backend changes)

## Dependencies Added

**File:** `pyproject.toml`
```toml
dependencies = [
    # ... existing ...
    "streamlit-mic-recorder>=0.1.0",  # NEW: Continuous audio recording
    "silero-vad>=5.1",                 # NEW: Voice Activity Detection
    "pyaudio>=0.12.0",                 # NEW: Audio I/O for VAD
    "numpy>=1.24.0",                  # NEW: Audio processing
]
```

## How to Run

```bash
# Terminal 1: Start backend
langgraph up --watch

# Terminal 2: Install new dependencies (first time only)
pip install streamlit-mic-recorder silero-vad pyaudio numpy

# Terminal 2: Run continuous voice UI
streamlit run ui/app_continuous_voice.py
```

## How It Works

### Step 1: Start Listening
1. Click **"🎙️ Start Listening"** button
2. VAD model initializes (~2 seconds first time)
3. Audio starts streaming to VAD
4. Watch **VAD Confidence** meter update in real-time

### Step 2: Speak Naturally
1. **VAD Confidence** rises during speech:
   - 0.0 - 0.3: Silence (red)
   - 0.3 - 0.5: Likely silence (orange)
   - 0.5 - 0.7: Likely speech (yellow)
   - 0.7 - 1.0: Definitely speech (green)

2. Audio accumulates during speech

### Step 3: Stop Speaking
1. After **600ms of silence**, segment completes
2. Audio automatically submitted to backend
3. "📤 Processing segment..." message appears
4. AI processes and responds

### Step 4: AI Response
1. "🤖 AI is thinking..." while processing
2. Response appears in conversation history
3. TTS auto-plays: "🔊 Playing audio..."
4. **Recording continues** automatically for next turn

## Key Settings

**In `ui/app_continuous_voice.py`:**
```python
# VAD Settings
VAD_THRESHOLD = 0.5      # Speech sensitivity (0.0-1.0)
SILENCE_DURATION_MS = 600   # Silence to end segment (ms)
CHUNK_DURATION_MS = 32     # Audio chunk size (ms)
SAMPLE_RATE = 16000      # Audio sample rate (Hz)
```

**Adjust for your environment:**
- **Noisy environment**: Raise `VAD_THRESHOLD` to 0.6 or 0.7
- **Fast speech**: Lower `SILENCE_DURATION_MS` to 400
- **Slower computers**: Raise `CHUNK_DURATION_MS` to 50

## Expected Behavior

### ✅ Working: Continuous VAD Detection
```
🎙️ Continuous Voice Conversation

Just start talking - VAD detects when you speak!

💬 Conversation
(Empty at start)

🎤 Recording Status
[🎙️ Start Listening] button

[Click start]

🔴 Listening...
VAD will detect when you speak
```

### ✅ Working: Speech Detection
```
💬 Conversation
👤 **You:** [Voice input]

🎤 Recording Status
🔴 Listening...
VAD Confidence: 0.23  (red/orange)
```

### ✅ Working: During Speech
```
💬 Conversation
👤 **You:** [Voice input]

🎤 Recording Status
🔴 Listening...
VAD Confidence: 0.87  (green/yellow)
```

### ✅ Working: After Silence (Segment Complete)
```
💬 Conversation
👤 **You:** [Voice input]
📤 Processing segment (1234 bytes)...

🤖 AI is thinking...

🤖 **AI:** The capital of France is Paris.
🔊 Playing audio...
```

### ✅ Working: Multi-Turn Conversation
```
💬 Conversation
👤 **You:** [Voice input]
🤖 **AI:** The capital of France is Paris.
👤 **You:** What time is it?
🤖 **AI:** The current time is...
```

## Troubleshooting

### Issue: "Input audio chunk is too short"
**Fixed:** Increased `CHUNK_DURATION_MS` from 30ms to 32ms
- Silero VAD requires minimum 512 samples (32ms @ 16kHz)
- Now provides 512 samples per chunk

### Issue: "TypeError: float() argument must be a real number"
**Fixed:** Convert numpy float to Python float before storing in session state
- `st.session_state.vad_confidence = float(confidence)`

### Issue: VAD not detecting speech
**Solutions:**
1. **Check microphone permissions** - Allow browser access
2. **Adjust threshold** - Lower `VAD_THRESHOLD` if background noise
3. **Speak clearly** - VAD needs clear speech signal
4. **Increase volume** - Low input may not trigger detection

### Issue: Recording doesn't stop
**Check:**
- Click **"⏹️ Stop Listening"** button
- Streamlit reruns after stop
- Audio stream properly closed

## Comparison: Old vs New

| Feature | Old UI (app_v4) | Continuous Voice |
|---------|-------------------|----------------|
| **Recording** | Manual "record → stop" | Automatic VAD detection |
| **VAD** | None | Real-time confidence meter |
| **UX** | Discrete segments | Natural conversation |
| **STT** | Batch per file | Batch per segment |
| **Complexity** | Simple (~300 lines) | Medium (~580 lines) |
| **Dependencies** | Existing only | +4 new packages |

## Key Improvements Over app_v4

1. **No more "record → stop"** - VAD detects automatically
2. **Real-time feedback** - See VAD confidence while speaking
3. **Natural conversation** - Multi-turn without button presses
4. **Better UX** - Push-to-talk instead of click-wait

## Limitations (What's NOT Possible)

❌ **True real-time** (<50ms latency):
   - Streamlit reruns add 100-300ms
   - Batch STT processing takes 1-3 seconds
   - Best achievable: 100-500ms total latency

❌ **Full-duplex** (simultaneous listen/speak):
   - Cannot listen while TTS plays
   - Alternates: listen → process → speak → listen

❌ **Streaming STT**:
   - SarvamAI/OpenAI APIs are batch-only
   - VAD segments simulate streaming via chunking

✅ **What IS Achieved:**
   - Continuous UX (no button presses after start)
   - VAD-driven automatic segmentation
   - Natural multi-turn conversation
   - "Near real-time" experience
   - **NO BACKEND CHANGES**

## Summary

This is **the best possible continuous voice experience using only Streamlit UI**:
- VAD automatically detects speech segments
- No "record → stop" cycle needed after initial start
- Natural conversation flow
- Pushes Streamlit to its limits (best effort with constraints)
