# VAD Testing Guide

## Quick Start
```bash
# 1. Start LangGraph backend
langgraph up --watch

# 2. Start continuous voice UI (new terminal)
streamlit run ui/app_continuous_voice.py
```

## Pre-Test Checklist

### 1. Check Backend is Running
```bash
curl http://localhost:2024/threads
# Should return: {"thread_id": "..."}
```

### 2. Check API Keys
Ensure `.env` has:
```bash
OPENAI_API_KEY=sk-...  # For STT/TTS or moderation
SARVAM_API_KEY=...      # For speech services
```

### 3. Check Microphone Permissions
- Windows: Settings → Privacy → Microphone → Allow apps to access
- Test with: Windows Sound Recorder or `streamlit run ui/app.py`

## Test Scenarios

### Test 1: Verify Audio Normalization
**Goal**: Confirm audio is properly normalized to [-1, 1]

**Steps**:
1. Open UI: http://localhost:8501
2. Click "Start Listening"
3. Speak clearly into microphone: "Testing one two three"
4. Check console (terminal running Streamlit)

**Expected Console Output**:
```
[VAD IN] Shape: (512,), Min: -0.XXX, Max: 0.XXX  # Values between -1.0 and 1.0
[VAD CONF] Final: 0.XXXX  # > 0.5 when speaking
```

**Pass Criteria**:
- Min/Max between -1.0 and 1.0 ✅
- Confidence > 0.5 when speaking clearly ✅

**Fail Signs**:
- Min/Max in scientific notation (e.g., 2.34e+04) → Audio not normalized
- Confidence stays < 0.05 → VAD not detecting speech

---

### Test 2: Verify Audio Level Meter
**Goal**: Confirm audio level meter shows movement when speaking

**Steps**:
1. Start listening
2. Watch "🎤 Audio" meter in UI
3. Speak: "This is a test"
4. Watch meter during speech

**Expected**:
- Meter stays at 0.000 when silent
- Meter rises to 0.1-0.3 when speaking
- Meter drops back to 0.000 after silence

**Pass Criteria**:
- Audio level > 0.05 when speaking ✅

**Fail Signs**:
- Meter stays at 0.000 → RMS calculation issue
- Meter always at 1.000 → Not normalized properly

---

### Test 3: Verify VAD State Transitions
**Goal**: Confirm VAD detects speech → silence → complete cycle

**Steps**:
1. Start listening
2. Watch "State:" indicator
3. Speak: "Hello world"
4. Stop speaking and wait 2 seconds

**Expected Console Output**:
```
[VAD STATE] → SPEECH (confidence 0.XXX >= 0.15)
[VAD STATE] → SILENCE (segment complete: XXXX bytes)
```

**Expected UI States**:
1. `👂 Waiting...` (before speaking)
2. `🗣️ SPEECH` (while speaking)
3. `⏸️ Waiting (1/56)` (counting silence chunks)
4. `🤐 SILENCE` (segment complete)

**Pass Criteria**:
- Sees "SPEECH" when talking starts ✅
- Sees "SILENCE (segment complete)" when talking stops ✅

**Fail Signs**:
- Never sees "SPEECH" → Threshold too high or no audio
- Always in "SPEECH" → Threshold too low
- Never sees "SILENCE" → Silence duration too long or background noise

---

### Test 4: Verify Backend Processing
**Goal**: Confirm transcribed text is sent to backend and AI responds

**Steps**:
1. Start listening
2. Speak: "What is two plus two?"
3. Wait for segment to complete
4. Watch UI messages

**Expected UI Messages**:
1. `📤 Processing segment (XXXXX bytes)...`
2. `🎤 Transcribing audio...`
3. `✅ Transcribed: what is two plus two...`
4. `🤖 AI is thinking...`
5. `✅ Response received in X.Xs`
6. `🔊 Generating audio...`
7. `🔊 Playing audio...`

**Expected Conversation**:
```
👤 You: [Voice input]
🤖 AI: Two plus two equals four.
```

**Pass Criteria**:
- Sees "Transcribing audio..." message ✅
- Sees transcribed text ✅
- Gets AI response ✅
- TTS auto-plays ✅

**Fail Signs**:
- "⚠️ Transcription failed" → STT API key issue or audio format
- "⚠️ No response generated" → Backend LLM error
- "❌ Error: ..." → Check error message

---

### Test 5: Verify Manual Fallback
**Goal**: Confirm text input works if VAD fails

**Steps**:
1. Start listening (optional)
2. Type in text box: "What is the capital of France?"
3. Click "Send Text"

**Expected UI Messages**:
1. `✅ Session ready: XXXXXXXX...` (if first message)
2. `🤖 AI is thinking...`
3. `✅ Response received`
4. `🔊 Generating audio...`

**Pass Criteria**:
- Can submit text while listening ✅
- Gets AI response ✅
- TTS auto-plays ✅

**Fail Signs**:
- Nothing happens → Backend not running
- Error in console → Check API keys

---

## Debug Tips

### Check Debug Info
Click "🔊 Audio Debug Info" expander to see:
- VAD confidence
- Audio level (RMS)
- VAD threshold

### Adjust Threshold
If VAD not detecting speech:
1. Lower "Speech Threshold" slider (try 0.10 or 0.05)
2. Try 0.02 for very quiet speech (may false trigger on noise)

If VAD always detecting speech:
1. Raise "Speech Threshold" slider (try 0.25 or 0.30)
2. Check for background noise (fan, traffic, etc.)

### Check Console Logs
Key console messages:
- `[VAD IN]` - Audio chunk stats (shape, min, max)
- `[VAD CONF]` - Final confidence score
- `[VAD STATE]` - State transitions
- `[VAD ERR]` - Errors (should not see this)
- `[SUBMIT]` - Transcription text

### Common Issues

**Issue**: "ModuleNotFoundError: No module named 'silero_vad'"
**Fix**: `pip install silero-vad`

**Issue**: "OSError: [Errno -9996] Invalid input device index"
**Fix**: Check microphone is connected and accessible

**Issue**: Confidence always 0.003-0.009
**Fix**: Audio not normalized - check code matches fix summary

**Issue**: "⚠️ Transcription failed"
**Fix**: Check API keys in `.env`:
- `SARVAM_API_KEY` (if using SarvamAI STT)
- `OPENAI_API_KEY` (if using OpenAI Whisper)

**Issue**: Backend returns error
**Fix**: Check backend is running:
```bash
curl http://localhost:2024/threads
```

## Performance Metrics

### Good Performance
- Confidence: 0.5-1.0 when speaking clearly
- Audio Level: 0.1-0.3 when speaking
- Transcription: < 2 seconds for 5-second audio
- Total Response: < 10 seconds (transcribe + LLM + TTS)

### Needs Tuning
- Confidence: 0.1-0.5 → Lower threshold or speak louder
- Audio Level: < 0.05 → Check microphone volume
- Transcription: > 5 seconds → Network latency or slow API
- Total Response: > 15 seconds → LLM timeout or slow TTS

## Next Steps After Testing

### If All Tests Pass
1. Adjust VAD threshold for your environment
2. Adjust silence duration (longer = more patient, shorter = faster)
3. Test with longer conversations
4. Check TTS quality for your use case

### If Tests Fail
1. Check console logs for error messages
2. Verify audio normalization (min/max between -1.0 and 1.0)
3. Verify API keys in `.env`
4. Test microphone with Windows Sound Recorder
5. Check `docs/VAD_FIX_SUMMARY.md` for detailed troubleshooting

## Additional Resources

- **VAD Fix Summary**: `docs/VAD_FIX_SUMMARY.md`
- **Unit Tests**: `tests/test_vad_audio_normalization.py`
- **Architecture**: `CLAUDE.md`
- **Speech Services**: `src/flow_agent/speech/`

## Reporting Issues

When reporting VAD issues, include:
1. Console output (especially `[VAD IN]` and `[VAD STATE]` messages)
2. Debug info values (confidence, audio level, threshold)
3. Test results (which tests passed/failed)
4. Microphone type and environment (quiet/noisy)
5. Python version and OS
