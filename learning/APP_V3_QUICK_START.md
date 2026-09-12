# Quick Start Guide: app_v3.py (SSE Streaming)

## Prerequisites

1. **LangGraph Backend Running:**
   ```bash
   langgraph up --watch
   ```
   - Should be running at `http://localhost:2024`
   - Check with: `curl http://localhost:2024/threads`

2. **Environment Variables Set:**
   - All required API keys in `.env` file
   - `DEPLOYMENT_URL` defaults to `http://localhost:2024`

## Running app_v3.py

**Terminal 1 - Start Backend:**
```bash
langgraph up --watch
```

**Terminal 2 - Start UI:**
```bash
streamlit run ui/app_v3.py
```

**Access:**
- Streamlit UI: http://localhost:8501
- LangGraph API: http://localhost:2024

## Testing SSE Streaming

### Test 1: Simple Text Query
1. Open http://localhost:8501
2. Enter text: "What is 2+2?"
3. Click "🚀 Run"
4. Watch for real-time event updates:
   - `📡 Event #1: Node 'input_validator' executing...`
   - `📡 Event #2: Node 'reasoning' executing...`
   - `✅ Run completed after X events`
5. Check results for: `✨ Processed via SSE streaming`

### Test 2: Multimodal Query
1. Upload an image
2. Enter text: "Describe this image"
3. Click "🚀 Run"
4. Verify streaming works with images
5. Check AI response includes image description

### Test 3: Voice Input
1. Click microphone icon
2. Record audio question
3. Click "🚀 Run"
4. Watch STT transcription: `✅ Transcribed: {text}`
5. Verify streaming events appear
6. Click "🔊 Read Aloud" to test TTS

### Test 4: Fallback Behavior
1. Stop backend (Ctrl+C in Terminal 1)
2. Submit query in UI
3. Verify fallback appears:
   - `⚠️ Streaming unavailable: Connection refused`
   - `🔄 Falling back to polling...`
4. Restart backend: `langgraph up --watch`
5. Verify queries work again

## Key Differences from app_v2.py

| Aspect | app_v2.py | app_v3.py |
|--------|-----------|-----------|
| **Status Updates** | "Status: **pending** · Poll #1" | "📡 Event #1: Node `reasoning` executing..." |
| **Update Frequency** | Every 1 second | Real-time (instant) |
| **Final Status** | "✅ Run completed successfully!" | "✅ Run completed after X events" |
| **Mode Indicator** | None | "✨ Processed via SSE streaming (X events)" |

## Troubleshooting

### Issue: "Streaming unavailable: Connection refused"
**Cause:** Backend not running or wrong port
**Fix:**
1. Check backend: `langgraph status`
2. Verify port: `DEPLOYMENT_URL="http://localhost:2024"`
3. Backend auto-falls back to polling - should still work

### Issue: No events appearing
**Cause:** SSE endpoint not responding
**Fix:** Check LangGraph logs: `docker logs -f <container_name>`

### Issue: Import error for streaming_client
**Cause:** Python path not set correctly
**Fix:** Ensure running from project root with `streamlit run ui/app_v3.py`

### Issue: "coroutine 'AsyncMockMixin._execute_mock_call' was never awaited"
**Cause:** AsyncMock in tests (development only)
**Fix:** Ignore - this is test-only warning, doesn't affect app

## Performance Comparison

**Polling (app_v2.py):**
- Network requests: ~10-20 per run (1 query/sec for 10-20 seconds)
- Latency: Up to 1 second delay for status updates
- UX: Stuttered updates

**SSE Streaming (app_v3.py):**
- Network requests: 1 connection per run
- Latency: Near-instant updates (< 100ms)
- UX: Smooth, responsive feedback

## Monitoring SSE Events

**Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Watch for log messages:**
```
INFO:ui.streaming_client:Connecting to SSE stream: http://localhost:2024/threads/...
INFO:ui.streaming_client:SSE connection established
INFO:ui.streaming_client:Run completed with status: success
```

## Switching Between Versions

**Compare side-by-side:**
```bash
# Terminal 2: app_v2.py (polling)
streamlit run ui/app_v2.py --port 8501

# Terminal 3: app_v3.py (streaming)
streamlit run ui/app_v3.py --port 8502
```

**Default to app_v3.py:**
```bash
# Update ui/app.py to symlink to app_v3.py
cd ui
mv app.py app_v1_original.py
cp app_v3.py app.py
```

## Next Steps

1. **Verify all tests pass:**
   ```bash
   pytest tests/ -v
   ```

2. **Test integration with existing workflow:**
   - Multimodal queries (text + images)
   - Voice input (STT + TTS)
   - Long-running queries (test streaming benefits)

3. **Monitor for issues:**
   - Check for SSE disconnections
   - Verify fallback works correctly
   - Test with slow/failed LLM calls

4. **Provide feedback:**
   - Report any bugs
   - Suggest improvements
   - Share performance comparisons
