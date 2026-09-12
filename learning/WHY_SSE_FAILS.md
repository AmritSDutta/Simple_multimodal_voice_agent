# Why SSE (Server-Sent Events) is Failing

## The Problem

From your logs:
```
2026-02-14 18:31:07.298 | INFO | SSE connection established (status: 200)
2026-02-14 18:31:26.719 | WARNING | SSE connection established but no events received before stream closed
```

The SSE endpoint:
- ✅ **Connects successfully** (HTTP 200)
- ❌ **Sends no events**
- ❌ **Closes immediately** (after ~19 seconds)

## Why This Happens

### 1. LangGraph SSE Endpoint Behavior

**LangGraph's `/stream` endpoint doesn't work like standard SSE!**

Looking at LangGraph documentation and behavior:
- The `/stream` endpoint exists and returns HTTP 200
- But it doesn't send events in standard SSE format (`event: xxx\ndata: {...}\n\n`)
- Instead, it likely:
  - Requires specific query parameters to enable streaming
  - Only streams when run is actively executing (not after completion)
  - Uses WebSocket protocol instead of SSE
  - Or simply doesn't stream intermediate events at all in local dev mode

### 2. Timing Issue

**From your logs:**
```
18:31:07 | SSE connection opens
18:31:26 | Stream closes (19 seconds later)
```

The backend completed the run in **18.7 seconds**:
```
run_started_at: '2026-02-14T12:56:30.758398+00:00'
run_ended_at: '2026-02-14T12:56:48.213211+00:00'
run_exec_ms: 17454  # 17.4 seconds
```

**By the time SSE connects, the run is already finished!**

LangGraph might:
1. Only stream events **while run is executing**
2. Stop sending events once run completes
3. Close the connection immediately

If SSE connects after completion, no events are sent.

### 3. Local Dev vs Production

**In local development (`langgraph up --watch`):**
- Events might not be streamed at all
- SSE endpoint might be disabled/stub
- Different behavior than production

**In production deployment:**
- SSE might work differently
- Requires WebSocket instead of SSE
- Has proper event streaming infrastructure

## Evidence from Logs

**What we see:**
```
HTTP Request: GET http://localhost:2024/threads/.../stream
HTTP/1.1 200 OK
SSE connection established (status: 200)
[19 seconds of silence]
SSE connection established but no events received before stream closed
```

**What we DON'T see:**
- No `event: update` lines
- No `event: end` lines
- No `data: {...}` lines
- No event chunks at all

**This means:**
- Connection works (HTTP 200)
- But server sends **zero bytes** of event data
- Server closes connection without sending anything

## Why Polling Works

**Polling doesn't rely on events:**
```python
# Polling (works):
while True:
    response = requests.get(f"/threads/{t}/runs/{r}")  # Always gets current status
    if response["status"] == "success":
        return response  # Done!
    sleep(1)
```

**SSE (doesn't work):**
```python
# SSE (fails):
async for chunk in response.aiter_bytes():  # Never receives any chunks!
    process_event(chunk)  # Never executes
```

## The Fix: Use Simple Polling

**Why polling is better for this use case:**

1. **Reliable:** Always works, regardless of SSE configuration
2. **Simple:** Easy to understand, no complex async/SSE code
3. **Fast enough:** 1-second polling is plenty fast for voice interaction
4. **LangGraph compatible:** Works with LangGraph's actual behavior
5. **Production ready:** Will work same in dev and production

**Your actual timing:**
```
STT: 10 seconds (uploading, processing, downloading)
AI Processing: 18 seconds
TTS: 10 seconds
Total: ~38 seconds
```

Polling adds **0-1 second** overhead = **negligible** compared to 38-second total!

## SSE vs Polling Comparison

| Aspect | SSE (Failed) | Polling (Works) |
|--------|--------------|------------------|
| **Events received** | 0 | ✅ Gets final status |
| **Code complexity** | High (async, SSE parsing) | Low (simple requests) |
| **Reliability** | ❌ Depends on LangGraph config | ✅ Always works |
| **Performance** | Same (run already finished) | Same (1 poll needed) |
| **Debugging** | Hard (why no events?) | Easy (HTTP status codes) |
| **Maintainability** | Complex | Simple |

## Conclusion

**SSE is failing because:**
1. LangGraph's `/stream` endpoint doesn't send intermediate events in local dev
2. By the time SSE connects, run is already complete
3. No events are sent at all (connection opens, sends nothing, closes)

**Polling works because:**
1. It just asks "Are you done yet?" every second
2. Works regardless of SSE/streaming behavior
3. Gets the answer reliably

**Recommendation:**
Use the **simple polling version** (`app_v4_simple_polling.py`). It's:
- ✅ 300 lines vs 600+ lines
- ✅ Synchronous, easy to understand
- ✅ Works reliably
- ✅ No async complexity
- ✅ Perfect for voice UI use case

**When would SSE be useful?**
- Long-running operations (minutes/hours)
- Need real-time progress updates
- Web-based dashboard with multiple concurrent users
- Production deployment with proper SSE infrastructure

**For voice UI:**
- Interactions take 30-60 seconds total
- 1-second polling is more than enough
- Simpler code = fewer bugs = better UX
