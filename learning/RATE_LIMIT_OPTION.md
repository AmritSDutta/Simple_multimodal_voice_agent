# Rate Limiting Implementation Options

## Objective

Add rate limiting to the LangGraph multimodal voice agent to prevent:
- LLM API throttling
- Excessive resource consumption
- Abuse from individual users

## Three Implementation Options

### Option 1: LangChain Rate Limiter (Recommended)

**Scope**: LLM API-level rate limiting (global throttle)

**Files to Change**: 3 (1 new, 2 modified)
**Lines Added**: ~20
**Works Locally**: ✅ Yes

#### File 1: `src/flow_agent/utils/rate_limiter.py` (NEW)

```python
"""Rate limiting utilities using LangChain's built-in rate limiters."""
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.language_models import BaseChatModel

from src.flow_agent.configurations.config import settings


def get_rate_limited_llm(
    llm: BaseChatModel,
    requests_per_second: float | None = None,
) -> BaseChatModel:
    """
    Wrap an LLM with an in-memory rate limiter.

    Args:
        llm: The LLM to wrap
        requests_per_second: Maximum requests per second (defaults to settings)

    Returns:
        The same LLM, wrapped with rate limiting
    """
    if requests_per_second is None:
        requests_per_second = settings.RATE_LIMIT_REQUESTS_PER_SECOND

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=requests_per_second
    )

    # Bind rate limiter to LLM
    return llm.bind(rate_limiter=rate_limiter)
```

#### File 2: `src/flow_agent/configurations/config.py` (MODIFY)

Add to `Settings` class:

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_SECOND: float = Field(default=2.0)
    """Maximum LLM requests per second. Default: 2 (1 request every 500ms)"""
```

#### File 3: `src/flow_agent/llms/LangChainChatLLM.py` (MODIFY)

Add import and wrap LLM before returning:

```python
from src.flow_agent.utils.rate_limiter import get_rate_limited_llm

def get_chat_llm(provider: str | None = None, use_for_summarization: bool = False) -> BaseChatModel:
    # ... existing provider selection logic ...

    if provider == "gemini":
        llm = ChatGoogleGenerativeAI(model=vision_model, ...)
    # ... other providers ...

    # Wrap with rate limiter (ADD THIS)
    llm = get_rate_limited_llm(llm)

    return llm
```

**Usage**: Automatically rate limits all LLM calls across the application.

---

### Option 2: Node-Level Rate Limiting

**Scope**: Per-user/per-thread rate limiting at graph entry

**Files to Change**: 3 (1 new, 2-3 modified)
**Lines Added**: ~60
**Works Locally**: ✅ Yes

#### File 1: `src/flow_agent/utils/rate_limiter.py` (NEW)

```python
"""In-memory rate limiter for node-level rate limiting."""
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict

from src.flow_agent.configurations.config import settings


@dataclass
class SimpleRateLimiter:
    """Thread-safe in-memory rate limiter using sliding window."""

    max_requests: int = field(default_factory=lambda: settings.RATE_LIMIT_MAX_REQUESTS)
    window_seconds: int = field(default_factory=lambda: settings.RATE_LIMIT_WINDOW_SECONDS)

    # Internal state
    _requests: Dict[str, list[datetime]] = field(default_factory=lambda: defaultdict(list))
    _lock: Lock = field(default_factory=Lock)

    def check_rate_limit(self, identifier: str) -> tuple[bool, str | None]:
        """
        Check if identifier is within rate limit.

        Args:
            identifier: Unique identifier (thread_id, IP, etc.)

        Returns:
            (allowed: bool, error_message: str | None)
        """
        with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.window_seconds)

            # Clean old requests for this identifier
            self._requests[identifier] = [
                req_time for req_time in self._requests[identifier]
                if req_time > cutoff
            ]

            # Check if under limit
            if len(self._requests[identifier]) >= self.max_requests:
                retry_after = int((self._requests[identifier][0] - cutoff).total_seconds())
                return False, f"Rate limit exceeded. Try again in {retry_after} seconds."

            # Record this request
            self._requests[identifier].append(now)
            return True, None


# Global rate limiter instance
_rate_limiter = SimpleRateLimiter()


def check_rate_limit(identifier: str) -> tuple[bool, str | None]:
    """
    Check rate limit for given identifier.

    Args:
        identifier: Unique identifier (thread_id, IP, user_id, etc.)

    Returns:
        (allowed: bool, error_message: str | None)
    """
    return _rate_limiter.check_rate_limit(identifier)
```

#### File 2: `src/flow_agent/configurations/config.py` (MODIFY)

Add to `Settings` class:

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Rate Limiting (Node-level)
    RATE_LIMIT_MAX_REQUESTS: int = Field(default=10)
    """Maximum requests per time window. Default: 10 requests"""

    RATE_LIMIT_WINDOW_SECONDS: int = Field(default=60)
    """Time window in seconds. Default: 60 seconds (1 minute)"""
```

#### File 3: `src/flow_agent/utils/state.py` (MODIFY)

Add `thread_id` to State:

```python
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retry_count: Annotated[int, add]
    issue: str
    final_report: str
    input_valid: bool
    conversation_summary: str
    thread_id: str | None  # ADD THIS
```

#### File 4: `src/flow_agent/utils/nodes.py` (MODIFY)

Modify `entry_node` function:

```python
from src.flow_agent.utils.rate_limiter import check_rate_limit

async def entry_node(state: State):
    messages: list[BaseMessage] = state.get("messages")

    # ===== ADD RATE LIMIT CHECK =====
    # Extract identifier from state (thread_id)
    thread_id = state.get("thread_id", None)
    identifier = thread_id or "unknown"

    # Check rate limit
    allowed, error_msg = check_rate_limit(identifier)
    if not allowed:
        # Reject request
        state["input_valid"] = False
        state["issue"] = error_msg
        return state
    # ===== END RATE LIMIT CHECK =====

    pii_redactor = PII_Redactor()
    redacted_message = await pii_redactor.do_pii_redaction([messages[-1]])
    # ... rest of function unchanged ...
```

**Usage**: Rejects requests from users who exceed threshold before processing.

---

### Option 3: LangGraph API Middleware

**Scope**: LangGraph API endpoint rate limiting

**Files to Change**: 1
**Lines Added**: ~5
**Works Locally**: ❌ No (LangGraph Cloud only)

#### File 1: `langgraph.json` (MODIFY)

**Before:**
```json
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": [...],
  "graphs": {
    "agent": "./src/flow_agent/graph.py:graph"
  },
  "env": ".env",
  "image_distro": "wolfi"
}
```

**After:**
```json
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": [...],
  "graphs": {
    "agent": "./src/flow_agent/graph.py:graph"
  },
  "env": ".env",
  "image_distro": "wolfi",
  "rate_limits": {
    "agent": {
      "requests_per_minute": 60,
      "concurrent_requests": 5
    }
  }
}
```

**Usage**: Platform-managed rate limiting. Only works with LangGraph Cloud deployment.

---

## Comparison Table

| Aspect | Option 1 (LangChain) | Option 2 (Node-Level) | Option 3 (LangGraph Cloud) |
|---------|---------------------|----------------------|-------------------------|
| **Files to create** | 1 | 1 | 0 |
| **Files to modify** | 2 | 2-3 | 1 |
| **Total lines added** | ~20 | ~60 | ~5 |
| **Where it limits** | LLM API calls | Graph entry (before LLM) | LangGraph API endpoint |
| **Scope** | Global (all LLM calls) | Per-identifier (thread/user) | Per-graph (global) |
| **Works with local** | ✅ Yes | ✅ Yes | ❌ No (Cloud only) |
| **Granular control** | Per-second | Per-request window | Per-minute |
| **Custom logic** | No | Yes (in code) | No |

---

## Recommendation

**Use Option 1 (LangChain Rate Limiter)** if you:
- ✅ Want the simplest solution
- ✅ Just need to prevent LLM API throttling
- ✅ Don't need per-user limits
- ✅ Running locally (`langgraph up`)

**Use Option 2 (Node-Level)** if you:
- ✅ Need per-user/per-thread rate limiting
- ✅ Want to reject requests early (before LLM processing)
- ✅ Have custom rate limiting logic
- ✅ Running locally

**Use Option 3 (LangGraph Cloud)** if you:
- ✅ Deploying to LangGraph Cloud (not local)
- ✅ Want platform-managed rate limits
- ✅ Need concurrent request limits

---

## Implementation Steps (Option 1 - Recommended)

1. Create `src/flow_agent/utils/rate_limiter.py`
2. Add `RATE_LIMIT_REQUESTS_PER_SECOND` to `src/flow_agent/configurations/config.py`
3. Modify `src/flow_agent/llms/LangChainChatLLM.py` to wrap LLM with rate limiter
4. Test: Run agent and verify LLM calls respect rate limit
5. Adjust `RATE_LIMIT_REQUESTS_PER_SECOND` based on your LLM provider limits

---

## Testing

```bash
# Test rate limiting by making multiple rapid requests
# Option 1 & 2: Should see requests throttled
# Option 3: Works only on LangGraph Cloud

pytest tests/ -k rate_limit -v
```

---

## Configuration Examples

### OpenAI Rate Limits
- **GPT-4o**: 10,000 TPM (tokens per minute), ~200-300 requests/minute
- **GPT-4o-mini**: Higher limits, ~500 requests/minute
- **Recommended setting**: 2-5 requests/second (120-300/minute)

### Google Gemini Rate Limits
- **Free tier**: 15 requests/minute
- **Paid tier**: 360 requests/minute
- **Recommended setting**: 0.25-6 requests/second

### SarvamAI Rate Limits
- Check dashboard for specific limits
- **Recommended setting**: Start with 1 request/second, adjust as needed

---

## Notes

- **Option 1 is thread-safe**: Uses LangChain's built-in asyncio support
- **Option 2 is thread-safe**: Uses threading.Lock for concurrent access
- **Option 2 persists in memory only**: Rate limit resets on restart
- For production, consider Redis-backed rate limiting (Option 2 extended)

---

## Selected Option

**Recommended**: Option 1 (LangChain Rate Limiter)

**Reason**: Minimal changes, works locally, effective for preventing LLM API throttling.
