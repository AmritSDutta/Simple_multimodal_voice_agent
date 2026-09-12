# Plan: Per-Provider Circuit Breaker Implementation

## Problem Statement
Currently, a single global `CircuitBreaker` instance is shared across all LLM providers (gemini, openai, ollama, zhipu, sarvam). If one provider fails 5 times, **all providers** get blocked for 30 seconds. This is incorrect behavior - failures in Ollama should not affect Gemini.

## Solution Overview
Implement per-provider circuit breakers using a simple dictionary pattern with dynamic decorator factory.

---

## Implementation Steps

### Step 1: Create Per-Provider Breakers Dictionary
**File:** `src/flow_agent/utils/circuit_breaker_llm.py`

Replace single global `breaker` with a dictionary:

```python
from aiobreaker import CircuitBreaker
from datetime import timedelta

# Per-provider circuit breakers
BREAKERS = {
    "openai": CircuitBreaker(fail_max=5, timeout_duration=timedelta(seconds=30)),
    "gemini": CircuitBreaker(fail_max=5, timeout_duration=timedelta(seconds=30)),
    "zhipu":  CircuitBreaker(fail_max=5, timeout_duration=timedelta(seconds=30)),
    "sarvam": CircuitBreaker(fail_max=5, timeout_duration=timedelta(seconds=30)),
    "ollama": CircuitBreaker(fail_max=5, timeout_duration=timedelta(seconds=30)),
}

def get_breaker(provider: str) -> CircuitBreaker:
    """Get circuit breaker for a provider."""
    return BREAKERS[provider.lower()]
```

### Step 2: Create Dynamic Call Function Factory
**File:** `src/flow_agent/utils/circuit_breaker_llm.py`

Add factory function that creates decorated call functions per provider:

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from aiobreaker import CircuitBreakerError

async def _should_retry(exc: BaseException) -> bool:
    """Retry everything EXCEPT when circuit is open."""
    return not isinstance(exc, CircuitBreakerError)

def make_call_fn(provider: str):
    """Factory that creates a breaker-protected call function for a provider."""
    breaker = get_breaker(provider)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    @breaker
    async def _call_llm(llm, messages):
        return await llm.ainvoke(messages)

    return _call_llm
```

### Step 3: Modify `call_llm_safely` to Use Provider Parameter
**File:** `src/flow_agent/utils/circuit_breaker_llm.py`

Update signature to accept provider and use dynamic call function:

```python
async def call_llm_safely(
    provider: str,  # NEW: Provider identifier
    llm: BaseChatModel | Runnable,
    conversation: List[BaseMessage],
    new_message: HumanMessage
) -> Any:
    """
    Safe LLM call with per-provider circuit breaker and retry.

    Args:
        provider: Provider identifier (e.g., "gemini", "openai")
        llm: LLM instance
        conversation: Previous messages
        new_message: New human message
    """
    full_context = conversation + [new_message]

    # Get provider-specific call function
    call_primary = make_call_fn(provider)

    try:
        response = await call_primary(llm, full_context)
        logging.info(f"[{provider}] response metadata: {response.response_metadata}")
        logging.info(f"[{provider}] usage metadata: {response.usage_metadata}")
        return response
    except Exception as e:
        logging.error(f"[{provider}] Primary failed: {e}. Trying fallback...")
        try:
            fb_provider = settings.FALLBACK_PROVIDER_IDENTIFIER
            fb_llm = await get_chat_llm(fb_provider)
            response = await fb_llm.ainvoke(full_context)
            logging.info(f"[{fb_provider}] Fallback response metadata: {response.response_metadata}")
            return response
        except Exception as ae:
            logging.error(f"[{fb_provider}] Fallback also failed: {ae}")
            raise ae
```

**Key Changes:**
- Added `provider: str` parameter
- Use `make_call_fn(provider)` instead of global `@breaker` decorator
- Enhanced logging with provider context `[gemini]`, `[ollama]`, etc.

### Step 4: Update Node Callers to Pass Provider
**File:** `src/flow_agent/utils/nodes.py`

Update reasoning and summarizer nodes:

**Before:**
```python
llm = await get_chat_llm()  # Returns only LLM
response = await call_llm_safely(llm, messages, multimodal_msg)
```

**After:**
```python
llm, provider = await get_chat_llm()  # Get both LLM and provider
response = await call_llm_safely(provider, llm, messages, multimodal_msg)
```

**In `call_langchain_reasoning_model` (around line 119):**
```python
llm, provider = await get_chat_llm()
# ... existing code ...
response = await call_llm_safely(provider, llm, messages, multimodal_msg)
```

**In `call_langchain_summarizer` (around line 313):**
```python
llm, provider = await get_chat_llm(is_summarizer=True)
response = await call_llm_safely(provider, llm, [], HumanMessage(content=summary_prompt))
```

### Step 5: Update `get_chat_llm` Return Type
**File:** `src/flow_agent/llms/LangChainChatLLM.py`

**No changes needed!** `get_vision_models()` and `get_summarization_models()` already return `(llm, provider)` tuples.

Just verify:
```python
async def get_vision_models(provider: str | None) -> tuple[BaseChatModel | None, Any]:
    # ... existing logic ...
    return llm, provider  # Already returns tuple!
```

### Step 6: Add Configuration (Optional)
**File:** `src/flow_agent/configurations/config.py`

Add breaker settings for tunability (optional but recommended):

```python
# Circuit Breaker Settings
MAX_TRY: int = 3
SLEEP_IN_SECONDS: int = 1
CIRCUIT_BREAKER_FAIL_MAX: int = 5        # NEW
CIRCUIT_BREAKER_TIMEOUT_SECONDS: int = 30   # NEW
```

Then use in breakers dict:
```python
BREAKERS = {
    "openai": CircuitBreaker(
        fail_max=settings.CIRCUIT_BREAKER_FAIL_MAX,
        timeout_duration=timedelta(seconds=settings.CIRCUIT_BREAKER_TIMEOUT_SECONDS)
    ),
    # ... other providers
}
```

### Step 7: Add Provider Alias Normalization
**File:** `src/flow_agent/utils/circuit_breaker_llm.py`

Handle aliases (from factory pattern):

```python
def normalize_provider(provider: str) -> str:
    """Normalize provider aliases to canonical names."""
    mapping = {
        "sarvamai": "sarvam",
        "sarvam_ai": "sarvam",
        "zai": "sarvam",
        "genai": "gemini",
        "gen_ai": "gemini",
        "google": "gemini",
    }
    return mapping.get(provider.lower(), provider.lower())

def get_breaker(provider: str) -> CircuitBreaker:
    """Get circuit breaker for a provider."""
    return BREAKERS[normalize_provider(provider)]
```

### Step 8: Update Tests
**File:** `tests/unit_tests/test_circuit_breaker_llm.py`

Add tests for per-provider isolation:

```python
class TestPerProviderBreakers:
    """Test per-provider circuit breaker isolation."""

    @pytest.mark.asyncio
    async def test_ollama_failure_does_not_affect_gemini(self):
        """Verify breaker isolation between providers."""
        # Simulate 5 failures on ollama
        ollama_breaker = get_breaker("ollama")
        gemini_breaker = get_breaker("gemini")

        for _ in range(5):
            try:
                await ollama_breaker.call_async(lambda: exec("raise Exception()"))
            except:
                pass

        # Ollama should be open, Gemini should be closed
        assert ollama_breaker.open()
        assert not gemini_breaker.open()

    @pytest.mark.asyncio
    async def test_get_breaker_normalizes_aliases(self):
        """Test provider alias normalization."""
        sarvam_breaker = get_breaker("sarvamai")
        assert sarvam_breaker is get_breaker("sarvam")
```

---

## Critical Files to Modify

| File | Changes |
|-------|----------|
| `src/flow_agent/utils/circuit_breaker_llm.py` | Add BREAKERS dict, get_breaker(), make_call_fn(), update call_llm_safely signature |
| `src/flow_agent/utils/nodes.py` | Update 2 callers to pass provider parameter |
| `src/flow_agent/configurations/config.py` | Add CIRCUIT_BREAKER_FAIL_MAX, CIRCUIT_BREAKER_TIMEOUT_SECONDS (optional) |
| `tests/unit_tests/test_circuit_breaker_llm.py` | Add isolation tests |

**No changes needed to:**
- `src/flow_agent/llms/LangChainChatLLM.py` - Already returns (llm, provider) tuples

---

## Verification Steps

### 1. Unit Tests
```bash
pytest tests/unit_tests/test_circuit_breaker_llm.py -v
```
Expected: New isolation tests pass

### 2. Integration Test
```bash
pytest tests/end_to_end/test_graph.py::test_graph_flow_text_only -v -s
```
Check logs contain provider-specific breaker usage:
```
INFO | [gemini] response metadata: {...}
INFO | [gemini] usage metadata: {...}
```

### 3. Manual Isolation Test
- Run agent with provider distribution
- Trigger failures on one provider (e.g., stop Ollama service)
- Verify other providers continue working
- Check logs show correct breaker states

---

## Semantics & Behavior

* Each provider has **independent** `CLOSED/OPEN/HALF_OPEN` state
* If **Zhipu** is down → only Zhipu trips open; **OpenAI/Gemini keep working**
* When a provider is **OPEN** → calls **fail fast** and go straight to fallback
* Tenacity **retries normal failures** (2-3 attempts), but **does not retry** when CB is OPEN
* Fallback provider uses its **own breaker** (not shared)

---

## Tuning Recommendations

| Setting | Range | Default | Notes |
|----------|-------|---------|--------|
| `fail_max` | 3-5 | 5 is standard; use 3 for faster recovery |
| `timeout_seconds` | 20-60 | 30s standard; increase for flaky networks |
| retry attempts | 2-3 | Keep low to avoid hammering degraded backend |

---

## Success Criteria

- ✅ Each provider has independent circuit breaker
- ✅ Failure in Ollama does NOT affect Gemini/OpenAI/Zhipu
- ✅ Logs show provider context: `[gemini]`, `[ollama]`
- ✅ Fallback mechanism still works
- ✅ Existing tests continue to pass
- ✅ New isolation tests verify per-provider behavior
