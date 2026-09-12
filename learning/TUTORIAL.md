# Multimodal Voice Agent — Comprehensive Tech Stack Tutorial

A section-per-tool guide to the `Simple_multimodal_voice_agent` codebase. Every file path, class name, function signature, and default value in this tutorial was verified against the actual source tree — not copied from older notes.

---

## How to read this tutorial

The project actually contains **two independent agent implementations** that share some building blocks:

1. A **LangGraph multimodal agent** (`src/flow_agent/`) with a Streamlit web UI (`ui/`). Text + image input, optional audio, security layers, multi-provider LLMs.
2. A **LiveKit real-time voice agent** (`src/livekit_*.py`). Speech-in / speech-out, tool calling, long-conversation context management.

Sections are grouped by concern. Each section follows the same shape:

> **What it is → Why the project uses it → Where it lives → Key APIs → Wired-in example → Pitfalls**

Sections marked **📝 Documented / not implemented** describe work that exists only in the planning docs under `learning/`, not in the code. They are included because the source docs cover them, but do not treat them as runnable.

A **known drift** appendix at the end lists every place where the older docs disagree with the code, so you can trust the code and ignore the stale claims.

---

## 0. Project at a glance

**Python:** `>=3.10` (`pyproject.toml`)
**Package name:** `simple_multimodal_voice_agent`
**License:** MIT

### The LangGraph pipeline

```
START
  │
  ▼
entry              # PII redaction on the incoming message
  │
  ▼
input_validator    # pattern scan + OpenAI moderation  (gated by IS_INPUT_VALIDATION_ENABLED)
  │
  ├── input_valid == False ──────────────────────────► END
  │
  ▼
reasoning          # call_langchain_reasoning_model  OR  call_gemini_reasoning_model
  │                # (chosen by REASONING_NODE_PREFERENCE)
  │
  ▼
should_summarize   # len(messages) >= SUMMARY_MESSAGE_THRESHOLD ?
  │
  ├── "summarizer" ► summarizer ─────────────────────► END
  │
  └── END
```

Defined in `src/flow_agent/graph.py`. See [Section 1](#1-langgraph) for the exact edge wiring.

### Quick start (LangGraph agent)

```bash
# 1. Install
uv sync                 # or: pip install -e .

# 2. spaCy model needed by Presidio PII redaction
python -m spacy download en_core_web_sm

# 3. Configure
cp .env.example .env    # then fill in the keys you use

# 4. Backend (Terminal 1)
langgraph up --watch     # serves the API on http://localhost:2024

# 5. Web UI (Terminal 2)
streamlit run ui/app.py  # http://localhost:8501
```

### Quick start (LiveKit voice agent)

```bash
# Terminal 1 — agent worker
python -m src.livekit_context_optimized dev

# Terminal 2 — voice console client
python -m src.livekit_context_optimized console
```

Full details in [Section 11](#11-livekit-agents).

### Environment variables

From `.env.example` plus the defaults in `src/flow_agent/configurations/config.py`:

| Variable | Used by | Notes |
|---|---|---|
| `ZAI_API_KEY` | Zhipu/GLM provider (`ChatOpenAI` → `api.z.ai`) | read via `settings.ZHIPU_KEY_STRING` |
| `OLLAMA_API_KEY` | Ollama provider | required if the Ollama provider is selected |
| `SARVAM_API_KEY` | Sarvam speech + `SarvamChat` summarizer | required for Sarvam STT/TTS |
| `OPENAI_API_KEY` | OpenAI provider + moderation API | moderation is on by default |
| `GOOGLE_API_KEY` | Gemini provider | `ChatGoogleGenerativeAI` |
| `GEMINI_API_KEY` | LiveKit agents using `google.LLM` | separate from `GOOGLE_API_KEY` |
| `GROQ_API_KEY` | LiveKit STT (`groq.STT`) and some LLM variants | LiveKit only |
| `TAVILY_API_KEY` | `TavilySearch` in the LiveKit agents | LiveKit only |
| `ARIZE_SPACE_ID` / `ARIZE_API_KEY` | Arize tracing | only if `ARIZE_TRACING_ENABLED=true` |
| `CONTEXT_SUMMARIZE_THRESHOLD_ITEMS` | LiveKit context manager | code default `10` |
| `CONTEXT_KEEP_LAST_N_TURNS` | LiveKit context manager | code default `3` |
| `CONTEXT_SUMMARIZE_EVERY_N_TURNS` | LiveKit context manager | code default `5` |

> ⚠️ The `CONTEXT_*` defaults in the **code** are `10 / 3 / 5`. Older docs claim `20 / 10 / 15`. The code wins.

---

# Part 1 — Core Orchestration

## 1. LangGraph

### What it is
LangGraph models an agent as a **state machine**: a `StateGraph` of nodes (functions) connected by edges, with a typed shared state that flows between them.

### Why the project uses it
The agent needs conditional routing (reject unsafe input, decide whether to summarize) and multi-turn message accumulation. LangGraph gives both with an explicit, inspectable graph that LangGraph Platform can serve as an API.

### Where it lives
- `src/flow_agent/graph.py` — graph definition
- `src/flow_agent/utils/state.py` — the shared `State`
- `src/flow_agent/utils/nodes.py` — node implementations
- `langgraph.json` — deployment manifest

### Key APIs

**The state** (`src/flow_agent/utils/state.py`):

```python
class State(TypedDict):
    retry_count: Annotated[int, add]
    messages: Annotated[list[BaseMessage], add_messages]
    issue: str
    final_report: str
    input_valid: bool
    conversation_summary: str
```

`Annotated[..., add_messages]` is a **reducer**: when a node returns `{"messages": [...]}`, LangGraph appends rather than overwrites. `Annotated[int, add]` sums. To bypass a reducer and replace the whole list, wrap the value in `Overwrite` (see the summarizer below).

**The graph** (`src/flow_agent/graph.py`):

```python
class Context(TypedDict):
    my_configurable_param: str

graph = (
    StateGraph(State, context_schema=Context)
    .add_node("entry", entry_node)
    .add_node("input_validator", call_input_validation)
    .add_node("reasoning",
              call_langchain_reasoning_model if settings.REASONING_NODE_PREFERENCE == 'langchain'
              else call_gemini_reasoning_model)
    .add_node("summarizer", call_langchain_summarizer)
    .add_edge(START, "entry")
    .add_edge("entry", "input_validator")
    .add_conditional_edges("input_validator", route_after_validation, {"reasoning": "reasoning", END: END})
    .add_conditional_edges("reasoning", should_summarize, {"summarizer": "summarizer", END: END})
    .add_edge("summarizer", END)
)
```

Note `graph` is a `StateGraph`, **not** compiled. Callers compile it themselves:

```python
compiled_graph = graph.compile()   # see tests/end_to_end/arize_evals/test_vision_eval.py
```

**Node signature.** Nodes that need runtime context take `(state, runtime)`:

```python
async def call_langchain_reasoning_model(state: State, runtime: Runtime[Context]) -> State:
    ...
```

`Runtime` comes from `langgraph.runtime`; the graph-level `Context` schema is imported from `langgraph_api.schema`.

**Routing functions** are plain functions returning a node name or `END`:

```python
async def route_after_validation(state: State) -> str:
    return "reasoning" if state["input_valid"] else END

async def should_summarize(state: State) -> str:
    message_count = len(state.get("messages", []))
    if message_count >= settings.SUMMARY_MESSAGE_THRESHOLD:
        return "summarizer"
    return END
```

**Replacing state past a reducer** — the summarizer collapses history and must overwrite, not append:

```python
from langgraph.types import Overwrite

return {
    "messages": Overwrite(new_messages),
    "conversation_summary": summary,
}
```

**Deployment manifest** (`langgraph.json`):

```json
{
  "dependencies": ["langgraph>=1.0.0", "...", "."],
  "graphs": { "agent": "./src/flow_agent/graph.py:graph" },
  "env": ".env",
  "image_distro": "wolfi"
}
```

Dependencies here are duplicated from `pyproject.toml` on purpose — the Docker build reads `langgraph.json`, not `pyproject.toml`.

### Wired-in example
`langgraph up --watch` serves the graph in `langgraph.json` on `http://localhost:2024`. The Streamlit UI talks to it over REST (see [Section 16](#16-streamlit-ui)).

### Pitfalls
- **`graph` is uncompiled.** `langgraph.json` points at it directly, which is correct, but in-process callers must call `.compile()`.
- **`langgraph_api.schema.Context`** is imported in `nodes.py` — this ties node code to the LangGraph API package being installed.
- **Env vars are set at import time** in `graph.py` (`LANGSMITH_TRACING_V2`, `LANGSMITH_PROJECT`), so importing the graph has side effects.
- Docker builds ignore `pyproject.toml`; keep both dependency lists in sync.

---

## 2. LangChain

### What it is
A provider-agnostic abstraction over chat models, messages, tools, and runnables. `BaseChatModel` is the common interface the project programs against.

### Why the project uses it
It lets one code path talk to OpenAI, Google, Zhipu, Ollama, Sarvam, and Mistral just by swapping the model object — the backbone of the weighted multi-provider strategy.

### Where it lives
- `src/flow_agent/llms/LangChainChatLLM.py` — provider construction
- `src/flow_agent/utils/nodes.py` — message construction and multimodal input
- `src/flow_agent/utils/circuit_breaker_llm.py` — `llm.ainvoke(...)`

### Key APIs

**Messages.** `HumanMessage`, `AIMessage`, `BaseMessage` from `langchain_core.messages`. The graph appends `AIMessage` replies through the `add_messages` reducer.

**The multimodal content format.** This is the shape the whole pipeline agrees on:

```python
HumanMessage(content=[
    {"type": "text", "text": "what is there in the image"},
    {
        "type": "image",
        "data": "<base64>",
        "metadata": {"filename": "olap.png"},
        "source_type": "base64",
        "mime_type": "image/png",
    },
])
```

Audio and video use the same envelope with `"type": "audio"` / `"type": "video"`. `entry_node` normalizes a bare string into `[{"type": "text", "text": ...}]`.

**Translating to a provider call.** `prepare_llm_input` converts that envelope into a data-URI `image_url` part and enforces `MAX_IMAGES_PER_REQUEST`:

```python
async def prepare_llm_input(text_prompt: str, media_b64s: list[Any] | None) -> list[dict]:
    message_content = [{"type": "text", "text": text_prompt}]
    if media_b64s:
        for b64_image in media_b64s[:settings.MAX_IMAGES_PER_REQUEST]:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
            })
    return message_content
```

**Built-in web search.** `nodes.py` enriches the prompt with DuckDuckGo results via LangChain Community:

```python
from langchain_community.tools import DuckDuckGoSearchResults
_tool = DuckDuckGoSearchResults(output_format="string", num_results=5)
```

`_enhance_with_tool_search` **skips search when an image is present** (`if mime_type: return searchable_text`).

**Rate limiting** (`InMemoryRateLimiter`) is described in `learning/RATE_LIMIT_OPTION.md` but not wired up — see [Section 6](#6-rate-limiting--documented--not-implemented).

### Pitfalls
- `prepare_llm_input` hardcodes `image/jpeg` in the data URI regardless of the incoming `mime_type`.
- Message content can be `str` **or** `list`; many bugs here come from assuming one shape. Guard with `isinstance(content, list)`.
- The `add_messages` reducer means a node returning `messages` *adds* — use `Overwrite` to replace.

---

# Part 2 — Models, Resilience & Configuration

## 3. Multi-Provider LLM Layer

### What it is
A weighted-random router that picks an LLM provider per request and constructs the right LangChain chat model.

### Why the project uses it
Cost optimization, load spreading, and provider diversity — plus a fallback if the chosen provider fails.

### Where it lives
`src/flow_agent/llms/LangChainChatLLM.py`

### Key APIs

```python
async def get_chat_llm(provider: str | None = None, is_summarizer: bool = False) -> BaseChatModel | Runnable
async def get_vision_models(provider: str | None) -> tuple[BaseChatModel | None, Any]
async def get_summarization_models(provider: str | None) -> tuple[BaseChatModel | None, Any]
```

`get_chat_llm` is the public entry point; it **returns only the model** (it unpacks and discards the provider tuple internally). The two `get_*_models` helpers return `(llm, provider)`.

Selection is random by weight, seeded for reproducibility:

```python
random.seed(1234)

def _get_random_provider():
    names = list(settings.VISION_PROVIDER_DISTRIBUTION.keys())
    weights = list(settings.VISION_PROVIDER_DISTRIBUTION.values())
    return random.choices(names, weights=weights, k=1)[0]
```

**The distributions** (`config.py`):

```python
VISION_PROVIDER_DISTRIBUTION = {"gemini": 0.7, "openai": 0.01, "zhipu": 0.1, "ollama": 0.2}

SUMMARIZATION_PROVIDER_DISTRIBUTION = {
    "mistral": 0.59, "ollama": 0.01, "sarvam": 0.2,
    "gemini": 0.1, "zhipu": 0.09, "openai": 0.01,
}
```

**Provider → model mapping** (vision path):

| Provider | Class | Model setting | Default |
|---|---|---|---|
| `gemini` | `ChatGoogleGenerativeAI` | `GEMINI_VISION_MODEL` | `gemini-2.5-flash-lite` |
| `openai` | `ChatOpenAI` | `OPENAI_VISION_MODEL` | `gpt-5-nano` |
| `zhipu` | `ChatOpenAI` (base `https://api.z.ai/api/paas/v4/`) | `ZHIPU_VISION_MODEL` | `GLM-4.6V-Flash` |
| `ollama` | `ChatOllama` (base `https://ollama.com`) | `OLLAMA_VISION_MODEL` | `qwen3-vl:235b-instruct-cloud` |

The summarization path adds two more:

| Provider | Class | Model setting | Default |
|---|---|---|---|
| `mistral` | `ChatMistralAI` | `MISTRAL_SUMMARIZATION_MODEL` | `mistral-medium-2508` |
| `sarvam` | `SarvamChat(reasoning_effort='low')` | — | — |
| `zhipu` | `ChatOpenAI` | `ZHIPU_SUMMARIZATION_MODEL` | `GLM-4.7-Flash` |
| `ollama` | `ChatOllama` | `OLLAMA_SUMMARIZATION_MODEL` | `nemotron-3-nano:30b-cloud` |

Note the **Zhipu providers are `ChatOpenAI` pointed at a different base URL and key** — a common trick when a vendor ships an OpenAI-compatible API.

### Pitfalls
- `get_chat_llm` returns a model, not `(model, provider)`. The circuit-breaker fallback relies on this and calls `.ainvoke` directly.
- Ollama requires `OLLAMA_API_KEY`; the code raises `ValueError` if it's unset **even when Ollama wasn't explicitly requested** (it's in the random distribution).
- `random.seed(1234)` makes distribution deterministic across runs — good for tests, less so for real load spreading.
- `src/flow_agent/llms/sarvam_openai.py` is a **standalone script, not a module**: it imports `flow_agent...` (missing the `src.` prefix) and fires an API call at import time. Don't import it.

---

## 4. Circuit Breaking & Retries (aiobreaker + Tenacity)

### What it is
Two stacked resilience layers: `aiobreaker` opens a circuit after repeated failures, and `tenacity` retries with exponential backoff — but *not* when the circuit is already open.

### Why the project uses it
LLM providers fail intermittently. Retrying smooths over blips; the circuit breaker stops hammering a provider that's clearly down.

### Where it lives
`src/flow_agent/utils/circuit_breaker_llm.py`

### Key APIs

```python
breaker = CircuitBreaker(fail_max=5, timeout_duration=timedelta(seconds=30))

async def _should_retry(exc: BaseException) -> bool:
    return not isinstance(exc, CircuitBreakerError)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception(_should_retry),
    reraise=True,
)
@breaker
async def _call_llm(llm, conversation):
    return await llm.ainvoke(conversation)

async def call_llm_safely(llm, conversation, new_message) -> Any:
    full_context = conversation + [new_message]
    try:
        response = await _call_llm(llm, full_context)
        return response
    except Exception as e:
        llm = await get_chat_llm(settings.FALLBACK_PROVIDER_IDENTIFIER)
        return await llm.ainvoke(full_context)
```

Decorator order matters: `@retry` wraps `@breaker`, so a retry is attempted **before** the breaker counts a failure, and an open circuit fails fast without consuming retries.

### Pitfalls
- 🔴 **The breaker is global.** One module-level instance is shared by every provider, so 5 failures on Ollama also block Gemini for 30 seconds. `learning/CIRCUIT_BREAKER_REFACTOR_PLAN.md` designs per-provider breakers — **it is not implemented.**
- `call_llm_safely` takes **three** arguments (`llm, conversation, new_message`). The refactor plan's `provider`-first signature does not exist.
- Fallback always jumps to `FALLBACK_PROVIDER_IDENTIFIER` (`gemini` by default), ignoring the weighted distribution.
- `from tenacity.asyncio import retry_if_exception` is imported, which is an unusual import path — verify it resolves in your environment.
- No concurrency guard: the shared breaker's behaviour under simultaneous async calls is untested.

---

## 5. Configuration with Pydantic Settings

### What it is
`pydantic-settings` loads typed configuration from environment variables and `.env`.

### Where it lives
`src/flow_agent/configurations/config.py`

### Key APIs

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,   # env names must match exactly
        extra="ignore",
    )
    ...

settings = Settings()          # module-level singleton imported everywhere
```

Notable settings groups:

```python
# Circuit breaker
MAX_TRY = 3
SLEEP_IN_SECONDS = 1

# Models
GEMINI_VISION_MODEL = 'gemini-2.5-flash-lite'
OPENAI_VISION_MODEL = 'gpt-5-nano'
ZHIPU_VISION_MODEL = 'GLM-4.6V-Flash'
OLLAMA_VISION_MODEL = 'qwen3-vl:235b-instruct-cloud'

# Routing
REASONING_NODE_PREFERENCE = 'langchain'       # 'langchain' | 'genai'
FALLBACK_PROVIDER_IDENTIFIER = 'gemini'

# Safety
MODERATION_API_CHECK_REQ = True
MODERATION_MODEL = 'omni-moderation-latest'
IS_PII_REDACTION_ENABLED = False              # disabled by default
IS_INPUT_VALIDATION_ENABLED = False           # disabled by default

# Speech
SPEECH_PROVIDER = "sarvam"

# Summarization
SUMMARY_MESSAGE_THRESHOLD = 10
SUMMARY_PROVIDER_PREFERENCE = 'langchain'

# Tracing
ENABLE_LANGSMITH_TRACING_V2 = "false"
TRACING_PROJECT_NAME = 'multimodal_voice_agent'
ARIZE_TRACING_ENABLED = False

MAX_IMAGES_PER_REQUEST = 2
PII_CONFIDENCE_THRESHOLD = 0.5
```

Because `case_sensitive=True`, an env var must be spelled exactly as the field (`SARVAM_STT_MODEL`, not `sarvam_stt_model`). `extra="ignore"` means unrelated env vars don't raise.

### Pitfalls
- **Importing the singleton everywhere makes testing painful.** Tests must patch `settings` in *each* module that imported it. See the three-way patch in `tests/end_to_end/arize_evals/test_vision_eval.py`:

  ```python
  with patch("src.flow_agent.graph.settings", custom_settings_with_gemma_3_12b), \
       patch("src.flow_agent.utils.nodes.settings", custom_settings_with_gemma_3_12b), \
       patch("src.flow_agent.llms.LangChainChatLLM.settings", custom_settings_with_gemma_3_12b):
  ```

  This is the "settings patching anti-pattern" called out in `learning/ARCHITECTURE_REVIEW.md`. Passing `Settings` as a parameter, or reading it through a `get_settings()` accessor, would remove the need.
- **Security defaults are off.** `IS_PII_REDACTION_ENABLED` and `IS_INPUT_VALIDATION_ENABLED` both default to `False`, so the safety layers are opt-in.
- The `Settings()` singleton is constructed at import time, so `.env` must exist (or all values must have defaults) before import succeeds.

---

## 6. Rate Limiting — 📝 Documented / not implemented

### What it is
`learning/RATE_LIMIT_OPTION.md` proposes three ways to throttle requests. **None is implemented** — there is no `rate_limiter.py` in the source tree.

### The three options compared

| Option | Where it limits | Works locally | Effort |
|---|---|---|---|
| 1. LangChain `InMemoryRateLimiter` | Global LLM API calls | ✅ | ~20 lines |
| 2. Node-level `SimpleRateLimiter` | Per thread/user at graph entry | ✅ | ~60 lines |
| 3. LangGraph Cloud `rate_limits` in `langgraph.json` | API endpoint | ❌ Cloud only | ~5 lines |

### Documented recommendation
Option 1 — wrap each model with `llm.bind(rate_limiter=InMemoryRateLimiter(requests_per_second=...))` inside `LangChainChatLLM.py`.

### If you implement it
Add a `RATE_LIMIT_REQUESTS_PER_SECOND` field to `Settings`, then wrap models at construction time. Option 2 additionally needs `thread_id` on `State` (`src/flow_agent/utils/state.py` currently has no such field) and a check in `entry_node`.

---

# Part 3 — Safety

## 7. Input Validation & OpenAI Moderation

### What it is
A two-stage gate: fast regex pattern matching for attack signatures, then an OpenAI moderation call for harmful content (text **and** images, using `omni-moderation-latest`).

### Why the project uses it
Defense in depth — cheap pattern screening catches obvious injection attempts before spending money on the moderation API.

### Where it lives
`src/flow_agent/utils/input_validation.py`, called from `call_input_validation` in `nodes.py`.

### Key APIs

```python
async def scan_for_vulnerability(user_input: HumanMessage | str) -> bool
async def get_moderation_api_feedback_on_input(user_input: BaseMessage | str) -> bool
```

`scan_for_vulnerability` returns `True` when input is **safe**. It lowercases and stringifies the message, then tests it against compiled regexes in `COMPILED_PATTERNS`, grouped by category:

```python
MALICIOUS_PATTERNS = {
    "shell_injection": [...],
    "docker_abuse": [...],
    "sql_injection": [...],
    "path_traversal": [...],
    "system_commands": [...],
    "code_execution": [...],
    "malicious_keywords": ["malware", "virus", "ransomware", "backdoor", ...],
    "windows_abuse": [...],   # powershell -enc, vssadmin delete shadows, rundll32, ...
}
```

Patterns are pre-compiled at import for speed (`re.compile(p, re.IGNORECASE)`) — except `malicious_keywords`, which is handled separately.

The moderation stage only runs when `settings.MODERATION_API_CHECK_REQ` is `True`:

```python
response = await client.moderations.create(model=settings.MODERATION_MODEL, input=prompt)
for res in response.results:
    if res.flagged:
        return False
```

Note that on **any exception** the moderation function returns `False` (fail-closed), so an OpenAI outage will block input when validation is enabled.

### Wiring
`call_input_validation` short-circuits when the feature is off:

```python
if not settings.IS_INPUT_VALIDATION_ENABLED:
    return {"input_valid": True}
```

When enabled and input is unsafe it returns `{"input_valid": False}` **plus** an `AIMessage`, and `route_after_validation` sends the run to `END`.

### Pitfalls
- **Off by default.** Set `IS_INPUT_VALIDATION_ENABLED=true` to activate.
- The image branch looks for `item.get("type") == "image_url"`, but the project's canonical content type is `"image"`. Images may bypass moderation — worth verifying before trusting it for image safety.
- Patterns are hardcoded strings; `ARCHITECTURE_REVIEW.md` recommends moving them to config.
- Fail-closed on API errors can cause confusing mass rejections during an OpenAI incident.

---

## 8. PII Redaction with Microsoft Presidio

### What it is
Presidio's `AnalyzerEngine` detects PII, and `AnonymizerEngine` replaces it. It runs twice: once on user input in `entry_node`, once on the model's response in `process_response`.

### Why the project uses it
Privacy protection for a voice/chat agent that may handle names, emails, phone numbers, etc.

### Where it lives
`src/flow_agent/utils/pii_redaction.py`

### Key APIs

```python
class PII_Redactor:
    def __init__(self, confidence_threshold: float = settings.PII_CONFIDENCE_THRESHOLD):
        self.analyzer = AnalyzerEngine(supported_languages=["en"])
        self.anonymizer = AnonymizerEngine()

    async def do_pii_redaction(self, messages: List[BaseMessage]) -> List[BaseMessage]
```

Internals:
- `_is_pii_data_detected(text)` runs `analyzer.analyze(...)` in a thread (`asyncio.to_thread`) and returns detected entities plus whether any score ≥ threshold.
- `_sanitize(text, analysis)` runs `anonymizer.anonymize(...)` in a thread.
- Handles both `str` content and list content, redacting only `{"type": "text"}` items — images are never touched.
- Copies messages with `message.model_copy()` to avoid mutating the original.

```python
async def _do_pii_redaction_on_message(self, message: BaseMessage) -> BaseMessage:
    new_message = message.model_copy()
    content = new_message.content
    if isinstance(content, str):
        analysis, detected = await self._is_pii_data_detected(content)
        if detected:
            new_message.content = await self._sanitize(content, analysis)
        return new_message
    ...
```

### Prerequisite
```bash
python -m spacy download en_core_web_sm
```
Presidio's analyzer needs a spaCy model; `presidio-analyzer` and `spacy` are declared in `pyproject.toml`.

### Pitfalls
- **Off by default** (`IS_PII_REDACTION_ENABLED = False`) — `do_pii_redaction` returns messages unchanged with a log line.
- English only (`supported_languages=["en"]`).
- `PII_Redactor()` is constructed in `entry_node` on **every request**, which means the spaCy pipeline may be re-initialized repeatedly — a latency concern.
- Logging (`logging.info(f"user req: ...")`) happens in `entry_node` before/around redaction; check ordering so raw PII doesn't reach logs.

---

## 9. MCP (Model Context Protocol) — 📝 Documented / not implemented

### What it is
MCP is a protocol for exposing external tools/data sources to models. `learning/MCP_INTEGRATION_SUMMARY.md` designs (but does not implement) MCP support.

### The core problem it solves
This project's vision models (`gemini-2.5-flash-lite`, `GLM-4.6V-Flash`, `qwen3-vl`) can *see* images but several cannot emit structured **tool calls**. MCP needs tool-calling models.

### Documented solution: two-stage pipeline

```
image + question
   │
   ▼
Stage 1: vision model describes the image
   │
   ▼
Stage 2: tool-capable model (OpenAI/Anthropic/Gemini) decides + calls tools
   │
   ▼
execute tools → combine results → final answer
```

### Documented files to add/change
- **New:** `src/flow_agent/llms/mcp_tools.py` using `MultiServerMCPClient` from `langchain-mcp-adapters`.
- `src/flow_agent/configurations/config.py` — add `ENABLE_MCP_TOOLS`, `MCP_ALLOWED_DIR`, `MCP_TOOL_TRIGGER_KEYWORDS`.
- `src/flow_agent/utils/nodes.py` — two-stage logic in `call_langchain_reasoning_model` plus helpers `_has_image_content`, `_might_need_tools`, `execute_tool_calls`.
- `langgraph.json` / `pyproject.toml` — add `langchain-mcp-adapters`.

### Alternative noted in the doc
Skip MCP and use plain LangChain tools. The project **already** does this — `nodes.py` uses `DuckDuckGoSearchResults` for web search.

### Pitfalls
- `langchain-mcp-adapters` is **not** in `pyproject.toml`.
- The `execute_tools`-as-a-separate-LangGraph-node pattern is described as the cleanest approach because it shows up in traces — but no such node exists in `graph.py` today.
- Tool-calling models are a *different* set from the vision models currently in `VISION_PROVIDER_DISTRIBUTION`.

---

# Part 4 — Speech & Voice

## 10. Speech Services (SarvamAI, OpenAI, Google GenAI)

### What it is
A provider-agnostic STT/TTS layer: an abstract interface, a factory with a provider registry, and three implementations.

### Why the project uses it
The same UI and graph can use Sarvam (default, strong Indian-language voices), OpenAI Whisper, or Google GenAI speech without code changes.

### Where it lives
- `src/flow_agent/speech/interface.py` — `SpeechService` ABC
- `src/flow_agent/speech/factory.py` — `get_speech_service`
- `src/flow_agent/speech/providers/{sarvam_ai,open_ai,gen_ai}.py`

### Key APIs

```python
class SpeechService(ABC):
    @abstractmethod
    async def speech_to_text(self, audio_bytes: bytes, file_extension: str = ".webm") -> str | None: ...
    @abstractmethod
    async def text_to_speech(self, text: str) -> list | None: ...
    @property
    @abstractmethod
    def provider_name(self) -> str: ...
```

```python
async def get_speech_service(provider: str | None = None, **kwargs) -> SpeechService
def list_available_providers() -> list[str]

_PROVIDERS = {"sarvam": SarvamAiSpeechService, "openai": OpenAiSpeechService, "gemini": GenAiSpeechService}
```

Provider names are normalized through an alias map, so `sarvamai`, `sarvam_ai`, and `zai` all resolve to `sarvam`, and `genai` / `gen_ai` / `google` resolve to `gemini`.

Default configuration comes from `Settings`:

| Setting | Default |
|---|---|
| `SARVAM_STT_MODEL` / `SARVAM_TTS_MODEL` | `saaras:v3` / `bulbul:v3` |
| `SARVAM_SPEAKER` / `SARVAM_LANGUAGE` | `shubh` / `en-IN` |
| `OPENAI_STT_MODEL` / `OPENAI_TTS_MODEL` | `whisper-1` / `gpt-4o-mini-tts` |
| `OPENAI_SPEAKER` | `coral` |
| `GENAI_STT_MODEL` / `GENAI_TTS_MODEL` | `gemini-3-flash-preview` / `gemini-2.5-flash-preview-tts` |
| `GENAI_SPEAKER` | `Kore` |

**SarvamAI** uses a job-based STT API — create a job, upload files, start, poll, download, then parse the JSON transcript:

```python
job = self.client.speech_to_text_job.create_job(language_code=..., model=...)
job.upload_files(file_paths=[temp_file_path])
job.start()
final_status = job.wait_until_complete()
job.download_outputs(output_dir=output_dir)
```

**OpenAI** writes audio to a temp file and calls `client.audio.transcriptions.create(...)`, then `client.audio.speech.create(...)`. Language is truncated to two chars (`"en-IN"` → `"en"`).

**Google GenAI** wraps raw PCM in a WAV header before transcription (`add_wav_header`) and requests audio output via `response_modalities=["AUDIO"]` + `SpeechConfig(voice_config=VoiceConfig(prebuilt_voice_config=...))`.

### Pitfalls
- **Keys are read at construction**, so `get_speech_service("sarvam")` raises if `SARVAM_API_KEY` is unset. `app.py` relies on the env var directly.
- All three implementations **swallow exceptions and return `None`** — callers must handle `None` explicitly (the UIs do).
- Sarvam STT is asynchronous and can be slow (job polling); the OpenAI path is a single request.
- `OPENAI_TTS_SAMPLE_RATE` is `24000` in `config.py` but the provider's constructor default is `22050` — the factory passes the setting, so config wins.
- The abstract signature defaults to `.webm` but Google's implementation defaults to `.wav`.

---

## 11. LiveKit Agents

### What it is
LiveKit Agents is a real-time voice framework: it combines a VAD, STT, LLM, and TTS into an `AgentSession` and handles audio, turns, and interruptions.

### Why the project uses it
To move from *push-to-talk over REST* (the Streamlit UI) to a *natural, interruptible, full-duplex* voice conversation.

### Where it lives
- `src/livekit_context_optimized.py` — the main, complete agent (context management + tools)
- `src/livekit_agent.py` — minimal agent (Groq LLM + Sarvam TTS)
- `src/livekit_agent_sarvam_llm.py` — Sarvam LLM via the OpenAI-compatible endpoint
- `src/flow_agent/livekit_agent_langgraph.py` — voice agent with 2 tools, simpler setup
- `src/livekit_screen_share.py` — screen-share detection
- `src/livekit_ai_expert.py` — **duplicate** of `livekit_context_optimized.py`
- `run_livekit_agent.py` — launcher script

### Key APIs

**Plugins in play** (from `livekit.plugins`): `groq`, `sarvam`, `silero`, `google`, `openai`, `langchain`.

**A typical agent** (`src/livekit_agent.py`):

```python
class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant...",
            vad=silero.VAD.load(),
            stt=groq.STT(model="whisper-large-v3-turbo", language="en"),
            llm=groq.LLM(model="llama-3.1-8b-instant"),
            tts=sarvam.TTS(target_language_code="en-IN", model="bulbul:v3", speaker="ishita"),
        )

    async def on_enter(self):
        await self.session.generate_reply()

async def entrypoint(ctx: JobContext):
    session = AgentSession()
    await session.start(agent=VoiceAgent(), room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

**Typed userdata.** For per-session state, declare a dataclass and parameterize `AgentSession`:

```python
@dataclass
class AgentUserData:
    ctx: Optional[JobContext] = None
    domain_state: Optional[SpecializedDomainState] = None

userdata = AgentUserData(ctx=ctx, domain_state=SpecializedDomainState())
session = AgentSession[AgentUserData](userdata=userdata)
await session.start(agent=OptimizedTaskAgent(), room=ctx.room)
```

Two rules follow from this, both learned the hard way (see `learning/LIVEKIT_FIX_SUMMARY.md`):
1. **Always construct `AgentSession` with typed userdata** — an untyped `AgentSession()` leaves `userdata` unset and `on_enter` raises `ValueError`.
2. **Access it via `context.userdata`, not `context.session.userdata`**, when the tool is typed `RunContext[AgentUserData]`.

**Function tools.** Declared with `@function_tool()`; parameters become the JSON schema:

```python
@function_tool()
async def save_preference(self, context: RunContext[AgentUserData], key: str, value: str) -> str:
    context.userdata.domain_state.set_user_preference(key, value)
    return f"Saved: {key} = {value}"

@function_tool()
async def get_context_stats(self, placeholder: str = "") -> str:
    stats = ContextManager.get_context_stats(self)
    ...
```

The `placeholder: str = ""` is deliberate: **LiveKit generates an invalid JSON schema for zero-parameter tools** (`'required'` present but `'properties'` missing), which providers reject with a 400. Adding one defaulted parameter fixes it. Note it cannot be named `_` — Pydantic rejects leading underscores.

**Other tools in the optimized agent:** `web_search` (Tavily, [Section 14](#14-tavily-web-search)) and `calculate` (safe `eval` with `{"__builtins__": {}}`, which compacts context before running).

### Running it

```bash
# Two terminals, from the project root
python -m src.livekit_context_optimized dev       # agent worker, hot reload
python -m src.livekit_context_optimized console   # voice test client

# Or the launcher script
python run_livekit_agent.py
```

### Pitfalls
- 🔴 **Run as a module, not a file.** `python src/livekit_context_optimized.py` fails with `ModuleNotFoundError: No module named 'src'` because the code imports `from src.livekit_context_manager import ...`. Use `python -m src.livekit_context_optimized`.
- 🔴 **The LiveKit dependency is missing from `pyproject.toml`.** There is no `livekit-agents` (or `livekit-plugins-*`) entry, yet every `src/livekit_*.py` imports it. You must install LiveKit separately before these run.
- `src/livekit_ai_expert.py` duplicates `src/livekit_context_optimized.py` verbatim — edit one, not both.
- `src/livekit_screen_share.py` **calls `await ctx.connect()` explicitly** and then keeps the entrypoint alive with `while ctx.room.isconnected`. That differs from the other agents, which let `session.start()` manage the room.
- `livekit_agent.py` declares `AgentSession()` with **no** userdata — fine because it never touches `self.session.userdata`.
- `livekit_agent_sarvam_llm.py` points `openai.LLM.with_ollama` at `https://api.sarvam.ai` with model `sarvam-m` — a naming quirk; it is not actually Ollama.

---

## 12. LiveKit Context Management

### What it is
A three-layer strategy to stop a long voice conversation from overflowing the context window:

1. **Interruption-based truncation** — built into LiveKit, free, automatic.
2. **Message filtering** — copy the context excluding tool calls/instructions/config noise.
3. **Proactive summarization** — periodically compress old history with an LLM.
   Plus **external state** so preferences and facts never enter the chat context at all.

### Why the project uses it
A voice session can run for hundreds of turns. Without compaction it hits token limits and either errors or degrades.

### Where it lives
- `src/livekit_context_manager.py` — `ContextManager`
- `src/specialized_domain_state.py` — `SpecializedDomainState`
- Used in `src/livekit_context_optimized.py`

### Key APIs

```python
class ContextManager:
    @staticmethod
    async def get_compacted_context(agent, keep_last_n_turns: int = 10, exclude_system: bool = True) -> llm.ChatContext

    @staticmethod
    async def summarize_if_needed(agent, threshold_items: int = 20, llm_for_summary: Optional[llm.LLM] = None, keep_last_turns: int = 3) -> llm.ChatContext

    @staticmethod
    def get_context_stats(agent) -> dict
```

`get_compacted_context` copies with exclusions then truncates:

```python
compacted_ctx = agent.chat_ctx.copy(
    exclude_function_call=True,
    exclude_instructions=True,
    exclude_config_update=True,
    exclude_handoff=True,
    exclude_empty_message=True,
)
if len(compacted_ctx.items) > keep_last_n_turns:
    compacted_ctx.truncate(max_items=keep_last_n_turns)
```

`summarize_if_needed` is a no-op below threshold, otherwise uses LiveKit's built-in `ChatContext._summarize`:

```python
if current_items <= threshold_items:
    return agent.chat_ctx
summary_ctx = await ctx_to_summarize._summarize(llm_v=llm_for_summary, keep_last_turns=keep_last_turns)
agent._chat_ctx = summary_ctx
```

`get_context_stats` returns `total_items`, `type_breakdown`, `has_system_messages`, `has_function_calls`.

**External state** (`SpecializedDomainState`, a dataclass) keeps data out of the context:

```python
state.set_user_preference("language", "en-IN")
state.get_user_preference("language")
state.add_task_result("analysis", {"rows": 100})
state.get_task_history("analysis")
state.set_domain_fact("user_level", "advanced")
state.update_user_profile({"name": "Alice"})
state.get_session_duration()
state.get_idle_time()
state.to_summary()
```

### How it's wired
The agent sets a turn counter and checks every N turns:

```python
self._SUMMARIZE_EVERY_N_TURNS    = int(os.getenv("CONTEXT_SUMMARIZE_EVERY_N_TURNS", "5"))
self._SUMMARIZE_THRESHOLD_ITEMS  = int(os.getenv("CONTEXT_SUMMARIZE_THRESHOLD_ITEMS", "10"))
self._KEEP_LAST_N_TURNS          = int(os.getenv("CONTEXT_KEEP_LAST_N_TURNS", "3"))

async def on_user_turn_ended(self):
    self._turn_counter += 1
    if self._turn_counter % self._SUMMARIZE_EVERY_N_TURNS == 0:
        await ContextManager.summarize_if_needed(
            agent=self, threshold_items=self._SUMMARIZE_THRESHOLD_ITEMS, keep_last_turns=3)
```

### Pitfalls
- 🔴 **Code defaults are `5 / 10 / 3`**, not the `15 / 20 / 10` the older docs advertise.
- `summarize_if_needed` calls the **private** `ChatContext._summarize` — a LiveKit version bump could break it.
- Summarization adds latency on the turn it fires; use a faster model via `llm_for_summary`.
- Aggressive summarization loses facts — put durable facts in `SpecializedDomainState`.
- `on_enter` calls summarization *before* the first reply, which is harmless at low item counts but worth knowing.

---

## 13. Silero VAD & Audio Processing — 📝 Documented (UI not implemented)

### What it is
`silero-vad` detects speech/silence; `pyaudio` captures audio; `numpy` processes it. In **LiveKit** agents, VAD is simply `silero.VAD.load()`. In the **documented Streamlit continuous-voice UI**, it was used manually.

### Where it's used for real
LiveKit agents only: `vad=silero.VAD.load()`.

### 📝 The documented Streamlit VAD work
`learning/CONTINUOUS_VOICE_QUICK_START.md`, `VAD_FIX_SUMMARY.md`, and `VAD_TESTING_GUIDE.md` describe `ui/app_continuous_voice.py` with a VAD confidence meter. **That file does not exist** and neither `silero-vad`, `pyaudio`, `numpy`, nor `streamlit-mic-recorder` is in `pyproject.toml`.

Documented constants:

```python
VAD_THRESHOLD = 0.5
SILENCE_DURATION_MS = 600
CHUNK_DURATION_MS = 32       # 512 samples @ 16 kHz — Silero's minimum
SAMPLE_RATE = 16000
```

### The bug worth learning from
int16 PCM was normalised by dividing by its own max, destroying amplitude information:

```python
# WRONG — normalizes loud and quiet speech to the same scale
max_val = np.max(np.abs(audio_chunk))
if max_val > 1.0:
    audio_chunk = audio_chunk / max_val
```

Correct — scale by the int16 range:

```python
if audio_chunk.dtype == np.int16:
    audio_chunk = audio_chunk.astype(np.float32) / 32768.0
audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
```

RMS on the normalized signal is then already in `[0, 1]`, so no arbitrary divisor is needed. Hysteresis (`silence_threshold = speech_threshold * 0.67`) prevents rapid toggling.

### Pitfalls
- Only the LiveKit VAD is real today; the continuous-voice Streamlit UI is aspirational.
- If you build it, remember the backend LLM does **not** accept audio — audio must be transcribed to text first (`prepare_llm_input` only handles images).

---

## 14. Tavily Web Search

### What it is
Tavily is a search API tuned for LLM consumption. Accessed here through LangChain's `TavilySearch` wrapper.

### Why the project uses it
The LiveKit voice agents need current information ("who won…", "latest news…") that the LLM's weights don't contain.

### Where it lives
- `src/livekit_context_optimized.py` (and its duplicate `src/livekit_ai_expert.py`)
- `src/flow_agent/livekit_agent_langgraph.py`

### Key APIs

```python
from langchain_tavily import TavilySearch

tavily_search = TavilySearch(max_results=3)

@function_tool()
async def web_search(self, context: RunContext, query: str) -> str:
    result = tavily_search.run(query)
    if isinstance(result, dict):
        # prefer the "answer" field, else concatenate results[].content
        ...
    else:
        cleaned = remove_urls(str(result))
        return cleaned if cleaned else result
```

A `remove_urls` helper strips links so voice output isn't polluted:

```python
def remove_urls(text: str) -> str:
    url_pattern = r'https?://[^\s]+|www\.[^\s]+'
    cleaned = re.sub(url_pattern, '', text)
    return re.sub(r'\s+', ' ', cleaned).strip()
```

**Note the contrast with the LangGraph agent**, which uses `DuckDuckGoSearchResults` (`nodes.py`), not Tavily. Two different search stacks coexist.

### Pitfalls
- **`langchain-tavily` is not in `pyproject.toml`**, and `TAVILY_API_KEY` is absent from `.env.example`. Add both.
- `TavilySearch` is instantiated at **module import**, so a missing key fails at import.
- The `dict`/`str` branching exists because Tavily's response shape varies by version — handle both.
- In the optimized agent, the documented context-compaction-before-search was removed as redundant (compaction already happens in `on_enter`).

---

## 15. Streaming: SSE vs Polling — 📝 Documented (client not present)

### What it is
`learning/APP_V3_IMPLEMENTATION_SUMMARY.md`, `SSE_STREAMING_FIX.md`, and `WHY_SSE_FAILS.md` describe adding real-time Server-Sent Events to the UI. **`ui/streaming_client.py` and `ui/app_v3.py` do not exist.**

### The documented architecture
A client module with:

```python
async def stream_run_events(deployment_url, thread_id, run_id, timeout=30.0) -> AsyncIterator[Dict]
async def poll_run_state(deployment_url, thread_id, run_id, poll_interval=1.0, max_attempts=None) -> Dict
```

with automatic fallback to polling if SSE fails.

### Why SSE was abandoned
The documented logs are worth understanding, because they mislead people into thinking the endpoint is broken:

```
SSE connection established (status: 200)
SSE connection established but no events received before stream closed
```

The connection succeeds but the server sends **zero bytes**. Causes identified:
1. LangGraph's `/stream` endpoint doesn't emit intermediate events in local dev.
2. The run had often **already finished** by the time SSE connected.
3. The `httpx-sse` parser saw nothing to parse.

The fix documented was to **use polling**, because a voice turn already takes ~30–60s and 1s polling overhead is negligible. That's exactly what the real UIs do today:

```python
while True:
    run_state = get_run_state(thread_id, run_id)
    status = run_state.get("status", "unknown")
    if status in ("success", "failed", "error", "cancelled"):
        break
    time.sleep(1)
```

### Pitfalls
- Don't resurrect `httpx-sse` for this endpoint without first confirming the backend actually emits events.
- If you need true real-time UX, prefer LiveKit ([Section 11](#11-livekit-agents)) over SSE on LangGraph.

---

# Part 5 — Interface, Observability & Quality

## 16. Streamlit UI

### What it is
A browser UI for the LangGraph agent supporting text, image upload, and recorded audio, with optional text-to-speech playback.

### Where it lives
- `ui/app.py` — original, self-contained (inlines Sarvam STT/TTS)
- `ui/app_v2.py` — refactored to use `src/flow_agent/speech`

Both are otherwise near-identical.

### Key APIs

```python
DEPLOYMENT_URL = "http://localhost:2024"
ASSISTANT_ID = "agent"
```

REST helpers against the LangGraph server:

```python
def create_thread() -> str                      # POST /threads
def submit_run(thread_id, input_data) -> str    # POST /threads/{id}/runs
def get_run_state(thread_id, run_id) -> dict    # GET  /threads/{id}/runs/{run_id}
def get_thread_state(thread_id) -> dict         # GET  /threads/{id}
```

`submit_run` sends `{"assistant_id": ASSISTANT_ID, "input": input_data, "stream_mode": "updates"}`.

Multimodal input is built to match the graph's content envelope:

```python
def build_multimodal_content(text, images=None, audio=None) -> list:
    content = [{"type": "text", "text": text}]
    # images → {"type": "image", "data": b64, "source_type": "base64", "mime_type": ...}
    # audio  → {"type": "audio", "data": b64, ...}
```

State is read back out of the thread with a recursive key search (`_find_key_recursive(thread_state, "final_report")`) because the payload is deeply nested.

**v2's refinement:** `speech_to_text_with_ui` / `text_to_speech_with_ui` wrap the async speech service with `asyncio.run(...)` and Streamlit feedback. Audio is transcribed to text and sent as text — raw audio is deliberately **not** forwarded (`audio=None`).

### Pitfalls
- `create_thread()` must POST `json={}` — an empty body causes a 422.
- `DEPLOYMENT_URL` is hardcoded to `localhost:2024` in both files.
- `_find_key_recursive` is a workaround for not knowing the state shape; it can return an unrelated key if the name is duplicated.
- `asyncio.run` inside a Streamlit callback is fragile if an event loop is already running.
- The `thread_id` persists in `st.session_state`, so all turns share one conversation — expected, but surprising when testing.

---

## 17. Observability & Evaluation (LangSmith, Arize, Phoenix)

### What it is
Three layers: LangSmith tracing for LangGraph runs, Arize (via OpenTelemetry) for instrumentation, and Phoenix/LLM-as-judge for evaluation tests.

### Where it lives
- `src/flow_agent/configurations/arize_config.py` — `configure_arize()`
- `src/flow_agent/graph.py` — sets LangSmith env vars at import
- `tests/end_to_end/arize_evals/` — evaluation tests

### Key APIs

**LangSmith** is enabled purely via environment variables, set from settings in `graph.py`:

```python
os.environ["LANGSMITH_TRACING_V2"] = settings.ENABLE_LANGSMITH_TRACING_V2
os.environ["LANGSMITH_PROJECT"] = settings.TRACING_PROJECT_NAME
```

Set `ENABLE_LANGSMITH_TRACING_V2=true` and provide `LANGSMITH_API_KEY`.

**Arize** is gated and configured by a function:

```python
def configure_arize():
    if not settings.ARIZE_TRACING_ENABLED:
        return
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    tracer_provider = register(
        space_id=os.getenv("ARIZE_SPACE_ID"),
        api_key=os.getenv("ARIZE_API_KEY"),
        project_name=settings.TRACING_PROJECT_NAME,
        set_global_tracer_provider=False,
        log_to_console=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
```

The OpenInference LangChain instrumentor also covers LangGraph.

**Evaluation** uses Phoenix's `ClassificationEvaluator` with an LLM judge. From `tests/end_to_end/arize_evals/test_vision_eval.py`:

```python
judge_llm = LLM(provider="google", model="gemini-2.5-flash-lite")
vision_eval = ClassificationEvaluator(
    name="vision_accuracy",
    llm=judge_llm,
    prompt_template=template,
    choices={"incorrect": 0.0, "correct": 1.0},
)
...
score = vision_eval.evaluate({"input": prompt, "output": answer})[0].score
assert score >= 0.9
```

The test loads `tests/resources/olap.png`, sends it through the compiled graph, then judges the answer.

### Pitfalls
- **`arize-phoenix` is not in `pyproject.toml`** even though the eval tests import `phoenix.evals`. Add it to run those tests.
- Arize and LangSmith are independently toggleable; enabling both sends traces to two places.
- The eval test patches `settings` in three modules (see [Section 5](#5-configuration-with-pydantic-settings)) — a symptom of the singleton pattern.
- Judge quality depends on the judge model; a `0.9` threshold is strict.

---

## 18. Testing & Code Quality

### Test layout

```
tests/
├── conftest.py                              # shared fixtures
├── resources/olap.png                       # fixture image
├── unit_tests/                              # fast, mocked
│   ├── test_langchain_llm.py
│   ├── test_circuit_breaker_llm.py
│   ├── test_pii_redaction.py
│   └── test_speech_services.py
├── integration/
│   └── test_long_conversation.py            # LiveKit context over 30 turns
├── end_to_end/
│   ├── test_graph.py
│   ├── test_summarization.py
│   ├── arize_evals/                         # LLM-as-judge evals
│   └── speech/test_speech.py
├── test_livekit_context_management.py
└── test_multiturn_memory.py
```

### Fixtures (`tests/conftest.py`)

```python
@pytest.fixture
def custom_settings():            # overrides via os.environ, restores after
    ...

@pytest.fixture
def custom_settings_with_gemma_3_12b():
    from src.flow_agent.configurations.config import settings
    return settings.model_copy(update={
        "MAX_TRY": 1,
        "SLEEP_IN_SECONDS": 0,
        "VISION_PROVIDER_DISTRIBUTION": {"gemini": 1.0, "openai": 0.0, "zhipu": 0.0, "ollama": 0.0},
        "GEMINI_VISION_MODEL": "gemma-3-12b-it",
    })

@pytest.fixture
def image_to_base64_fixture():
    ...
```

`custom_settings_with_gemma_3_12b` is the safe way to make tests deterministic — pin the distribution to one provider.

### Running

```bash
pytest                                  # all tests
pytest -v                               # verbose
pytest -m "not slow"                    # skip slow tests (marker declared in pyproject)
pytest tests/unit_tests/ -v             # fast unit tests only
pytest tests/test_livekit_context_management.py tests/integration/test_long_conversation.py -v
```

### Tooling (`pyproject.toml`)

```bash
ruff check .          # lint (selects E, F, T201, UP)
ruff check --fix .
mypy src/             # type checking
```

Dev dependencies include `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`, `flake8`.

### Pitfalls
- `pytest-asyncio` is required for the `@pytest.mark.asyncio` tests; without it they're skipped as unawaited.
- The `slow` marker is declared but you must tag tests with `@pytest.mark.slow` to use it.
- `eval()`-based tests (`calculate`) are safe in production code due to the empty `__builtins__`, but keep an eye on it.
- Many end-to-end tests patch module-level `settings`; if you refactor to a `get_settings()` accessor, those patches must be updated together.

---

# Appendices

## A. Consolidated troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` (LiveKit) | Ran `python src/livekit_context_optimized.py` | Use `python -m src.livekit_context_optimized dev` |
| `ModuleNotFoundError` for `livekit`/`langchain_tavily`/`phoenix` | Dependencies not in `pyproject.toml` | Install separately (see Appendix C) |
| `ValueError: AgentSession userdata is not set` | `AgentSession()` created without userdata | `AgentSession[AgentUserData](userdata=...)` |
| Provider 400 `invalid JSON schema for tool` | Zero-parameter `@function_tool()` | Add a defaulted param: `placeholder: str = ""` |
| `NameError: Fields must not use names with leading underscores` | Placeholder named `_` | Rename to a normal identifier |
| Input 422 when creating a thread | Empty POST body | `requests.post(url, json={})` |
| No SSE events, connection closes | LangGraph `/stream` sends nothing in dev | Use polling |
| Ollama `ValueError: OLLAMA_API_KEY ... not set` | Ollama in the random distribution | Set the key or zero its weight |
| PII/validation seemingly does nothing | `IS_PII_REDACTION_ENABLED` / `IS_INPUT_VALIDATION_ENABLED` default `False` | Enable explicitly |
| Presidio fails to start | Missing spaCy model | `python -m spacy download en_core_web_sm` |
| Summarization never fires | Turn counter not incrementing / threshold too high | Check `on_user_turn_ended`, `CONTEXT_SUMMARIZE_THRESHOLD_ITEMS` |

## B. Glossary

- **Node** — an async function in the LangGraph graph operating on `State`.
- **Reducer** — an annotation (`add_messages`, `add`) controlling how a state key accumulates.
- **`Overwrite`** — wrapper that replaces a value instead of merging through a reducer.
- **Provider distribution** — weighted random map choosing which LLM to call.
- **Circuit breaker** — fails fast after N errors; here a single global `aiobreaker` instance.
- **VAD** — Voice Activity Detection; decides speech vs silence.
- **AgentSession** — LiveKit's per-user voice session holding VAD/STT/LLM/TTS.
- **userdata** — typed per-session state attached to an `AgentSession`.
- **STT / TTS** — speech-to-text / text-to-speech.
- **`@function_tool()`** — LiveKit decorator exposing a method as an LLM-callable tool.

## C. Known drift & discrepancies

Where the `learning/*.md` docs and the code disagree. **The code is authoritative.**

| # | Claim in docs | Reality in code |
|---|---|---|
| 1 | `ui/app_v3.py`, `app_v3_audio_only.py`, `app_continuous_voice.py`, `app_v4_simple_polling.py`, `ui/streaming_client.py` | **Do not exist.** `ui/` has only `app.py` and `app_v2.py` |
| 2 | Circuit breaker refactored to per-provider | **Still global.** Single `breaker = CircuitBreaker(...)` in `circuit_breaker_llm.py` |
| 3 | `call_llm_safely(provider, llm, conversation, new_message)` | Actual signature: `call_llm_safely(llm, conversation, new_message)` |
| 4 | Config at `src/flow_agent/config.py` | Actual: `src/flow_agent/configurations/config.py` |
| 5 | Gemini model `gemma-3-27b-it` | Actual default: `gemini-2.5-flash-lite` |
| 6 | Speech providers `sarvam.py` / `openai.py` / `genai.py` | Actual: `sarvam_ai.py` / `open_ai.py` / `gen_ai.py` |
| 7 | Context defaults `20 / 10 / 15` | Code defaults: `10 / 3 / 5` |
| 8 | LiveKit agents runnable after `uv sync` | `livekit-agents` + plugins are **absent** from `pyproject.toml` |
| 9 | Tavily available | `langchain-tavily` absent from `pyproject.toml`; `TAVILY_API_KEY` absent from `.env.example` |
| 10 | Phoenix evals available | `arize-phoenix` absent from `pyproject.toml` |
| 11 | VAD Streamlit UI with confidence meter | Only LiveKit `silero.VAD.load()` exists; no VAD UI |
| 12 | Docs referenced as `docs/*.md` | Now under `learning/` |
| 13 | `N` unit tests / `N` integration tests (various) | Counts in old docs are unreliable — run `pytest` for the truth |
| 14 | Mistral support implied by distribution | `langchain-mistralai` is **not** in `pyproject.toml` despite `ChatMistralAI` being imported |

### Documented-but-absent dependencies to add if you use those features

```toml
# voice agents
livekit-agents
livekit-plugins-silero
livekit-plugins-groq
livekit-plugins-sarvam
livekit-plugins-google
livekit-plugins-openai
# web search in LiveKit agents
langchain-tavily
# evaluation
arize-phoenix
# mistral summarizer (already referenced in LangChainChatLLM.py)
langchain-mistralai
# documented / experimental
silero-vad
pyaudio
numpy
streamlit-mic-recorder
httpx-sse
langchain-mcp-adapters
```

## D. Source-doc → tutorial section map

| `learning/` doc | Tutorial section |
|---|---|
| `QWEN.md` | Overview, [5](#5-configuration-with-pydantic-settings), [16](#16-streamlit-ui) |
| `ARCHITECTURE_REVIEW.md` | [4](#4-circuit-breaking--retries-aiobreaker--tenacity), [5](#5-configuration-with-pydantic-settings) |
| `CIRCUIT_BREAKER_REFACTOR_PLAN.md` | [4](#4-circuit-breaking--retries-aiobreaker--tenacity) (planned) |
| `project_assessment.md` | [3](#3-multi-provider-llm-layer), [18](#18-testing--code-quality) |
| `RATE_LIMIT_OPTION.md` | [6](#6-rate-limiting--documented--not-implemented) |
| `MCP_INTEGRATION_SUMMARY.md` | [9](#9-mcp-model-context-protocol--documented--not-implemented) |
| `LIVEKIT_IMPLEMENTATION_SUMMARY.md` | [12](#12-livekit-context-management) |
| `LIVEKIT_CONTEXT_OPTIMIZATION.md` | [12](#12-livekit-context-management) |
| `LIVEKIT_QUICK_REFERENCE.md` | [12](#12-livekit-context-management) |
| `LIVEKIT_FIX_SUMMARY.md` | [11](#11-livekit-agents) |
| `LIVEKIT_TESTING_COMMANDS.md` | [11](#11-livekit-agents), [18](#18-testing--code-quality) |
| `RUN_AGENT_INSTRUCTIONS.md` | [11](#11-livekit-agents) |
| `TAVILY_INTEGRATION.md` | [14](#14-tavily-web-search) |
| `VAD_FIX_SUMMARY.md` | [13](#13-silero-vad--audio-processing--documented-ui-not-implemented) |
| `VAD_TESTING_GUIDE.md` | [13](#13-silero-vad--audio-processing--documented-ui-not-implemented) |
| `CONTINUOUS_VOICE_QUICK_START.md` | [13](#13-silero-vad--audio-processing--documented-ui-not-implemented) |
| `APP_V3_IMPLEMENTATION_SUMMARY.md` | [15](#15-streaming-sse-vs-polling--documented-client-not-present) |
| `APP_V3_QUICK_START.md` | [15](#15-streaming-sse-vs-polling--documented-client-not-present) |
| `SSE_STREAMING_FIX.md` | [15](#15-streaming-sse-vs-polling--documented-client-not-present) |
| `WHY_SSE_FAILS.md` | [15](#15-streaming-sse-vs-polling--documented-client-not-present) |
| `COMMAND_REFERENCE.md` | [11](#11-livekit-agents) |

---

*Tutorial generated from the actual source tree. Where this document and `learning/*.md` disagree, trust this document and Appendix C.*
