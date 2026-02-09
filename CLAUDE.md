# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangGraph-based multimodal voice agent template built with Python. The project implements a reasoning agent that can process both text and multimodal inputs (images, audio, video) using various LLM providers including Google Gemini, OpenAI, Zhipu (Zai), and Ollama. The agent features a Streamlit web interface for user interaction and supports both development and production deployment.

**Key Security Features:**
- Multi-layer input validation with regex pattern matching
- OpenAI moderation API integration for text and image content
- PII redaction using Microsoft Presidio
- Enterprise-grade security with comprehensive threat detection

## Configuration

### Application Settings (`src/flow_agent/config.py`)
The `Settings` class (using Pydantic BaseSettings) loads configuration from environment variables and `.env` file:

**Circuit Breaker Settings:**
- `MAX_TRY: int = 3` - Maximum retry attempts for LLM calls
- `SLEEP: int = 1` - Initial sleep time for exponential backoff

**Model Configuration:**
- `GEMINI_VISION_MODEL: str = 'gemma-3-27b-it'`
- `OPENAI_VISION_MODEL: str = 'gpt-5-nano'`
- `ZHIPU_VISION_MODEL: str = 'GLM-4.6V-Flash'`
- `OLLAMA_VISION_MODEL: str = 'qwen3-vl:235b-instruct-cloud'`

**Provider Distribution:**
- `PROVIDER_DISTRIBUTION: dict` - Weighted distribution for random provider selection
  - `ollama`: 0.5 (default)
  - `zhipu`: 0.2
  - `gemini`: 0.2
  - `openai`: 0.1
- `FALLBACK_PROVIDER_IDENTIFIER: str = 'openai'` - Fallback provider on circuit breaker failure

**Node Selection:**
- `REASONING_NODE_PREFERENCE: str = 'langchain'` - Choose between `'langchain'` or `'genai'` reasoning node

**Security Settings:**
- `MODERATION_API_CHECK_REQ: bool = True` - Enable/disable OpenAI moderation API
- `MODERATION_MODEL: str = 'omni-moderation-latest'` - Model for text+image moderation

**Speech Service Configuration:**
- `SPEECH_PROVIDER: str = 'sarvam'` - Default speech provider (options: `'sarvam'`, `'openai'`, `'gemini'`)
- `SARVAM_STT_MODEL`, `SARVAM_TTS_MODEL`, `SARVAM_LANGUAGE`, `SARVAM_SPEAKER` - SarvamAI settings
- `GENAI_STT_MODEL`, `GENAI_TTS_MODEL`, `GENAI_LANGUAGE`, `GENAI_SPEAKER` - Google GenAI settings
- `OPENAI_STT_MODEL`, `OPENAI_TTS_MODEL`, `OPENAI_LANGUAGE`, `OPENAI_SPEAKER` - OpenAI settings

**Base URLs:**
- `OLLAMA_BASE_URL: str = "https://ollama.com"` (remote server)
- `ZHIPU_BASE_URL: str = "https://api.z.ai/api/paas/v4/"`

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

### Code Quality
```bash
# Run linting
ruff check .

# Auto-fix lint issues
ruff check --fix .

# Type checking
mypy src/
```

### Running the Agent

**Development Mode (with hot-reload):**
```bash
# Start LangGraph with Docker Compose (Postgres + Redis included)
# Automatically builds, starts services, and watches for code changes
langgraph up --watch
```

The `langgraph up` command:
- Creates a Docker Compose setup with Postgres (state storage) and Redis (caching)
- Builds and starts all services
- `--watch` enables hot-reload for rapid development
- Exposes the API at `http://localhost:8123` by default

**Production Build:**
```bash
# Build for production (without watch mode)
langgraph build
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/path/to/test_file.py

# Run with verbose output
pytest -v
```

**Test Structure:**
- `tests/conftest.py` - Test configuration (adds src to Python path)
  - `resources_path` fixture - Path to test resources directory
  - `image_to_base64_fixture` - Helper for encoding test images
  - `custom_settings` fixture - Override settings via environment variables during tests
- `tests/unit_tests/` - Unit test directory (currently has `test_langchain_llm.py`)
- `tests/end_to_end/` - End-to-end test directory (currently has `test_graph.py`)

**Test Fixtures Usage:**
```python
async def test_something(custom_settings):
    custom_settings.set("REASONING_NODE_PREFERENCE", "genai")
    # ... test code using the custom setting
```

## Architecture

### Graph Structure (`src/flow_agent/graph.py`)
The agent is defined as a LangGraph StateGraph with the following nodes:

- **entry**: Entry node that checks if the thread has already ended
- **input_validator**: Security node that scans for malicious content and runs moderation checks
- **reasoning**: Main reasoning node that calls the LLM via LangChain interface

**Graph Flow:**
```
START → entry → conditional edge → input_validator → conditional edge → reasoning → END
                    ↓                                      ↓
                   END (if closed)                     END (if invalid)
```

The first conditional edge (`should_continue`) checks the `ended_once` state flag to prevent thread reuse.
The second conditional edge (`route_after_validation`) checks the `input_valid` flag to route to reasoning or END.

### State Management (`src/flow_agent/utils/state.py`)
The `State` TypedDict contains:
- `messages`: Annotated list of BaseMessage (uses `add_messages` reducer)
- `retry_count`: Annotated int (uses `add` reducer)
- `issue`: String for issue summary
- `final_report`: String for final results
- `ended_once`: Boolean flag to prevent re-execution of closed threads
- `input_valid`: Boolean flag for routing after input validation

### Nodes (`src/flow_agent/utils/nodes.py`)
- `entry_node`: Checks `ended_once` flag to prevent thread reuse
- `should_continue`: Conditional edge function that returns END if thread is closed
- `call_input_validation`: Validates user input for malicious patterns and moderation compliance
- `route_after_validation`: Conditional edge function that routes to reasoning or END based on `input_valid`
- `call_langchain_reasoning_model`: Main reasoning node using LangChain's unified interface (currently active)
- `call_gemini_reasoning_model`: Alternative reasoning node using native Google GenAI SDK
- `prepare_llm_input`: Helper to prepare multimodal content for LLM (text + up to 4 images)
- `call_llm_safely`: Circuit breaker with exponential backoff and fallback to alternative provider
- `process_response`: Helper function to format LLM responses and update state

### LLM Integration

The project supports multiple LLM providers through two interfaces:

#### 1. LangChain Interface (`src/flow_agent/llms/LangChainChatLLM.py`)
Unified interface supporting multiple providers via `get_chat_llm(provider)`:

| Provider | Model | Notes |
|----------|-------|-------|
| `ollama` | `qwen3-vl:235b-instruct-cloud` | Vision-capable, remote server at https://ollama.com |
| `zhipu` / `zai` | `GLM-4.6V-Flash` | Vision-capable, requires `ZAI_API_KEY` |
| `openai` | `gpt-5-nano` | Standard OpenAI model |
| `gemini` | `gemma-3-27b-it` | Via ChatGoogleGenerativeAI |

**Provider Selection:**
- If `provider` is `None`, uses weighted random distribution via `_get_random_provider()`
- Distribution weights are configured in `settings.PROVIDER_DISTRIBUTION` (default: ollama 50%, zhipu 20%, gemini 20%, openai 10%)
- DuckDuckGo search tool is bound for all providers except `gemini` and `zhipu`

#### 2. Native Google GenAI (`src/flow_agent/llms/genai_agent.py`)
- Uses Google GenAI SDK with `AsyncClient` and `AsyncChat`
- Default model: `gemini-2.5-flash-lite`
- Safety settings configured for all harm categories (BLOCK_LOW_AND_ABOVE)
- System prompt defined in `_GENAI_SUMMARIZER_PROMPT`

### Multimodal Support

The system processes multimodal content from `HumanMessage` content arrays:

**Expected Structure:**
```python
HumanMessage(content=[
    {'type': 'text', 'text': 'what is there in the image'},
    {
        'type': 'image',
        'data': 'iVBORw0KGgoAAAANS...',  # base64 encoded
        'metadata': {'filename': 'example.png'},
        'source_type': 'base64',
        'mime_type': 'image/png'
    }
])
```

**Content Extraction Logic** (in `call_langchain_reasoning_model` and `call_gemini_reasoning_model`):
- Extracts text prompt from `content[i]['type'] == 'text'`
- Extracts media from `content[i]['type'] in ['image', 'audio', 'video']`
- Supports up to 4 images per request (LangChain interface)
- Converts base64 data for LLM consumption
- Handles both string and list content formats for backward compatibility

**Note on Audio Processing:**
Audio is transcribed to text via SarvamAI STT in the UI before being sent to the graph. The graph receives the transcribed text as part of the human message, not raw audio data.

### Web Interface (`ui/app.py`)

Streamlit-based UI that communicates with the LangGraph API via REST:

**REST Endpoints Used:**
- `POST /threads` - Create new thread
- `POST /threads/{thread_id}/runs` - Submit run with `assistant_id: "agent"`
- `GET /threads/{thread_id}/runs/{run_id}` - Poll run status
- `GET /threads/{thread_id}` - Retrieve final thread state

**UI Features:**
- Text area for issue/context input
- Image upload (multiple files supported)
- Voice recording via `st.audio_input()`
- Thread lifecycle management (creates new thread on first run)
- Real-time status polling during execution
- Final report extraction from nested thread state using recursive key search

**Speech Integration:**
The UI now uses the factory pattern for speech services. Provider is configurable via `settings.SPEECH_PROVIDER`:

- **SarvamAI (default)**: Job-based STT API with TTS
  - STT Model: `saaras:v3`, Language: `en-IN`
  - TTS Model: `bulbul:v3`, Speaker: `shubh`
  - Creates STT job, uploads audio, polls for completion, downloads transcript JSON
  - Returns list of audio data (base64 or URLs) for TTS

- **OpenAI**: Whisper-1 for STT, gpt-4o-mini-tts for TTS
- **Google GenAI**: gemini-3-flash-preview for STT, gemini-2.5-flash-preview-tts for TTS

**Deployment Configuration:**
- `DEPLOYMENT_URL`: Default `http://localhost:8123` (Docker Compose port)
  - **Note**: The ui/app.py file may have a different default port (e.g., 2024) - ensure this matches your LangGraph API port
- `ASSISTANT_ID`: `"agent"` (must match `langgraph.json` graph name)

**Debug Mode:**
The UI includes debug expanders that show raw JSON for:
- `final_report` - Final agent output
- STT transcripts
- Audio data (base64 or URLs)

### Logging (`src/flow_agent/logging_config.py`)
- Custom color formatter for console output
- Configured to silence noisy libraries (uvicorn, langgraph, httpx, etc.)
- Forced setup with `force=True` to override uvicorn/langgraph defaults

### Input Validation (`src/flow_agent/utils/input_validation.py`)
Multi-layer security validation for all user inputs:

**Pattern-Based Detection:**
- Shell command injection (command chaining, backticks, sudo)
- Docker abuse (privileged mode, volume mounting, shell access)
- SQL injection patterns (UNION SELECT, exec functions)
- Path traversal attempts (../, /etc/passwd, Windows system directories)
- System commands (rm -rf, dd, shutdown, kill)
- Code execution patterns (script tags, JavaScript protocols)
- Windows-specific abuse (PowerShell encoding, registry tampering, LOLBINs)

**API-Based Moderation:**
- OpenAI moderation API (`omni-moderation-latest` model)
- Supports both text and image content
- Enabled/disabled via `settings.MODERATION_API_CHECK_REQ`
- Returns flagged categories for audit trails

**Usage:**
```python
from src.flow_agent.utils.input_validation import scan_for_vulnerability

is_safe = await scan_for_vulnerability(human_message)
# Returns False if malicious patterns detected or moderation fails
```

### PII Redaction (`src/flow_agent/utils/pii_redaction.py`)
Privacy protection using Microsoft Presidio:

**Features:**
- Detects PII entities (emails, phone numbers, SSN, credit cards, etc.)
- Configurable confidence threshold (default: 0.5)
- Handles both string and list content formats
- Non-destructive (creates message copies)

**Usage:**

```python
from src.flow_agent.utils.pii_redaction import PII_Redactor

redactor = PII_Redactor(confidence_threshold=0.5)
redacted_messages = await redactor.do_pii_redaction(messages)
```

**Note:** PII redaction is implemented but not currently wired into the graph by default. To enable, add a redaction node before `input_validator`.

### Speech Services (`src/flow_agent/speech/`)
Factory pattern for multi-provider STT/TTS with abstract interface:

**Interface Definition (`interface.py`):**
```python
class SpeechService(ABC):
    @abstractmethod
    async def speech_to_text(self, audio_bytes: bytes, file_extension: str = ".webm") -> str | None

    @abstractmethod
    async def text_to_speech(self, text: str) -> list | None  # Returns base64 or URLs
```

**Available Providers (`factory.py`):**

| Provider | STT Model | TTS Model | Language | Speaker |
|----------|-----------|-----------|----------|---------|
| `sarvam` | `saaras:v3` | `bulbul:v3` | `en-IN` | `shubh` |
| `openai` | `whisper-1` | `gpt-4o-mini-tts` | `en-IN` | `coral` |
| `gemini` | `gemini-3-flash-preview` | `gemini-2.5-flash-preview-tts` | `en-IN` | `Kore` |

**Usage:**
```python
from src.flow_agent.speech.factory import get_speech_service

# Use default provider (from settings.SPEECH_PROVIDER)
speech = await get_speech_service()

# Explicit provider selection
speech = await get_speech_service("sarvam")
speech = await get_speech_service("gemini", language="hi-IN")

# Transcribe audio
transcript = await speech.speech_to_text(audio_bytes, file_extension=".webm")

# Generate speech
audio_data = await speech.text_to_speech("Hello world")
```

**Provider Aliases:** `sarvamai`, `sarvam_ai`, `zai` → `sarvam`; `genai`, `gen_ai`, `google` → `gemini`

### Code Style Configuration (`pyproject.toml`)
**Ruff Configuration:**
- Enabled rule sets: E (pycodestyle), F (pyflakes), UP (pyupgrade), T201 (print statements)
- Relaxed rules:
  - `UP006`, `UP007`, `UP035`: Allows `typing_extensions` imports
  - `E501`: No line length enforcement
- Per-file ignores: Tests exclude upgrade rules (`"tests/*" = ["UP"]`)

## LangGraph Configuration (`langgraph.json`)
- Graph entry point: `./src/flow_agent/graph.py:graph`
- Graph name: `agent` (used as `assistant_id` in UI)
- Environment file: `.env`
- Image distribution: `wolfi`

**Important - Dependencies for Docker Deployment:**
For Docker deployment (`langgraph up`), dependencies **must** be explicitly listed in the `dependencies` array in `langgraph.json`. The Docker build process does NOT automatically infer dependencies from `pyproject.toml`. However, `langgraph dev` (local development) works fine with `pyproject.toml` alone.

Current `langgraph.json` dependencies include:
- LangGraph stack: `langgraph`, `langgraph-api`, `langgraph-cli`
- LangChain integrations: `langchain-core`, `langchain-google-genai`, `langchain-ollama`, `langchain-openai`, `langchain-community`
- Utilities: `pydantic`, `pydantic-settings`, `python-dotenv`, `requests`
- UI: `streamlit`, `sarvamai`
- Security: `presidio-analyzer`, `presidio-anonymizer` (for PII redaction)
- Tools: `ddgs` (DuckDuckGo search)

When adding new dependencies, remember to:
1. Add to `pyproject.toml` for local development
2. **Also add to `langgraph.json`** for Docker deployment

## Environment Variables
The project uses a `.env` file for configuration (not tracked in git):
- `ZAI_API_KEY` - For Zhipu/Zai provider
- `OLLAMA_API_KEY` - For Ollama provider
- `OPENAI_API_KEY` - For OpenAI provider (also used as fallback, moderation API)
- `GOOGLE_API_KEY` - For Google Gemini models (used by both LangChain and native GenAI SDK)
- `SARVAM_API_KEY` - For SarvamAI speech services (STT/TTS)
- `LANGSMITH_API_KEY` - Optional, for LangSmith tracing and monitoring
- `LANGCHAIN_PROJECT` - LangSmith project name (e.g., `multimodal_voice_agent`)

**Additional Optional Variables:**
- `MODERATION_API_CHECK_REQ` - Enable/disable moderation API (default: True)
- `SPEECH_PROVIDER` - Default speech provider (default: sarvam)

## Running the Streamlit UI

**Step 1: Start the LangGraph backend**
```bash
langgraph up --watch
```

**Step 2: Start the Streamlit UI (in a separate terminal)**
```bash
streamlit run ui/app.py
```

**Access:**
- LangGraph API: http://localhost:8123
- API Docs: http://localhost:8123/docs
- Streamlit UI: http://localhost:8501

## Important Notes

### Security Architecture
The agent implements defense-in-depth with three security layers:

1. **Input Validation Node** (`src/flow_agent/utils/input_validation.py`):
   - Pattern-based detection for 100+ attack vectors
   - Compiled regex patterns for performance
   - Categories: shell injection, docker abuse, SQL injection, path traversal, system commands, code execution, Windows abuse

2. **Moderation API** (OpenAI `omni-moderation-latest`):
   - Configurable via `settings.MODERATION_API_CHECK_REQ`
   - Processes both text and images
   - Returns flagged categories for compliance logging

3. **PII Redaction** (Microsoft Presidio):
   - Implemented but not enabled by default
   - To enable: add PII redaction node before `input_validator` in graph
   - Configurable confidence threshold

### Thread Lifecycle
Threads are designed to be single-use. Once `ended_once` is set to True, subsequent calls to the thread will return a message directing the user to create a new thread.

### Provider Selection
Provider selection is controlled in two ways:

1. **Automatic**: Set `provider=None` in `get_chat_llm()` to use weighted random distribution configured in `settings.PROVIDER_DISTRIBUTION`

2. **Explicit**: Pass a specific provider string to `get_chat_llm(provider)`:
   - `'ollama'`, `'openai'`, `'gemini'`, `'zhipu'`, or `'zai'`

### Node Selection
The graph selects the reasoning node based on `settings.REASONING_NODE_PREFERENCE`:
- `'langchain'` (default) - Uses `call_langchain_reasoning_model`
- `'genai'` - Uses `call_gemini_reasoning_model`

**Important gotcha**: When testing different nodes, you must patch settings in multiple places because the graph is compiled at import time:
```python
# In tests, patch both modules
with patch("src.flow_agent.graph.settings", custom_settings), \
     patch("src.flow_agent.utils.nodes.settings", custom_settings):
    # ... test code
```

To change the node for development, edit `src/flow_agent/config.py`:
```python
REASONING_NODE_PREFERENCE: str = 'genai'  # or 'langchain'
```

### Circuit Breaker Behavior
The `call_llm_safely` function implements:
1. Up to `MAX_TRY` retry attempts with exponential backoff
2. Falls back to `settings.FALLBACK_PROVIDER_IDENTIFIER` (default: `'openai'`) on final failure
3. Doubles sleep time between retries

### Response Processing
The `process_response` helper function sets `final_report` to the full `summary` string (corrected from the previous `summary[-1]` bug).

### Debug Mode
The UI includes debug expanders that show raw JSON:
```python
with st.expander("🔍 Debug: Raw final_report"):
    st.code(json.dumps(final_report_raw, indent=2, default=str))
```
This is invaluable for debugging but exposes internal state structure.

### Graph Modification Notes

**Adding PII Redaction:**
To enable PII redaction, modify `src/flow_agent/graph.py`:

```python
from src.flow_agent.utils.pii_redaction import PII_Redactor

# Add node after entry
.add_node("pii_redaction", call_pii_redaction)

# Update edges
.add_conditional_edges("entry", should_continue, {"pii_redaction": "pii_redaction", END: END})
.add_conditional_edges("pii_redaction", route_after_redaction, {"input_validator": "input_validator", END: END})
```

**Disabling Moderation API:**
Set in `.env` or environment:
```bash
MODERATION_API_CHECK_REQ=False
```

**Custom Moderation Model:**
```bash
MODERATION_MODEL=text-moderation-latest  # Text only
MODERATION_MODEL=omni-moderation-latest  # Text + Image (default)
```
