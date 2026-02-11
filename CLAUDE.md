# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangGraph-based multimodal voice agent template built with Python. The project implements a reasoning agent that can process both text and multimodal inputs (images, audio, video) using various LLM providers including Google Gemini, OpenAI, Zhipu (Zai), SarvamAI, and Ollama. The agent features a Streamlit web interface for user interaction and supports both development and production deployment.

**Key Security Features:**
- Multi-layer input validation with regex pattern matching
- OpenAI moderation API integration for text and image content
- PII redaction using Microsoft Presidio (enabled in entry node)
- Enterprise-grade security with comprehensive threat detection

**Key Features:**
- Automatic conversation summarization after configurable message threshold
- Multi-provider weighted distribution for cost optimization
- Factory pattern for speech services (SarvamAI, OpenAI, Google GenAI)
- LangSmith and Arize Phoenix tracing support

## Configuration

### Application Settings (`src/flow_agent/config.py`)
The `Settings` class (using Pydantic BaseSettings) loads configuration from environment variables and `.env` file:

**Circuit Breaker Settings:**
- `MAX_TRY: int = 3` - Maximum retry attempts for LLM calls
- `SLEEP_IN_SECONDS: int = 1` - Initial sleep time for exponential backoff

**Vision/Reasoning Models:**
- `GEMINI_VISION_MODEL: str = 'gemma-3-27b-it'`
- `OPENAI_VISION_MODEL: str = 'gpt-5-nano'`
- `ZHIPU_VISION_MODEL: str = 'GLM-4.6V-Flash'`
- `OLLAMA_VISION_MODEL: str = 'qwen3-vl:235b-instruct-cloud'`

**Summarization Models:**
- `ZHIPU_SUMMARIZATION_MODEL: str = 'GLM-4.7-Flash'`
- `OLLAMA_SUMMARIZATION_MODEL: str = 'nemotron-3-nano:30b-cloud'`

**Provider Distribution:**
- `VISION_PROVIDER_DISTRIBUTION: dict` - Weighted distribution for vision/reasoning
  - `gemini`: 0.7
  - `ollama`: 0.2
  - `zhipu`: 0.1
  - `openai`: 0.01
- `SUMMARIZATION_PROVIDER_DISTRIBUTION: dict` - Weighted distribution for summarization
  - `ollama`: 0.5
  - `sarvam`: 0.3
  - `gemini`: 0.1
  - `zhipu`: 0.09
  - `openai`: 0.01
- `FALLBACK_PROVIDER_IDENTIFIER: str = 'openai'` - Fallback provider on circuit breaker failure

**Node Selection:**
- `REASONING_NODE_PREFERENCE: str = 'langchain'` - Choose between `'langchain'` or `'genai'` reasoning node
- `SUMMARY_PROVIDER_PREFERENCE: str = 'langchain'` - Choose between `'langchain'` or `'genai'` summarizer

**Security Settings:**
- `MODERATION_API_CHECK_REQ: bool = True` - Enable/disable OpenAI moderation API
- `MODERATION_MODEL: str = 'omni-moderation-latest'` - Model for text+image moderation
- `IS_PII_REDACTION_ENABLED: bool = False` - Enable/disable PII redaction
- `PII_CONFIDENCE_THRESHOLD: float = 0.5` - PII detection threshold

**Summarization Settings:**
- `SUMMARY_MESSAGE_THRESHOLD: int = 4` - Trigger summarization after N messages
- `MAX_IMAGES_PER_REQUEST: int = 2` - Maximum images to retain during summarization

**Speech Service Configuration:**
- `SPEECH_PROVIDER: str = 'sarvam'` - Default speech provider (options: `'sarvam'`, `'openai'`, `'gemini'`)
- `SARVAM_STT_MODEL`, `SARVAM_TTS_MODEL`, `SARVAM_LANGUAGE`, `SARVAM_SPEAKER` - SarvamAI settings
- `GENAI_STT_MODEL`, `GENAI_TTS_MODEL`, `GENAI_LANGUAGE`, `GENAI_SPEAKER` - Google GenAI settings
- `OPENAI_STT_MODEL`, `OPENAI_TTS_MODEL`, `OPENAI_LANGUAGE`, `OPENAI_SPEAKER` - OpenAI settings

**Tracing Configuration:**
- `ENABLE_LANGSMITH_TRACING_V2: str = "false"` - Enable LangSmith tracing
- `TRACING_PROJECT_NAME: str = 'multimodal_voice_agent'` - LangSmith project name
- `ARIZE_TRACING_ENABLED: bool = False` - Enable Arize Phoenix tracing

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

# Download Spacy English model (required for PII redaction)
python -m spacy download en_core_web_sm
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

# Run without warnings
pytest -p no:warnings
```

**Test Structure (97 tests total):**
- `tests/conftest.py` - Test configuration (adds src to Python path)
  - `resources_path` fixture - Path to test resources directory
  - `image_to_base64_fixture` - Helper for encoding test images
  - `custom_settings` fixture - Override settings via environment variables during tests
- `tests/unit_tests/` - Unit test directory (79 tests)
  - LLM provider selection (14 tests)
  - PII redaction (15 tests)
  - Speech services (45 tests)
  - Multiturn memory (5 tests)
- `tests/end_to_end/` - End-to-end test directory (18 tests)
  - `arize_evals/` - LLM-as-judge evaluations using Arize Phoenix
  - Graph flow tests
  - Summarization trigger and retention logic
  - Media handling and image limiting
  - Arize Phoenix evaluations (4 tests)

**Test Fixtures Usage:**
```python
async def test_something(custom_settings):
    custom_settings.set("REASONING_NODE_PREFERENCE", "genai")
    # ... test code using the custom setting
```

### LLM-as-Judge Evaluations (Arize Phoenix)

The project includes evaluations using Arize Phoenix for testing vision understanding and conversation quality:

**Location:** `tests/end_to_end/arize_evals/`

**Vision Evaluation** (`test_vision_eval.py`):
- Tests multimodal understanding with image inputs (e.g., OLAP/data cube images)
- Uses Phoenix `ClassificationEvaluator` with OpenAI GPT-5-nano as judge LLM
- Scores "correct" (1.0) or "incorrect" (0.0) based on vision accuracy
- Threshold: 0.9 (requires high confidence)

**Session Evaluations** (`test_session_eval.py`):
- **Multi-turn correctness** (`test_multi_turn_session_correctness`):
  - Evaluates if agent provides accurate, coherent responses across conversation turns
  - Uses Phoenix `llm_classify` with Google GenAI (`gemma-3-27b-it`) as judge
  - Rails: ["correct", "incorrect"]

- **Goal achievement** (`test_multi_turn_session_goal_achievement`):
  - Tests if user's goals were met by conversation end
  - Uses Phoenix `llm_classify` with Google GenAI as judge
  - Rails: ["achieved", "not_achieved"]

- **Single-turn correctness** (`test_single_turn_correctness`):
  - Simpler test for single QA interactions
  - Uses OpenAI-style Sarvam-M as judge model
  - Rails: ["correct", "incorrect"]

**Running Evaluations:**
```bash
# Run all Arize evals
pytest tests/end_to_end/arize_evals/ -v

# Run specific test
pytest tests/end_to_end/arize_evals/test_vision_eval.py::test_graph_flow_with_image -v
```

**Evaluation Prompts:**

The session evaluations use structured prompts that evaluate:
- **Correctness**: Factual accuracy, addressing questions directly, maintaining context
- **Goal Achievement**: Whether explanations resolved user's doubts, logical progression

**Example Output:**
```
=== Session Correctness Evaluation ===
Label: correct
Explanation: The AI provided accurate information about list comprehensions...
======================================

=== Goal Achievement Evaluation ===
Label: achieved
Explanation: The agent successfully explained the difference between list and tuple...
====================================
```

**Additional Evaluation Resources:**
- `langsmith_evaluation_guide.py` - Reference guide for setting up LangSmith evaluations with custom evaluators for multimodal accuracy, safety, and voice quality

## Architecture

### Graph Structure (`src/flow_agent/graph.py`)
The agent is defined as a LangGraph StateGraph with the following nodes:

- **entry**: Entry node that performs PII redaction and initializes conversation summary
- **input_validator**: Security node that scans for malicious content and runs moderation checks
- **reasoning**: Main reasoning node that calls the LLM via LangChain interface (or native GenAI SDK)
- **summarizer**: Optional node that summarizes older messages when conversation grows long

**Graph Flow:**
```
START → entry → input_validator → conditional edge → reasoning → conditional edge → summarizer → END
                                          ↓                                      ↓
                                         END (if invalid)                     END (if < threshold)
```

The first conditional edge (`route_after_validation`) checks the `input_valid` flag to route to reasoning or END.
The second conditional edge (`should_summarize`) checks the message count to optionally route to summarizer.

### State Management (`src/flow_agent/utils/state.py`)
The `State` TypedDict contains:
- `messages`: Annotated list of BaseMessage (uses `add_messages` reducer)
- `retry_count`: Annotated int (uses `add` reducer)
- `issue`: String for issue summary
- `final_report`: String for final results
- `input_valid`: Boolean flag for routing after input validation
- `conversation_summary`: String for storing conversation summaries

### Nodes (`src/flow_agent/utils/nodes.py`)

**Entry Node:**
- `entry_node`: Performs PII redaction on the latest message, initializes `conversation_summary` if needed

**Validation Nodes:**
- `call_input_validation`: Validates user input for malicious patterns and moderation compliance
- `route_after_validation`: Conditional edge function that routes to reasoning or END based on `input_valid`

**Reasoning Nodes:**
- `call_langchain_reasoning_model`: Main reasoning node using LangChain's unified interface (default)
- `call_gemini_reasoning_model`: Alternative reasoning node using native Google GenAI SDK
- `prepare_llm_input`: Helper to prepare multimodal content for LLM (text + up to 2 images)
- `call_llm_safely`: Circuit breaker with exponential backoff and fallback to alternative provider
- `process_response`: Helper function to format LLM responses and update state

**Summarization Nodes:**
- `call_langchain_summarizer`: Summarizes older messages using weighted provider distribution
- `should_summarize`: Conditional edge that checks if message count exceeds `SUMMARY_MESSAGE_THRESHOLD`

### Conversation Memory & Summarization

The agent maintains conversation history with automatic summarization:

**Trigger:** After `SUMMARY_MESSAGE_THRESHOLD` messages (default: 4)

**Retention Strategy:**
- All human messages (questions) are kept
- Last 2 AI responses are retained
- Older messages are summarized
- Up to `MAX_IMAGES_PER_REQUEST` images are preserved
- Summaries stored in `conversation_summary` state field and included in subsequent LLM calls

**Summarization Flow:**
1. Conditional edge checks message count
2. If threshold exceeded, routes to `summarizer` node
3. Summarizer uses `SUMMARIZATION_PROVIDER_DISTRIBUTION` for LLM selection
4. Old messages removed, summary stored, recent messages retained

### LLM Integration

The project supports multiple LLM providers through two interfaces:

#### 1. LangChain Interface (`src/flow_agent/llms/LangChainChatLLM.py`)
Unified interface supporting multiple providers via `get_chat_llm(provider, use_for_summarization=False)`:

| Provider | Vision Model | Summarization Model | Notes |
|----------|-------------|---------------------|-------|
| `gemini` | `gemma-3-27b-it` | (same) | Via ChatGoogleGenerativeAI |
| `ollama` | `qwen3-vl:235b-instruct-cloud` | `nemotron-3-nano:30b-cloud` | Vision-capable, remote server at https://ollama.com |
| `zhipu` / `zai` | `GLM-4.6V-Flash` | `GLM-4.7-Flash` | Vision-capable, requires `ZAI_API_KEY` |
| `sarvam` | (not vision-capable) | `sarvam-1` | SarvamAI LLM integration |
| `openai` | `gpt-5-nano` | (same) | Standard OpenAI model |

**Provider Selection:**
- If `provider` is `None`, uses weighted random distribution via `_get_random_provider()`
- Distribution weights differ for vision vs summarization (see `VISION_PROVIDER_DISTRIBUTION` and `SUMMARIZATION_PROVIDER_DISTRIBUTION`)
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
- Supports up to 2 images per request (configurable via `MAX_IMAGES_PER_REQUEST`)
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
The UI uses the factory pattern for speech services. Provider is configurable via `settings.SPEECH_PROVIDER`:

- **SarvamAI (default)**: Job-based STT API with TTS
  - STT Model: `saaras:v3`, Language: `en-IN`
  - TTS Model: `bulbul:v3`, Speaker: `shubh`
  - Creates STT job, uploads audio, polls for completion, downloads transcript JSON
  - Returns list of audio data (base64 or URLs) for TTS

- **OpenAI**: Whisper-1 for STT, gpt-4o-mini-tts for TTS
- **Google GenAI**: gemini-3-flash-preview for STT, gemini-2.5-flash-preview-tts for TTS

**Deployment Configuration:**
- `DEPLOYMENT_URL`: Default `http://localhost:8123` (Docker Compose port)
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
- **Enabled in entry_node** - PII redaction runs on every request by default

**Usage:**

```python
from src.flow_agent.utils.pii_redaction import PII_Redactor

redactor = PII_Redactor(confidence_threshold=0.5)
redacted_messages = await redactor.do_pii_redaction(messages)
```

**Configuration:**
- Enable/disable via `IS_PII_REDACTION_ENABLED` in `.env` or settings
- Adjust confidence threshold via `PII_CONFIDENCE_THRESHOLD`

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

### Tracing (`src/flow_agent/utils/arize_config.py`)

**Arize Phoenix Tracing:**
- Enabled via `ARIZE_TRACING_ENABLED: bool = False` in settings
- Uses OpenInference instrumentation for LangChain
- Exports traces via OTLP to Phoenix collector

**LangSmith Tracing:**
- Enabled via `ENABLE_LANGSMITH_TRACING_V2: str = "false"` in settings
- Project name configured via `TRACING_PROJECT_NAME: str = 'multimodal_voice_agent'`
- Requires `LANGSMITH_API_KEY` environment variable

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
- Utilities: `pydantic`, `pydantic-settings`, `python-dotenv`, `requests`, `google-genai`, `typing-extensions`
- UI: `streamlit`, `sarvamai`
- Security: `presidio-analyzer`, `presidio-anonymizer`, `spacy`
- Tools: `ddgs` (DuckDuckGo search)
- Tracing: `openinference-instrumentation-langchain`, `arize-otel`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`
- LLM: `langchain-sarvam-integration`

When adding new dependencies, remember to:
1. Add to `pyproject.toml` for local development
2. **Also add to `langgraph.json`** for Docker deployment

**Project Root Files:**
- `langsmith_evaluation_guide.py` - Reference guide for LangSmith evaluations with custom evaluators
- `.env.example` - Environment variable template
- `pyproject.toml` - Dependencies and tool configuration
- `langgraph.json` - LangGraph configuration and Docker dependencies

## Environment Variables
The project uses a `.env` file for configuration (not tracked in git):
- `ZAI_API_KEY` - For Zhipu/Zai provider
- `OLLAMA_API_KEY` - For Ollama provider
- `OPENAI_API_KEY` - For OpenAI provider (also used as fallback, moderation API)
- `GOOGLE_API_KEY` - For Google Gemini models (used by both LangChain and native GenAI SDK)
- `SARVAM_API_KEY` - For SarvamAI speech services and LLM
- `LANGSMITH_API_KEY` - Optional, for LangSmith tracing and monitoring
- `LANGCHAIN_PROJECT` - LangSmith project name (e.g., `multimodal_voice_agent`)
- `ARIZE_SPACE_ID` - Optional, for Arize Phoenix tracing
- `ARIZE_API_KEY` - Optional, for Arize Phoenix tracing and evaluation

**Additional Optional Variables:**
- `MODERATION_API_CHECK_REQ` - Enable/disable moderation API (default: True)
- `IS_PII_REDACTION_ENABLED` - Enable PII redaction in entry node (default: False)
- `PII_CONFIDENCE_THRESHOLD` - PII detection confidence threshold (default: 0.5)
- `SPEECH_PROVIDER` - Default speech provider (default: sarvam)
- `SUMMARY_MESSAGE_THRESHOLD` - Messages before summarization (default: 4)
- `MAX_IMAGES_PER_REQUEST` - Maximum images to retain during summarization (default: 2)
- `REASONING_NODE_PREFERENCE` - Reasoning node: 'langchain' or 'genai' (default: langchain)
- `SUMMARY_PROVIDER_PREFERENCE` - Summarizer node: 'langchain' or 'genai' (default: langchain)

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
   - Enabled in entry_node by default (configurable via `IS_PII_REDACTION_ENABLED`)
   - Detects emails, phone numbers, SSN, credit cards, URLs, IP addresses, etc.
   - Configurable confidence threshold

### Provider Selection
Provider selection is controlled in two ways:

1. **Automatic**: Set `provider=None` in `get_chat_llm()` to use weighted random distribution
   - Use `VISION_PROVIDER_DISTRIBUTION` for vision/reasoning tasks
   - Use `SUMMARIZATION_PROVIDER_DISTRIBUTION` for summarization tasks

2. **Explicit**: Pass a specific provider string to `get_chat_llm(provider)`:
   - `'ollama'`, `'openai'`, `'gemini'`, `'sarvam'`, `'zhipu'`, or `'zai'`

### Node Selection
The graph selects nodes based on settings:

**Reasoning Node:** `settings.REASONING_NODE_PREFERENCE`
- `'langchain'` (default) - Uses `call_langchain_reasoning_model`
- `'genai'` - Uses `call_gemini_reasoning_model`

**Summarizer Node:** `settings.SUMMARY_PROVIDER_PREFERENCE`
- `'langchain'` (default) - Uses `call_langchain_summarizer`
- `'genai'` - Not currently implemented in graph

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

### Debug Mode
The UI includes debug expanders that show raw JSON:
```python
with st.expander("🔍 Debug: Raw final_report"):
    st.code(json.dumps(final_report_raw, indent=2, default=str))
```
This is invaluable for debugging but exposes internal state structure.

### Graph Modification Notes

**Disabling PII Redaction:**
Set in `.env` or environment:
```bash
IS_PII_REDACTION_ENABLED=False
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

**Adjusting Summarization Threshold:**
```bash
SUMMARY_MESSAGE_THRESHOLD=6  # Summarize after 6 messages instead of 4
```
