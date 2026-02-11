# 🎙️ Multimodal Voice Agent

> *An AI agent that sees images, hears your voice, and talks back. Skynet, but friendlier.*

Meet your new favorite conversational companion: a LangGraph-based multimodal voice agent that actually listens—literally. Whether you want to type, upload photos, or just talk at your computer like a sci-fi protagonist, this agent has you covered.

## ✨ Features

- **🗣️ Voice Input & Output** — Talk to your computer. It talks back. Welcome to the future.
- **📷 Image Analysis** — Upload images and the AI will tell you what's in them. (Spoiler: it's probably a cat.)
- **🧠 Multiple LLM Brains** — Choose between OpenAI, Google Gemini, Zhipu/Zai, SarvamAI, or Ollama. Variety is the spice of life.
- **🔄 Conversation Memory** — Automatic summarization keeps long conversations fresh without losing context.
- **🔍 Built-in Search** — DuckDuckGo integration means it can fact-check itself. Imagine if humans could do that.
- **🛡️ Enterprise Security** — Input validation, OpenAI moderation, and PII redaction using Microsoft Presidio.
- **📊 LLM-as-Judge Evaluations** — Built-in Arize Phoenix evaluations for vision and conversation quality testing.
- **🎨 Pretty Web Interface** — Streamlit-powered UI that won't hurt your eyes.

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher (we're living in the future, no more Python 2.7)
- A sense of adventure
- API keys for the services you want to use

### Installation

```bash
# Install dependencies
uv sync

# Or if you're a traditionalist
pip install -e .

# Download Spacy English model (required for PII redaction)
python -m spacy download en_core_web_sm
```

### Configuration

Create a `.env` file from the example template:

```bash
# Copy the template
cp .env.example .env

# Edit with your actual API keys
```

```bash
# LLM Provider API Keys
ZAI_API_KEY=your_key_here          # For Zhipu/Zai
OLLAMA_API_KEY=your_key_here       # For Ollama
OPENAI_API_KEY=your_key_here       # For OpenAI (also used for moderation API)
GOOGLE_API_KEY=your_key_here       # For Google Gemini
SARVAM_API_KEY=your_key_here       # For SarvamAI (LLM + Speech)

# Optional: LangSmith Tracing
LANGSMITH_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=multimodal_voice_agent

# Optional: Arize Phoenix Tracing & Evaluation
ARIZE_SPACE_ID=your_arize_space_id
ARIZE_API_KEY=your_arize_api_key

# Optional: Feature Flags
MODERATION_API_CHECK_REQ=True      # Enable OpenAI moderation API
IS_PII_REDACTION_ENABLED=False     # Enable PII redaction (slow but thorough)
SPEECH_PROVIDER=sarvam              # Default speech provider (sarvam/openai/gemini)
SUMMARY_PROVIDER_PREFERENCE=langchain  # Summarization backend (langchain/genai)
SUMMARY_MESSAGE_THRESHOLD=4        # Trigger summarization after N messages
```

**⚠️ Important Note for Docker Deployment:**

Dependencies must be explicitly listed in `langgraph.json` for Docker builds (`langgraph up`). The Docker build process does NOT automatically read from `pyproject.toml`. However, `langgraph dev` (local development) works fine with `pyproject.toml` alone.

When adding new packages:
1. Add to `pyproject.toml` (for local dev)
2. **Also add to `langgraph.json`** (for Docker deployment)

### Running the Agent

#### Docker Compose (Recommended)

LangGraph's built-in Docker Compose setup includes Postgres for state storage and Redis for caching:

```bash
# Terminal 1: Start LangGraph backend with hot-reload
langgraph up --watch
```

The `--watch` flag enables automatic reloading when you change code.

```bash
# Terminal 2: Start the web UI
streamlit run ui/app.py
```

**Access:**
- **LangGraph API**: http://localhost:8123
- **API Docs**: http://localhost:8123/docs
- **Streamlit UI**: http://localhost:8501

#### Production Build

```bash
# Build without watch mode for production
langgraph build
```

## 🏗️ Architecture

The agent is built as a LangGraph StateGraph with defense-in-depth security:

```
START → entry → conditional edge → input_validator → conditional edge → reasoning → conditional edge → summarizer → END
                    ↓                                      ↓                                       ↓
                   END (if closed)                     END (if invalid)                        END (if < threshold)
```

**What's happening here?**

- **entry** — Checks if you've already had this conversation. No double-dipping.
- **input_validator** — Multi-layer security: pattern-based vulnerability scanning + OpenAI moderation API
- **reasoning** — The LLM does its thing, processing your text/images with style.
- **summarizer** — When conversations get long, automatically summarizes older messages while retaining images.
- **conditional edges** — Smart routing based on thread status, validation results, and message count.

### Conversation Memory & Summarization

The agent maintains conversation history with automatic summarization:

- **Trigger**: After `SUMMARY_MESSAGE_THRESHOLD` messages (default: 4)
- **Retention**:
  - All human messages (questions) are kept
  - Last 2 AI responses are retained
  - Older messages are summarized
  - Up to `MAX_IMAGES_PER_REQUEST` images are preserved
- **Storage**: Summaries stored in `conversation_summary` state field

### Multimodal Content

The agent accepts content in this format:

```python
HumanMessage(content=[
    {'type': 'text', 'text': 'what is there in the image'},
    {
        'type': 'image',
        'data': 'base64_encoded_data',
        'metadata': {'filename': 'mystery_cat.png'},
        'source_type': 'base64',
        'mime_type': 'image/png'
    }
])
```

### Security Features

The agent implements enterprise-grade security with three layers:

1. **Input Validation Node** (`src/flow_agent/utils/input_validation.py`):
   - Pattern-based detection for 100+ attack vectors
   - Compiled regex patterns for performance
   - Categories: shell injection, docker abuse, SQL injection, path traversal, system commands, code execution, Windows abuse

2. **Moderation API** (OpenAI `omni-moderation-latest`):
   - Configurable via `settings.MODERATION_API_CHECK_REQ`
   - Processes both text and images
   - Returns flagged categories for compliance logging

3. **PII Redaction** (Microsoft Presidio):
   - Disabled by default (set `IS_PII_REDACTION_ENABLED=True` to enable)
   - Detects emails, phone numbers, SSN, credit cards, URLs, IP addresses, etc.
   - To enable: Set `IS_PII_REDACTION_ENABLED=True` in `.env`

## 🧩 LLM Provider Options

### Vision/Reasoning Models

| Provider | Model | Vibe |
|----------|-------|------|
| `gemini` | `gemma-3-27b-it` | The heavy lifter (70% weight) |
| `ollama` | `qwen3-vl:235b-instruct-cloud` | The cloud power house (20% weight) |
| `zhipu` / `zai` | `GLM-4.6V-Flash` | Vision-capable and ready (10% weight) |
| `openai` | `gpt-5-nano` | The expensive option (1% weight) |

**Automatic provider selection** uses weighted distribution (configurable in `src/flow_agent/config.py`):

```python
VISION_PROVIDER_DISTRIBUTION = {
    "gemini": 0.7,
    "ollama": 0.2,
    "zhipu": 0.1,
    "openai": 0.01,
}
```

### Summarization Models

Separate distribution optimized for summarization tasks:

| Provider | Model | Vibe |
|----------|-------|------|
| `ollama` | `nemotron-3-nano:30b-cloud` | The summarization specialist (50% weight) |
| `sarvam` | `sarvam-1` | The newcomer (30% weight) |
| `gemini` | `gemma-3-27b-it` | Reliable workhorse (10% weight) |
| `zhipu` / `zai` | `GLM-4.7-Flash` | Speed demon (9% weight) |
| `openai` | `gpt-5-nano` | When money is no object (1% weight) |

```python
SUMMARIZATION_PROVIDER_DISTRIBUTION = {
    "ollama": 0.5,
    "sarvam": 0.3,
    "gemini": 0.1,
    "zhipu": 0.09,
    "openai": 0.01,
}
```

**Manual selection**: Edit `src/flow_agent/config.py` to adjust weights or specify a preferred provider.

## 🗣️ Speech Features

Multi-provider speech services using a factory pattern:

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
speech = await get_speech_service("gemini", language="hi-IN")

# Transcribe audio
transcript = await speech.speech_to_text(audio_bytes, file_extension=".webm")

# Generate speech
audio_data = await speech.text_to_speech("Hello world")
```

## 🛠️ Development

### Code Quality

```bash
# Linting
ruff check .

# Auto-fix (let the robot fix your code)
ruff check --fix .

# Type checking
mypy src/
```

### Testing

```bash
# Run all tests
pytest

# Run with verbose output (for when you need to feel important)
pytest -v

# Run specific test file
pytest tests/unit_tests/test_pii_redaction.py

# Run without warnings
pytest -p no:warnings
```

**Test Coverage (101 tests total):**
- **End-to-end tests**: 20 tests
  - Graph flow tests (text + multimodal)
  - Summarization trigger and retention logic (10 tests)
  - Media handling and image limiting
  - Speech services (3 tests)
  - Multiturn memory (5 tests)
  - Arize Phoenix evaluations (4 tests)
- **Unit tests**: 81 tests
  - LLM provider selection (14 tests)
  - PII redaction (15 tests)
  - Speech services (51 tests)
  - Circuit breaker (1 test)

### Evaluations

The project includes LLM-as-judge evaluations using Arize Phoenix for testing vision understanding and conversation quality:

**Vision Evaluation** (`tests/end_to_end/arize_evals/test_vision_eval.py`):
- Tests multimodal understanding with image inputs
- Uses `ClassificationEvaluator` with OpenAI GPT-5-nano as judge
- Scores "correct" (1.0) or "incorrect" (0.0) based on vision accuracy

**Session Evaluations** (`tests/end_to_end/arize_evals/test_session_eval.py`):
- **Multi-turn correctness**: Evaluates conversation quality across turns
- **Goal achievement**: Tests if user goals are met by conversation end
- **Single-turn correctness**: Simple QA evaluation
- Uses Google GenAI (`gemma-3-27b-it`) or Sarvam-M as judge models
- Uses Phoenix `llm_classify` with rails for deterministic outputs

Run evaluations:
```bash
pytest tests/end_to_end/arize_evals/ -v
```

**Additional Evaluation Resources:**
- `langsmith_evaluation_guide.py` - Guide for setting up LangSmith evaluations with custom evaluators for multimodal accuracy, safety, and voice quality

## 📁 Project Structure

```
.
├── src/flow_agent/
│   ├── graph.py                 # LangGraph definition
│   ├── config.py                # Pydantic settings
│   ├── utils/
│   │   ├── state.py                # State management (includes conversation_summary)
│   │   ├── nodes.py                # Processing nodes (reasoning, summarizer)
│   │   ├── input_validation.py     # Security: vulnerability scanning
│   │   ├── pii_redaction.py        # Privacy: PII redaction
│   │   ├── circuit_breaker_llm.py  # Circuit breaker with retry logic
│   │   └── arize_config.py         # Arize Phoenix tracing configuration
│   ├── llms/
│   │   └── LangChainChatLLM.py  # Multi-provider interface with summarization support
│   ├── speech/
│   │   ├── interface.py         # Abstract speech service interface
│   │   ├── factory.py           # Provider factory
│   │   ├── sarvam.py            # SarvamAI implementation
│   │   ├── openai.py            # OpenAI implementation
│   │   └── genai.py             # Google GenAI implementation
│   └── logging_config.py        # Color-coded logging
├── ui/
│   └── app.py                   # Streamlit web interface
├── tests/
│   ├── unit_tests/              # 79 unit tests
│   ├── end_to_end/
│   │   ├── arize_evals/         # LLM-as-judge evaluations (4 tests)
│   │   └── ...                  # Other end-to-end tests
│   └── conftest.py              # Test fixtures
├── langgraph.json               # LangGraph configuration
├── .env.example                 # Environment variable template
└── pyproject.toml               # Dependencies & tool config
```

## ⚙️ Configuration

### LangGraph Configuration (`langgraph.json`)

- Graph entry point: `./src/flow_agent/graph.py:graph`
- Graph name: `agent`
- Environment file: `.env`
- Distribution: `wolfi` (sounds cool, right?)

### Application Settings (`src/flow_agent/config.py`)

Key settings configurable via environment variables:

**Circuit Breaker:**
- `MAX_TRY`: Maximum retry attempts (default: 3)
- `SLEEP_IN_SECONDS`: Initial backoff time (default: 1)

**Vision/Reasoning Models:**
- `GEMINI_VISION_MODEL`, `OPENAI_VISION_MODEL`, `ZHIPU_VISION_MODEL`, `OLLAMA_VISION_MODEL`
- `VISION_PROVIDER_DISTRIBUTION`: Weighted distribution for automatic provider selection

**Summarization Models:**
- `ZHIPU_SUMMARIZATION_MODEL`, `OLLAMA_SUMMARIZATION_MODEL`
- `SUMMARIZATION_PROVIDER_DISTRIBUTION`: Weighted distribution for summarization
- `SUMMARY_MESSAGE_THRESHOLD`: Trigger summarization after N messages (default: 4)
- `MAX_IMAGES_PER_REQUEST`: Maximum images to retain (default: 2)

**Node Selection:**
- `REASONING_NODE_PREFERENCE`: `'langchain'` or `'genai'`
- `SUMMARY_PROVIDER_PREFERENCE`: `'langchain'` or `'genai'`

**Security:**
- `MODERATION_API_CHECK_REQ`: Enable/disable moderation API (default: True)
- `MODERATION_MODEL`: OpenAI moderation model (default: `omni-moderation-latest`)
- `IS_PII_REDACTION_ENABLED`: Enable/disable PII redaction (default: False)
- `PII_CONFIDENCE_THRESHOLD`: PII detection threshold (default: 0.5)

**Speech:**
- `SPEECH_PROVIDER`: Default speech provider (`sarvam`/`openai`/`gemini`)
- Provider-specific settings (STT/TTS models, language, speaker, pace, sample rate)

## 🤝 Contributing

Found a bug? Have a feature idea? Open an issue or submit a PR. We don't bite (usually).

## 📄 License

MIT — Do whatever you want with this code. Just don't blame us if your AI becomes sentient.

---

**Built with ❤️ and too much coffee**
