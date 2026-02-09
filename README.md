# 🎙️ Multimodal Voice Agent

> *An AI agent that sees images, hears your voice, and talks back. Skynet, but friendlier.*

Meet your new favorite conversational companion: a LangGraph-based multimodal voice agent that actually listens—literally. Whether you want to type, upload photos, or just talk at your computer like a sci-fi protagonist, this agent has you covered.

## ✨ Features

- **🗣️ Voice Input & Output** — Talk to your computer. It talks back. Welcome to the future.
- **📷 Image Analysis** — Upload images and the AI will tell you what's in them. (Spoiler: it's probably a cat.)
- **🧠 Multiple LLM Brains** — Choose between OpenAI, Google Gemini, Zhipu/Zai, or Ollama. Variety is the spice of life.
- **🔍 Built-in Search** — DuckDuckGo integration means it can fact-check itself. Imagine if humans could do that.
- **🛡️ Enterprise Security** — Input validation, OpenAI moderation, and PII redaction using Microsoft Presidio.
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

# Speech Service API Keys
SARVAM_API_KEY=your_sarvam_key     # For SarvamAI speech (STT/TTS)

# Optional: LangSmith Tracing
LANGSMITH_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=multimodal_voice_agent

# Optional: Feature Flags
MODERATION_API_CHECK_REQ=True      # Enable OpenAI moderation API
SPEECH_PROVIDER=sarvam              # Default speech provider (sarvam/openai/gemini)
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
START → entry → conditional edge → input_validator → conditional edge → reasoning → END
                    ↓                                      ↓
                   END (if closed)                     END (if invalid)
```

**What's happening here?**

- **entry** — Checks if you've already had this conversation. No double-dipping.
- **input_validator** — Multi-layer security: pattern-based vulnerability scanning + OpenAI moderation API
- **reasoning** — The LLM does its thing, processing your text/images with style.
- **conditional edges** — Smart routing based on thread status and validation results.

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
   - Implemented but not enabled by default
   - Detects emails, phone numbers, SSN, credit cards, URLs, IP addresses, etc.
   - To enable: add PII redaction node before `input_validator` in graph

## 🧩 LLM Provider Options

| Provider | Model | Vibe |
|----------|-------|------|
| `ollama` | `qwen3-vl:235b-instruct-cloud` | The hipster choice |
| `zhipu` / `zai` | `GLM-4.6V-Flash` | Vision-capable and ready |
| `openai` | `gpt-5-nano` | The classic |
| `gemini` | `gemma-3-27b-it` | Google's contribution |

**Automatic provider selection** uses weighted distribution (configurable in `src/flow_agent/config.py`):
- `ollama`: 50% (default)
- `zhipu`: 20%
- `gemini`: 20%
- `openai`: 10%

**Manual selection**: Edit `src/flow_agent/utils/nodes.py` to specify a provider.

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
```

**Test Coverage:**
- 72 tests across 3 test files
- Unit tests for LLM provider selection
- Unit tests for PII redaction (15 tests)
- Unit tests for speech services (45 tests)

## 📁 Project Structure

```
.
├── src/flow_agent/
│   ├── graph.py                 # LangGraph definition
│   ├── config.py                # Pydantic settings
│   ├── utils/
│   │   ├── state.py             # State management
│   │   ├── nodes.py             # Processing nodes
│   │   ├── input_validation.py  # Security: vulnerability scanning
│   │   └── pii_redaction.py     # Privacy: PII redaction
│   ├── llms/
│   │   └── LangChainChatLLM.py  # Multi-provider interface
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
│   ├── unit_tests/              # 72 unit tests
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

- **Circuit Breaker**: `MAX_TRY`, `SLEEP` (retry behavior)
- **Models**: `GEMINI_VISION_MODEL`, `OPENAI_VISION_MODEL`, etc.
- **Provider Distribution**: `PROVIDER_DISTRIBUTION` (weighted random selection)
- **Security**: `MODERATION_API_CHECK_REQ`, `MODERATION_MODEL`
- **Speech**: `SPEECH_PROVIDER`, model and language settings per provider

## 🤝 Contributing

Found a bug? Have a feature idea? Open an issue or submit a PR. We don't bite (usually).

## 📄 License

MIT — Do whatever you want with this code. Just don't blame us if your AI becomes sentient.

---

**Built with ❤️ and too much coffee**
