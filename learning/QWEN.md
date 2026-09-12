# Qwen Code Context: Simple Multimodal Voice Agent

## Project Overview

The Simple Multimodal Voice Agent is an advanced AI application built with LangGraph that combines multimodal input processing (text, images, audio) with voice input/output capabilities. The project serves as a starter template for creating sophisticated conversational AI agents with enterprise-grade security features.

### Key Technologies
- **LangGraph**: Core orchestration framework for the agent workflow
- **LangChain**: LLM integration and multimodal processing
- **Streamlit**: Web interface for user interaction
- **Pydantic**: Configuration and settings management
- **Presidio**: PII redaction and privacy protection
- **SarvamAI**: Speech-to-text and text-to-speech services

### Architecture
The agent follows a stateful graph architecture with the following nodes:
1. **Entry Node**: Processes incoming messages and performs PII redaction
2. **Input Validator**: Security layer with vulnerability scanning and OpenAI moderation
3. **Reasoning Node**: LLM processing with multimodal content support
4. **Conditional Edges**: Smart routing based on validation results

## Building and Running

### Prerequisites
- Python 3.10+
- API keys for selected LLM providers (OpenAI, Google Gemini, Zhipu/Zai, or Ollama)
- SarvamAI API key for speech services

### Setup
```bash
# Install dependencies
uv sync
# OR
pip install -e .

# Download Spacy English model for PII redaction
python -m spacy download en_core_web_sm
```

### Configuration
1. Copy `.env.example` to `.env`
2. Add your API keys for the services you plan to use
3. Configure settings in `src/flow_agent/config.py`

### Running the Agent
```bash
# Terminal 1: Start LangGraph backend with hot-reload
langgraph up --watch

# Terminal 2: Start the web UI
streamlit run ui/app.py
```

**Access Points:**
- LangGraph API: http://localhost:8123
- API Docs: http://localhost:8123/docs
- Streamlit UI: http://localhost:8501

## Development Conventions

### Code Quality
- **Linting**: `ruff check .`
- **Auto-fix**: `ruff check --fix .`
- **Type checking**: `mypy src/`

### Testing
- **Run all tests**: `pytest`
- **Run with verbose output**: `pytest -v`
- **Run specific test file**: `pytest tests/unit_tests/test_pii_redaction.py`

### Security Features
The agent implements defense-in-depth security with three layers:
1. **Input Validation**: Pattern-based detection for 100+ attack vectors
2. **Moderation API**: OpenAI's omni-moderation-latest for text and image content
3. **PII Redaction**: Microsoft Presidio for detecting and redacting sensitive information

### LLM Provider Options
The agent supports multiple LLM providers with automatic weighted distribution:
- **Ollama**: 50% (default) - `qwen3-vl:235b-instruct-cloud`
- **Zhipu/Zai**: 20% - `GLM-4.6V-Flash`
- **Google Gemini**: 20% - `gemma-3-27b-it`
- **OpenAI**: 10% - `gpt-5-nano`

### Speech Services
Multi-provider speech services using a factory pattern:
- **SarvamAI**: Default provider with Saaras STT and Bulbul TTS models
- **OpenAI**: Whisper STT and TTS models
- **Google Gemini**: GenAI STT and TTS models

## Project Structure
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

## Configuration

### LangGraph Configuration (`langgraph.json`)
- Graph entry point: `./src/flow_agent/graph.py:graph`
- Graph name: `agent`
- Environment file: `.env`
- Distribution: `wolfi`

### Application Settings (`src/flow_agent/config.py`)
Key configurable settings:
- Circuit breaker: `MAX_TRY`, `SLEEP`
- Models: `GEMINI_VISION_MODEL`, `OPENAI_VISION_MODEL`, etc.
- Provider distribution: `PROVIDER_DISTRIBUTION` (weighted random selection)
- Security: `MODERATION_API_CHECK_REQ`, `MODERATION_MODEL`
- Speech: `SPEECH_PROVIDER`, model and language settings per provider

## Special Considerations

### Docker Deployment
Dependencies must be explicitly listed in `langgraph.json` for Docker builds. The Docker build process does NOT automatically read from `pyproject.toml`.

### Multimodal Content Format
The agent accepts content in the following format:
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

### Circuit Breaker Implementation
The agent includes a trivial circuit breaker with exponential backoff in the `call_llm_safely` function to handle API failures gracefully.