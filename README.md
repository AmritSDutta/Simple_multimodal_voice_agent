# 🎙️ Multimodal Voice Agent

> *An AI agent that sees images, hears your voice, and talks back. Skynet, but friendlier.*

Meet your new favorite conversational companion: a LangGraph-based multimodal voice agent that actually listens—literally. Whether you want to type, upload photos, or just talk at your computer like a sci-fi protagonist, this agent has you covered.

## ✨ Features

- **🗣️ Voice Input & Output** — Talk to your computer. It talks back. Welcome to the future.
- **📷 Image Analysis** — Upload images and the AI will tell you what's in them. (Spoiler: it's probably a cat.)
- **🧠 Multiple LLM Brains** — Choose between OpenAI, Google Gemini, Zhipu/Zai, or Ollama. Variety is the spice of life.
- **🔍 Built-in Search** — DuckDuckGo integration means it can fact-check itself. Imagine if humans could do that.
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
```

### Configuration

Create a `.env` file with your secrets:

```bash
# For Zhipu/Zai
ZAI_API_KEY=your_key_here

# For Ollama
OLLAMA_API_KEY=your_key_here

# For speech magic
SARVAM_API_KEY=your_sarvam_key
```

### Running the Agent

```bash
# Terminal 1: Start the LangGraph server
langgraph dev

# Terminal 2: Start the web UI
streamlit run ui/app.py
```

Then open `http://localhost:8501` and start conversing.

## 🏗️ Architecture

The agent is built as a LangGraph StateGraph (fancy talk for "state machine on steroids"):

```
START → entry → conditional edge → reasoning → END
                    ↓
                   END (if thread closed)
```

**What's happening here?**

- **entry** — Checks if you've already had this conversation. No double-dipping.
- **reasoning** — The LLM does its thing, processing your text/images with style.
- **conditional edge** — Decides whether to continue or call it quits.

### Multimodal Content

The agent accepts content in this format:

```python
HumanMessage(content=[
    {'type': 'text', 'text': 'what is there in the image'},
    {
        'type': 'image',
        'data': 'base64_encoded_data',  # because why not?
        'metadata': {'filename': 'mystery_cat.png'},
        'source_type': 'base64',
        'mime_type': 'image/png'
    }
])
```

## 🧩 LLM Provider Options

| Provider | Model | Vibe |
|----------|-------|------|
| `ollama` | `qwen3-vl:235b-instruct-cloud` | The hipster choice |
| `zhipu` / `zai` | `GLM-4.6V-Flash` | Vision-capable and ready |
| `openai` | `gpt-5-nano` | The classic |
| `gemini` | `gemma-3-27b-it` | Google's contribution |

**Switch providers** by editing `src/flow_agent/utils/nodes.py:81`:

```python
provider: str = 'zai'  # Change me!
```

## 🗣️ Speech Features

Powered by [SarvamAI](https://sarvam.ai/):

- **Speech-to-Text**: Model `saaras:v3` transcribes your voice into text (English/India)
- **Text-to-Speech**: Model `bulbul:v3` with speaker `shubh` reads responses back to you

Because typing is *so* 2019.

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
```

## 📁 Project Structure

```
.
├── src/flow_agent/
│   ├── graph.py          # LangGraph definition
│   ├── utils/
│   │   ├── state.py      # State management
│   │   └── nodes.py      # Processing nodes
│   ├── llms/
│   │   ├── LangChainChatLLM.py    # Multi-provider interface
│   │   └── genai_agent.py          # Native Google GenAI
│   └── logging_config.py  # Color-coded logging
├── ui/
│   └── app.py             # Streamlit web interface
├── tests/                 # Currently empty, like my motivation
├── langgraph.json         # LangGraph configuration
└── pyproject.toml         # Dependencies & tool config
```

## ⚙️ Configuration

### LangGraph Configuration (`langgraph.json`)

- Graph entry point: `./src/flow_agent/graph.py:graph`
- Graph name: `agent`
- Environment file: `.env`
- Distribution: `wolfi` (sounds cool, right?)

## 🤝 Contributing

Found a bug? Have a feature idea? Open an issue or submit a PR. We don't bite (usually).

## 📄 License

MIT — Do whatever you want with this code. Just don't blame us if your AI becomes sentient.

---

**Built with ❤️ and too much coffee**
