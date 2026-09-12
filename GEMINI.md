# Multimodal Voice Agent

## Project Overview

This project is a multimodal voice agent built with Python and LangGraph. It can process text, images, and voice input, and it can respond with voice output. The agent is designed to be a conversational companion that can understand and respond to a variety of inputs.

The key technologies used in this project are:

*   **Python:** The primary programming language.
*   **LangGraph:** A framework for building stateful, multi-actor applications with LLMs.
*   **Streamlit:** Used to create the web interface for the agent.
*   **Docker:** For containerization and deployment.
*   **Pytest:** For testing.
*   **Ruff:** For linting and code formatting.
*   **MyPy:** For static type checking.

The agent's architecture is based on a LangGraph StateGraph, which includes nodes for input validation, reasoning (LLM processing), and summarization. The agent supports multiple LLM providers, including OpenAI, Google Gemini, and others, and it has built-in features for conversation memory, search, and security.

## Building and Running

### Prerequisites

*   Python 3.10 or higher
*   API keys for the desired LLM and speech services

### Installation

1.  Install dependencies:
    ```bash
    uv sync
    ```
    or
    ```bash
    pip install -e .
    ```
2.  Download the Spacy English model for PII redaction:
    ```bash
    python -m spacy download en_core_web_sm
    ```

### Configuration

1.  Create a `.env` file from the example:
    ```bash
    cp .env.example .env
    ```
2.  Edit the `.env` file to add your API keys.

### Running the Agent

The recommended way to run the agent is with Docker Compose:

1.  Start the LangGraph backend:
    ```bash
    langgraph up --watch
    ```
2.  Start the web UI:
    ```bash
    streamlit run ui/app.py
    ```

The LangGraph API will be available at `http://localhost:8123` and the Streamlit UI at `http://localhost:8501`.

## Development Conventions

### Code Quality

*   **Linting:** `ruff check .`
*   **Auto-fixing:** `ruff check --fix .`
*   **Type checking:** `mypy src/`

### Testing

*   Run all tests: `pytest`
*   Run tests with verbose output: `pytest -v`
*   Run a specific test file: `pytest tests/unit_tests/test_pii_redaction.py`
