# Dockerfile for Simple Multimodal Voice Agent
# Build: docker build -t multimodal-voice-agent .
# Run:   docker run -p 2024:2024 -p 8501:8501 --env-file .env multimodal-voice-agent

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANGGRAPH_ENV=production \
    DEPLOYMENT_URL=http://localhost:2024

WORKDIR /app

# Install uv for faster dependency management
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/
COPY langgraph.json ./
COPY ui/ ./ui/

# Create startup script
RUN echo '#!/bin/sh\n\
echo "Starting LangGraph API server on port 2024..."\n\
uv run langgraph serve &\n\
LANGPID=$!\n\
echo "Starting Streamlit UI on port 8501..."\n\
uv run streamlit run ui/app.py --server.port=8501 --server.address=0.0.0.0 &\n\
UIPID=$!\n\
# Handle shutdown\n\
trap "kill $LANGPID $UIPID 2>/dev/null" EXIT INT TERM\n\
# Wait for any process to exit\n\
wait -n\n' > /app/start.sh && \
    chmod +x /app/start.sh

# Create a non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose ports
# 2024: LangGraph API server
# 8501: Streamlit UI
EXPOSE 2024 8501

# Start both services
CMD ["/app/start.sh"]
