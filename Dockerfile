# Use latest stable Python 3.13 slim image as base
FROM python:3.13-slim

# Metadata labels
LABEL maintainer="nawatech-dev@example.com"
LABEL description="Nawatech FAQ Chatbot with RAG and LangChain"
LABEL version="1.0.0"

# Environment variables for better Python and pip behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies and clean apt cache in one RUN to optimize image size
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Create necessary directories and set ownership
RUN mkdir -p /app/data /app/logs && chown -R appuser:appuser /app

# Copy app source code with correct ownership
COPY --chown=appuser:appuser . .

# Create __init__.py files as needed
RUN touch /app/chatbot/__init__.py && touch /app/config/__init__.py

# Switch to non-root user
USER appuser

# Expose service port
EXPOSE 8501

# Healthcheck for Streamlit app endpoint (verify endpoint exists in app)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Execute the Streamlit app with production-ready options
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
