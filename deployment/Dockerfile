FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY daeelm_agents/ ./daeelm_agents/
COPY bridge/ ./bridge/
COPY skill_hub/ ./skill_hub/
COPY deployment/ ./deployment/
COPY data/ ./data/
COPY setup.py .
COPY pyproject.toml .

# Set Python path
ENV PYTHONPATH=/app

# Default command to run TaskRunner
CMD ["python", "-c", "from src.core.runner import TaskRunner; TaskRunner.run()"]
