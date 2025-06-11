# syntax=docker/dockerfile:1
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY services/tick-fetcher-decoupler/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage with debugging tools
FROM base as development
RUN pip install debugpy
ENV PYTHONPATH=/app
EXPOSE 5678 8080
CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "cmd/fetcher.py"]

# Production stage
FROM base as production
# Copy only necessary files
COPY shared/ /app/shared/
COPY grift/ /app/grift/
COPY cmd/ /app/cmd/
COPY services/tick-fetcher-decoupler/ /app/services/tick-fetcher-decoupler/

ENV PYTHONPATH=/app
EXPOSE 8080

CMD ["python", "cmd/fetcher.py"]
