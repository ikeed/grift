# syntax=docker/dockerfile:1
FROM python:3.11-slim as base

# Install build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements first to leverage Docker cache
COPY services/tick_fetcher_decoupler/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development
RUN pip install --no-cache-dir debugpy

# Copy entrypoint script and make it executable
COPY services/tick_fetcher_decoupler/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["cmd/fetcher.py"]

# Production stage
FROM base as production
# Copy only necessary files
COPY shared/ /app/shared/
COPY grift/ /app/grift/
COPY cmd/ /app/cmd/
COPY services/tick_fetcher_decoupler/ /app/services/tick_fetcher_decoupler/

ENV PYTHONPATH=/app
EXPOSE 8080

CMD ["python", "cmd/fetcher.py"]
