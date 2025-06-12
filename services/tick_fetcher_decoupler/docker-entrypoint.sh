#!/bin/sh
set -e

echo "[entrypoint] Starting with DEBUG_WAIT=${DEBUG_WAIT:-false}"

# If we have credentials JSON, write it to a file
if [ ! -z "$GOOGLE_APPLICATION_CREDENTIALS_JSON" ]; then
    echo "[entrypoint] Writing application credentials..."
    mkdir -p /tmp/gcloud
    echo "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > /tmp/gcloud/credentials.json
    export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcloud/credentials.json
    echo "[entrypoint] Credentials written successfully"
fi

# Start the application
if [ "${DEBUG_WAIT:-false}" = "true" ]; then
    echo "[entrypoint] Starting in debug wait mode..."
    # Start with debugger in wait mode
    python ${PYTHONARGS} -m debugpy --listen 0.0.0.0:5680 --wait-for-client /app/services/tick_fetcher_decoupler/main.py
else
    echo "[entrypoint] Starting in normal mode..."
    # Start without debugger
    python ${PYTHONARGS} /app/services/tick_fetcher_decoupler/main.py
fi
