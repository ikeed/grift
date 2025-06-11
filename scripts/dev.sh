#!/bin/bash
set -e

# Get the user's home directory explicitly
HOME_DIR="$HOME"
EMULATOR_DIR="$HOME_DIR/.grift/emulator"
PUBSUB_PORT=8085
FIRESTORE_PORT=8081

# Function to check if a port is in use
check_port() {
    local port="$1"
    if lsof -i ":$port" > /dev/null 2>&1; then
        return 0  # Port is in use
    fi
    return 1  # Port is free
}

# Function to kill process using a port
kill_port_process() {
    local port="$1"
    local pid=$(lsof -t -i ":$port" 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo "Killing process using port $port (PID: $pid)"
        kill -9 "$pid" 2>/dev/null || true
    fi
}

# Function to wait for a file to exist
wait_for_file() {
    local file="$1"
    local timeout="$2"
    local count=0

    while [ ! -f "$file" ] && [ $count -lt $timeout ]; do
        sleep 1
        count=$((count + 1))
        echo -n "."
    done
    echo ""

    if [ ! -f "$file" ]; then
        echo "Timeout waiting for $file"
        return 1
    fi
    return 0
}

# Function to wait for an emulator to be ready
wait_for_emulator() {
    local name="$1"
    local port="$2"
    local logfile="$3"
    local ready_message="$4"
    local timeout=30
    local count=0

    echo -n "Waiting for $name emulator to start"
    while [ $count -lt $timeout ]; do
        if grep -q "$ready_message" "$logfile" 2>/dev/null; then
            echo " Ready!"
            return 0
        fi
        sleep 1
        count=$((count + 1))
        echo -n "."
    done
    echo " Timeout!"
    echo "Last few lines of $logfile:"
    tail -n 5 "$logfile"
    return 1
}

# Function to verify emulator is responding
verify_emulator() {
    local name="$1"
    local port="$2"

    if ! nc -z localhost "$port" 2>/dev/null; then
        echo "Error: $name emulator is not responding on port $port"
        return 1
    fi
    return 0
}

# Function to check Java version
check_java() {
    if ! command -v java &> /dev/null; then
        echo "Error: Java not found"
        echo "Please install Java using:"
        echo "  brew install openjdk@17"
        echo "Then link it:"
        echo "  sudo ln -sfn $(brew --prefix)/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk"
        exit 1
    fi

    local java_version=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | cut -d'.' -f1)
    if [ "$java_version" -lt "17" ]; then
        echo "Error: Java 17 or higher is required (found version $java_version)"
        exit 1
    fi
}

# Function to ensure required components are installed
ensure_components() {
    echo "Checking required components..."

    # Install Firestore emulator if needed
    if ! gcloud components list --quiet --filter='id:cloud-firestore-emulator' --format='get(state.name)' | grep -q "Installed"; then
        echo "Installing Cloud Firestore emulator..."
        gcloud components install cloud-firestore-emulator --quiet
    fi

    # Install beta components if needed
    if ! gcloud components list --quiet --filter='id:beta' --format='get(state.name)' | grep -q "Installed"; then
        echo "Installing beta components..."
        gcloud components install beta --quiet
    fi
}

# Initialize local development environment
init_local_dev() {
    echo "Setting up local development environment..."

    # Check Java version
    check_java

    # Ensure required components are installed
    ensure_components

    # Create emulator data directories with full paths
    echo "Creating emulator directories..."
    PUBSUB_DIR="$EMULATOR_DIR/pubsub"
    FIRESTORE_DIR="$EMULATOR_DIR/firestore"

    mkdir -p "$PUBSUB_DIR"
    mkdir -p "$FIRESTORE_DIR"

    # Clean up any existing processes
    echo "Cleaning up existing processes..."
    pkill -f "gcloud.*emulators" || true
    kill_port_process $PUBSUB_PORT
    kill_port_process $FIRESTORE_PORT
    sleep 2

    # Check if ports are available
    if check_port $PUBSUB_PORT; then
        echo "Error: Port $PUBSUB_PORT is still in use. Please free it up first."
        exit 1
    fi

    if check_port $FIRESTORE_PORT; then
        echo "Error: Port $FIRESTORE_PORT is still in use. Please free it up first."
        exit 1
    fi

    echo "Starting emulators..."

    # Start Pub/Sub emulator with absolute paths and explicit Java heap settings
    echo "Starting Pub/Sub emulator..."
    JAVA_TOOL_OPTIONS="-Xmx512m" gcloud beta emulators pubsub start \
        --host-port="0.0.0.0:$PUBSUB_PORT" \
        --data-dir="$PUBSUB_DIR" \
        --log-http \
        --verbosity=debug \
        > /tmp/pubsub-emulator.log 2>&1 &

    # Start Firestore emulator with --quiet flag
    echo "Starting Firestore emulator..."
    JAVA_TOOL_OPTIONS="-Xmx512m" gcloud beta emulators firestore start \
        --host-port="0.0.0.0:$FIRESTORE_PORT" \
        --quiet \
        > /tmp/firestore-emulator.log 2>&1 &

    # Wait for emulators to be ready
    wait_for_emulator "Pub/Sub" "$PUBSUB_PORT" "/tmp/pubsub-emulator.log" "Server started, listening on $PUBSUB_PORT" || {
        echo "Error starting Pub/Sub emulator. Full log:"
        cat /tmp/pubsub-emulator.log
        exit 1
    }

    wait_for_emulator "Firestore" "$FIRESTORE_PORT" "/tmp/firestore-emulator.log" "Dev App Server is now running" || {
        echo "Error starting Firestore emulator. Full log:"
        cat /tmp/firestore-emulator.log
        exit 1
    }

    # Export emulator environment variables
    echo "Setting up environment variables..."
    # First set up Pub/Sub environment
    eval "$(gcloud beta emulators pubsub env-init)"
    # Also set Firestore environment
    export FIRESTORE_EMULATOR_HOST="localhost:$FIRESTORE_PORT"

    echo "Setting up Pub/Sub topics..."
    topics=(
        "w.latent.raw"
        "w.latent.5s"
        "w.latent.15s"
        "M.latent.5s"
        "M.latent.15s"
    )

    # Check and create topics
    for topic in "${topics[@]}"; do
        echo -n "Checking topic: $topic ... "
        if PUBSUB_EMULATOR_HOST="${PUBSUB_EMULATOR_HOST}" gcloud beta pubsub topics describe "$topic" &>/dev/null; then
            echo "exists ✓"
        else
            echo "creating..."
            if PUBSUB_EMULATOR_HOST="${PUBSUB_EMULATOR_HOST}" gcloud beta pubsub topics create "$topic" &>/dev/null; then
                echo "  ✓ Created topic: $topic"
            else
                echo "  ✗ Failed to create topic: $topic"
                echo "Error details:"
                PUBSUB_EMULATOR_HOST="${PUBSUB_EMULATOR_HOST}" gcloud beta pubsub topics create "$topic"
            fi
        fi
    done

    echo "Local environment setup complete!"
    echo "Environment variables set:"
    echo "  PUBSUB_EMULATOR_HOST=$PUBSUB_EMULATOR_HOST"
    echo "  FIRESTORE_EMULATOR_HOST=$FIRESTORE_EMULATOR_HOST"
    echo ""
    echo "Emulator status:"
    echo "  - Pub/Sub:   $(verify_emulator "Pub/Sub" "$PUBSUB_PORT" && echo "Running ✓" || echo "Not running ✗") (port $PUBSUB_PORT)"
    echo "  - Firestore: $(verify_emulator "Firestore" "$FIRESTORE_PORT" && echo "Running ✓" || echo "Not running ✗") (port $FIRESTORE_PORT)"
    echo ""
    echo "Emulator directories:"
    echo "  - Pub/Sub:   $PUBSUB_DIR"
    echo "  - Firestore: $FIRESTORE_DIR"
    echo ""
    echo "Emulator logs:"
    echo "  - Pub/Sub:   /tmp/pubsub-emulator.log"
    echo "  - Firestore: /tmp/firestore-emulator.log"
}

# Start the service in development mode
start_dev() {
    echo "Starting tick-fetcher-decoupler in development mode..."

    # Ensure emulators are running
    if ! nc -z localhost $PUBSUB_PORT 2>/dev/null || ! nc -z localhost $FIRESTORE_PORT 2>/dev/null; then
        echo "Error: Emulators are not running. Please run './scripts/dev.sh init' first"
        exit 1
    fi

    # Export emulator environment variables
    echo "Setting up environment variables..."
    # For Pub/Sub we can use env-init
    eval "$(gcloud beta emulators pubsub env-init)"
    # For Firestore we need to set it manually
    export FIRESTORE_EMULATOR_HOST="localhost:$FIRESTORE_PORT"

    echo "Environment variables set:"
    echo "  PUBSUB_EMULATOR_HOST=$PUBSUB_EMULATOR_HOST"
    echo "  FIRESTORE_EMULATOR_HOST=$FIRESTORE_EMULATOR_HOST"

    # Build and start the services
    docker-compose up --build
}

# Run tests
run_tests() {
    echo "Running tests..."
    python -m pytest tests/services/tick-fetcher-decoupler
}

# Clean up development environment
cleanup() {
    echo "Cleaning up..."
    docker-compose down
    pkill -f "gcloud.*emulators" || true
    rm -f /tmp/pubsub-emulator.log /tmp/firestore-emulator.log
}

# Show help
show_help() {
    echo "Usage: $0 [command]"
    echo "Commands:"
    echo "  init    - Initialize local development environment"
    echo "  start   - Start the service in development mode"
    echo "  test    - Run tests"
    echo "  clean   - Clean up development environment"
    echo ""
    echo "Prerequisites:"
    echo "  - Java 8+ JRE (brew install openjdk@17)"
    echo "  - Google Cloud SDK (brew install --cask google-cloud-sdk)"
}

# Main script
case "$1" in
    "init")
        init_local_dev
        ;;
    "start")
        start_dev
        ;;
    "test")
        run_tests
        ;;
    "clean")
        cleanup
        ;;
    *)
        show_help
        ;;
esac
