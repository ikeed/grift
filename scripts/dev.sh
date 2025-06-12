#!/bin/bash
set -e

# Function to start emulators
start_emulators() {
    echo "Starting emulators..."
    # Start pubsub emulator in the background
    gcloud beta emulators pubsub start --project=grift-forex --host-port=localhost:8085 &
    PUBSUB_PID=$!

    # Give the emulator time to start
    sleep 5

    # Set environment variables
    $(gcloud beta emulators pubsub env-init)

    # Save PID for cleanup
    echo $PUBSUB_PID > /tmp/pubsub_emulator.pid
}

# Function to stop emulators
stop_emulators() {
    echo "Stopping emulators..."
    if [ -f /tmp/pubsub_emulator.pid ]; then
        kill $(cat /tmp/pubsub_emulator.pid) 2>/dev/null || true
        rm /tmp/pubsub_emulator.pid
    fi
}

# Check for gcloud and debug PATH if not found
if ! command -v gcloud &> /dev/null; then
    echo "Error: 'gcloud' command not found"
    echo "Current PATH: $(echo $PATH)"
    # Try to find gcloud in common locations
    POSSIBLE_LOCATIONS=(
        "/Users/Craig.Burnett/google-cloud-sdk/bin/gcloud"
        "/usr/local/google-cloud-sdk/bin/gcloud"
        "/usr/local/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/bin/gcloud"
    )

    for loc in "${POSSIBLE_LOCATIONS[@]}"; do
        echo "Checking for gcloud at: $loc"
        if [ -f "$loc" ]; then
            echo "Found gcloud at $loc"
            export PATH="$(dirname "$loc"):$PATH"
            echo "Updated PATH: $PATH"
            if command -v gcloud &> /dev/null; then
                echo "Successfully added gcloud to PATH"
                break
            fi
        fi
    done

    # If we still can't find gcloud, exit with error
    if ! command -v gcloud &> /dev/null; then
        echo "Could not find gcloud in any standard location."
        echo "Please check your Google Cloud SDK installation and ensure it's in your PATH"
        exit 1
    fi
fi

# Check if user is authenticated with gcloud
if ! gcloud auth list --filter=status:ACTIVE --format="get(account)" 2>/dev/null | grep -q "@"; then
    echo "Not authenticated with gcloud. Please login first..."
    gcloud auth login
    echo "Please run the script again after authentication."
    exit 1
fi

# Check for application default credentials
if [ ! -f "$HOME/.config/gcloud/application_default_credentials.json" ]; then
    echo "Application default credentials not found. Setting them up..."
    gcloud auth application-default login
    echo "Please run the script again after setting up credentials."
    exit 1
fi

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
    tail -n 15 "$logfile"
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
    echo "Java version $java_version found.  you're good to go!"
}

# Function to ensure required components are installed
ensure_components() {
    echo "Updating Cloud SDK and components..."

    # First update the SDK itself
    gcloud components update --quiet || {
        echo "Failed to update Cloud SDK. Please run:"
        echo "  gcloud components update"
        echo "manually, or run gcloud init to ensure your SDK is properly configured."
        exit 1
    }

    # Update all required components
    gcloud components update beta cloud-firestore-emulator pubsub-emulator --quiet || {
        echo "Failed to update required components. Please try running:"
        echo "  gcloud components update beta cloud-firestore-emulator pubsub-emulator"
        echo "manually, or run gcloud init to ensure your SDK is properly configured."
        exit 1
    }

    # Verify installations
    for component in "beta" "cloud-firestore-emulator" "pubsub-emulator"; do
        if ! gcloud components list --filter="id:$component" --format="get(state.name)" | grep -q "Installed"; then
            echo "Error: Component $component is not properly installed."
            echo "Please run: gcloud components install $component"
            exit 1
        fi
    done

    echo "All required components are up to date."
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
        --quiet \
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
    # Set up Pub/Sub environment with explicit data directory
    eval "$(CLOUDSDK_EMULATOR_PUBSUB_DATA_DIR=$PUBSUB_DIR gcloud beta emulators pubsub env-init --quiet)" || {
        echo "Failed to initialize Pub/Sub environment. Setting manually..."
        export PUBSUB_EMULATOR_HOST="localhost:$PUBSUB_PORT"
    }
    # Set Firestore environment
    export FIRESTORE_EMULATOR_HOST="localhost:$FIRESTORE_PORT"

    # Verify environment variables are set
    if [ -z "$PUBSUB_EMULATOR_HOST" ]; then
        echo "Warning: PUBSUB_EMULATOR_HOST not set, using default"
        export PUBSUB_EMULATOR_HOST="localhost:$PUBSUB_PORT"
    fi

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

    # Read and export application default credentials content
    if [ -f "$HOME/.config/gcloud/application_default_credentials.json" ]; then
        export GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat "$HOME"/.config/gcloud/application_default_credentials.json)"
    else
        echo "Error: Application default credentials not found. Please run 'gcloud auth application-default login' first."
        exit 1
    fi

    echo "Environment variables set:"
    echo "  PUBSUB_EMULATOR_HOST=$PUBSUB_EMULATOR_HOST"
    echo "  FIRESTORE_EMULATOR_HOST=$FIRESTORE_EMULATOR_HOST"
    echo "  Application credentials loaded ✓"

    # Build and start the services with Python frozen modules disabled
    PYTHONARGS="-Xfrozen_modules=off" docker-compose up --build
}

# Run tests
run_tests() {
    echo "Running tests..."
    python -m pytest tests/services/tick_fetcher_decoupler
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
