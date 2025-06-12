# GRIFT (Generic Real-time Instrument Forecasting Technology)

A distributed system for real-time currency strength modeling and forecasting.

## Local Development Setup

### Prerequisites

1. **Java 17+**
   ```bash
   brew install openjdk@17
   sudo ln -sfn $(brew --prefix)/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk
   ```

2. **Google Cloud SDK**
   ```bash
   brew install --cask google-cloud-sdk
   ```

3. **Docker Desktop**
   - Install from [Docker's website](https://www.docker.com/products/docker-desktop)
   - Ensure it's running before starting development

### First-Time Setup

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd Grift
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r services/tick_fetcher_decoupler/requirements.txt
   ```

3. **Initialize development environment**
   ```bash
   chmod +x scripts/dev.sh  # Make script executable
   ./scripts/dev.sh init    # Sets up emulators and required infrastructure
   ```

   This will:
   - Install required Google Cloud components
   - Start local Pub/Sub emulator
   - Start local Firestore emulator
   - Create necessary Pub/Sub topics
   - Configure environment variables

### Development Workflow

1. **Start the service**
   ```bash
   ./scripts/dev.sh start
   ```
   This will:
   - Verify emulators are running
   - Build and start the service containers
   - Enable hot-reloading for development

2. **Access development tools**
   - Debug port: 5678 (attach with VS Code or PyCharm)
   - Health check: http://localhost:8080/healthz
   - Metrics: http://localhost:8080/metrics

3. **Clean up resources**
   ```bash
   ./scripts/dev.sh clean
   ```
   This stops all containers and emulators.

### Debugging

- **View emulator logs**:
  ```bash
  tail -f /tmp/pubsub-emulator.log
  tail -f /tmp/firestore-emulator.log
  ```

- **Container logs**: Available through Docker Desktop or:
  ```bash
  docker-compose logs -f tick_fetcher_decoupler
  ```

- **Debug with VS Code**:
  1. Start the service with `./scripts/dev.sh start`
  2. Attach debugger to port 5678
  3. Set breakpoints and debug as normal

### Common Issues

1. **Port conflicts**
   - Pub/Sub emulator uses port 8085
   - Firestore emulator uses port 8081
   - Debug port uses 5678
   - Run `./scripts/dev.sh clean` to free up ports

2. **Emulator not starting**
   - Check Java installation: `java -version`
   - Ensure ports are free: `lsof -i :8080` and `lsof -i :8085`
   - Check logs in /tmp/

### Project Structure

- `cmd/` - Service entry points
- `shared/` - Common libraries and utilities
- `services/` - Individual service implementations
- `infra/` - Infrastructure configurations
- `scripts/` - Development and deployment tools

## Architecture

For detailed architecture information, see:
- [Cloud Architecture](docs/grift_cloud_pipeline.md)
- [Architecture Overview](architecture.md)

## License

[License details here]
