# Project Guidelines for Junie (AI Agent)

These guidelines define the **project layout**, **code conventions**, **testing strategy**, **infrastructure setup**, and **tooling** for the GRIFT codebase, implemented entirely in Python.

---

## 1. Directory Structure

```
grift/                         # Root of repository
├── cmd/                       # Entry-point scripts (CLIs & services)
│   ├── fetcher.py             # Ingestion & decoupling loop
│   ├── rollup.py              # Rollup & blending service
│   ├── solver.py              # Matrix solver (nonlinear optimization)
│   ├── forecaster.py          # Forecasting & trajectory generation
│   └── signal_detector.py     # Entry/exit signal logic
├── grift/                     # Core library packages
│   ├── config.py              # Configuration loader (pydantic, env)
│   ├── logger.py              # Structured logging setup
│   ├── pubsub_client.py       # Pub/Sub publisher/subscriber wrappers
│   ├── firestore_client.py    # Firestore checkpoint utilities
│   ├── redis_client.py        # Redis-based ephemeral buffers
│   └── decoupler.py           # Your existing ContinuousDecoupler class
├── infra/                     # Infrastructure-as-Code
│   ├── terraform/             # Terraform modules & envs
│   ├── k8s/                   # Kubernetes manifests (GKE)
│   └── cloudrun/              # Cloud Run service configs (if used)
├── tests/                     # Test suites
│   ├── unit/                  # pytest unit tests by module
│   ├── integration/           # integration tests with Pub/Sub emulator
│   ├── smoke/                 # smoke tests for endpoints
│   └── fixtures/              # sample JSON/CSV tick data
├── tools/                     # Developer tooling
│   ├── pubsub_simulator/      # Embedded Pub/Sub simulator in Python
│   └── dev_setup.sh           # Local bootstrap script
├── docs/                      # Documentation (architecture, design)
│   └── guidelines.md          # This file
├── .github/                   # GitHub Actions workflows
│   └── main.yml
├── .gitignore
├── LICENSE
└── README.md
```

---

## 2. Naming Conventions

- **Python packages** and modules: **snake_case** (e.g., `decoupler.py`, `pubsub_client.py`).
- **Script entry points** in `cmd/`: **snake_case** with `.py` suffix.
- **Terraform modules**: lowercase with underscores (e.g., `gke_module`).
- **Kubernetes YAML**: `deployment-*.yaml`, `service-*.yaml`.
- **Cloud Run configs**: `service-*.yaml`.

---

## 3. Configuration Management

- Use **pydantic** models in `config.py` to validate and load environment variables.
- Support a `.env` file for local development, loaded via `python-dotenv`.
- Secrets (e.g., API keys) stored in **Secret Manager**, injected into env vars at runtime.
- Provide `config.example.env` as a template.

---

## 4. Testing Strategy

### 4.1 Unit Tests
- Use **pytest**. Place tests under `tests/unit/` mirroring module structure.
- Write tests as `test_*.py`, using fixtures in `tests/fixtures/`.
- Mock external systems (Pub/Sub, Firestore, Redis) with `unittest.mock` or `pytest-mock`.

### 4.2 Integration Tests
- Use the **embedded Pub/Sub simulator** from `tools/pubsub_simulator/`.
- Tests in `tests/integration/` should:
  1. Start the simulator as a subprocess.
  2. Publish sample `w.latent.raw` messages.
  3. Assert rollup outputs or downstream API responses.

### 4.3 Smoke Tests
- Located in `tests/smoke/`.
- Validate each service’s HTTP or CLI interface responds correctly.

---

## 5. Infrastructure-as-Code (Terraform)

- Organize modules under `infra/terraform/modules/`:
  - `gke/`, `pubsub/`, `firestore/`, `redis/`, `cloudrun/`.
- Define environments under `infra/terraform/envs/{dev,staging,prod}` with `terraform.tfvars`.
- Use separate workspaces per environment.

---

## 6. Pub/Sub Simulation

- The Python-based simulator in `tools/pubsub_simulator/` should implement:
  - HTTP endpoints matching Pub/Sub publish & pull APIs.
  - In-memory message queues with ack semantics.
- Integration tests override `PUBSUB_EMULATOR_HOST` to point at the simulator.

---

## 7. CI/CD Guidelines

- GitHub Actions in `.github/main.yml`:
  1. **Lint & format**: `black .`, `flake8 .`, `isort .`
  2. **Unit tests**: `pytest --maxfail=1 --disable-warnings -q`
  3. **Build Docker images**: for each service under `cmd/`, tag and push to Artifact Registry.
  4. **Terraform**: `terraform init && terraform plan && terraform apply --auto-approve --var-file=infra/terraform/envs/dev.tfvars`.
  5. **Deploy services**: to GKE (`kubectl apply -f infra/k8s/`) or Cloud Run (`gcloud run deploy ...`).
  6. **Integration & smoke tests**: against deployed dev cluster.

---

## 8. Documentation

- Keep `README.md` updated with:
  - Local dev setup
  - Running services and tests
  - Deploying infra & code
- Store architecture diagrams in `docs/` (Mermaid or PlantUML).

---

*End of guidelines.md*
