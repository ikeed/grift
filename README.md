# GRIFT Cloud Pipeline

This repository contains the implementation of the GRIFT cloud pipeline, a system for processing financial data from OANDA FX Tick API.

## Project Structure

The project follows a modular structure with the following main components:

- **tick-fetcher-decoupler**: A python service that fetches tick data from OANDA FX Tick API, decouples it, and publishes w-vectors to Pub/Sub.
- **w-rollup-aggregator**: A Python Dataflow pipeline that reads raw w-vectors, windows them, computes elementwise averages, and writes to separate Pub/Sub topics.
- **matrix_solver_fn**: A Python Dataflow pipeline that reads adjacent pairs of rollup vectors, invokes a solver, and publishes to Pub/Sub.
- **matrix_blender**: A component that accumulates a sliding window of matrices and emits a blended result.

## Getting Started

### Prerequisites

- Python 3.13 or later
- Google Cloud SDK
- Access to OANDA FX Tick API

### Installation

1. Clone this repository
2. Set up the required GCP resources (Pub/Sub topics, Firestore, Secret Manager)
3. Deploy the components as described in the deployment instructions

## Deployment

Each component has its own deployment instructions:

- **tick-fetcher-decoupler**: Deploy as a GKE container
- **w-rollup-aggregator**: Deploy as a Dataflow job
- **matrix_solver_fn**: Deploy as a Dataflow job or Cloud Function
- **matrix_blender**: Deploy as a Dataflow job or GKE container

## Configuration

The project uses Secret Manager for storing sensitive information like OANDA API tokens. Separate secrets are maintained for development and production environments.

## Documentation

For more detailed information, refer to the following documents:

- [Architecture](architecture.md): Detailed architecture plan for the GRIFT cloud pipeline
- [Guidelines](guidelines.md): Guidelines for organizing the codebase, infrastructure, and CI/CD pipelines