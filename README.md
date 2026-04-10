[![CI](https://github.com/Trojan3877/DeepSequence-Recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/Trojan3877/DeepSequence-Recommender/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-service-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-demo-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Kubernetes](https://img.shields.io/badge/k8s-manifests-informational)
![Prometheus](https://img.shields.io/badge/metrics-prometheus-E6522C?logo=prometheus&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

# DeepSequence Recommender

DeepSequence Recommender is a sequence-aware recommendation service built around a FastAPI application, a modular sequence-processing pipeline, and Prometheus-compatible instrumentation.

This repository is designed to demonstrate production-style ML service design rather than a notebook-only recommender prototype.

---

## Overview

This project focuses on the engineering side of recommendation systems:

- serving recommendations through an API
- separating routing, preprocessing, configuration, and modeling concerns
- exposing service-level metrics for observability
- supporting local development, container execution, and Kubernetes deployment
- providing a Streamlit demo surface for interactive exploration
- providing a cleaner foundation for future benchmarking and model evolution

Instead of presenting recommendation logic only in notebooks, this repo frames the work as a runnable service.

---

## What is implemented today

The current repository includes:

- a FastAPI service entry point with `/health`, `/recommendations`, and `/metrics` endpoints
- a Streamlit interactive demo (`streamlit_app.py`)
- startup-time model and processor initialization with validation
- sequence preprocessing and vocabulary handling
- recommendation API routing
- Prometheus metrics exposure through `/metrics`
- Docker and Kubernetes deployment assets
- automated unit tests and API smoke tests with CI wiring

This means a reviewer can inspect the repo as an application, not just a model artifact.

---

## Architecture

```text
Client request
    ↓
FastAPI application  ─────────────────────────────────────────────────┐
    ↓                                                                  │
Sequence processor → sequence model → top-k recommendations           │
    ↓                                                                  │
/metrics endpoint → Prometheus / monitoring stack                      │
                                                                       │
Streamlit app (streamlit_app.py) ── loads same model ─────────────────┘
```

---

## Local Setup

### Prerequisites

- Python 3.11+
- pip

### 1. Clone and install

```bash
git clone https://github.com/Trojan3877/DeepSequence-Recommender.git
cd DeepSequence-Recommender
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env if you want to override defaults (optional for local demo)
```

Required environment variables (all have safe defaults for local use):

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `change-this-secret` | JWT signing secret |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection (optional) |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server (optional) |
| `ENVIRONMENT` | `development` | Runtime environment label |
| `LOG_LEVEL` | `INFO` | Python logging level |

### 3. Run the FastAPI service

```bash
uvicorn app.main:app --reload --port 8000
```

Then open:
- **API docs (Swagger):** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health
- **Recommendations health:** http://localhost:8000/recommendations/health
- **Metrics (Prometheus):** http://localhost:8000/metrics

#### Example recommendation request

```bash
curl -X POST http://localhost:8000/recommendations/ \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_42", "item_sequence": ["item_1", "item_5", "item_12"], "top_k": 5}'
```

### 4. Run the Streamlit demo

```bash
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser. Select items from the catalogue, choose how many recommendations you want, and click **Get Recommendations**.

---

## Docker

### Build and run locally

```bash
# Build the image
docker build -t deepsequence-recommender .

# Run the container (API only)
docker run --rm -p 8000:8000 --env-file .env deepsequence-recommender
```

### Run with Docker Compose (API + Redis)

```bash
cp .env.example .env
docker compose up --build
```

The API will be available at http://localhost:8000.

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# Unit tests only
pytest tests/test_recommender.py -v

# API smoke tests only
pytest tests/test_smoke_api.py -v
```

---

## Kubernetes Deployment

Manifests are in `k8s/`:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

---

## CI/CD

Every push and pull request triggers the GitHub Actions pipeline (`.github/workflows/ci.yml`):

1. `flake8` lint
2. `black` format check
3. `mypy` type checking
4. `bandit` security scan
5. `pytest` unit tests
6. `pytest` API smoke tests

---

## Architecture Detail

```text
app/
 ├── main.py              – FastAPI app, startup validation, /health, Prometheus mount
 ├── core/
 │    ├── config.py       – Pydantic settings loaded from environment
 │    ├── model.py        – DeepSequenceModel (BiLSTM + attention)
 │    ├── data_processor.py – SequenceProcessor (vocab, padding, encoding)
 │    ├── metrics.py      – Prometheus metrics definitions
 │    └── security.py     – JWT helpers
 └── api/
      └── routes.py       – /recommendations endpoints + /recommendations/health

streamlit_app.py           – Interactive Streamlit demo (same model, no server required)

tests/
 ├── test_recommender.py  – Core unit tests (processor + model)
 └── test_smoke_api.py    – API smoke tests (root, health, recommendations)

k8s/
 ├── deployment.yaml
 ├── service.yaml
 └── hpa.yaml
```

Why is this stronger than a notebook recommender?  
Because it demonstrates service boundaries, startup initialization, observability, deployment assets, and testability — plus an interactive demo surface that requires no separate setup.

What would you improve next for enterprise use?  
Externalize model loading from storage, add benchmark automation, add authentication on production routes, publish quality and latency reports tied to CI, and add a training pipeline.
