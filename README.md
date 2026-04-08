[![CI](https://github.com/Trojan3877/DeepSequence-Recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/Trojan3877/DeepSequence-Recommender/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-service-009688?logo=fastapi&logoColor=white)
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
- providing a cleaner foundation for future benchmarking and model evolution

Instead of presenting recommendation logic only in notebooks, this repo frames the work as a runnable service.

---

## What is implemented today

The current repository includes:

- a FastAPI service entry point
- startup-time model and processor initialization
- sequence preprocessing and vocabulary handling
- recommendation API routing
- Prometheus metrics exposure through `/metrics`
- Docker and Kubernetes deployment assets
- automated tests and CI wiring referenced from the repository root

This means a reviewer can inspect the repo as an application, not just a model artifact.

---

## Architecture

```text
Client request
    ↓
FastAPI application
    ↓
Sequence processor → sequence model → top-k recommendations
    ↓
/metrics endpoint → Prometheus / monitoring stack

Why is this stronger than a notebook recommender?
Because it demonstrates service boundaries, startup initialization, observability, deployment assets, and testability.
What makes the repo production-style even if it is still a portfolio project?
It is organized like a service, with routing, config, model initialization, and metrics exposed in a way that resembles real backend ML systems.
What would you improve next for enterprise use?
I would externalize model loading, add benchmark automation, add authentication or rate limiting, and publish quality and performance reports tied to CI.