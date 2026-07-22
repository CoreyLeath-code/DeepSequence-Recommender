"""Smoke tests for the DeepSequence Recommender FastAPI endpoints.

These tests use FastAPI's TestClient so no running server is required.
They exercise the full request/response cycle including startup initialisation.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Keep smoke tests independent of developer-generated artifacts in models/current.
os.environ["ENVIRONMENT"] = "development"
os.environ["MODEL_BUNDLE_PATH"] = "models/test-bundle-does-not-exist"

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


class TestRootEndpoints:
    def test_root_returns_200(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200

    def test_root_payload(self, client: TestClient) -> None:
        data = client.get("/").json()
        assert data["service"] == "DeepSequence Recommender"
        assert "version" in data

    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_payload(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_recommendations_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/recommendations/health")
        assert response.status_code == 200

    def test_recommendations_health_model_loaded(self, client: TestClient) -> None:
        data = client.get("/recommendations/health").json()
        assert data["model_loaded"] is True
        assert data["vocab_size"] > 0


class TestRecommendEndpoint:
    def test_recommend_success(self, client: TestClient) -> None:
        payload = {
            "user_id": "smoke_user",
            "item_sequence": ["item_1", "item_2", "item_3"],
            "top_k": 5,
        }
        response = client.post("/recommendations/", json=payload)
        assert response.status_code == 200

    def test_recommend_response_schema(self, client: TestClient) -> None:
        payload = {
            "user_id": "smoke_user",
            "item_sequence": ["item_1", "item_2", "item_3"],
            "top_k": 5,
        }
        data = client.post("/recommendations/", json=payload).json()
        assert data["user_id"] == "smoke_user"
        assert isinstance(data["recommendations"], list)
        assert isinstance(data["latency_ms"], float)

    def test_recommend_top_k_respected(self, client: TestClient) -> None:
        payload = {
            "user_id": "smoke_user",
            "item_sequence": ["item_1", "item_2"],
            "top_k": 3,
        }
        data = client.post("/recommendations/", json=payload).json()
        assert len(data["recommendations"]) <= 3

    def test_recommend_empty_sequence(self, client: TestClient) -> None:
        payload = {
            "user_id": "empty_seq_user",
            "item_sequence": [],
            "top_k": 5,
        }
        response = client.post("/recommendations/", json=payload)
        assert response.status_code == 422

    def test_recommend_unknown_items(self, client: TestClient) -> None:
        payload = {
            "user_id": "smoke_user",
            "item_sequence": ["unknown_xyz", "another_unknown"],
            "top_k": 5,
        }
        response = client.post("/recommendations/", json=payload)
        assert response.status_code == 422

    def test_recommend_rejects_top_k_beyond_contract(self, client: TestClient) -> None:
        response = client.post(
            "/recommendations/",
            json={"user_id": "u", "item_sequence": ["item_1"], "top_k": 51},
        )
        assert response.status_code == 422

    def test_response_exposes_model_provenance(self, client: TestClient) -> None:
        response = client.post(
            "/recommendations/",
            json={"user_id": "u", "item_sequence": ["item_1"], "top_k": 3},
        )
        assert response.json()["model_version"] == "development-untrained"
        assert response.json()["fallback"] is False


class TestFeedbackEndpoint:
    def test_feedback_is_accepted_without_logging_raw_user_id(
        self, client: TestClient, caplog
    ) -> None:
        payload = {
            "impression_id": "imp-1",
            "user_id": "private-user",
            "item_id": "item_3",
            "event_type": "click",
            "position": 2,
            "model_version": "development-untrained",
        }
        with caplog.at_level("INFO"):
            response = client.post("/recommendations/feedback", json=payload)
        assert response.status_code == 202
        assert "private-user" not in caplog.text
        assert "anonymous_user_id" in caplog.text
