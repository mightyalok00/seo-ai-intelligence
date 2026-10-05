"""
Edge-Case & Input Validation Tests for FastAPI REST Endpoints.

Author: Alok Agarwal (mightyalok00)
License: MIT
"""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

def test_api_invalid_intent_payload():
    # Empty payload should return 422 Unprocessable Entity
    response = client.post("/api/v1/intent/classify", json={})
    assert response.status_code == 422

def test_api_invalid_ranking_payload():
    # Negative word count / invalid probability bounds should fail validation
    payload = {
        "target_keyword": "python",
        "word_count": -500  # Invalid negative count
    }
    response = client.post("/api/v1/predict/ranking", json=payload)
    assert response.status_code == 422

def test_api_404_not_found():
    response = client.get("/api/v1/non_existent_endpoint_xyz")
    assert response.status_code == 404

def test_api_root_discovery():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "documentation" in data
