from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_intent_classify_api():
    payload = {"keywords": ["what is machine learning", "buy vps hosting cheap"]}
    response = client.post("/api/v1/intent/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_keywords"] == 2
    assert data["predictions"][0]["intent"] == "Informational"
    assert data["predictions"][1]["intent"] == "Transactional"

def test_predict_ranking_api():
    payload = {
        "target_keyword": "data science tutorial",
        "word_count": 1500,
        "internal_links_count": 10,
        "external_links_count": 4,
        "keyword_in_title": 1,
        "keyword_in_h1": 1,
        "keyword_in_url": 1,
        "keyword_density": 1.5,
        "semantic_coverage": 0.8,
        "search_intent_match": 0.9,
        "page_speed_score": 85.0,
        "has_schema": 1,
        "image_alt_ratio": 0.9,
        "domain_authority_proxy": 50.0
    }
    response = client.post("/api/v1/predict/ranking", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "overall_seo_score" in data
    assert "ranking_prediction" in data
    assert 0.0 <= data["ranking_prediction"]["top_10_probability"] <= 1.0
