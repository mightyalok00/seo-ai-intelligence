from src.models.ctr_forecaster import CTRForecaster
from src.models.ranking_predictor import RankingPredictor


def test_ranking_predictor():
    predictor = RankingPredictor()
    test_features = {
        "domain_authority_proxy": 65.0,
        "backlink_count": 350,
        "word_count": 1800,
        "keyword_in_title": 1,
        "keyword_in_h1": 1,
        "keyword_in_url": 1,
        "keyword_density": 1.8,
        "semantic_coverage": 0.85,
        "search_intent_match": 0.90,
        "internal_links_count": 14,
        "page_speed_score": 88.0,
        "has_schema": 1,
        "image_alt_ratio": 0.95
    }

    res = predictor.predict_probability(test_features)
    assert 0.0 <= res["top_10_probability"] <= 1.0
    assert "ranking_tier" in res
    assert len(res["feature_impacts"]) > 0

def test_ctr_forecaster():
    forecaster = CTRForecaster()
    forecast = forecaster.forecast_traffic(ranking_prob=0.85, monthly_search_volume=5000)
    assert forecast["forecast_monthly_clicks"] > 0
    assert len(forecast["serp_ctr_curve"]) == 10
