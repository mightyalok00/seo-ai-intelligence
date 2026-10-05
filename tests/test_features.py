import pytest
from src.features.extractor import SEOFeatureExtractor
from src.features.scorer import SEOScorer

def test_feature_extractor():
    extractor = SEOFeatureExtractor()
    mock_crawl = {
        "url": "https://example.com/python-course",
        "title": "Complete Python Course 2026",
        "meta_description": "Learn python from scratch with tutorials.",
        "h1_tags": ["Python Course"],
        "h2_tags": ["Module 1", "Module 2", "FAQs"],
        "word_count": 1200,
        "internal_links_count": 8,
        "external_links_count": 3,
        "images_count": 4,
        "images_missing_alt": 0,
        "has_schema": True,
        "full_text": "This python course covers everything you need to master python programming.",
        "pagespeed": {"performance_score": 85}
    }

    features = extractor.extract_features(mock_crawl, target_keyword="python course")
    assert features["keyword_in_title"] == 1
    assert features["keyword_in_h1"] == 1
    assert features["keyword_in_url"] == 1
    assert features["word_count"] == 1200
    assert features["has_schema"] == 1

def test_seo_scorer():
    scorer = SEOScorer()
    features = {
        "keyword_in_title": 1,
        "keyword_in_h1": 1,
        "h1_is_single": 1,
        "has_schema": 1,
        "title_length_optimal": 1,
        "meta_length_optimal": 1,
        "image_alt_ratio": 1.0,
        "word_count": 1400,
        "h2_count": 4,
        "semantic_coverage": 0.8,
        "internal_links_count": 12,
        "page_speed_score": 90.0
    }
    res = scorer.calculate_score(features, search_intent_match=0.9)
    assert 0 <= res["overall_seo_score"] <= 100
    assert res["overall_seo_score"] >= 80.0
