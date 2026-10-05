"""
Unit and Integration Tests for Recommendation Prioritizer, AI Analyst, and Semantic Matcher.

Author: Alok Agarwal (mightyalok00)
License: MIT
"""

import pytest

from src.nlp.semantic_matcher import SemanticMatcher
from src.recommendations.ai_analyst import AISEOAnalyst
from src.recommendations.prioritizer import RecommendationPrioritizer


def test_recommendation_prioritizer_full_coverage():
    prioritizer = RecommendationPrioritizer()
    features = {
        "keyword_in_title": 0,
        "keyword_in_h1": 0,
        "h1_is_single": 0,
        "has_schema": 0,
        "internal_links_count": 2,
        "word_count": 500,
        "image_alt_ratio": 0.5,
        "images_count": 4,
        "page_speed_score": 55.0
    }
    intent_data = {"intent": "Transactional", "confidence": 0.60}
    missing_topics = ["Neural Networks", "Gradient Boosting", "Cross-validation"]

    matrix = prioritizer.prioritize(
        features=features,
        current_prob=0.45,
        intent_data=intent_data,
        missing_topics=missing_topics
    )

    assert "prioritized_actions" in matrix
    assert len(matrix["prioritized_actions"]) >= 5
    assert matrix["potential_ranking_probability"] > matrix["current_ranking_probability"]
    assert all("roi_score" in a for a in matrix["prioritized_actions"])

@pytest.mark.asyncio
async def test_ai_seo_analyst_generation():
    analyst = AISEOAnalyst()
    report = await analyst.generate_seo_report(
        url="https://example.com/python-guide",
        target_keyword="python guide",
        ranking_probability=0.68,
        seo_score=78.5,
        intent_data={"intent": "Informational", "confidence": 0.95},
        feature_impacts=[{"feature": "word_count", "shap_value": 0.12}],
        prioritized_actions=[{"rank": 1, "title": "Add schema", "expected_impact_pct": 5.0, "effort": "Low"}],
        missing_topics=["Asyncio", "FastAPI"]
    )

    assert "executive_summary" in report
    assert len(report["recommended_title_tags"]) == 3
    assert len(report["recommended_meta_descriptions"]) == 2
    assert len(report["recommended_h2_structure"]) >= 5
    assert len(report["faq_schema_questions"]) == 2

def test_semantic_matcher():
    matcher = SemanticMatcher()
    sim = matcher.compute_similarity("machine learning python algorithms", "deep learning neural networks in python")
    assert 0.0 <= sim <= 1.0

    gap = matcher.analyze_content_gaps(
        target_text="Basic python tutorial syntax variables loops",
        competitor_texts=["Advanced python data science pandas numpy scikit-learn machine learning"]
    )
    assert "missing_topics" in gap
    assert "coverage_pct" in gap
