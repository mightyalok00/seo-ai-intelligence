"""
Live Integration Test for SEO-AI-MLOps End-to-End Pipeline.
"""

import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.crawler.async_crawler import SEOCrawler
from src.features.extractor import SEOFeatureExtractor
from src.features.scorer import SEOScorer
from src.models.ranking_predictor import RankingPredictor
from src.nlp.intent_classifier import SearchIntentClassifier
from src.recommendations.ai_analyst import AISEOAnalyst
from src.recommendations.prioritizer import RecommendationPrioritizer


async def main():
    print("=" * 70)
    print("Starting Live End-to-End System Audit Test...")
    print("=" * 70)

    # 1. Crawl
    crawler = SEOCrawler()
    res = await crawler.crawl_url("https://example.com", fetch_pagespeed=False)
    assert res["success"] is True
    print(f"[1/7] Async Crawl: OK (URL: {res['url']}, Words: {res['word_count']})")

    # 2. NLP Intent
    intent_clf = SearchIntentClassifier()
    intent = intent_clf.predict("example domain documentation")
    print(f"[2/7] NLP Search Intent: {intent['intent']} (Confidence: {intent['confidence']*100:.0f}%)")

    # 3. Features
    extractor = SEOFeatureExtractor()
    features = extractor.extract_features(res, target_keyword="example domain documentation", domain_authority_proxy=55.0)
    features["search_intent_match"] = intent["confidence"]
    print(f"[3/7] Feature Extractor: OK ({len(features)} numerical features extracted)")

    # 4. Scorer
    scorer = SEOScorer()
    score = scorer.calculate_score(features, search_intent_match=intent["confidence"])
    print(f"[4/7] 6-Pillar Health Score: {score['overall_seo_score']}/100")

    # 5. XGBoost Ranking Predictor + TreeSHAP
    predictor = RankingPredictor()
    pred = predictor.predict_probability(features)
    print(f"[5/7] ML Ranking Probability: {pred['top_10_percentage']}% ({pred['ranking_tier']})")
    print(f"      Top TreeSHAP Attribution: {pred['feature_impacts'][0]['feature']} -> {pred['feature_impacts'][0]['estimated_impact']}")

    # 6. Action Prioritizer
    prioritizer = RecommendationPrioritizer()
    matrix = prioritizer.prioritize(features, pred["top_10_probability"], intent)
    print(f"[6/7] ROI Action Prioritizer: {len(matrix['prioritized_actions'])} prioritized tasks identified")

    # 7. AI Analyst
    analyst = AISEOAnalyst()
    ai_report = await analyst.generate_seo_report(
        res["url"], "example domain documentation", pred["top_10_probability"],
        score["overall_seo_score"], intent, pred["feature_impacts"], matrix["prioritized_actions"]
    )
    print(f"[7/7] GenAI Diagnostic Analyst: OK ({len(ai_report['recommended_title_tags'])} titles, {len(ai_report['recommended_h2_structure'])} headings)")

    print("\n" + "=" * 70)
    print("ALL 7 END-TO-END PIPELINE STAGES PASSED WITH ZERO ERRORS!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
