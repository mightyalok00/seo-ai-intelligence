from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.schemas import AnalyzeUrlRequest, CrawlRequest
from src.crawler.async_crawler import SEOCrawler
from src.features.extractor import SEOFeatureExtractor
from src.features.scorer import SEOScorer
from src.models.ctr_forecaster import CTRForecaster
from src.models.ranking_predictor import RankingPredictor
from src.nlp.intent_classifier import SearchIntentClassifier
from src.nlp.semantic_matcher import SemanticMatcher
from src.recommendations.ai_analyst import AISEOAnalyst
from src.recommendations.prioritizer import RecommendationPrioritizer
from src.utils.database import CrawledPage, SEOPrediction, get_db

router = APIRouter(prefix="/crawl", tags=["Crawler & Technical Auditing"])

crawler = SEOCrawler()
feature_extractor = SEOFeatureExtractor()
scorer = SEOScorer()
intent_clf = SearchIntentClassifier()
semantic_matcher = SemanticMatcher()
ranking_predictor = RankingPredictor()
ctr_forecaster = CTRForecaster()
prioritizer = RecommendationPrioritizer()
ai_analyst = AISEOAnalyst()

@router.post("")
async def crawl_single_url(req: CrawlRequest, db: Session = Depends(get_db)):
    """Crawls a URL and returns raw DOM, metadata, heading hierarchy, and performance."""
    result = await crawler.crawl_url(req.url, fetch_pagespeed=req.fetch_pagespeed)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Crawling failed"))
    return result

@router.post("/analyze-full")
async def analyze_full_url(req: AnalyzeUrlRequest, db: Session = Depends(get_db)):
    """End-to-end audit: Crawl -> Feature Extraction -> NLP Intent -> ML Prediction -> ROI Actions -> AI Report."""
    # 1. Crawl Target
    crawl_result = await crawler.crawl_url(req.url, fetch_pagespeed=True)
    if not crawl_result.get("success"):
        raise HTTPException(status_code=400, detail=f"Failed to crawl URL: {crawl_result.get('error')}")

    # 2. Competitor Crawl & Content Gap Analysis
    competitor_texts = []
    for comp_url in (req.competitor_urls or [])[:3]:
        comp_crawl = await crawler.crawl_url(comp_url, fetch_pagespeed=False)
        if comp_crawl.get("success") and comp_crawl.get("full_text"):
            competitor_texts.append(comp_crawl["full_text"])

    gap_analysis = semantic_matcher.analyze_content_gaps(crawl_result.get("full_text", ""), competitor_texts)

    # 3. Intent Classification
    intent_data = intent_clf.predict(req.target_keyword)

    # 4. Feature Extraction
    features = feature_extractor.extract_features(
        crawl_data=crawl_result,
        target_keyword=req.target_keyword,
        domain_authority_proxy=req.domain_authority_proxy or 45.0
    )
    features["search_intent_match"] = intent_data.get("confidence", 0.8)

    # 5. Deterministic Scoring
    score_data = scorer.calculate_score(features, search_intent_match=intent_data.get("confidence", 0.8))
    overall_seo_score = score_data["overall_seo_score"]

    # 6. ML Ranking Prediction
    prediction = ranking_predictor.predict_probability(features)

    # 7. CTR & Traffic Forecasting
    traffic_forecast = ctr_forecaster.forecast_traffic(
        ranking_prob=prediction["top_10_probability"],
        monthly_search_volume=req.monthly_search_volume or 2400
    )

    # 8. Impact vs Effort Task Prioritization
    action_matrix = prioritizer.prioritize(
        features=features,
        current_prob=prediction["top_10_probability"],
        intent_data=intent_data,
        missing_topics=gap_analysis.get("missing_topics", [])
    )

    # 9. AI SEO Analyst Report
    ai_report = await ai_analyst.generate_seo_report(
        url=req.url,
        target_keyword=req.target_keyword,
        ranking_probability=prediction["top_10_probability"],
        seo_score=overall_seo_score,
        intent_data=intent_data,
        feature_impacts=prediction.get("feature_impacts", []),
        prioritized_actions=action_matrix.get("prioritized_actions", []),
        missing_topics=gap_analysis.get("missing_topics", [])
    )

    # 10. Persist to DB
    try:
        page_record = db.query(CrawledPage).filter(CrawledPage.url == req.url).first()
        if not page_record:
            page_record = CrawledPage(url=req.url)
            db.add(page_record)
        page_record.title = crawl_result.get("title", "")
        page_record.meta_description = crawl_result.get("meta_description", "")
        page_record.word_count = crawl_result.get("word_count", 0)
        page_record.seo_score = overall_seo_score
        page_record.features_json = features
        db.commit()

        pred_record = SEOPrediction(
            url=req.url,
            target_keyword=req.target_keyword,
            ranking_probability=prediction["top_10_probability"],
            seo_score=overall_seo_score,
            search_intent=intent_data.get("intent", ""),
            shap_importance=prediction.get("feature_impacts", []),
            recommendations=action_matrix.get("prioritized_actions", [])
        )
        db.add(pred_record)
        db.commit()
    except Exception:
        db.rollback()

    return {
        "url": req.url,
        "target_keyword": req.target_keyword,
        "overall_seo_score": overall_seo_score,
        "score_breakdown": score_data["pillars"],
        "ranking_prediction": prediction,
        "traffic_forecast": traffic_forecast,
        "search_intent": intent_data,
        "content_gap_analysis": gap_analysis,
        "action_matrix": action_matrix,
        "ai_analyst_report": ai_report,
        "extracted_features": features,
        "crawl_summary": {
            "status_code": crawl_result.get("status_code"),
            "title": crawl_result.get("title"),
            "word_count": crawl_result.get("word_count"),
            "internal_links": crawl_result.get("internal_links_count"),
            "external_links": crawl_result.get("external_links_count"),
            "has_schema": crawl_result.get("has_schema"),
            "pagespeed": crawl_result.get("pagespeed")
        }
    }
