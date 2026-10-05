from fastapi import APIRouter
from src.models.ranking_predictor import RankingPredictor
from src.features.scorer import SEOScorer
from api.schemas import RankingPredictionRequest

router = APIRouter(prefix="/predict", tags=["ML Ranking Prediction"])

ranking_predictor = RankingPredictor()
scorer = SEOScorer()

@router.post("/ranking")
async def predict_ranking(req: RankingPredictionRequest):
    """Predicts Google Top 10 probability and returns feature attribution breakdown."""
    features_dict = req.model_dump()
    prediction = ranking_predictor.predict_probability(features_dict)
    score_data = scorer.calculate_score(features_dict, search_intent_match=req.search_intent_match)

    return {
        "target_keyword": req.target_keyword,
        "overall_seo_score": score_data["overall_seo_score"],
        "score_breakdown": score_data["pillars"],
        "ranking_prediction": prediction
    }

@router.get("/metrics")
async def get_model_benchmarks():
    """Returns benchmark performance metrics comparing XGBoost, Random Forest, and Logistic Regression."""
    return {
        "best_model": "XGBoost",
        "benchmarks": ranking_predictor.metrics.get("comparison", {}),
        "feature_importances": ranking_predictor.metrics.get("feature_importance", [])
    }
