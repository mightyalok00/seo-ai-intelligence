from fastapi import APIRouter

from api.schemas import RecommendationRequest
from src.recommendations.ai_analyst import AISEOAnalyst
from src.recommendations.prioritizer import RecommendationPrioritizer

router = APIRouter(prefix="/recommendations", tags=["Action Prioritization & AI Analyst"])

prioritizer = RecommendationPrioritizer()
ai_analyst = AISEOAnalyst()

@router.post("/prioritize")
async def prioritize_recommendations(req: RecommendationRequest):
    """Calculates ROI Impact vs Effort for identified SEO bottlenecks and generates AI action plan."""
    action_matrix = prioritizer.prioritize(
        features=req.features,
        current_prob=req.current_probability,
        intent_data=req.intent_data or {"intent": "Informational", "confidence": 0.85},
        missing_topics=req.missing_topics or []
    )

    ai_report = await ai_analyst.generate_seo_report(
        url=req.url,
        target_keyword=req.target_keyword,
        ranking_probability=req.current_probability,
        seo_score=req.features.get("overall_seo_score", 75.0),
        intent_data=req.intent_data or {"intent": "Informational", "confidence": 0.85},
        feature_impacts=[],
        prioritized_actions=action_matrix.get("prioritized_actions", []),
        missing_topics=req.missing_topics or []
    )

    return {
        "action_matrix": action_matrix,
        "ai_analyst_report": ai_report
    }
