from fastapi import APIRouter

from api.schemas import IntentClassificationRequest
from src.nlp.intent_classifier import SearchIntentClassifier

router = APIRouter(prefix="/intent", tags=["NLP & Search Intent"])
intent_clf = SearchIntentClassifier()

@router.post("/classify")
async def classify_intent(req: IntentClassificationRequest):
    """Classifies search queries into Informational, Commercial, Transactional, or Navigational intents."""
    predictions = intent_clf.predict_batch(req.keywords)
    return {
        "total_keywords": len(predictions),
        "predictions": predictions
    }
