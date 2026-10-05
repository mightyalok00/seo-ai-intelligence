from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.utils.database import SEOPrediction, get_db

router = APIRouter(prefix="/reports", tags=["History & Reports"])

@router.get("/history")
async def get_audit_history(limit: int = 20, db: Session = Depends(get_db)):
    """Fetches recent website audit and ranking prediction history."""
    predictions = db.query(SEOPrediction).order_by(SEOPrediction.created_at.desc()).limit(limit).all()
    return [
        {
            "id": p.id,
            "url": p.url,
            "target_keyword": p.target_keyword,
            "ranking_probability": p.ranking_probability,
            "seo_score": p.seo_score,
            "search_intent": p.search_intent,
            "created_at": p.created_at.isoformat() if p.created_at else None
        }
        for p in predictions
    ]
