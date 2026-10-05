from fastapi import APIRouter
from src.nlp.keyword_clustering import KeywordClusterer
from api.schemas import KeywordClusteringRequest

router = APIRouter(prefix="/keywords", tags=["Keyword Clustering"])

@router.post("/cluster")
async def cluster_keywords(req: KeywordClusteringRequest):
    """Clusters keyword lists into semantic topic categories with automated topic names."""
    clusterer = KeywordClusterer(method=req.method or "agglomerative")
    result = clusterer.cluster_keywords(req.keywords, min_cluster_size=req.min_cluster_size or 2)
    return result
