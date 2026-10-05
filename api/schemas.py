from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# 1. Crawl & Audit Schemas
class CrawlRequest(BaseModel):
    url: str = Field(..., json_schema_extra={"example": "https://example.com/python-course"})
    fetch_pagespeed: bool = Field(True, description="Fetch or estimate Core Web Vitals")
    target_keyword: Optional[str] = Field(None, json_schema_extra={"example": "python course"})

class AnalyzeUrlRequest(BaseModel):
    url: str = Field(..., json_schema_extra={"example": "https://example.com/python-course"})
    target_keyword: str = Field(..., json_schema_extra={"example": "python course"})
    competitor_urls: Optional[List[str]] = Field(default=[], json_schema_extra={"example": ["https://competitor.com/learn-python"]})
    domain_authority_proxy: Optional[float] = Field(45.0, ge=0, le=100)
    monthly_search_volume: Optional[int] = Field(2400, ge=0)

# 2. NLP Schemas
class IntentClassificationRequest(BaseModel):
    keywords: List[str] = Field(..., min_length=1, json_schema_extra={"example": ["what is random forest", "best seo software", "buy ahrefs"]})

class KeywordClusteringRequest(BaseModel):
    keywords: List[str] = Field(..., min_length=2, json_schema_extra={"example": ["learn python", "python course", "pandas dataframe", "pandas tutorial"]})
    min_cluster_size: Optional[int] = Field(2, ge=1)
    method: Optional[str] = Field("agglomerative", description="agglomerative or kmeans")

# 3. Prediction Schemas
class RankingPredictionRequest(BaseModel):
    target_keyword: str = Field(..., json_schema_extra={"example": "python machine learning"})
    word_count: int = Field(1200, ge=0)
    internal_links_count: int = Field(6, ge=0)
    external_links_count: int = Field(2, ge=0)
    keyword_in_title: int = Field(1, ge=0, le=1)
    keyword_in_h1: int = Field(1, ge=0, le=1)
    keyword_in_url: int = Field(1, ge=0, le=1)
    keyword_density: float = Field(1.6, ge=0.0)
    semantic_coverage: float = Field(0.75, ge=0.0, le=1.0)
    search_intent_match: float = Field(0.85, ge=0.0, le=1.0)
    page_speed_score: float = Field(80.0, ge=0.0, le=100.0)
    has_schema: int = Field(1, ge=0, le=1)
    image_alt_ratio: float = Field(0.9, ge=0.0, le=1.0)
    domain_authority_proxy: Optional[float] = Field(45.0, ge=0, le=100)

class RecommendationRequest(BaseModel):
    url: str
    target_keyword: str
    current_probability: float
    features: Dict[str, Any]
    intent_data: Optional[Dict[str, Any]] = None
    missing_topics: Optional[List[str]] = []
