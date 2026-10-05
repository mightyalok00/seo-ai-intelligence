"""
FastAPI Main Application Entrypoint.

This module initializes the FastAPI REST backend, mounts all modular sub-routers,
configures secure CORS middleware, and handles lifecycle database startup hooks.

Author: Alok Agarwal (mightyalok00)
License: MIT
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.clustering import router as clustering_router
from api.routes.crawl import router as crawl_router
from api.routes.intent import router as intent_router
from api.routes.prediction import router as prediction_router
from api.routes.recommendations import router as recommendations_router
from api.routes.reports import router as reports_router
from src.utils.config import settings
from src.utils.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and shutdown event management."""
    # Initialize SQLite/PostgreSQL schema tables
    init_db()
    yield

app = FastAPI(
    title="SEO-AI-MLOps Intelligence Platform API",
    description="Production REST API for SERP Ranking Prediction, NLP Search Intent, TreeSHAP Explainability, and ROI Prioritization.",
    version="1.0.0",
    lifespan=lifespan
)

# Secure CORS Middleware Configuration
ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
]

# Allow custom environment overrides if specified
custom_origins = os.getenv("ALLOWED_ORIGINS", "")
if custom_origins:
    ALLOWED_ORIGINS.extend([o.strip() for o in custom_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Register modular sub-routers
app.include_router(crawl_router, prefix="/api/v1")
app.include_router(intent_router, prefix="/api/v1")
app.include_router(clustering_router, prefix="/api/v1")
app.include_router(prediction_router, prefix="/api/v1")
app.include_router(recommendations_router, prefix="/api/v1")
app.include_router(reports_router, prefix="/api/v1")

@app.get("/", tags=["Health"])
async def root():
    """Root health and discovery endpoint."""
    return {
        "platform": "SEO-AI-MLOps Intelligence Platform",
        "status": "operational",
        "version": "1.0.0",
        "documentation": "/docs",
        "openapi_spec": "/openapi.json"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness probe for container orchestrators."""
    return {"status": "healthy", "service": "seo-ai-mlops-backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
