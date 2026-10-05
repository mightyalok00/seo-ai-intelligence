from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.utils.config import settings
from src.utils.database import init_db
from api.routes.crawl import router as crawl_router
from api.routes.intent import router as intent_router
from api.routes.clustering import router as clustering_router
from api.routes.prediction import router as prediction_router
from api.routes.recommendations import router as recommendations_router
from api.routes.reports import router as reports_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables
    init_db()
    yield

app = FastAPI(
    title="SEO-AI-MLOps Intelligence Platform",
    description="Machine Learning & AI Platform for SERP Ranking Prediction, NLP Search Intent, and Prioritized SEO Decisions.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(crawl_router, prefix="/api/v1")
app.include_router(intent_router, prefix="/api/v1")
app.include_router(clustering_router, prefix="/api/v1")
app.include_router(prediction_router, prefix="/api/v1")
app.include_router(recommendations_router, prefix="/api/v1")
app.include_router(reports_router, prefix="/api/v1")

@app.get("/", tags=["Health"])
async def root():
    return {
        "platform": "SEO-AI-MLOps Intelligence Platform",
        "status": "operational",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
