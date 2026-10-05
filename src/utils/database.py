import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.utils.config import settings

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class CrawledPage(Base):
    __tablename__ = "crawled_pages"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String(1024), unique=True, index=True, nullable=False)
    status_code = Column(Integer, default=200)
    title = Column(String(512), nullable=True)
    meta_description = Column(Text, nullable=True)
    h1_count = Column(Integer, default=0)
    h2_count = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    internal_links_count = Column(Integer, default=0)
    external_links_count = Column(Integer, default=0)
    images_count = Column(Integer, default=0)
    images_without_alt = Column(Integer, default=0)
    has_schema = Column(Boolean, default=False)
    page_speed_score = Column(Float, default=75.0)
    seo_score = Column(Float, default=0.0)
    features_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class SEOPrediction(Base):
    __tablename__ = "seo_predictions"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String(1024), index=True, nullable=False)
    target_keyword = Column(String(256), nullable=False)
    ranking_probability = Column(Float, nullable=False)
    seo_score = Column(Float, nullable=False)
    search_intent = Column(String(64), nullable=True)
    shap_importance = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
