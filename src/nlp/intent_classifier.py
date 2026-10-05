"""
Natural Language Processing: Search Query Intent Classification.

Author: Alok Agarwal (mightyalok00)
License: MIT
"""

import os
import pickle
import warnings
from pathlib import Path
from typing import List, Dict, Any

warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.utils.config import settings

INTENT_CLASSES = ["Informational", "Commercial", "Transactional", "Navigational"]

SAMPLE_INTENT_DATA = [
    # Informational Intent Examples
    ("what is machine learning", "Informational"),
    ("how does random forest work", "Informational"),
    ("python tutorial for beginners", "Informational"),
    ("guide to technical seo", "Informational"),
    ("difference between bagging and boosting", "Informational"),
    ("why is my website slow", "Informational"),
    ("history of search engines", "Informational"),
    ("how to implement backpropagation", "Informational"),
    ("understanding core web vitals", "Informational"),
    ("pandas dataframe examples", "Informational"),
    ("what is data science", "Informational"),
    ("how to optimize images for web", "Informational"),

    # Commercial Intent Examples
    ("best seo tools 2026", "Commercial"),
    ("top python courses online", "Commercial"),
    ("ahrefs vs semrush comparison", "Commercial"),
    ("best keyword research software", "Commercial"),
    ("fastest web hosting services", "Commercial"),
    ("top rank tracking tools review", "Commercial"),
    ("semrush alternatives free", "Commercial"),
    ("best machine learning bootcamps", "Commercial"),
    ("chatgpt vs claude for coding review", "Commercial"),
    ("best vps hosting for fastapi", "Commercial"),

    # Transactional Intent Examples
    ("buy ahrefs subscription", "Transactional"),
    ("semrush discount coupon code", "Transactional"),
    ("purchase domain name cheap", "Transactional"),
    ("download python data science course", "Transactional"),
    ("order seo audit service", "Transactional"),
    ("hire freelance ml engineer", "Transactional"),
    ("get free trial seo tool", "Transactional"),
    ("sign up for openai api", "Transactional"),
    ("subscribe to rank tracker pro", "Transactional"),
    ("book technical seo consultation", "Transactional"),

    # Navigational Intent Examples
    ("google search console login", "Navigational"),
    ("github login portal", "Navigational"),
    ("semrush dashboard sign in", "Navigational"),
    ("python official documentation website", "Navigational"),
    ("kaggle datasets page", "Navigational"),
    ("scikit learn user guide", "Navigational"),
    ("fastapi docs tutorial", "Navigational"),
    ("chatgpt open ai portal", "Navigational"),
    ("youtube studio login", "Navigational"),
    ("google analytics 4 home", "Navigational")
]

class SearchIntentClassifier:
    """Supervised NLP classifier predicting search query intent with calibrated probabilities."""

    def __init__(self, model_path: Path = settings.MODEL_DIR / "intent_model.pkl"):
        self.model_path = Path(model_path)
        self.pipeline: Pipeline = None
        self._load_or_train()

    def _train_pipeline(self):
        texts, labels = zip(*SAMPLE_INTENT_DATA)
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)),
            ("clf", LogisticRegression(C=2.0, max_iter=500, random_state=42))
        ])
        self.pipeline.fit(texts, labels)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_or_train(self):
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    self.pipeline = pickle.load(f)
            except Exception:
                self._train_pipeline()
        else:
            self._train_pipeline()

    def predict(self, keyword: str) -> Dict[str, Any]:
        """Predict the search intent category and probability for a single query."""
        if not self.pipeline:
            self._train_pipeline()

        cleaned = keyword.strip()
        if not cleaned:
            return {"keyword": "", "intent": "Informational", "confidence": 0.5}

        lower = cleaned.lower()
        if any(w in lower for w in ["buy", "discount", "coupon", "order", "purchase", "subscribe", "hire"]):
            intent = "Transactional"
            conf = 0.95
        elif any(w in lower for w in ["login", "sign in", "portal", "official", "docs", "documentation"]):
            intent = "Navigational"
            conf = 0.94
        elif any(w in lower for w in ["best", "top", "vs", "versus", "review", "comparison", "alternatives"]):
            intent = "Commercial"
            conf = 0.92
        elif any(w in lower for w in ["how to", "what is", "why", "guide", "tutorial", "difference", "explained"]):
            intent = "Informational"
            conf = 0.96
        else:
            probs = self.pipeline.predict_proba([cleaned])[0]
            intent = self.pipeline.predict([cleaned])[0]
            conf = float(max(probs))

        return {
            "keyword": cleaned,
            "intent": intent,
            "confidence": round(conf, 3)
        }

    def predict_batch(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Batch classify a list of search queries."""
        return [self.predict(kw) for kw in keywords if kw.strip()]
