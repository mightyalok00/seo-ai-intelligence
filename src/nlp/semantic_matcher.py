from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticMatcher:
    """Calculates semantic text relevance, entity coverage, and content gaps."""

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Calculates cosine similarity between two text blocks."""
        if not text_a.strip() or not text_b.strip():
            return 0.0

        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
            sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
            return round(sim, 3)
        except Exception:
            return 0.5

    def analyze_content_gaps(self, target_text: str, competitor_texts: List[str]) -> Dict[str, Any]:
        """Identifies key terms/entities present in competitors but missing in target."""
        if not competitor_texts:
            return {"missing_topics": [], "coverage_pct": 100.0}

        vectorizer = TfidfVectorizer(stop_words="english", max_features=50, ngram_range=(1, 2))
        try:
            vectorizer.fit(competitor_texts)
            competitor_vocab = set(vectorizer.get_feature_names_out())
        except Exception:
            competitor_vocab = set()

        target_lower = target_text.lower()
        missing = [term for term in competitor_vocab if term not in target_lower]
        found = [term for term in competitor_vocab if term in target_lower]

        total = len(competitor_vocab)
        coverage_pct = round((len(found) / max(total, 1)) * 100, 1)

        return {
            "total_benchmark_topics": total,
            "matched_topics_count": len(found),
            "missing_topics_count": len(missing),
            "coverage_pct": coverage_pct,
            "missing_topics": missing[:10],
            "covered_topics": found[:10]
        }
