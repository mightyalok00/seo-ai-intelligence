from typing import Dict, Any

class SEOScorer:
    """Calculates granular category scores and an overall SEO Score (0-100)."""

    def calculate_score(self, features: Dict[str, Any], search_intent_match: float = 0.8) -> Dict[str, Any]:
        """Calculates multi-pillar SEO score:
        - Technical SEO: 20%
        - Content Quality: 25%
        - Search Intent Alignment: 20%
        - Semantic Coverage: 15%
        - Internal Linking & Structure: 10%
        - Performance & Core Web Vitals: 10%
        """
        # 1. Technical SEO (0-100)
        tech_score = 100.0
        if not features.get("h1_is_single", 0):
            tech_score -= 15
        if not features.get("has_schema", 0):
            tech_score -= 20
        if not features.get("title_length_optimal", 0):
            tech_score -= 15
        if not features.get("meta_length_optimal", 0):
            tech_score -= 10
        if features.get("image_alt_ratio", 1.0) < 0.8:
            tech_score -= 15
        tech_score = max(20.0, min(100.0, tech_score))

        # 2. Content Quality (0-100)
        word_count = features.get("word_count", 0)
        if word_count >= 1500:
            content_score = 95.0
        elif word_count >= 1000:
            content_score = 85.0
        elif word_count >= 600:
            content_score = 70.0
        elif word_count >= 300:
            content_score = 50.0
        else:
            content_score = 30.0

        if features.get("h2_count", 0) >= 3:
            content_score += 5.0

        # Keyword alignment
        kw_title = features.get("keyword_in_title", 0)
        kw_h1 = features.get("keyword_in_h1", 0)
        if kw_title:
            content_score += 5.0
        if kw_h1:
            content_score += 5.0
        content_score = min(100.0, content_score)

        # 3. Search Intent Alignment (0-100)
        intent_score = max(0.0, min(100.0, search_intent_match * 100))

        # 4. Semantic Coverage (0-100)
        semantic_score = max(0.0, min(100.0, features.get("semantic_coverage", 0.5) * 100))

        # 5. Internal Linking & Architecture (0-100)
        internal_links = features.get("internal_links_count", 0)
        if internal_links >= 10:
            linking_score = 95.0
        elif internal_links >= 5:
            linking_score = 80.0
        elif internal_links >= 2:
            linking_score = 65.0
        else:
            linking_score = 40.0

        # 6. Performance (0-100)
        perf_score = float(features.get("page_speed_score", 75.0))

        # Weighted Total
        overall_score = round(
            (tech_score * 0.20) +
            (content_score * 0.25) +
            (intent_score * 0.20) +
            (semantic_score * 0.15) +
            (linking_score * 0.10) +
            (perf_score * 0.10),
            1
        )

        return {
            "overall_seo_score": overall_score,
            "pillars": {
                "technical_seo": round(tech_score, 1),
                "content_quality": round(content_score, 1),
                "search_intent": round(intent_score, 1),
                "semantic_coverage": round(semantic_score, 1),
                "internal_linking": round(linking_score, 1),
                "performance": round(perf_score, 1)
            }
        }
