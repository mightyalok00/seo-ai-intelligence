from typing import Dict, Any, List

class RecommendationPrioritizer:
    """Prioritizes SEO optimization tasks using Expected Impact % / Implementation Effort ROI matrix."""

    def prioritize(self, features: Dict[str, Any], current_prob: float, intent_data: Dict[str, Any], missing_topics: List[str] = None) -> Dict[str, Any]:
        candidates = []

        # 1. Search Intent Check
        intent_conf = intent_data.get("confidence", 0.8)
        intent_type = intent_data.get("intent", "Informational")
        if intent_conf < 0.75:
            candidates.append({
                "title": f"Align Content with {intent_type} Search Intent",
                "description": f"The page does not clearly fulfill {intent_type} intent. Restructure intro, add actionable guides/comparisons, and match user expectations.",
                "category": "Search Intent",
                "expected_impact_pct": 11.5,
                "effort": "Medium",
                "effort_score": 2.0
            })

        # 2. Missing Semantic Entities
        if missing_topics and len(missing_topics) > 0:
            top_missing = ", ".join(missing_topics[:4])
            candidates.append({
                "title": "Inject Missing Benchmark Semantic Entities",
                "description": f"Incorporate core benchmark subtopics into H2/H3 headers and body copy: {top_missing}.",
                "category": "Content Quality",
                "expected_impact_pct": 8.0,
                "effort": "Low",
                "effort_score": 1.0
            })

        # 3. Keyword in Title
        if not features.get("keyword_in_title", 0):
            candidates.append({
                "title": "Include Primary Keyword in <title> Tag",
                "description": "Place the exact target keyword near the front of the title tag (under 60 characters).",
                "category": "On-Page SEO",
                "expected_impact_pct": 5.5,
                "effort": "Very Low",
                "effort_score": 0.5
            })

        # 4. Keyword in H1 / Single H1
        if not features.get("keyword_in_h1", 0) or not features.get("h1_is_single", 0):
            candidates.append({
                "title": "Standardize H1 Structure & Inject Primary Keyword",
                "description": "Ensure exactly one H1 tag exists on the page containing the exact keyword query.",
                "category": "On-Page SEO",
                "expected_impact_pct": 4.5,
                "effort": "Very Low",
                "effort_score": 0.5
            })

        # 5. Schema Markup
        if not features.get("has_schema", 0):
            candidates.append({
                "title": "Implement JSON-LD Schema Structured Data",
                "description": "Add Article, FAQPage, or Product schema to unlock rich SERP snippet features.",
                "category": "Technical SEO",
                "expected_impact_pct": 3.5,
                "effort": "Very Low",
                "effort_score": 0.5
            })

        # 6. Internal Linking
        if features.get("internal_links_count", 0) < 5:
            candidates.append({
                "title": "Build Contextual Internal Links",
                "description": f"Page currently has only {features.get('internal_links_count', 0)} internal links. Link to this page from 5-8 relevant topical cluster articles.",
                "category": "Site Architecture",
                "expected_impact_pct": 3.8,
                "effort": "Medium",
                "effort_score": 2.0
            })

        # 7. Word Count & Depth
        if features.get("word_count", 0) < 800:
            candidates.append({
                "title": "Expand Content Depth and Comprehensiveness",
                "description": f"Current word count ({features.get('word_count', 0)}) is below the recommended 1,200+ word benchmark for competitive rankings.",
                "category": "Content Depth",
                "expected_impact_pct": 6.0,
                "effort": "High",
                "effort_score": 3.0
            })

        # 8. Missing Alt Tags
        if features.get("image_alt_ratio", 1.0) < 0.9 and features.get("images_count", 0) > 0:
            candidates.append({
                "title": "Add Descriptive Alt Attributes to Images",
                "description": "Improve accessibility and image search indexing by providing descriptive alt tags.",
                "category": "Accessibility & On-Page",
                "expected_impact_pct": 2.0,
                "effort": "Very Low",
                "effort_score": 0.5
            })

        # 9. PageSpeed
        if features.get("page_speed_score", 100) < 70:
            candidates.append({
                "title": "Optimize Core Web Vitals & Asset Loading",
                "description": f"Performance score is {features.get('page_speed_score', 0)}/100. Compress WebP images, defer unused JS, and leverage CDN caching.",
                "category": "Performance",
                "expected_impact_pct": 4.0,
                "effort": "High",
                "effort_score": 3.0
            })

        # Compute ROI Score = Expected Impact / Effort Score
        for item in candidates:
            roi_value = round(item["expected_impact_pct"] / max(item["effort_score"], 0.5), 2)
            item["roi_score"] = roi_value
            if roi_value >= 8.0:
                item["roi_tier"] = "Immediate Quick-Win (Very High ROI)"
            elif roi_value >= 4.0:
                item["roi_tier"] = "High Priority"
            elif roi_value >= 2.0:
                item["roi_tier"] = "Medium Priority"
            else:
                item["roi_tier"] = "Strategic / Long-term"

        # Sort by ROI descending
        candidates.sort(key=lambda x: x["roi_score"], reverse=True)

        for rank, item in enumerate(candidates, 1):
            item["rank"] = rank

        total_potential_gain = sum(c["expected_impact_pct"] for c in candidates[:4])
        potential_prob = min(0.96, current_prob + (total_potential_gain / 100.0) * 0.7)

        return {
            "current_ranking_probability": round(current_prob, 4),
            "potential_ranking_probability": round(potential_prob, 4),
            "estimated_probability_gain_pct": round((potential_prob - current_prob) * 100, 1),
            "total_issues_identified": len(candidates),
            "prioritized_actions": candidates
        }
