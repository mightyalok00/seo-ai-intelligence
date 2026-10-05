"""
Structured Feature Engineering Engine for Machine Learning SERP Models.

This module converts unstructured crawled web content, metadata tags, heading
hierarchies, and keyword alignments into numerical feature vectors suitable for
supervised machine learning models (XGBoost, Random Forest, Logistic Regression).

Author: Alok Agarwal (mightyalok00)
License: MIT
"""

import re
from urllib.parse import urlparse
from typing import Dict, Any, List

class SEOFeatureExtractor:
    """
    Extracts structured machine learning features from raw crawled page artifacts.
    """

    def extract_features(
        self,
        crawl_data: Dict[str, Any],
        target_keyword: str = "",
        domain_authority_proxy: float = 40.0
    ) -> Dict[str, Any]:
        """
        Convert crawled DOM dictionary and target keyword into a numerical feature dictionary.

        Args:
            crawl_data (Dict[str, Any]): Dictionary containing raw parsed DOM from SEOCrawler.
            target_keyword (str): The primary search query targeted by the content.
            domain_authority_proxy (float): Estimated domain authority proxy score (1-100).

        Returns:
            Dict[str, Any]: Standardized numerical features for model inference.
        """
        url = crawl_data.get("url", "")
        title = crawl_data.get("title", "")
        meta_desc = crawl_data.get("meta_description", "")
        h1_tags = crawl_data.get("h1_tags", [])
        h2_tags = crawl_data.get("h2_tags", [])
        word_count = crawl_data.get("word_count", 0)
        internal_links = crawl_data.get("internal_links_count", 0)
        external_links = crawl_data.get("external_links_count", 0)
        images_count = crawl_data.get("images_count", 0)
        images_missing_alt = crawl_data.get("images_missing_alt", 0)
        has_schema = 1 if crawl_data.get("has_schema", False) else 0
        pagespeed = crawl_data.get("pagespeed", {})
        perf_score = pagespeed.get("performance_score", 75.0)

        # Keyword matching features across document zones
        kw_clean = target_keyword.strip().lower()
        title_lower = title.lower()
        url_lower = url.lower()
        h1_text = " ".join(h1_tags).lower()
        full_text = crawl_data.get("full_text", "").lower()

        keyword_in_title = 1 if kw_clean and kw_clean in title_lower else 0
        keyword_in_h1 = 1 if kw_clean and kw_clean in h1_text else 0
        keyword_in_url = 1 if kw_clean and (
            kw_clean.replace(" ", "-") in url_lower or
            kw_clean.replace(" ", "_") in url_lower or
            kw_clean in url_lower
        ) else 0
        keyword_in_meta = 1 if kw_clean and kw_clean in meta_desc.lower() else 0

        # Title length & Meta description length optimality benchmarks
        title_length = len(title)
        title_length_optimal = 1 if 30 <= title_length <= 60 else 0
        meta_length = len(meta_desc)
        meta_length_optimal = 1 if 120 <= meta_length <= 160 else 0

        # Content structural depth
        h1_count = len(h1_tags)
        h1_is_single = 1 if h1_count == 1 else 0
        h2_count = len(h2_tags)

        # Exact keyword density calculation in body copy
        kw_density = 0.0
        if kw_clean and word_count > 0:
            kw_occurrences = len(re.findall(r"\b" + re.escape(kw_clean) + r"\b", full_text))
            kw_words = len(kw_clean.split())
            kw_density = round((kw_occurrences * kw_words / max(word_count, 1)) * 100, 2)

        # Image accessibility ratio (images with valid alt attribute)
        image_alt_ratio = 1.0
        if images_count > 0:
            image_alt_ratio = round((images_count - images_missing_alt) / images_count, 2)

        # Semantic heuristic coverage (Ratio of query terms represented in text)
        semantic_coverage = 0.5
        if kw_clean:
            tokens = [t for t in kw_clean.split() if len(t) > 2]
            if tokens:
                found = sum(1 for t in tokens if t in full_text)
                semantic_coverage = round(found / len(tokens), 2)

        return {
            # On-page keyword alignments
            "keyword_in_title": keyword_in_title,
            "keyword_in_h1": keyword_in_h1,
            "keyword_in_url": keyword_in_url,
            "keyword_in_meta": keyword_in_meta,
            "keyword_density": kw_density,
            "semantic_coverage": semantic_coverage,

            # Content breadth & structure
            "word_count": word_count,
            "title_length": title_length,
            "title_length_optimal": title_length_optimal,
            "meta_length": meta_length,
            "meta_length_optimal": meta_length_optimal,
            "h1_count": h1_count,
            "h1_is_single": h1_is_single,
            "h2_count": h2_count,

            # Linking & Media assets
            "internal_links_count": internal_links,
            "external_links_count": external_links,
            "images_count": images_count,
            "image_alt_ratio": image_alt_ratio,

            # Technical authority & performance
            "has_schema": has_schema,
            "page_speed_score": perf_score,
            "domain_authority_proxy": domain_authority_proxy
        }
