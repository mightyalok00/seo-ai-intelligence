"""
Synthetic Benchmark Dataset Generator for SEO SERP Ranking Models.

This module generates statistically parameterized benchmark datasets modeling
real-world search engine result page dynamics (authority, content depth, intent
matching, semantic coverage, and performance signals).

Author: Alok Agarwal (mightyalok00)
License: MIT
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import settings

FEATURE_COLUMNS = [
    "domain_authority_proxy",
    "backlink_count",
    "word_count",
    "keyword_in_title",
    "keyword_in_h1",
    "keyword_in_url",
    "keyword_density",
    "semantic_coverage",
    "search_intent_match",
    "internal_links_count",
    "page_speed_score",
    "has_schema",
    "image_alt_ratio"
]

class SEODatasetGenerator:
    """Generates realistic statistical SEO benchmark datasets for ranking prediction."""

    def __init__(self, n_samples: int = 3000, random_seed: int = 42):
        self.n_samples = n_samples
        self.random_seed = random_seed

    def generate(self) -> pd.DataFrame:
        np.random.seed(self.random_seed)
        n = self.n_samples

        # 1. Authority & Backlinks
        da = np.random.beta(a=3, b=3, size=n) * 85 + 10  # 10 to 95
        backlinks = np.random.exponential(scale=250, size=n) * (da / 30)

        # 2. On-page & Content
        word_count = np.random.gamma(shape=3.5, scale=400, size=n) + 200 # 200 to ~3500
        kw_title = np.random.binomial(n=1, p=0.65, size=n)
        kw_h1 = np.random.binomial(n=1, p=0.70, size=n)
        kw_url = np.random.binomial(n=1, p=0.55, size=n)
        kw_density = np.random.normal(loc=1.8, scale=0.8, size=n).clip(0.1, 5.0)

        # 3. Semantic & Intent Signals
        semantic_coverage = np.random.beta(a=4, b=2, size=n) # 0.2 to 1.0
        search_intent_match = np.random.beta(a=5, b=2, size=n) # 0.3 to 1.0

        # 4. Architecture & Technical
        internal_links = np.random.poisson(lam=8, size=n) + np.random.binomial(1, 0.4, size=n) * 5
        page_speed = np.random.normal(loc=72, scale=18, size=n).clip(25, 100)
        has_schema = np.random.binomial(n=1, p=0.48, size=n)
        image_alt_ratio = np.random.beta(a=6, b=2, size=n)

        df = pd.DataFrame({
            "domain_authority_proxy": np.round(da, 1),
            "backlink_count": np.round(backlinks, 0).astype(int),
            "word_count": np.round(word_count, 0).astype(int),
            "keyword_in_title": kw_title,
            "keyword_in_h1": kw_h1,
            "keyword_in_url": kw_url,
            "keyword_density": np.round(kw_density, 2),
            "semantic_coverage": np.round(semantic_coverage, 3),
            "search_intent_match": np.round(search_intent_match, 3),
            "internal_links_count": internal_links,
            "page_speed_score": np.round(page_speed, 1),
            "has_schema": has_schema,
            "image_alt_ratio": np.round(image_alt_ratio, 2)
        })

        # Standardized logit z calculation with centered signals for realistic class balance (~40% positives)
        z = (
            ((df["domain_authority_proxy"] - 50) / 20) * 0.8 +
            ((np.log1p(df["backlink_count"]) - 5.0) / 2.0) * 0.7 +
            ((df["word_count"] - 1400) / 800) * 0.6 +
            (df["keyword_in_title"] - 0.65) * 1.0 +
            (df["keyword_in_h1"] - 0.70) * 0.8 +
            (df["keyword_in_url"] - 0.55) * 0.6 +
            ((df["semantic_coverage"] - 0.65) / 0.2) * 1.1 +
            ((df["search_intent_match"] - 0.70) / 0.2) * 1.2 +
            ((df["internal_links_count"] - 8) / 4.0) * 0.5 +
            ((df["page_speed_score"] - 72) / 18.0) * 0.5 +
            (df["has_schema"] - 0.48) * 0.5 - 0.2
        )

        probs = 1 / (1 + np.exp(-z))
        noise = np.random.normal(0, 0.08, size=n)
        final_probs = np.clip(probs + noise, 0.01, 0.99)

        df["target_probability"] = np.round(final_probs, 4)
        df["is_top_10"] = (final_probs >= 0.50).astype(int)

        return df

    def save_dataset(self, output_path: Path = settings.DATA_DIR / "processed" / "seo_ranking_dataset.csv"):
        df = self.generate()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df
