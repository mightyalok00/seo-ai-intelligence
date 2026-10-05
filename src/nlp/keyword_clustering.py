"""
Unsupervised Keyword Semantic Clustering & Topic Hub Modeling.

This module provides the `KeywordClusterer` class, clustering target keyword lists
into semantic topic clusters using TF-IDF n-grams and Agglomerative / K-Means
clustering with automated cluster title generation and Silhouette validation.

Author: Alok Agarwal (mightyalok00)
License: MIT
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

class KeywordClusterer:
    """
    Groups keyword universes into semantic topic hubs with automated cluster titles.
    """

    def __init__(self, method: str = "agglomerative"):
        """
        Initialize the clusterer with desired algorithm.

        Args:
            method (str): Clustering strategy ('agglomerative' or 'kmeans').
        """
        self.method = method

    def cluster_keywords(
        self,
        keywords: List[str],
        n_clusters: int = None,
        min_cluster_size: int = 2
    ) -> Dict[str, Any]:
        """
        Cluster a list of keywords into semantic topic groups.

        Args:
            keywords (List[str]): List of keywords to group.
            n_clusters (int, optional): Explicit number of clusters.
            min_cluster_size (int): Minimum cluster size heuristic.

        Returns:
            Dict[str, Any]: Structured clusters with automated titles and Silhouette score.
        """
        cleaned_keywords = list(dict.fromkeys([kw.strip() for kw in keywords if kw.strip()]))

        if len(cleaned_keywords) < 2:
            return {
                "total_keywords": len(cleaned_keywords),
                "total_clusters": 1 if cleaned_keywords else 0,
                "silhouette_score": 0.0,
                "clusters": [{"cluster_id": 0, "name": "General", "keywords": cleaned_keywords}]
            }

        # Vectorize using TF-IDF word and character n-grams
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer="word", min_df=1)
        X = vectorizer.fit_transform(cleaned_keywords).toarray()

        # Heuristic determination of optimal cluster count
        if not n_clusters:
            n_clusters = max(2, min(len(cleaned_keywords) // min_cluster_size, 8))
            n_clusters = min(n_clusters, len(cleaned_keywords) - 1)

        if self.method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            labels = model.fit_predict(X)
        else:
            # Agglomerative clustering with cosine distance metric
            model = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
            labels = model.fit_predict(X)

        # Silhouette score computation
        try:
            sil_score = float(silhouette_score(X, labels, metric="cosine"))
        except Exception:
            sil_score = 0.5

        # Group keywords by cluster identifier
        cluster_groups: Dict[int, List[str]] = {}
        for kw, lbl in zip(cleaned_keywords, labels):
            lbl_int = int(lbl)
            if lbl_int not in cluster_groups:
                cluster_groups[lbl_int] = []
            cluster_groups[lbl_int].append(kw)

        # Format clusters and assign representative topic names
        formatted_clusters = []
        for cluster_id, kw_list in sorted(cluster_groups.items()):
            topic_name = self._generate_cluster_name(kw_list)
            formatted_clusters.append({
                "cluster_id": cluster_id,
                "name": topic_name,
                "keyword_count": len(kw_list),
                "keywords": kw_list
            })

        return {
            "total_keywords": len(cleaned_keywords),
            "total_clusters": len(formatted_clusters),
            "silhouette_score": round(sil_score, 3),
            "clusters": formatted_clusters
        }

    def _generate_cluster_name(self, kw_list: List[str]) -> str:
        """
        Derives the most representative cluster title from common high-frequency tokens.
        """
        word_counts: Dict[str, int] = {}
        for kw in kw_list:
            for w in kw.split():
                w_clean = w.lower().strip(".,!?:")
                if len(w_clean) > 2 and w_clean not in ["and", "for", "the", "with", "how", "best", "top"]:
                    word_counts[w_clean] = word_counts.get(w_clean, 0) + 1

        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = [w.capitalize() for w, _ in sorted_words[:3]]
        return " & ".join(top_words) if top_words else kw_list[0].title()
