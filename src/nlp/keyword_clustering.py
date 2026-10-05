import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

class KeywordClusterer:
    """Clusters keyword lists into semantic topic clusters and labels them."""

    def __init__(self, method: str = "agglomerative"):
        self.method = method

    def cluster_keywords(self, keywords: List[str], n_clusters: int = None, min_cluster_size: int = 2) -> Dict[str, Any]:
        """Cluster a list of keywords into topic groups."""
        cleaned_keywords = list(dict.fromkeys([kw.strip() for kw in keywords if kw.strip()]))
        
        if len(cleaned_keywords) < 2:
            return {
                "total_keywords": len(cleaned_keywords),
                "total_clusters": 1 if cleaned_keywords else 0,
                "silhouette_score": 0.0,
                "clusters": [{"cluster_id": 0, "name": "General", "keywords": cleaned_keywords}]
            }

        # Vectorize using TF-IDF character & word n-grams
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer="word", min_df=1)
        X = vectorizer.fit_transform(cleaned_keywords).toarray()

        # Determine optimal clusters count if not provided
        if not n_clusters:
            n_clusters = max(2, min(len(cleaned_keywords) // min_cluster_size, 8))
            n_clusters = min(n_clusters, len(cleaned_keywords) - 1)

        if self.method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            labels = model.fit_predict(X)
        else:
            # Agglomerative clustering with cosine distance
            model = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
            labels = model.fit_predict(X)

        # Silhouette score
        try:
            sil_score = float(silhouette_score(X, labels, metric="cosine"))
        except Exception:
            sil_score = 0.5

        # Group keywords by cluster
        cluster_groups: Dict[int, List[str]] = {}
        for kw, lbl in zip(cleaned_keywords, labels):
            lbl_int = int(lbl)
            if lbl_int not in cluster_groups:
                cluster_groups[lbl_int] = []
            cluster_groups[lbl_int].append(kw)

        # Generate topic names for each cluster
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
        """Derives the most representative cluster title from common terms."""
        word_counts: Dict[str, int] = {}
        for kw in kw_list:
            for w in kw.split():
                w_clean = w.lower().strip(".,!?:")
                if len(w_clean) > 2 and w_clean not in ["and", "for", "the", "with", "how", "best", "top"]:
                    word_counts[w_clean] = word_counts.get(w_clean, 0) + 1

        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = [w.capitalize() for w, _ in sorted_words[:3]]
        return " & ".join(top_words) if top_words else kw_list[0].title()
