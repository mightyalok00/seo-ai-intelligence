import pytest
from src.nlp.intent_classifier import SearchIntentClassifier
from src.nlp.keyword_clustering import KeywordClusterer

def test_intent_classifier():
    clf = SearchIntentClassifier()
    info_res = clf.predict("what is linear regression")
    assert info_res["intent"] == "Informational"

    trans_res = clf.predict("buy ahrefs pro plan")
    assert trans_res["intent"] == "Transactional"

    comm_res = clf.predict("best seo software comparison")
    assert comm_res["intent"] == "Commercial"

def test_keyword_clustering():
    clusterer = KeywordClusterer()
    keywords = [
        "learn python online", "best python tutorial", "python beginner course",
        "buy seo tool", "purchase semrush account"
    ]
    res = clusterer.cluster_keywords(keywords, min_cluster_size=2)
    assert res["total_clusters"] >= 2
    assert res["total_keywords"] == 5
