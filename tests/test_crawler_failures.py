"""
Edge-Case & Failure-Path Regression Tests for Async SEO Crawler.

Author: Alok Agarwal (mightyalok00)
License: MIT
"""

import pytest

from src.crawler.async_crawler import SEOCrawler


@pytest.mark.asyncio
async def test_crawler_invalid_domain():
    crawler = SEOCrawler(timeout=3.0)
    res = await crawler.crawl_url("https://this-is-a-completely-non-existent-domain-xyz123456.org", fetch_pagespeed=False)
    assert res["success"] is False
    assert res["status_code"] == 0
    assert "error" in res

@pytest.mark.asyncio
async def test_crawler_url_without_scheme():
    crawler = SEOCrawler(timeout=5.0)
    res = await crawler.crawl_url("example.com", fetch_pagespeed=False)
    assert res["success"] is True
    assert res["url"].startswith("http")

@pytest.mark.asyncio
async def test_crawler_pagespeed_heuristic_fallback():
    crawler = SEOCrawler()
    metrics = crawler.pagespeed_client._heuristic_performance("https://example.com/test-page")
    assert "performance_score" in metrics
    assert "lcp_seconds" in metrics
    assert 0 <= metrics["performance_score"] <= 100
    assert metrics["source"] == "heuristic_fallback"
