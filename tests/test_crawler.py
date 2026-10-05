import pytest

from src.crawler.async_crawler import SEOCrawler


@pytest.mark.asyncio
async def test_crawler_parsing():
    crawler = SEOCrawler()
    # Test with public stable webpage
    res = await crawler.crawl_url("https://example.com", fetch_pagespeed=False)
    assert res["success"] is True
    assert res["status_code"] == 200
    assert "Example Domain" in res["title"]
    assert res["word_count"] > 0
    assert isinstance(res["internal_links_count"], int)
