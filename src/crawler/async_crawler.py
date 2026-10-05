"""
Asynchronous SEO Web Crawler and Technical DOM Auditor.

This module provides the core `SEOCrawler` class designed for non-blocking,
high-throughput crawling of target web pages. It extracts comprehensive on-page
structural metadata (headings, canonicals, robots directives, schema, images,
word counts, internal/external links) and measures Core Web Vitals.

Author: Alok Agarwal (mightyalok00)
License: MIT
"""

import re
import httpx
from bs4 import BeautifulSoup
import trafilatura
from urllib.parse import urlparse, urljoin
from typing import Dict, Any, List, Optional
from src.crawler.pagespeed_client import PageSpeedClient

class SEOCrawler:
    """
    High-performance asynchronous crawler extracting technical on-page signals.

    Attributes:
        timeout (float): Request timeout in seconds.
        headers (dict): HTTP headers including customizable User-Agent.
        pagespeed_client (PageSpeedClient): Client for Core Web Vitals audit.
    """

    def __init__(self, timeout: float = 15.0, user_agent: Optional[str] = None):
        """
        Initialize the SEOCrawler with custom connection limits and user agent.

        Args:
            timeout (float): Max seconds to wait for HTTP response. Default: 15.0.
            user_agent (str, optional): Custom HTTP User-Agent header string.
        """
        self.timeout = timeout
        self.headers = {
            "User-Agent": user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 SEO-AI-MLOps/1.0"
        }
        self.pagespeed_client = PageSpeedClient()

    async def crawl_url(self, url: str, fetch_pagespeed: bool = True) -> Dict[str, Any]:
        """
        Crawl a target URL asynchronously and parse on-page structural metrics.

        Args:
            url (str): Target web page URL.
            fetch_pagespeed (bool): Whether to retrieve Core Web Vitals / performance score.

        Returns:
            Dict[str, Any]: Parsed DOM features, heading hierarchy, links, and speed metrics.
        """
        # Ensure proper URL protocol prefix
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"

        parsed_origin = urlparse(url)
        domain = parsed_origin.netloc

        try:
            # Asynchronous HTTP GET request following redirects
            async with httpx.AsyncClient(headers=self.headers, follow_redirects=True, timeout=self.timeout) as client:
                response = await client.get(url)
                status_code = response.status_code
                html = response.text
                final_url = str(response.url)
        except Exception as e:
            return {
                "success": False,
                "url": url,
                "error": f"Connection failed: {str(e)}",
                "status_code": 0
            }

        # Parse HTML DOM with BeautifulSoup
        soup = BeautifulSoup(html, "lxml" if "lxml" in BeautifulSoup.__dict__ else "html.parser")

        # 1. Meta & Header extraction
        title_tag = soup.find("title")
        title = title_tag.get_text().strip() if title_tag else ""

        meta_desc_tag = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
        if not meta_desc_tag:
            meta_desc_tag = soup.find("meta", attrs={"property": "og:description"})
        meta_description = meta_desc_tag.get("content", "").strip() if meta_desc_tag else ""

        # Robots & Canonical URL directives
        robots_tag = soup.find("meta", attrs={"name": re.compile(r"^robots$", re.I)})
        robots_content = robots_tag.get("content", "").lower() if robots_tag else ""
        is_indexable = "noindex" not in robots_content

        canonical_tag = soup.find("link", attrs={"rel": "canonical"})
        canonical_url = canonical_tag.get("href", "") if canonical_tag else ""

        # 2. Heading Structure (H1, H2, H3)
        h1_tags = [h.get_text().strip() for h in soup.find_all("h1") if h.get_text().strip()]
        h2_tags = [h.get_text().strip() for h in soup.find_all("h2") if h.get_text().strip()]
        h3_tags = [h.get_text().strip() for h in soup.find_all("h3") if h.get_text().strip()]

        # 3. Clean Text Extraction & Word Count (Trafilatura for noise removal)
        extracted_text = trafilatura.extract(html) or ""
        if not extracted_text:
            extracted_text = soup.get_text(separator=" ", strip=True)

        words = re.findall(r"\b\w+\b", extracted_text)
        word_count = len(words)

        # 4. Internal and External Hyperlink Graph
        internal_links = []
        external_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue
            full_url = urljoin(final_url, href)
            link_domain = urlparse(full_url).netloc
            if link_domain == domain or not link_domain:
                internal_links.append(full_url)
            else:
                external_links.append(full_url)

        # 5. Image Alt Accessibility Coverage
        images = soup.find_all("img")
        images_count = len(images)
        images_missing_alt = sum(1 for img in images if not img.get("alt", "").strip())

        # 6. Structured Data (JSON-LD Schema)
        schemas = soup.find_all("script", attrs={"type": "application/ld+json"})
        has_schema = len(schemas) > 0

        # 7. Core Web Vitals & PageSpeed Performance
        pagespeed_metrics = {}
        if fetch_pagespeed:
            pagespeed_metrics = await self.pagespeed_client.get_metrics(final_url)

        return {
            "success": True,
            "url": final_url,
            "status_code": status_code,
            "is_indexable": is_indexable,
            "canonical_url": canonical_url,
            "title": title,
            "meta_description": meta_description,
            "h1_tags": h1_tags,
            "h2_tags": h2_tags,
            "h3_tags": h3_tags,
            "word_count": word_count,
            "extracted_text_preview": extracted_text[:1000],
            "full_text": extracted_text,
            "internal_links_count": len(internal_links),
            "external_links_count": len(external_links),
            "internal_links_sample": list(set(internal_links))[:10],
            "images_count": images_count,
            "images_missing_alt": images_missing_alt,
            "has_schema": has_schema,
            "pagespeed": pagespeed_metrics
        }
