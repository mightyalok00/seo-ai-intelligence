# 🚀 SEO-AI-MLOps: Machine Learning Powered SEO Intelligence & Ranking Prediction Platform

[![CI Pipeline](https://github.com/mightyalok00/seo-ai-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/mightyalok00/seo-ai-intelligence/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An end-to-end Machine Learning and Generative AI system that models search ranking signals, classifies search intent, extracts competitor content gaps, and computes an ROI-ranked action plan (Impact vs. Effort) for organic growth.**

---

## 🏛️ System Architecture

```
                                      CLIENT / USER
                                            │
                             ┌──────────────┴──────────────┐
                             ▼                             ▼
                     Streamlit Dashboard              FastAPI REST API
                       (Port: 8501)                     (Port: 8000)
                             │                             │
                             └──────────────┬──────────────┘
                                            ▼
                                   Asynchronous Crawler
                               (httpx + BeautifulSoup + CWV)
                                            │
                                            ▼
                               Feature Engineering Engine
                         (DOM, Headings, Meta, Links, Density)
                                            │
                 ┌──────────────────────────┼──────────────────────────┐
                 ▼                          ▼                          ▼
          NLP Search Intent        Keyword Clusterer          Supervised ML Predictor
       (Informational/Commercial)  (Agglomerative/HDBSCAN)    (Calibrated XGBoost / RF)
                 │                          │                          │
                 └──────────────────────────┼──────────────────────────┘
                                            ▼
                               TreeSHAP Attribution Matrix
                               (Exact Feature Contributions)
                                            │
                                            ▼
                               Decision & Prioritizer Matrix
                             (Expected Impact % / Effort ROI)
                                            │
                                            ▼
                                   GenAI SEO Analyst
                           (Groq / Gemini / OpenAI / Ollama)
```

---

## 🌟 Core Capabilities

1. **High-Throughput Asynchronous Crawler (`src/crawler/`):**
   * Extracts titles, metas, heading hierarchies (H1–H3), canonicals, robots directives, image alt distributions, and JSON-LD schema.
   * Integrates Google PageSpeed Insights & Core Web Vitals (LCP, CLS, INP) with graceful heuristic fallbacks.

2. **Multidimensional SEO Health Scoring (`src/features/`):**
   * Computes a deterministic 0–100 score across 6 weighted pillars:
     * **Technical SEO (20%)** • **Content Quality (25%)** • **Search Intent Alignment (20%)**
     * **Semantic Coverage (15%)** • **Internal Linking (10%)** • **Performance / CWV (10%)**

3. **NLP Search Intent & Semantic Clustering (`src/nlp/`):**
   * **Intent Classification:** Classifies search queries into *Informational*, *Commercial*, *Transactional*, or *Navigational* with calibrated confidence.
   * **Semantic Topic Clustering:** Groups keyword universes using TF-IDF / embeddings with automated cluster titling and Silhouette validation.
   * **Competitor Content Gap Detection:** Scrapes Top-10 competitors and flags missing semantic entities.

4. **Supervised ML Ranking Predictor & TreeSHAP Attribution (`src/models/`):**
   * Evaluates **XGBoost**, **Random Forest**, and **Logistic Regression** across ROC-AUC, PR-AUC, F1, Accuracy, and Brier Loss.
   * Computes **exact TreeSHAP values** (`pred_contribs=True`) for transparent feature attribution on every prediction.
   * Models Google SERP CTR distributions to forecast monthly organic clicks.

5. **Impact vs. Effort Prioritizer & GenAI Analyst (`src/recommendations/`):**
   * Ranks identified bottlenecks by **ROI (`Expected SEO Impact % / Implementation Effort`)**.
   * Generates executive diagnostic reports, optimized title/meta tags, and FAQPage JSON-LD schemas via Groq, Gemini, OpenAI, or local Ollama.

---

## 📊 Supervised Model Benchmarks

| Model Architecture | ROC-AUC | F1 Score | Precision | Recall | Accuracy | Brier Score Loss |
|---|---|---|---|---|---|---|
| **XGBoost Classifier (Production)** | **0.962** | **0.894** | **0.901** | **0.887** | **89.8%** | **0.076** |
| **Random Forest** | 0.948 | 0.872 | 0.885 | 0.860 | 87.5% | 0.088 |
| **Logistic Regression (Baseline)** | 0.912 | 0.835 | 0.840 | 0.830 | 83.7% | 0.118 |

---

## 🔬 Scientific Methodology & Data Framing

> [!NOTE]
> **Methodology Positioning:** This platform does not claim to decode Google's proprietary search ranking algorithm. Rather, it **models observable search ranking propensity** from structural on-page, semantic, and authority signals.
> 
> The baseline models are trained on an **empirically parameterized benchmark dataset** (`src/models/synthetic_data.py`) derived from published industry SERP correlation distributions (word count, backlink decay, intent alignment, and Core Web Vitals). The training pipeline is fully extensible to ingest verified Google Search Console (GSC) or proprietary rank logs.

---

## 🛠️ Quickstart Installation

### 1. Clone & Setup
```bash
git clone https://github.com/mightyalok00/seo-ai-intelligence.git
cd seo-ai-intelligence

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
# Optional: Add GROQ_API_KEY, GEMINI_API_KEY, or GOOGLE_PAGESPEED_API_KEY in .env
```

---

## 🚀 Running the Platform

### Start the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
*Access in browser:* **`http://localhost:8501`**

### Start the FastAPI REST Server
```bash
uvicorn api.main:app --reload --port 8000
```
*Interactive Swagger Documentation:* **`http://localhost:8000/docs`**

---

## 📡 REST API Examples

### 1. End-to-End Audit & Ranking Prediction
```bash
curl -X POST "http://localhost:8000/api/v1/crawl/analyze-full" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://en.wikipedia.org/wiki/Machine_learning",
       "target_keyword": "machine learning tutorial",
       "competitor_urls": ["https://en.wikipedia.org/wiki/Artificial_intelligence"],
       "domain_authority_proxy": 85.0,
       "monthly_search_volume": 12000
     }'
```

### 2. Search Intent Batch Classification
```bash
curl -X POST "http://localhost:8000/api/v1/intent/classify" \
     -H "Content-Type: application/json" \
     -d '{
       "keywords": [
         "what is random forest",
         "best seo software 2026",
         "buy ahrefs subscription",
         "google search console login"
       ]
     }'
```

---

## 🧪 Automated Testing

Run the automated test suite covering unit tests and API integration tests:
```bash
pytest -v
```

---

## 🐳 Docker Containerization

```bash
docker build -t seo-ai-intelligence .
docker run -p 8000:8000 -p 8501:8501 seo-ai-intelligence
```

---

## 📄 License
MIT License. Created by [Alok Agarwal](https://github.com/mightyalok00).
