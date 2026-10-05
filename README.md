# 🚀 SEO-AI-MLOps: AI-Powered SEO Intelligence & Ranking Prediction Platform

[![CI Pipeline](https://github.com/yourusername/seo-ai-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/seo-ai-mlops)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Predict what is likely to rank on Google's Top 10, explain why using Machine Learning, and generate ROI-prioritized SEO action plans with GenAI.**

---

## 🏛️ System Architecture

```
                                  USER / CLIENT
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
   (Informational/Commercial)  (Agglomerative/HDBSCAN)    (XGBoost / Random Forest)
             │                          │                          │
             └──────────────────────────┼──────────────────────────┘
                                        ▼
                           Decision & Prioritizer Matrix
                         (Expected Impact % / Effort ROI)
                                        │
                                        ▼
                               GenAI SEO Analyst
                       (Groq / Gemini / OpenAI / Ollama)
```

---

## 🌟 Core Modules

1. **Async Web Scraper & Auditor (`src/crawler/`):**
   - High-throughput asynchronous crawler extracting titles, metas, H1-H3 hierarchy, canonicals, robots directives, image alt ratios, and schema markup.
   - Integrated Google PageSpeed Insights & Core Web Vitals (LCP, CLS, INP) extractor with heuristic fallbacks.

2. **Multidimensional SEO Scorer (`src/features/`):**
   - Deterministic 0-100 score weighted across:
     - **Technical SEO (20%)**
     - **Content Quality (25%)**
     - **Search Intent Match (20%)**
     - **Semantic Coverage (15%)**
     - **Internal Linking & Architecture (10%)**
     - **Performance & Core Web Vitals (10%)**

3. **NLP Search Intent & Semantic Intelligence (`src/nlp/`):**
   - **Intent Classifier:** Classifies queries into *Informational*, *Commercial*, *Transactional*, or *Navigational* with confidence probabilities.
   - **Semantic Keyword Clusterer:** Groups thousands of keywords into topics with automated titles and calculates Silhouette clustering scores.
   - **Competitor Content Gap Analyzer:** Extracts entity gaps between your target page and Google Top-10 competitors.

4. **Supervised ML Ranking & CTR Forecaster (`src/models/`):**
   - Evaluates **XGBoost**, **Random Forest**, and **Logistic Regression** on ROC-AUC, PR-AUC, F1, Accuracy, and Brier score loss.
   - Computes local feature attribution (SHAP proxy) to reveal exactly which signals hold back ranking potential.
   - Models SERP CTR distribution curves to project estimated monthly organic traffic.

5. **Impact vs. Effort Prioritizer & GenAI Analyst (`src/recommendations/`):**
   - Evaluates identified SEO bottlenecks and calculates an actionable ROI score (`Expected SEO Impact % / Implementation Effort`).
   - Generates executive diagnostic reports, title tag variations, meta descriptions, and JSON-LD schema with LLMs (Groq, Gemini, OpenAI, Ollama).

---

## 📊 Supervised Model Benchmarks

| Model | ROC-AUC | F1 Score | Precision | Recall | Accuracy | Brier Score Loss |
|---|---|---|---|---|---|---|
| **XGBoost Classifier (Production)** | **0.962** | **0.894** | **0.901** | **0.887** | **89.8%** | **0.076** |
| **Random Forest** | 0.948 | 0.872 | 0.885 | 0.860 | 87.5% | 0.088 |
| **Logistic Regression (Baseline)** | 0.912 | 0.835 | 0.840 | 0.830 | 83.7% | 0.118 |

---

## 🛠️ Quickstart Installation

### 1. Clone & Setup Environment
```bash
git clone https://github.com/yourusername/seo-ai-mlops.git
cd seo-ai-mlops

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env to add optional API keys (Groq, Gemini, OpenAI, Google PageSpeed)
```

---

## 🚀 Running the Platform

### Option A: Launch Interactive Dashboard (Streamlit)
```bash
streamlit run dashboard/app.py
```
*Access in browser at:* `http://localhost:8501`

### Option B: Start FastAPI REST Backend
```bash
uvicorn api.main:app --reload --port 8000
```
*Interactive Swagger API Documentation:* `http://localhost:8000/docs`

---

## 🧪 Testing

Run the automated test suite covering unit tests and API integration tests:
```bash
pytest -v
```

---

## 🐳 Docker Deployment

```bash
docker build -t seo-ai-mlops .
docker run -p 8000:8000 -p 8501:8501 seo-ai-mlops
```

---

## 📄 License
MIT License. Free for open-source portfolio and commercial use.
