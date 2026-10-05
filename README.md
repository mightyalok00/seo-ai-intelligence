# 🚀 SEO-AI-MLOps: Machine Learning Powered SEO Intelligence & Ranking Prediction Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B.svg)](https://seo-ai-intelligence.streamlit.app/)\n[![CI Pipeline](https://github.com/mightyalok00/seo-ai-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/mightyalok00/seo-ai-intelligence/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.2-009688.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A research-grade Machine Learning and Generative AI platform that models search ranking signals, classifies search intent, extracts competitor content gaps, and computes an ROI-prioritized action plan (Impact vs. Effort) for organic search growth.**

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

## 🌟 Core Modules

1. **High-Throughput Asynchronous Crawler (`src/crawler/`):**
   * Extracts titles, metas, heading hierarchies (H1–H3), canonicals, robots directives, image alt ratios, and schema markup.
   * Integrates Google PageSpeed Insights & Core Web Vitals (LCP, CLS, INP) with heuristic fallbacks.

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
   * Integrates real-data fine-tuning with **Google Search Console (GSC)** performance extracts.

5. **Impact vs. Effort Prioritizer & GenAI Analyst (`src/recommendations/`):**
   * Ranks identified bottlenecks by **ROI (`Expected SEO Impact % / Implementation Effort`)**.
   * Generates executive diagnostic reports, optimized title/meta tags, and FAQPage JSON-LD schemas via Groq, Gemini, OpenAI, or local Ollama.

---

## 📊 Dual Benchmark Provenance & Validation Results

To ensure scientific rigor and transparent provenance, all results are deterministic and reproducible via `python scripts/reproduce_benchmarks.py`:

### Benchmark 1: Controlled SERP Parameterized Benchmark (N=3,500)
* **Dataset Provenance:** Statistically parameterized SERP distribution modeling empirical search feature decays (`src/models/synthetic_data.py`).
* **Validation Strategy:** Stratified 80/20 train/test holdout with random_seed=42.

| Model Architecture | ROC-AUC | F1 Score | Precision | Recall | Accuracy | Brier Score Loss |
|---|---|---|---|---|---|---|
| **XGBoost Classifier (Production)** | **0.969** | **0.932** | **0.923** | **0.942** | **91.3%** | **0.068** |
| **Logistic Regression (L2 Baseline)** | 0.981 | 0.939 | 0.932 | 0.946 | 92.1% | 0.052 |
| **Random Forest (Ensemble)** | 0.943 | 0.898 | 0.853 | 0.949 | 86.3% | 0.111 |

### Benchmark 2: Observational Google Search Console (GSC) Extract (N=1,200)
* **Dataset Provenance:** Real-world anonymized search queries, impressions, CTR, and SERP positions (`src/models/gsc_pipeline.py`).
* **Ground-Truth Target:** `is_top_10 = (avg_position <= 10.0)`.

| Model Architecture | ROC-AUC | F1 Score | Precision | Recall (Top 10) | Brier Score Loss |
|---|---|---|---|---|---|
| **XGBoost (Fine-Tuned on GSC Logs)** | **0.760** | **0.794** | **0.691** | **93.5%** | **0.173** |

*Interpretation: The GSC observational benchmark captures real-world noisy SERP fluctuations and algorithmic non-stationarity, demonstrating robust generalization on real search queries.*

---

## 🔬 1-Command Benchmark Reproduction

Execute the automated reproduction script to re-train all models, compute calibration curves, and generate the official experiment manifest:

```bash
python scripts/reproduce_benchmarks.py
```
*Manifest output:* [`experiments/benchmark_reproduction_manifest.json`](file:///e:/Seo/experiments/benchmark_reproduction_manifest.json)

---

## 🌐 Live Demo\n\n**Try the deployed SEO AI Intelligence dashboard:**\n\n👉 https://seo-ai-intelligence.streamlit.app/\n\nThe live application runs the Streamlit dashboard from `dashboard/app.py`.\n\n---\n\n## 🛠️ Quickstart Installation

### 1. Clone & Setup
```bash
git clone https://github.com/mightyalok00/seo-ai-intelligence.git
cd seo-ai-intelligence

python -m venv venv
# Windows:
.\\venv\\Scripts\\activate
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

## 🚀 Running Locally

### Option A: Start Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
*Access in browser:* **`http://localhost:8501`**

### Option B: Start FastAPI REST Server
```bash
uvicorn api.main:app --reload --port 8000
```
*Interactive Swagger Documentation:* **`http://localhost:8000/docs`**

---

## ☁️ Streamlit Community Cloud Deployment

Use the following deployment settings:

- **Repository:** `mightyalok00/seo-ai-intelligence`
- **Branch:** `main`
- **Main file path:** `dashboard/app.py`
- **Python:** `3.12`
- **Secrets:** optional; add API keys only when using the corresponding GenAI/PageSpeed integrations.\n\n**Live app:** https://seo-ai-intelligence.streamlit.app/

The project pins Streamlit and FastAPI to compatible versions in `requirements.txt`. Streamlit Community Cloud will detect dependency changes committed to GitHub and re-resolve the environment automatically.

---

## 🐳 Docker & Docker Compose Deployment

Run both the FastAPI Backend (`port 8000`) and the Streamlit Dashboard (`port 8501`) as isolated microservices:

```bash
docker compose up --build -d
```

* **FastAPI Backend:** `http://localhost:8000/docs`
* **Streamlit Dashboard:** `http://localhost:8501`

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

## 🧪 Automated Testing Suite

Run the full automated test suite covering unit tests, GSC pipelines, calibration, and API integration tests:
```bash
pytest -v
```

---

## 📄 License
MIT License. Created by [Alok Agarwal](https://github.com/mightyalok00).
