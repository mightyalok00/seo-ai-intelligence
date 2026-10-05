import sys
from pathlib import Path

# Add project root directory to sys.path so 'src' and other root modules can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import json

# Setup page config
st.set_page_config(
    page_title="SEO-AI-MLOps Intelligence Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366F1, #8B5CF6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1E293B;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-high {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Lazy import engine components
from src.crawler.async_crawler import SEOCrawler
from src.features.extractor import SEOFeatureExtractor
from src.features.scorer import SEOScorer
from src.nlp.intent_classifier import SearchIntentClassifier
from src.nlp.keyword_clustering import KeywordClusterer
from src.nlp.semantic_matcher import SemanticMatcher
from src.models.ranking_predictor import RankingPredictor
from src.models.ctr_forecaster import CTRForecaster
from src.recommendations.prioritizer import RecommendationPrioritizer
from src.recommendations.ai_analyst import AISEOAnalyst

@st.cache_resource
def load_components():
    return {
        "crawler": SEOCrawler(),
        "extractor": SEOFeatureExtractor(),
        "scorer": SEOScorer(),
        "intent_clf": SearchIntentClassifier(),
        "clusterer": KeywordClusterer(),
        "matcher": SemanticMatcher(),
        "ranking_model": RankingPredictor(),
        "ctr_model": CTRForecaster(),
        "prioritizer": RecommendationPrioritizer(),
        "analyst": AISEOAnalyst()
    }

comp = load_components()

# Sidebar Navigation
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400&auto=format&fit=crop&q=60", use_container_width=True)
    st.markdown("### ⚡ SEO-AI-MLOps Engine")
    st.markdown("Predictive Machine Learning & GenAI Decision Platform.")
    st.divider()
    app_mode = st.radio(
        "Navigation",
        [
            "🎯 Live URL Audit & ML Predictor",
            "🧠 Search Intent Classifier",
            "🗂️ Semantic Keyword Clusterer",
            "📈 ML Benchmarks & Feature Importance",
            "🤖 AI Content & Schema Generator"
        ]
    )
    st.divider()
    st.caption("Active Model: **XGBoost Classifier v2.0**")
    st.caption("Status: 🟢 **Pipeline Online**")

# Main Title Header
st.markdown('<div class="main-title">SEO-AI-MLOps Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Machine Learning Ranking Probability • NLP Search Intent • Prioritized ROI Action Engine</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 1: Live URL Audit & ML Predictor
# ---------------------------------------------------------
if app_mode == "🎯 Live URL Audit & ML Predictor":
    st.subheader("🌐 Website Crawl, Ranking Prediction & Action Matrix")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        target_url = st.text_input("Target URL to Audit", value="https://en.wikipedia.org/wiki/Machine_learning")
    with col2:
        target_keyword = st.text_input("Primary Target Keyword", value="machine learning tutorial")

    with st.expander("⚙️ Advanced Parameters (Competitors & Search Volume)", expanded=False):
        c_col1, c_col2, c_col3 = st.columns(3)
        with c_col1:
            comp_url = st.text_input("Benchmark Competitor URL", value="https://en.wikipedia.org/wiki/Artificial_intelligence")
        with c_col2:
            domain_da = st.slider("Domain Authority Proxy (DA)", 1, 100, 55)
        with c_col3:
            search_vol = st.number_input("Monthly Search Volume", min_value=100, max_value=500000, value=6600)

    if st.button("🚀 Run Live End-to-End SEO Audit", type="primary", use_container_width=True):
        with st.spinner("Crawling target page, extracting DOM features, running NLP intent and XGBoost models..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 1. Crawl
            crawl_data = loop.run_until_complete(comp["crawler"].crawl_url(target_url, fetch_pagespeed=True))
            
            if not crawl_data.get("success"):
                st.error(f"Failed to crawl URL: {crawl_data.get('error')}")
            else:
                # 2. Competitor analysis
                comp_texts = []
                if comp_url:
                    comp_crawl = loop.run_until_complete(comp["crawler"].crawl_url(comp_url, fetch_pagespeed=False))
                    if comp_crawl.get("success") and comp_crawl.get("full_text"):
                        comp_texts.append(comp_crawl["full_text"])
                
                gap_analysis = comp["matcher"].analyze_content_gaps(crawl_data.get("full_text", ""), comp_texts)

                # 3. Intent & Features
                intent_res = comp["intent_clf"].predict(target_keyword)
                features = comp["extractor"].extract_features(crawl_data, target_keyword, domain_authority_proxy=float(domain_da))
                features["search_intent_match"] = intent_res.get("confidence", 0.8)

                # 4. Scoring & ML Prediction
                score_res = comp["scorer"].calculate_score(features, search_intent_match=intent_res.get("confidence", 0.8))
                prediction = comp["ranking_model"].predict_probability(features)
                traffic_forecast = comp["ctr_model"].forecast_traffic(prediction["top_10_probability"], monthly_search_volume=search_vol)
                
                # 5. Prioritization & AI Analyst
                action_matrix = comp["prioritizer"].prioritize(
                    features=features,
                    current_prob=prediction["top_10_probability"],
                    intent_data=intent_res,
                    missing_topics=gap_analysis.get("missing_topics", [])
                )
                ai_report = loop.run_until_complete(
                    comp["analyst"].generate_seo_report(
                        url=target_url,
                        target_keyword=target_keyword,
                        ranking_probability=prediction["top_10_probability"],
                        seo_score=score_res["overall_seo_score"],
                        intent_data=intent_res,
                        feature_impacts=prediction.get("feature_impacts", []),
                        prioritized_actions=action_matrix.get("prioritized_actions", []),
                        missing_topics=gap_analysis.get("missing_topics", [])
                    )
                )

                st.success("✅ Audit, Machine Learning Prediction, and AI Analysis Complete!")

                # --- TOP KPI METRIC CARDS ---
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric(
                        label="Overall SEO Score",
                        value=f"{score_res['overall_seo_score']} / 100",
                        delta=f"{'+' if score_res['overall_seo_score'] >= 75 else ''}{round(score_res['overall_seo_score']-70, 1)} vs Benchmark"
                    )
                with m2:
                    st.metric(
                        label="Top-10 Ranking Probability",
                        value=f"{prediction['top_10_percentage']}%",
                        delta=f"+{action_matrix['estimated_probability_gain_pct']}% Potential Gain"
                    )
                with m3:
                    st.metric(
                        label="Search Intent",
                        value=intent_res["intent"],
                        delta=f"{int(intent_res['confidence']*100)}% Confidence"
                    )
                with m4:
                    st.metric(
                        label="Forecast Monthly Traffic",
                        value=f"{traffic_forecast['forecast_monthly_clicks']:,} clicks",
                        delta=f"Rank ~#{traffic_forecast['expected_ranking_position']}"
                    )

                st.divider()

                # --- 2-COLUMN DETAILED VIEW ---
                d_col1, d_col2 = st.columns([1, 1])

                with d_col1:
                    st.markdown("### 📊 SEO Pillar Breakdown")
                    pillars = score_res["pillars"]
                    fig_radar = go.Figure(data=go.Scatterpolar(
                        r=[pillars["technical_seo"], pillars["content_quality"], pillars["search_intent"],
                           pillars["semantic_coverage"], pillars["internal_linking"], pillars["performance"]],
                        theta=["Technical", "Content Quality", "Search Intent", "Semantic Coverage", "Internal Links", "Performance"],
                        fill="toself",
                        line_color="#6366F1"
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=False,
                        margin=dict(l=40, r=40, t=20, b=20),
                        height=320
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                with d_col2:
                    st.markdown("### 🎯 SERP CTR Projection Curve")
                    ctr_df = pd.DataFrame(traffic_forecast["serp_ctr_curve"])
                    fig_ctr = px.bar(
                        ctr_df,
                        x="position",
                        y="potential_monthly_clicks",
                        title=f"Estimated Monthly Clicks by SERP Rank (Vol: {search_vol:,})",
                        color="potential_monthly_clicks",
                        color_continuous_scale="Viridis",
                        labels={"position": "Google SERP Rank Position", "potential_monthly_clicks": "Monthly Clicks"}
                    )
                    fig_ctr.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_ctr, use_container_width=True)

                # --- PRIORITIZED ACTION MATRIX ---
                st.markdown("### ⭐ Prioritized ROI Action Plan (Expected Impact vs. Effort)")
                st.caption("Fix high-ROI bottlenecks first to achieve the fastest ranking gains.")

                actions = action_matrix.get("prioritized_actions", [])
                if actions:
                    act_df = pd.DataFrame(actions)[["rank", "title", "category", "expected_impact_pct", "effort", "roi_tier", "description"]]
                    act_df.columns = ["Rank", "Action Item", "Category", "Impact (+%)", "Effort", "ROI Tier", "Details"]
                    st.dataframe(act_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No critical bottlenecks identified on this page!")

                # --- AI ANALYST SUMMARY ---
                st.markdown("### 🤖 AI SEO Analyst Diagnostic Report")
                st.info(ai_report.get("executive_summary", ""))

                c_tag1, c_tag2 = st.columns(2)
                with c_tag1:
                    st.markdown("#### 💡 Recommended Title Tag Variations")
                    for t in ai_report.get("recommended_title_tags", []):
                        st.markdown(f"- **{t}**")
                with c_tag2:
                    st.markdown("#### 📝 Recommended Meta Descriptions")
                    for m in ai_report.get("recommended_meta_descriptions", []):
                        st.markdown(f"- _{m}_")

# ---------------------------------------------------------
# TAB 2: NLP Search Intent Classifier
# ---------------------------------------------------------
elif app_mode == "🧠 Search Intent Classifier":
    st.subheader("🧠 NLP Search Intent Classifier")
    st.write("Classifies search queries into **Informational**, **Commercial**, **Transactional**, or **Navigational** intents.")

    default_keywords = (
        "what is random forest in python\n"
        "best seo software 2026\n"
        "buy ahrefs pro subscription\n"
        "google search console login\n"
        "how to calculate precision and recall\n"
        "semrush vs ahrefs pricing comparison\n"
        "order cloud gpu server\n"
        "fastapi documentation tutorial"
    )

    kw_input = st.text_area("Enter Keywords (One per line)", value=default_keywords, height=180)
    
    if st.button("Classify Intent Batch", type="primary"):
        keywords_list = [k.strip() for k in kw_input.split("\n") if k.strip()]
        predictions = comp["intent_clf"].predict_batch(keywords_list)
        df_intent = pd.DataFrame(predictions)
        
        c1, c2 = st.columns([3, 2])
        with c1:
            st.dataframe(df_intent, use_container_width=True)
        with c2:
            intent_counts = df_intent["intent"].value_counts().reset_index()
            intent_counts.columns = ["Intent", "Count"]
            fig_intent = px.pie(intent_counts, names="Intent", values="Count", hole=0.4, title="Intent Distribution", color_discrete_sequence=px.colors.qualitative.Prism)
            st.plotly_chart(fig_intent, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: Semantic Keyword Clusterer
# ---------------------------------------------------------
elif app_mode == "🗂️ Semantic Keyword Clusterer":
    st.subheader("🗂️ Semantic Keyword Clusterer")
    st.write("Automatically groups large keyword lists into semantic clusters with automated topic headers.")

    sample_cluster_text = (
        "python for beginners\n"
        "learn python programming online\n"
        "python course with certificate\n"
        "pandas dataframe tutorial\n"
        "pandas groupby syntax\n"
        "how to use pandas in python\n"
        "xgboost hyperparameter tuning\n"
        "xgboost classification tutorial\n"
        "gradient boosting with xgboost\n"
        "technical seo audit checklist\n"
        "how to fix core web vitals\n"
        "page speed optimization guide"
    )

    cluster_input = st.text_area("Keywords to Cluster", value=sample_cluster_text, height=220)
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        method = st.selectbox("Clustering Algorithm", ["agglomerative", "kmeans"])
    with col_c2:
        min_size = st.slider("Minimum Cluster Size", 1, 5, 2)

    if st.button("Cluster Keywords", type="primary"):
        kw_list = [k.strip() for k in cluster_input.split("\n") if k.strip()]
        result = comp["clusterer"].cluster_keywords(kw_list, min_cluster_size=min_size)
        
        st.success(f"Generated **{result['total_clusters']}** clusters across **{result['total_keywords']}** keywords (Silhouette Score: **{result['silhouette_score']}**)")
        
        for c in result["clusters"]:
            with st.expander(f"📁 Cluster #{c['cluster_id'] + 1}: {c['name']} ({c['keyword_count']} keywords)", expanded=True):
                for kw in c["keywords"]:
                    st.markdown(f"• `{kw}`")

# ---------------------------------------------------------
# TAB 4: ML Benchmarks & Feature Importance
# ---------------------------------------------------------
elif app_mode == "📈 ML Benchmarks & Feature Importance":
    st.subheader("📈 Supervised Model Benchmarks & SHAP Explainability")
    
    benchmarks = comp["ranking_model"].metrics.get("comparison", {})
    if benchmarks:
        bench_df = pd.DataFrame(benchmarks).T.reset_index()
        bench_df.columns = ["Model", "ROC-AUC", "F1 Score", "Precision", "Recall", "Accuracy", "Brier Loss"]
        st.markdown("#### Model Performance Evaluation Matrix")
        st.dataframe(bench_df.style.highlight_max(axis=0, color="#3730A3"), use_container_width=True)

    st.divider()
    st.markdown("#### 🌟 Global Feature Importance (XGBoost)")
    importances = comp["ranking_model"].metrics.get("feature_importance", [])
    if importances:
        imp_df = pd.DataFrame(importances)
        fig_imp = px.bar(
            imp_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Relative Feature Importance in Top-10 SERP Prediction",
            color="importance",
            color_continuous_scale="Purples"
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=450)
        st.plotly_chart(fig_imp, use_container_width=True)

# ---------------------------------------------------------
# TAB 5: AI Content & Schema Generator
# ---------------------------------------------------------
elif app_mode == "🤖 AI Content & Schema Generator":
    st.subheader("🤖 AI Content Optimizer & Schema Generator")
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        topic = st.text_input("Target Topic / Keyword", "Python Data Science Tutorial")
        page_intent = st.selectbox("Intended Search Intent", ["Informational", "Commercial", "Transactional", "Navigational"])
    with col_a2:
        missing_entities = st.text_input("Entities to Inject", "Cross-validation, Pandas, Scikit-learn, XGBoost")

    if st.button("Generate AI Outline & JSON-LD Schema", type="primary"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        entities_list = [e.strip() for e in missing_entities.split(",") if e.strip()]
        report = loop.run_until_complete(
            comp["analyst"].generate_seo_report(
                url="https://example.com/generated-outline",
                target_keyword=topic,
                ranking_probability=0.72,
                seo_score=82.0,
                intent_data={"intent": page_intent, "confidence": 0.9},
                feature_impacts=[],
                prioritized_actions=[],
                missing_topics=entities_list
            )
        )

        st.markdown("### 📋 Optimized Heading Hierarchy (H2 / H3)")
        for h in report.get("recommended_h2_structure", []):
            st.markdown(f"#### 🔹 {h}")

        st.divider()
        st.markdown("### 📜 FAQPage JSON-LD Structured Data")
        faqs = report.get("faq_schema_questions", [])
        json_ld = {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": [
                {
                    "@type": "Question",
                    "name": f["question"],
                    "acceptedAnswer": {"@type": "Answer", "text": f["answer"]}
                }
                for f in faqs
            ]
        }
        st.code(json.dumps(json_ld, indent=2), language="json")
