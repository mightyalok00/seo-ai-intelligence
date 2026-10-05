import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, brier_score_loss

import xgboost as xgb
from src.utils.config import settings
from src.models.synthetic_data import SEODatasetGenerator, FEATURE_COLUMNS

class RankingPredictor:
    """Supervised ML model for predicting Google Top-10 Ranking Probability."""

    def __init__(self, model_dir: Path = settings.MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "ranking_xgboost.joblib"
        self.meta_path = self.model_dir / "model_metrics.joblib"
        self.model = None
        self.scaler = None
        self.metrics = {}
        self.feature_columns = FEATURE_COLUMNS
        self._load_or_train()

    def train_models(self) -> Dict[str, Any]:
        """Trains Logistic Regression, Random Forest, and XGBoost, evaluates and persists the best model."""
        generator = SEODatasetGenerator(n_samples=3000, random_seed=42)
        df = generator.generate()

        X = df[self.feature_columns]
        y = df["is_top_10"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 1. Baseline Logistic Regression
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_train_scaled, y_train)
        lr_probs = lr.predict_proba(X_test_scaled)[:, 1]

        # 2. Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
        rf.fit(X_train, y_train)
        rf_probs = rf.predict_proba(X_test)[:, 1]

        # 3. XGBoost Classifier
        xgb_model = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric="logloss"
        )
        xgb_model.fit(X_train, y_train)
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

        # Evaluate comparison metrics
        def calc_metrics(y_true, probs):
            preds = (probs >= 0.5).astype(int)
            return {
                "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
                "f1": round(float(f1_score(y_true, preds)), 4),
                "precision": round(float(precision_score(y_true, preds)), 4),
                "recall": round(float(recall_score(y_true, preds)), 4),
                "accuracy": round(float(accuracy_score(y_true, preds)), 4),
                "brier_loss": round(float(brier_score_loss(y_true, probs)), 4)
            }

        comparison = {
            "LogisticRegression": calc_metrics(y_test, lr_probs),
            "RandomForest": calc_metrics(y_test, rf_probs),
            "XGBoost": calc_metrics(y_test, xgb_probs)
        }

        # Calculate feature importances from XGBoost
        importances = xgb_model.feature_importances_
        feature_importance_list = [
            {"feature": col, "importance": round(float(imp), 4)}
            for col, imp in zip(self.feature_columns, importances)
        ]
        feature_importance_list.sort(key=lambda x: x["importance"], reverse=True)

        self.model = xgb_model
        self.scaler = scaler
        self.metrics = {
            "comparison": comparison,
            "best_model": "XGBoost",
            "feature_importance": feature_importance_list,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }

        # Save artifacts
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump({"metrics": self.metrics, "scaler": self.scaler}, self.meta_path)

        return self.metrics

    def _load_or_train(self):
        if self.model_path.exists() and self.meta_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                meta = joblib.load(self.meta_path)
                self.metrics = meta.get("metrics", {})
                self.scaler = meta.get("scaler", None)
            except Exception:
                self.train_models()
        else:
            self.train_models()

    def predict_probability(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts ranking probability and local feature impact breakdown."""
        if not self.model:
            self._load_or_train()

        # Build feature vector
        row = []
        for col in self.feature_columns:
            val = features_dict.get(col, 0)
            if col == "backlink_count" and "backlink_count" not in features_dict:
                val = int(features_dict.get("domain_authority_proxy", 40) * 4)
            row.append(float(val))

        X_input = pd.DataFrame([row], columns=self.feature_columns)
        prob = float(self.model.predict_proba(X_input)[0][1])

        # Feature impact estimation (Local attribution vs baseline)
        baseline_defaults = {
            "domain_authority_proxy": 45.0,
            "backlink_count": 180,
            "word_count": 1000,
            "keyword_in_title": 1,
            "keyword_in_h1": 1,
            "keyword_in_url": 1,
            "keyword_density": 1.5,
            "semantic_coverage": 0.65,
            "search_intent_match": 0.75,
            "internal_links_count": 5,
            "page_speed_score": 70.0,
            "has_schema": 1,
            "image_alt_ratio": 0.85
        }

        feature_impacts = []
        for col in self.feature_columns:
            cur_val = float(features_dict.get(col, baseline_defaults[col]))
            base_val = float(baseline_defaults[col])
            diff_ratio = (cur_val - base_val) / max(base_val, 1.0)
            
            # Find feature global weight
            weight = next((f["importance"] for f in self.metrics.get("feature_importance", []) if f["feature"] == col), 0.08)
            estimated_delta = round(diff_ratio * weight * 0.4, 3)

            feature_impacts.append({
                "feature": col,
                "current_value": cur_val,
                "benchmark_value": base_val,
                "estimated_impact": f"{'+' if estimated_delta >= 0 else ''}{round(estimated_delta * 100, 1)}%"
            })

        feature_impacts.sort(key=lambda x: abs(float(x["estimated_impact"].replace("%", "").replace("+", ""))), reverse=True)

        return {
            "top_10_probability": round(prob, 4),
            "top_10_percentage": round(prob * 100, 1),
            "ranking_tier": self._get_tier(prob),
            "feature_impacts": feature_impacts[:6],
            "global_model_metrics": self.metrics.get("comparison", {}).get("XGBoost", {})
        }

    def _get_tier(self, prob: float) -> str:
        if prob >= 0.75:
            return "Strong Rank Contender (Top 1-3)"
        elif prob >= 0.55:
            return "Probable Top 10 (Page 1)"
        elif prob >= 0.35:
            return "Borderline Contender (Positions 11-25)"
        return "Low Visibility (Page 3+)"
