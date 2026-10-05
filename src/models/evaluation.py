"""
Model Evaluation, Calibration & Reliability Diagnostic Suite.

This module provides comprehensive evaluation utilities for classification models:
- Probability Calibration Curves (Reliability Diagrams)
- Precision-Recall (PR) Curves & Average Precision
- ROC-AUC with False Positive / True Positive rates
- Confusion Matrix computation
- Temporal (Time-Aware) Cross-Validation

Author: Alok Agarwal (mightyalok00)
License: MIT
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    classification_report
)
from sklearn.model_selection import TimeSeriesSplit

class ModelEvaluator:
    """
    Computes rigorous statistical diagnostic curves and calibration metrics.
    """

    @staticmethod
    def compute_calibration(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        """
        Compute calibration curve (reliability diagram) measuring predicted probability vs true frequency.

        Args:
            y_true (np.ndarray): True binary labels (0 or 1).
            y_probs (np.ndarray): Predicted probabilities (0 to 1).
            n_bins (int): Number of discretization bins. Default: 10.

        Returns:
            Dict[str, Any]: Fraction of positives, mean predicted value, and Brier score loss.
        """
        prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy="uniform")
        brier = float(brier_score_loss(y_true, y_probs))

        return {
            "fraction_of_positives": [round(float(p), 4) for p in prob_true],
            "mean_predicted_value": [round(float(p), 4) for p in prob_pred],
            "brier_score_loss": round(brier, 4)
        }

    @staticmethod
    def compute_pr_and_roc(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, Any]:
        """
        Compute ROC and Precision-Recall curve coordinates with AUC metrics.
        """
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc_val = float(auc(fpr, tpr))

        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        avg_precision = float(average_precision_score(y_true, y_probs))

        preds = (y_probs >= 0.5).astype(int)
        cm = confusion_matrix(y_true, preds).tolist()

        return {
            "roc_auc": round(roc_auc_val, 4),
            "average_precision_score": round(avg_precision, 4),
            "confusion_matrix": {
                "true_negative": cm[0][0],
                "false_positive": cm[0][1],
                "false_negative": cm[1][0],
                "true_positive": cm[1][1]
            },
            "roc_curve": {
                "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr)//20)]],
                "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr)//20)]]
            },
            "pr_curve": {
                "recall": [round(float(x), 4) for x in recall[::max(1, len(recall)//20)]],
                "precision": [round(float(x), 4) for x in precision[::max(1, len(precision)//20)]]
            }
        }

    @staticmethod
    def evaluate_temporal_backtesting(
        model,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Executes time-aware (chronological) rolling window cross-validation
        to prevent future data leakage in temporal search rank dynamics.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_te)[:, 1]
            preds = (probs >= 0.5).astype(int)

            fold_results.append({
                "fold": fold,
                "train_size": len(X_tr),
                "test_size": len(X_te),
                "roc_auc": round(float(auc(*roc_curve(y_te, probs)[:2])), 4),
                "brier_loss": round(float(brier_score_loss(y_te, probs)), 4)
            })

        return fold_results
