"""
Unit Tests for Model Evaluation & Reliability Diagnostics.

Author: Alok Agarwal (mightyalok00)
License: MIT
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.models.evaluation import ModelEvaluator

def test_calibration_curve():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 1, 0])
    y_probs = np.array([0.1, 0.2, 0.35, 0.8, 0.9, 0.7, 0.25, 0.85, 0.65, 0.15])

    calib = ModelEvaluator.compute_calibration(y_true, y_probs, n_bins=5)
    assert "fraction_of_positives" in calib
    assert "mean_predicted_value" in calib
    assert "brier_score_loss" in calib
    assert 0.0 <= calib["brier_score_loss"] <= 1.0

def test_pr_and_roc_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_probs = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.85])

    res = ModelEvaluator.compute_pr_and_roc(y_true, y_probs)
    assert "roc_auc" in res
    assert "average_precision_score" in res
    assert "confusion_matrix" in res
    assert res["roc_auc"] >= 0.8

def test_temporal_backtesting():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 4), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series(np.random.binomial(1, 0.5, size=100))

    model = LogisticRegression()
    folds = ModelEvaluator.evaluate_temporal_backtesting(model, X, y, n_splits=3)
    assert len(folds) == 3
    assert all("roc_auc" in f for f in folds)
