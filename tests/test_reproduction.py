"""
Unit and Integration Tests for Benchmark Reproduction Pipeline.

Author: Alok Agarwal (mightyalok00)
License: MIT
"""

from scripts.reproduce_benchmarks import run_reproduction_pipeline


def test_reproduction_manifest_generation():
    manifest = run_reproduction_pipeline()
    assert manifest is not None
    assert manifest["manifest_version"] == "1.0.0"
    assert "benchmark_1_controlled_serp" in manifest["benchmarks"]
    assert "benchmark_2_gsc_observational" in manifest["benchmarks"]

    bench1 = manifest["benchmarks"]["benchmark_1_controlled_serp"]
    assert bench1["sample_count"] == 3500
    assert "XGBoost" in bench1["models"]
    assert bench1["models"]["XGBoost"]["roc_auc"] > 0.90

    bench2 = manifest["benchmarks"]["benchmark_2_gsc_observational"]
    assert bench2["sample_count"] == 1200
    assert bench2["evaluation_metrics"]["roc_auc"] > 0.70
