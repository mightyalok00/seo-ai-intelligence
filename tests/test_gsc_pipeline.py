"""
Unit and Integration Tests for Google Search Console (GSC) Data Pipeline.

Author: Alok Agarwal (mightyalok00)
License: MIT
"""

from pathlib import Path

from src.models.gsc_pipeline import GSCDataPipeline


def test_gsc_data_pipeline_generation(tmp_path: Path):
    pipeline = GSCDataPipeline(output_dir=tmp_path)
    csv_file = tmp_path / "test_gsc.csv"
    df = pipeline.create_sample_gsc_dataset(csv_file)

    assert df is not None
    assert len(df) == 1200
    assert "impressions" in df.columns
    assert "clicks" in df.columns
    assert "is_top_10" in df.columns
    assert csv_file.exists()

def test_gsc_training_pipeline(tmp_path: Path):
    pipeline = GSCDataPipeline(output_dir=tmp_path)
    metrics = pipeline.train_on_gsc_data()

    assert "roc_auc" in metrics
    assert "f1_score" in metrics
    assert metrics["roc_auc"] > 0.65
    assert metrics["sample_count"] > 0
