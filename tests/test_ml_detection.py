"""Tasks 2.5/2.6 tests: ML detection with leakage-free splits.
Uses a small dataset built on the fly (fast) - not the saved models."""
import numpy as np
import pandas as pd
import pytest

from analysis.feature_extraction import FEATURE_COLUMNS, build_feature_dataset
from detection import ml_detection as ml


@pytest.fixture(scope="module")
def data():
    return build_feature_dataset(runs_per_class=3, duration_s=1200, path=None, seed=77)


@pytest.fixture(scope="module")
def split(data):
    return ml.split_by_run(data, seed=1)


@pytest.fixture(scope="module")
def models(split):
    train, _ = split
    return ml.train_isolation_forest(train, seed=0), ml.train_random_forest(train, seed=0)


def test_split_has_no_run_leakage(split):
    train, test = split
    assert set(train.run_id).isdisjoint(test.run_id)
    assert all(r.startswith("multi_") is False for r in train.run_id.unique())
    # every class appears in both sets
    cls = lambda d: {ml.run_class(r) for r in d.run_id.unique()} - {"multi"}
    assert cls(train) == cls(test)


def test_run_class_parsing():
    assert ml.run_class("bearing_degradation_03") == "bearing_degradation"
    assert ml.run_class("healthy_00") == "healthy"
    assert ml.run_class("multi_imbalance+cavitation") == "multi"


def test_isolation_forest_trained_on_healthy_only(models, split):
    iforest, _ = models
    train, _ = split
    assert iforest["n_train"] == (train.fault_label == "healthy").sum()
    scores = ml.anomaly_scores(iforest, train[train.fault_label == "healthy"])
    assert np.mean(scores > iforest["threshold"]) == pytest.approx(0.01, abs=0.011)


def test_isolation_forest_flags_severe_mechanical_faults(models, split):
    iforest, _ = models
    _, test = split
    ev = ml.evaluate_isolation_forest(iforest, test)
    assert ev["false_positive_rate"] < 0.08
    assert ev["roc_auc"] > 0.8
    severe = test[test.fault_label.isin(["bearing_degradation", "flow_restriction"]) & (test.fault_severity > 0.5)]
    assert np.mean(ml.anomaly_scores(iforest, severe) > iforest["threshold"]) > 0.8


def test_random_forest_metrics(models, split):
    _, rf = models
    _, test = split
    ev = ml.evaluate_random_forest(rf, test, min_severity=0.3)
    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        assert 0.0 <= ev[k] <= 1.0
    assert ev["accuracy"] > 0.85 and ev["f1_macro"] > 0.8
    assert ev["false_positive_rate"] < 0.05
    assert ev["confusion"].to_numpy().sum() == ev["windows"]
    assert set(ev["report"].index) >= {"healthy", "macro avg"}


def test_rf_training_excludes_low_severity_windows(models, split):
    _, rf = models
    train, _ = split
    expected = ((train.fault_label == "healthy") | (train.fault_severity >= ml.TRAIN_MIN_SEVERITY)).sum()
    assert rf["n_train"] == expected


def test_feature_importance_ranks_engineering_features(models, split):
    _, rf = models
    _, test = split
    imp = ml.feature_importance(rf, test, top=10)
    assert len(imp) == 10 and set(imp.feature) <= set(FEATURE_COLUMNS)
    assert imp.permutation.iloc[0] > 0


def test_save_load_and_live_predict(models, split, tmp_path):
    iforest, rf = models
    _, test = split
    p_if = ml.save_bundle(iforest, tmp_path / "if.joblib")
    p_rf = ml.save_bundle(rf, tmp_path / "rf.joblib", {"train_runs": ["x"]})
    assert ml.MLDetector.available(p_if, p_rf)
    det = ml.MLDetector(p_if, p_rf)
    assert "sklearn_version" in det.rf["meta"]
    row = test.iloc[-1][FEATURE_COLUMNS].to_dict()
    out = det.predict(row)
    assert len(out) == 1 and out.predicted_fault.iloc[0] in rf["classes"]
    probs = out[[c for c in out if c.startswith("p_")]].to_numpy()
    assert probs.sum() == pytest.approx(1.0)
    # batch prediction identical to direct model prediction
    batch = det.predict(test)
    np.testing.assert_array_equal(batch.predicted_fault.to_numpy(), rf["model"].predict(ml._X(test)))


def test_predict_tolerates_missing_features(models, tmp_path):
    iforest, rf = models
    det = ml.MLDetector(ml.save_bundle(iforest, tmp_path / "a.joblib"), ml.save_bundle(rf, tmp_path / "b.joblib"))
    out = det.predict({"speed_rpm_mean": 1450.0, "vibration_rms_mm_s_mean": 1.8})
    assert np.isfinite(out.anomaly_score.iloc[0])


def test_saved_production_models_load_if_present():
    if not ml.MLDetector.available():
        pytest.skip("production models not trained yet (python -m detection.ml_detection)")
    det = ml.MLDetector()
    assert det.rf["features"] == FEATURE_COLUMNS
    assert len(det.rf["train_runs"]) >= 16 and det.iforest["threshold"] > 0
    # trained runs come from the training dataset only (validation set uses separate seeds;
    # run names repeat across datasets, so leakage is controlled by seed, see training_run_specs)
    train_ids = set(pd.read_csv("data/processed/features.csv").run_id)
    assert set(det.rf["train_runs"]) <= train_ids
