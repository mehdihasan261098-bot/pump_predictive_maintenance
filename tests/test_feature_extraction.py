"""Task 2.3 tests: window feature extraction."""
import math

import numpy as np
import pandas as pd
import pytest

from analysis.feature_extraction import (FEATURE_COLUMNS, META_COLUMNS, _slope_per_min, extract_features,
                                         simulate_run, training_run_specs)
from simulation.fault_simulator import FaultScenario, FaultType, generate_fault_dataset
from simulation.sensor_simulator import OperatingProfile


def run(fault=None, sev=1.0, duration=600, seed=1):
    scen = [FaultScenario(fault, shape="step", max_severity=sev)] if fault else []
    df = generate_fault_dataset(scen, duration, seed=seed, out_dir=None)
    return extract_features(df, seed=seed)


@pytest.fixture(scope="module")
def healthy():
    return run()


def test_window_count_and_columns(healthy):
    assert len(healthy) == 10                                    # 600 s / 60 s
    assert list(healthy.columns[:4]) == META_COLUMNS[:4]
    assert all(c in healthy for c in FEATURE_COLUMNS)
    assert not healthy[FEATURE_COLUMNS].isna().any().any()


def test_no_ground_truth_leakage():
    leaky = [c for c in FEATURE_COLUMNS if c.startswith(("true_", "sev_", "fault_", "active_", "run_"))]
    assert leaky == []
    assert len(set(FEATURE_COLUMNS) & set(META_COLUMNS)) == 0


def test_slope_hand_calculation():
    t = np.arange(0, 60, 1.0)
    assert _slope_per_min(t, 50 + 0.05 * t) == pytest.approx(3.0)      # 0.05 degC/s = 3 degC/min
    y = 50 + 0.05 * t
    y[::5] = np.nan
    assert _slope_per_min(t, y) == pytest.approx(3.0)
    assert math.isnan(_slope_per_min(t[:2], y[:2]))


def test_healthy_feature_values(healthy):
    m = healthy[FEATURE_COLUMNS].mean()
    assert m.flow_m3h_mean == pytest.approx(30, abs=1.5)
    assert m.head_m_mean == pytest.approx(25, abs=1)
    assert m.vibration_rms_mm_s_mean == pytest.approx(1.8, abs=0.15)
    assert m.vib_dominant_order == pytest.approx(1.0, abs=0.05)
    assert abs(m.flow_deficit_m3h_mean) < 0.3
    assert m.bearing_temp_rise_c == pytest.approx(30, abs=1)


def test_non_running_windows_excluded():
    df = generate_fault_dataset([], 600, seed=2, out_dir=None,
                                profile=OperatingProfile(events=[(200.0, "stop", 0)]))
    feats = extract_features(df)
    assert feats.window_end_s.max() <= 200


@pytest.mark.parametrize("fault,feature,direction", [
    (FaultType.BEARING_DEGRADATION, "vib_env_bpfo_ratio", 5.0),
    (FaultType.IMBALANCE, "vib_amp_1x_mm_s", 3.0),
    (FaultType.MISALIGNMENT, "vib_ratio_2x_1x", 2.5),
    (FaultType.CAVITATION, "vib_acc_hf_band_rms", 10.0),
    (FaultType.FLOW_RESTRICTION, "bep_ratio_mean", -0.5),
    (FaultType.SEAL_LEAKAGE, "flow_deficit_m3h_mean", None),
    (FaultType.OVERHEATING, "bearing_temp_rise_c", 1.3),
])
def test_fault_signature_in_features(fault, feature, direction, healthy):
    f = run(fault, 1.0, seed=3)
    late, ref = f[feature].iloc[-3:].mean(), healthy[feature].mean()
    if fault is FaultType.SEAL_LEAKAGE:
        assert late == pytest.approx(3.0, abs=0.4)
    elif direction < 0:
        assert late < -direction * ref
    else:
        assert late > direction * ref


def test_labels_carried_from_ground_truth():
    df = generate_fault_dataset([FaultScenario(FaultType.IMBALANCE, onset_s=300, shape="step")], 600, seed=4,
                                out_dir=None)
    f = extract_features(df)
    assert (f[f.window_end_s < 300].fault_label == "healthy").all()
    assert (f[f.window_start_s > 300].fault_label == "imbalance").all()
    assert (f[f.window_start_s > 300].sev_imbalance == 1.0).all()


def test_training_specs_are_independent_runs():
    specs = training_run_specs(runs_per_class=2, duration_s=600)
    ids = [s["run_id"] for s in specs]
    assert len(ids) == len(set(ids)) == 8 * 2 + 3
    assert len({s["seed"] for s in specs}) == len(specs)
    df = simulate_run(specs[2])
    assert df.run_id.unique().tolist() == [specs[2]["run_id"]]


def test_deterministic():
    a, b = run(seed=7), run(seed=7)
    pd.testing.assert_frame_equal(a, b)
