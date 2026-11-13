"""Tasks 2.7/2.8 tests: health index, trend, simulated RUL, maintenance."""
import numpy as np
import pandas as pd
import pytest

from analysis.feature_extraction import extract_features
from database.database_manager import DatabaseManager
from detection.rule_based import RuleBasedDiagnoser
from predictive_maintenance.health_monitor import HealthMonitor, damage, run_monitor, rul_validation
from simulation.fault_simulator import FaultScenario, FaultType, generate_fault_dataset
from simulation.sensor_simulator import OperatingProfile

T0 = pd.Timestamp("2026-01-01 08:00")


def feats(scen=None, duration=900, seed=3, profile=None):
    return extract_features(generate_fault_dataset(scen or [], duration, seed=seed, out_dir=None, profile=profile))


@pytest.fixture(scope="module")
def degradation():
    f = feats([FaultScenario(FaultType.IMBALANCE, onset_s=300, ramp_s=2700)], duration=3600, seed=8)
    return run_monitor(f)


# ------------------------------------------------------------ damage mapping
def test_damage_anchors_rising_indicator():
    assert damage(2.0, 2.2, 4.5, 7.1) == 0.0
    assert damage(4.5, 2.2, 4.5, 7.1) == pytest.approx(0.45)
    assert damage(7.1, 2.2, 4.5, 7.1) == pytest.approx(0.8)
    assert damage(9.7, 2.2, 4.5, 7.1) == pytest.approx(1.0)
    assert damage(50.0, 2.2, 4.5, 7.1) == 1.0
    assert damage(3.35, 2.2, 4.5, 7.1) == pytest.approx(0.225)


def test_damage_anchors_falling_indicator_and_nan():
    assert damage(6.0, 5.0, 4.0, 1.5) == 0.0           # NPSH margin above good
    assert damage(4.0, 5.0, 4.0, 1.5) == pytest.approx(0.45)
    assert damage(1.5, 5.0, 4.0, 1.5) == pytest.approx(0.8)
    assert damage(float("nan"), 5.0, 4.0, 1.5) == 0.0


def test_condition_bands():
    m = HealthMonitor()
    assert [m.condition(v) for v in (100, 85, 84.9, 65, 64.9, 40, 39.9, 0)] == \
        ["Healthy", "Healthy", "Caution", "Caution", "Degraded", "Degraded", "Critical", "Critical"]


# ------------------------------------------------------------ health index
@pytest.mark.parametrize("profile", [None, OperatingProfile(events=[(0, "speed", 1300.0), (0, "valve", 0.75)])])
def test_healthy_pump_scores_high(profile):
    hist = run_monitor(feats(profile=profile, duration=1800))
    assert hist.health_index.min() >= 85
    assert (hist.condition == "Healthy").all() and (hist.urgency == "Routine").all()


def test_hi_decreases_with_severity(degradation):
    by = degradation.groupby(pd.cut(degradation.fault_severity, [-0.01, 0.05, 0.4, 0.7, 1.0]),
                             observed=True).health_index.mean().to_numpy()
    assert all(np.diff(by) < 0)
    assert degradation.health_index.iloc[-1] < 40
    assert degradation.condition.iloc[-1] == "Critical" and degradation.urgency.iloc[-1] == "Immediate"


def test_component_breakdown_explains_fault():
    m = HealthMonitor()
    f = feats([FaultScenario(FaultType.SEAL_LEAKAGE, shape="step")]).iloc[-1].to_dict()
    hi, comps, details = m.health_index(f)
    assert min(comps, key=comps.get) == "hydraulic"
    assert details["flow_deficit_m3h"]["damage"] > 0.5
    assert comps["vibration"] == 100.0


def test_each_fault_lowers_health_index():
    m = HealthMonitor()
    for fault in FaultType:
        dur = 2400 if fault is FaultType.OVERHEATING else 600
        f = feats([FaultScenario(fault, shape="step")], duration=dur).iloc[-1].to_dict()
        assert m.health_index(f)[0] < 65, fault


def test_ewma_smoothing():
    m = HealthMonitor()
    good = feats().iloc[-1].to_dict()
    bad = feats([FaultScenario(FaultType.IMBALANCE, shape="step")]).iloc[-1].to_dict()
    m.update(T0, good)
    r = m.update(T0 + pd.to_timedelta(60, unit="s"), bad)
    a = m.h.ewma_alpha
    assert r.health_index == pytest.approx(a * r.health_index_raw + (1 - a) * 100.0, abs=0.01)


# ------------------------------------------------------------ trend and RUL
def test_rul_linear_trend_hand_calculation():
    m = HealthMonitor()
    t = np.arange(-14, 1) / 60.0                     # 15 windows, 1 min apart, in hours
    hi = 80 - 60.0 * t                               # past values higher: falls 60 points/h, now 80
    slope, rul, model = m._trend_and_rul(t, hi)
    assert slope == pytest.approx(-60.0, rel=1e-6)
    assert model == "linear"
    assert rul == pytest.approx((80 - 40) / 60.0, rel=1e-3)   # 0.667 h


def test_rul_not_estimated_without_trend():
    m = HealthMonitor()
    t = np.arange(-14, 1) / 60.0
    slope, rul, model = m._trend_and_rul(t, np.full(15, 95.0))
    assert rul is None and "no significant" in model
    slope, rul, model = m._trend_and_rul(t[:3], np.array([90, 89, 88.0]))
    assert rul is None and model == "insufficient history"


def test_rul_validation_converges(degradation):
    val = rul_validation(degradation)
    assert len(val) > 5
    late = val[val.actual_rul_h <= val.actual_rul_h.max() / 2]
    assert late.error_h.abs().mean() < 0.25        # < 15 min in the final half (simulated)


# ------------------------------------------------------------ urgency / maintenance
def test_urgency_and_schedule_rules():
    m = HealthMonitor()
    dx = RuleBasedDiagnoser().diagnose({"speed_rpm_mean": 1450.0})
    assert m._urgency(95, dx, None) == "Routine"
    assert m._urgency(80, dx, None) == "Planned"
    assert m._urgency(60, dx, None) == "Urgent"
    assert m._urgency(30, dx, None) == "Immediate"
    assert m._urgency(95, dx, 10.0) == "Urgent"
    assert m._schedule(T0, "Planned", None) == T0 + pd.to_timedelta(7, unit="D")
    assert m._schedule(T0, "Planned", 20.0) == T0 + pd.to_timedelta(10, unit="h")


def test_recommendations_from_diagnosis():
    m = HealthMonitor()
    r = m.update(T0, feats([FaultScenario(FaultType.CAVITATION, shape="step")]).iloc[-1].to_dict())
    assert any("suction" in s.lower() for s in r.recommendations)
    assert r.primary_fault == "cavitation"


def test_persist_health_and_deduplicated_work_orders():
    m = HealthMonitor()
    rows = feats([FaultScenario(FaultType.MISALIGNMENT, onset_s=120, shape="step")], duration=900).to_dict("records")
    with DatabaseManager(":memory:") as db:
        ids = [m.persist(m.update(r["timestamp"], r), db) for r in rows]
        assert db.counts()["health_history"] == len(rows)
        orders = db.get_maintenance()
        assert sum(i is not None for i in ids) == len(orders)
        assert orders.component.is_unique                  # one open order per component
        assert "Coupling / shaft alignment" in set(orders.component)
        assert orders.notes.str.contains("simulated estimate").all()
