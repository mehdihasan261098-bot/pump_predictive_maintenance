"""Task 3.1/4.1 tests: dashboard engine = full pipeline integration, rerun safety."""
import pandas as pd
import pytest

from dashboard.engine import SimulationEngine
from simulation.fault_simulator import FaultType


@pytest.fixture
def eng(tmp_path):
    e = SimulationEngine(db_path=str(tmp_path / "e.db"), start_time="2026-03-01 10:00:00", speedup=60.0)
    yield e
    e.close()


def test_rerun_safety_wall_clock_ticks(eng):
    assert eng.tick(1000.0) == 0                 # first tick only sets the reference
    assert eng.tick(1000.0) == 0                 # rerun at the same instant -> nothing
    n = eng.tick(1001.0)                         # 1 s wall x 60 = 60 samples
    assert n == 60
    assert eng.tick(1001.0) == 0
    assert eng.db.counts()["measurements"] == 60
    assert eng.collector.data.timestamp.is_unique


def test_full_pipeline_healthy(eng):
    eng.advance(600)
    c = eng.db.counts()
    assert c["measurements"] == 600 and c["health_history"] == 10
    assert eng.latest_health.health_index >= 85 and eng.latest_diagnosis.primary_fault == "healthy"
    assert eng.alarms.active().empty


def test_full_pipeline_fault_raises_alarms_and_diagnosis(eng):
    eng.advance(120)
    eng.set_fault(FaultType.IMBALANCE, 1.0)
    eng.advance(300)
    assert eng.latest_diagnosis.primary_fault == "imbalance"
    keys = set(eng.alarms.active().alarm_key)
    assert {"vibration_hi", "vibration_hihi", "diag_imbalance"} <= keys
    rules = eng.db.get_diagnoses(method="rule")
    assert "imbalance" in set(rules.fault)
    assert eng.latest_health.health_index < 65
    if eng.ml is not None:
        assert eng.latest_ml["predicted_fault"] == "imbalance"
        assert not eng.db.get_diagnoses(method="ml_classifier").empty


def test_operator_controls_and_estop(eng):
    eng.advance(60)
    eng.set_valve(0.6)
    eng.advance(60)
    assert eng.latest()["flow_m3h"] < 27
    eng.emergency_stop()
    eng.advance(30)
    assert eng.state == "Emergency stop" and eng.latest()["speed_rpm"] == 0
    assert "emergency_stop" in set(eng.alarms.active().alarm_key)
    eng.start()
    eng.advance(5)
    assert eng.state == "Emergency stop"         # latched
    assert eng.reset_emergency_stop() and eng.state == "Stopped"
    eng.start()
    eng.advance(10)
    assert eng.state == "Running"


def test_gradual_fault_ramp(eng):
    eng.ramp_fault("seal_leakage", 1.0, 600)
    eng.advance(300)
    assert eng.fault_severities()["seal_leakage"] == pytest.approx(0.5, abs=0.01)


def test_vibration_snapshot_cached_and_reference(eng):
    eng.advance(10)
    a, b = eng.vibration_snapshot(), eng.vibration_snapshot()
    assert a is b
    ref = eng.vibration_snapshot(healthy_reference=True)
    assert len(ref.velocity_mm_s) == int(eng.cfg.vibration.sampling_rate_hz)


def test_restart_continues_after_existing_data(tmp_path):
    path = str(tmp_path / "r.db")
    e1 = SimulationEngine(db_path=path, start_time="2026-03-01 10:00:00")
    e1.advance(120)
    last = e1.db.get_measurements(limit=1, latest_first=True).timestamp.iloc[0]
    e1.close()
    e2 = SimulationEngine(db_path=path, start_time="2026-03-01 10:00:00")
    e2.advance(60)
    assert e2.db.counts()["measurements"] == 180          # no overwrite / no duplicate timestamps
    assert e2.now > last
    e2.close()
