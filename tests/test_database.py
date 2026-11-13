"""Task 1.7 tests: SQLite persistence."""
import json
import math
import sqlite3

import pandas as pd
import pytest

from acquisition.data_collector import DataCollector, SimulatedSource
from database.database_manager import DatabaseManager
from simulation.fault_simulator import FaultScenario, FaultSimulator, FaultType
from simulation.sensor_simulator import OperatingProfile, SensorSimulator

T0 = pd.Timestamp("2026-01-01 08:00:00")


@pytest.fixture
def db():
    d = DatabaseManager(":memory:")
    yield d
    d.close()


@pytest.fixture(scope="module")
def acquired():
    sim = SensorSimulator(seed=2, dropout_probability=0.0)
    faults = FaultSimulator([FaultScenario(FaultType.IMBALANCE, onset_s=200, shape="step")])
    col = DataCollector(SimulatedSource(sim, faults, OperatingProfile.standard(600)))
    col.acquire(600)
    return col.data


def test_schema_created(tmp_path):
    path = tmp_path / "x" / "pump.db"
    with DatabaseManager(path) as d:
        assert set(d.counts()) == {"measurements", "operating_status", "diagnoses", "health_history",
                                   "alarms", "maintenance_records"}
    assert path.exists()


def test_measurement_roundtrip(db, acquired):
    assert db.insert_measurements(acquired) == 600
    back = db.get_measurements()
    assert len(back) == 600
    assert back.timestamp.iloc[0] == pd.Timestamp(acquired.timestamp.iloc[0])
    pd.testing.assert_series_equal(back.flow_m3h.reset_index(drop=True),
                                   acquired.flow_m3h.reset_index(drop=True), check_names=False)
    assert set(back.fault_label) == {"healthy", "imbalance"}
    assert back.head_m.notna().any()


def test_duplicate_measurements_ignored(db, acquired):
    db.insert_measurements(acquired.iloc[:100])
    assert db.insert_measurements(acquired.iloc[50:150]) == 50
    assert db.counts()["measurements"] == 150


def test_nan_stored_as_null(db):
    df = pd.DataFrame([{"timestamp": T0, "time_s": 1.0, "flow_m3h": math.nan, "speed_rpm": 1450.0}])
    db.insert_measurements(df)
    assert db.query("SELECT flow_m3h FROM measurements").flow_m3h.isna().all()


def test_date_filter_and_columns(db, acquired):
    db.insert_measurements(acquired)
    w = db.get_measurements(start=T0 + pd.to_timedelta(100, unit="s"), end=T0 + pd.to_timedelta(199, unit="s"),
                            columns=["flow_m3h"])
    assert len(w) == 100 and list(w.columns) == ["timestamp", "flow_m3h"]
    latest = db.get_measurements(limit=1, latest_first=True)
    assert latest.timestamp.iloc[0] == acquired.timestamp.max()
    with pytest.raises(ValueError):
        db.get_measurements(columns=["flow; DROP TABLE measurements"])


def test_collector_sink_and_status_transitions(db):
    sim = SensorSimulator(seed=3, dropout_probability=0.0)
    col = DataCollector(SimulatedSource(sim, profile=OperatingProfile.standard(300)))
    col.sinks.append(db.as_sink())
    col.acquire_to_time(150)
    col.acquire_to_time(150)              # rerun
    col.acquire_to_time(300)
    assert db.counts()["measurements"] == 300
    states = db.get_status_history().state.tolist()
    assert states == ["Stopped", "Starting", "Running", "Shutdown", "Stopped"]
    db.as_sink()(col.data)                # replaying history adds nothing
    assert db.counts()["operating_status"] == 5


def test_diagnosis_and_health(db):
    db.insert_diagnosis(T0, "rule", "cavitation", "Critical", 0.9, {"npsh_margin": 0.8}, "NPSH low", "Check suction")
    db.insert_diagnosis(T0, "rule", "cavitation", "Critical", 0.9)          # duplicate ignored
    db.insert_diagnosis(T0, "ml_classifier", "cavitation", "Warning", 0.7)
    assert len(db.get_diagnoses()) == 2 and len(db.get_diagnoses(method="rule")) == 1
    assert json.loads(db.get_diagnoses(method="rule").evidence.iloc[0]) == {"npsh_margin": 0.8}
    db.insert_health(T0, 72.0, "Caution", 120.0, {"vibration": 60})
    assert db.insert_health(T0, 50.0, "Degraded") is None                   # same timestamp ignored
    h = db.get_health_history()
    assert h.health_index.iloc[0] == 72.0 and h.rul_hours.iloc[0] == 120.0


def test_check_constraints(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_health(T0, 150.0, "Healthy")
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_alarm("x", "Urgent!!", "bad priority", T0)
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_diagnosis(T0, "guess", "x", "Normal")


def test_alarm_lifecycle(db):
    aid = db.insert_alarm("vib_high", "Critical", "Vibration high", T0, 9.5, 7.1)
    assert db.get_alarms(statuses=("active",)).id.tolist() == [aid]
    assert db.update_alarm(aid, status="acknowledged", acknowledged_at=T0 + pd.to_timedelta(30, unit="s"),
                           acknowledged_by="op1")
    db.update_alarm(aid, status="cleared", cleared_at=T0 + pd.to_timedelta(60, unit="s"))
    a = db.get_alarms().iloc[0]
    assert a.status == "cleared" and a.acknowledged_by == "op1"
    assert a.acknowledged_at == T0 + pd.to_timedelta(30, unit="s")
    assert db.get_alarms(statuses=("active", "acknowledged")).empty
    with pytest.raises(ValueError):
        db.update_alarm(aid, **{"status = 'x'; --": 1})


def test_maintenance_records(db):
    mid = db.add_maintenance("Mechanical seal", "Replace seal", "Urgent", created_at=T0)
    db.update_maintenance(mid, status="completed", completed_at=T0 + pd.to_timedelta(1, unit="D"),
                          technician="tech A", cost=850.0)
    assert db.get_maintenance(status="open").empty
    rec = db.get_maintenance(status="completed").iloc[0]
    assert rec.cost == 850.0 and rec.completed_at == T0 + pd.to_timedelta(1, unit="D")


def test_export_csv(db, acquired, tmp_path):
    db.insert_measurements(acquired)
    p = db.export_csv("measurements", tmp_path / "m.csv", start=T0 + pd.to_timedelta(500, unit="s"))
    assert len(pd.read_csv(p)) == 101
    with pytest.raises(ValueError):
        db.export_csv("sqlite_master", tmp_path / "bad.csv")
