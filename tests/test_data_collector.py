"""Task 1.6 tests: data acquisition, QC and rerun safety."""
import math

import pandas as pd
import pytest

from acquisition.data_collector import DataCollector, SimulatedSource, clean_dataset
from simulation.fault_simulator import FaultScenario, FaultSimulator, FaultType
from simulation.sensor_simulator import SENSOR_NAMES, SensorFailure, SensorSimulator


def collector(failures=None, dropout=0.0, faults=None, **kw):
    sim = SensorSimulator(seed=4, warm_start=True, dropout_probability=dropout, failures=failures)
    return DataCollector(SimulatedSource(sim, faults), **kw)


def rec(t, **values):
    base = {"timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(t, unit="s"), "time_s": float(t),
            "bearing_temp_c": 55.0, "casing_temp_c": 35.0, "vibration_rms_mm_s": 1.8,
            "vibration_accel_m_s2": 0.4, "suction_pressure_bar": 1.2, "discharge_pressure_bar": 3.65,
            "flow_m3h": 30.0, "motor_current_a": 7.26, "motor_power_kw": 3.32, "speed_rpm": 1450.0}
    base.update(values)
    return base


def test_acquire_produces_validated_records_with_kpis():
    c = collector()
    df = c.acquire(60)
    assert len(df) == 60
    for col in [*SENSOR_NAMES, "state", "fault_label", "qc_flags", "head_m", "sec_kwh_m3"]:
        assert col in df
    assert (df.qc_issue_count == 0).all()
    assert df.head_m.mean() == pytest.approx(25.0, abs=0.3)


def test_rerun_safety_acquire_to_time():
    c = collector()
    c.acquire_to_time(120)
    c.acquire_to_time(120)
    c.acquire_to_time(119.5)
    assert len(c.data) == 120
    c.acquire_to_time(130)
    assert len(c.data) == 130
    assert c.data.timestamp.is_unique


def test_duplicate_timestamps_rejected():
    c = collector()
    c.ingest([rec(1), rec(2), rec(2), rec(1)])
    assert len(c.data) == 2 and c.stats["duplicates"] == 2


def test_out_of_range_value_flagged_and_filled():
    c = collector()
    c.ingest([rec(1), rec(2, discharge_pressure_bar=99.0)])
    row = c.data.iloc[-1]
    assert row.discharge_pressure_bar == 3.65
    assert "discharge_pressure_bar:range+filled" in row.qc_flags


def test_short_gap_filled_long_gap_left_missing():
    c = collector(max_gap_fill=2)
    c.ingest([rec(0)] + [rec(t, flow_m3h=math.nan) for t in range(1, 6)] + [rec(6)])
    flow = c.data.flow_m3h.tolist()
    assert flow[1:3] == [30.0, 30.0]                     # filled
    assert all(math.isnan(v) for v in flow[3:6])         # gap too long
    assert flow[6] == 30.0
    assert c.stats["missing"] == 5 and c.stats["filled"] == 2


def test_temperature_spike_rejected_but_persistent_step_accepted():
    c = collector(max_gap_fill=3)
    c.ingest([rec(0), rec(1, bearing_temp_c=80.0), rec(2)])
    assert c.data.bearing_temp_c.tolist() == [55.0, 55.0, 55.0]
    assert "bearing_temp_c:rate+filled" in c.data.qc_flags.iloc[1]
    c.ingest([rec(t, bearing_temp_c=85.0) for t in range(3, 10)])
    temps = c.data.bearing_temp_c.tolist()
    assert temps[3:6] == [55.0, 55.0, 55.0]              # first 3 treated as suspect
    assert temps[-1] == 85.0                             # persistent level accepted
    assert any("rate_step" in f for f in c.data.qc_flags)


def test_sensor_failures_are_caught():
    fails = [SensorFailure("flow_m3h", "dead", 100, 200),
             SensorFailure("discharge_pressure_bar", "offset", 300, magnitude=50.0)]
    c = collector(failures=fails)
    df = c.acquire(400)
    dead = df[(df.time_s > 100) & (df.time_s <= 200)]
    assert dead.flow_m3h.isna().sum() == len(dead) - 3   # 3 filled
    assert df[df.time_s > 305].discharge_pressure_bar.isna().all()


def test_fault_labels_recorded():
    faults = FaultSimulator([FaultScenario(FaultType.CAVITATION, onset_s=50, shape="step")])
    df = collector(faults=faults).acquire(100)
    assert (df[df.time_s <= 50].fault_label == "healthy").all()
    assert (df[df.time_s > 51].fault_label == "cavitation").all()


def test_sinks_receive_batches_and_csv_export(tmp_path):
    c = collector()
    seen = []
    c.sinks.append(lambda b: seen.append(len(b)))
    c.acquire(10)
    c.acquire(5)
    assert seen == [10, 5]
    path = c.export_csv(tmp_path / "out.csv")
    back = pd.read_csv(path)
    assert len(back) == 15 and "qc_flags" in back


def test_buffer_trimming():
    c = collector(buffer_seconds=100)
    for _ in range(30):
        c.acquire(10)
    assert c.data.time_s.min() >= 300 - 100 - 10
    assert c.data.time_s.max() == 300


def test_clean_dataset_offline():
    raw = pd.DataFrame([rec(3), rec(1), rec(2), rec(2), rec(4, flow_m3h=-5.0)])
    cleaned, stats = clean_dataset(raw)
    assert cleaned.time_s.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert stats["duplicates"] == 1 and stats["range"] == 1
    assert cleaned.flow_m3h.iloc[-1] == 30.0
