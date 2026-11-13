"""Task 3.8 tests: alarm lifecycle, hysteresis, suppression, escalation."""
import math

import pandas as pd
import pytest

from alarms.alarm_manager import AlarmManager
from database.database_manager import DatabaseManager
from detection.rule_based import RuleBasedDiagnoser

T0 = pd.Timestamp("2026-01-01 08:00")


def t(s):
    return T0 + pd.to_timedelta(s, unit="s")


NORMAL_REC = {"bearing_temp_c": 55.0, "casing_temp_c": 35.0, "vibration_rms_mm_s": 1.8, "vibration_accel_m_s2": 0.3,
              "suction_pressure_bar": 1.2, "discharge_pressure_bar": 3.65, "flow_m3h": 30.0, "motor_current_a": 7.3,
              "motor_power_kw": 3.3, "speed_rpm": 1450.0}


def rec(**kw):
    return {**NORMAL_REC, **kw}


@pytest.fixture
def am():
    db = DatabaseManager(":memory:")
    yield AlarmManager(db)
    db.close()


def keys(am):
    a = am.active()
    return {} if a.empty else dict(zip(a.alarm_key, a.status))


def test_normal_operation_no_alarms(am):
    assert am.evaluate(t(0), rec()) == []
    assert am.active().empty


def test_raise_ack_clear_reset_lifecycle(am):
    ev = am.evaluate(t(0), rec(bearing_temp_c=72.0))
    assert ev[0]["event"] == "raised" and ev[0]["priority"] == "Warning"
    aid = ev[0]["id"]
    assert keys(am) == {"bearing_temp_hi": "active"}
    assert am.reset(aid, t(5)) is False                        # cannot reset while condition present
    assert am.acknowledge(aid, t(10), "op1")
    assert keys(am)["bearing_temp_hi"] == "acknowledged"
    am.evaluate(t(20), rec(bearing_temp_c=60.0))
    assert keys(am)["bearing_temp_hi"] == "cleared"            # latched until reset
    assert am.reset(aid, t(30))
    assert am.active().empty
    h = am.history().iloc[0]
    assert h.status == "reset" and h.acknowledged_by == "op1"
    assert h.raised_at == t(0) and h.cleared_at == t(20) and h.reset_at == t(30)


def test_duplicate_suppression(am):
    for s in range(10):
        am.evaluate(t(s), rec(vibration_rms_mm_s=5.0))
    assert len(am.history()) == 1


def test_hysteresis_prevents_chatter(am):
    am.evaluate(t(0), rec(bearing_temp_c=70.5))                # raise above 70
    am.evaluate(t(1), rec(bearing_temp_c=69.0))                # inside 2 degC deadband -> still on
    assert keys(am)["bearing_temp_hi"] == "active"
    am.evaluate(t(2), rec(bearing_temp_c=67.9))                # below 70 - 2 -> clears
    assert keys(am)["bearing_temp_hi"] == "cleared"


def test_reactivation_counts_occurrences(am):
    am.evaluate(t(0), rec(flow_m3h=10.0))
    am.evaluate(t(1), rec(flow_m3h=30.0))
    am.evaluate(t(2), rec(flow_m3h=10.0))
    h = am.history()
    assert len(h) == 1 and h.occurrences.iloc[0] == 2 and h.status.iloc[0] == "active"


def test_priority_levels_hi_and_hihi(am):
    am.evaluate(t(0), rec(vibration_rms_mm_s=8.0))
    a = am.active()
    assert set(a.alarm_key) == {"vibration_hi", "vibration_hihi"}
    assert a.iloc[0].priority == "Critical"                     # sorted most severe first
    assert am.counts()["Critical"] == 1 and am.counts()["Warning"] == 1


def test_running_only_alarms_suppressed_when_stopped(am):
    am.evaluate(t(0), rec(flow_m3h=0.0, suction_pressure_bar=1.3), running=False)
    assert "flow_lo" not in keys(am)


def test_missing_value_keeps_state_and_raises_sensor_info(am):
    am.evaluate(t(0), rec(bearing_temp_c=75.0))
    am.evaluate(t(1), rec(bearing_temp_c=math.nan))
    k = keys(am)
    assert k["bearing_temp_hi"] == "active" and k["sensor_bearing_temp_c"] == "active"
    am.evaluate(t(2), rec(bearing_temp_c=75.0))
    assert "sensor_bearing_temp_c" not in keys(am)             # Info alarms auto-reset


def test_emergency_stop_alarm(am):
    ev = am.evaluate(t(0), state="Emergency stop")
    assert ev[0]["priority"] == "Critical" and "not safety-rated" in ev[0]["message"]
    am.evaluate(t(5), state="Stopped")
    assert keys(am)["emergency_stop"] == "cleared"


def test_diagnosis_alarm_escalation():
    db = DatabaseManager(":memory:")
    am = AlarmManager(db)
    dx = RuleBasedDiagnoser()
    base = {"speed_rpm_mean": 1450.0, "motor_power_kw_mean": 3.3, "flow_deficit_m3h_mean": 1.0,
            "npsh_margin_mean": 7.0}
    am.evaluate(t(0), diagnosis=dx.diagnose(base))
    assert am.active().set_index("alarm_key").loc["diag_seal_leakage", "priority"] == "Warning"
    am.evaluate(t(60), diagnosis=dx.diagnose({**base, "flow_deficit_m3h_mean": 2.5}))
    a = am.active().set_index("alarm_key").loc["diag_seal_leakage"]
    assert a.priority == "Critical" and a.status == "active"
    assert len(am.history()) == 1                              # escalated, not duplicated
    db.close()


def test_health_and_ml_alarms(am):
    class H:
        health_index = 50.0
    am.evaluate(t(0), health=H(), ml={"is_anomaly": True, "anomaly_score": 0.7})
    k = keys(am)
    assert {"health_caution", "health_degraded", "ml_anomaly"} <= set(k) and "health_critical" not in k


def test_state_survives_restart(tmp_path):
    path = tmp_path / "a.db"
    db = DatabaseManager(path)
    AlarmManager(db).evaluate(t(0), rec(vibration_rms_mm_s=5.0))
    db.close()
    db2 = DatabaseManager(path)
    am2 = AlarmManager(db2)
    am2.evaluate(t(10), rec(vibration_rms_mm_s=5.0))
    assert len(am2.history()) == 1                             # no duplicate after restart
    db2.close()


def test_acknowledge_and_reset_all(am):
    am.evaluate(t(0), rec(vibration_rms_mm_s=5.0, flow_m3h=10.0))
    assert am.acknowledge_all(t(1)) == 2
    am.evaluate(t(2), rec())
    assert am.reset_all_cleared(t(3)) == 2 and am.active().empty
