"""Task 1.3 tests: virtual sensor system."""
import math

import numpy as np
import pandas as pd
import pytest

from simulation.pump_model import FaultEffects
from simulation.sensor_simulator import (SENSOR_NAMES, OperatingProfile, SensorFailure,
                                         SensorSimulator, generate_healthy_baseline)


@pytest.fixture(scope="module")
def baseline():
    return generate_healthy_baseline(duration_s=900, seed=1, path=None)


def steady(seed=0, n=300, **kw):
    sim = SensorSimulator(seed=seed, warm_start=True, dropout_probability=0.0, **kw)
    return sim.run(n, OperatingProfile.continuous(0.0), include_truth=True)


def test_record_structure_and_timestamps(baseline):
    for col in ["timestamp", "time_s", "state", "valve_opening", *SENSOR_NAMES]:
        assert col in baseline
    ts = pd.to_datetime(baseline["timestamp"])
    assert ts.is_monotonic_increasing and ts.is_unique
    assert (ts.diff().dropna().dt.total_seconds() == 1.0).all()


def test_profile_covers_all_operating_phases(baseline):
    assert {"Stopped", "Starting", "Running", "Shutdown"} <= set(baseline["state"])


def test_noise_matches_configuration():
    df = steady(n=600)
    sim_cfg = SensorSimulator().cfg.sensors.noise_std
    for name in ("flow_m3h", "discharge_pressure_bar", "motor_current_a", "speed_rpm"):
        resid = df[name] - df[f"true_{name}"]
        assert abs(resid.mean()) < 4 * sim_cfg[name] / math.sqrt(len(df))
        assert resid.std() == pytest.approx(sim_cfg[name], rel=0.15)


def test_reproducible_with_seed():
    a, b = steady(seed=7, n=50), steady(seed=7, n=50)
    pd.testing.assert_frame_equal(a, b)


def test_sensors_are_physically_consistent_under_throttling():
    sim = SensorSimulator(seed=0, warm_start=True, dropout_probability=0.0)
    prof = OperatingProfile(events=[(100.0, "valve", 0.5)])
    df = sim.run(200, prof)
    before, after = df[df.time_s < 100], df[df.time_s > 110]
    assert after.flow_m3h.mean() < before.flow_m3h.mean() - 5
    assert after.discharge_pressure_bar.mean() > before.discharge_pressure_bar.mean()
    assert after.motor_current_a.mean() < before.motor_current_a.mean()
    # head from measured pressures must equal pump head (H = dP / rho g)
    dp = after.discharge_pressure_bar - after.suction_pressure_bar
    assert (dp.mean() * 1e5 / 9810) == pytest.approx(29.25, abs=0.1)


def test_stopped_pump_reads_zero_flow_and_speed(baseline):
    stopped = baseline[(baseline.state == "Stopped") & (baseline.time_s < 10)]
    assert (stopped.flow_m3h.fillna(0) == 0).all()
    assert (stopped.speed_rpm.fillna(0) == 0).all()
    assert (baseline[SENSOR_NAMES[2:]].fillna(0) >= -1.0).all().all()


def test_dropout_rate():
    sim = SensorSimulator(seed=3, warm_start=True, dropout_probability=0.05)
    df = sim.run(500, OperatingProfile.continuous(0.0))
    frac = df[SENSOR_NAMES].isna().to_numpy().mean()
    assert 0.035 < frac < 0.065


@pytest.mark.parametrize("mode", ["dead", "stuck", "drift", "offset", "spike"])
def test_sensor_failure_modes(mode):
    f = SensorFailure("bearing_temp_c", mode, start_s=50, end_s=150,
                      magnitude=20.0, rate=0.1, probability=0.5)
    df = steady(n=200, failures=[f])
    win = df[(df.time_s > 50) & (df.time_s < 150)]
    err = win.bearing_temp_c - win.true_bearing_temp_c
    if mode == "dead":
        assert win.bearing_temp_c.isna().all()
    elif mode == "stuck":
        assert win.bearing_temp_c.nunique() == 1
    elif mode == "drift":
        assert err.iloc[-1] == pytest.approx(0.1 * (win.time_s.iloc[-1] - 50), abs=1.0)
    elif mode == "offset":
        assert err.mean() == pytest.approx(20.0, abs=0.2)
    elif mode == "spike":
        assert (err.abs() > 15).mean() == pytest.approx(0.5, abs=0.2)
    after = df[df.time_s >= 151]
    assert (after.bearing_temp_c - after.true_bearing_temp_c).abs().max() < 1.5


def test_invalid_failure_definition_rejected():
    with pytest.raises(ValueError):
        SensorFailure("no_such_sensor", "dead", 0)
    with pytest.raises(ValueError):
        SensorFailure("flow_m3h", "melted", 0)


def test_pump_fault_effects_reach_sensors():
    sim = SensorSimulator(seed=0, warm_start=True, dropout_probability=0.0)
    df = sim.run(120, OperatingProfile.continuous(0.0),
                 effects_fn=lambda t: FaultEffects(vibration_add_mm_s=5.0) if t >= 60 else None)
    assert df[df.time_s > 61].vibration_rms_mm_s.mean() > 2.5 * df[df.time_s < 60].vibration_rms_mm_s.mean()


def test_emergency_stop_in_profile():
    sim = SensorSimulator(seed=0, warm_start=True, dropout_probability=0.0)
    df = sim.run(60, OperatingProfile(events=[(20.0, "estop", 0)]))
    assert df.state.iloc[-1] == "Emergency stop"
    assert df.flow_m3h.iloc[-1] == 0.0 and np.isclose(df.motor_current_a.iloc[-1], 0, atol=0.2)
