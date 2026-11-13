"""Task 1.2 tests: pump/motor model against independent hand calculations."""
import math

import pytest

from analysis import hydraulic_calculations as hc
from simulation.pump_model import FaultEffects, PumpModel, PumpState


@pytest.fixture
def warm():
    return PumpModel(warm_start=True)


# ---------------------------------------------------------------- hydraulics
def test_unit_conversions():
    assert hc.m3h_to_m3s(36.0) == pytest.approx(0.01)
    assert hc.m3s_to_m3h(0.01) == pytest.approx(36.0)
    assert hc.bar_to_pa(2.5) == pytest.approx(250_000)
    assert hc.pa_to_bar(101_300) == pytest.approx(1.013)


def test_hydraulic_power_hand_calc():
    # 1000 * 9.81 * (30/3600) * 25 / 1000 = 2.04375 kW
    assert hc.hydraulic_power_kw(30, 25) == pytest.approx(2.04375)


def test_head_pressure_roundtrip():
    # dP = 2.4525 bar -> H = 245250 / 9810 = 25 m
    assert hc.head_from_pressure_m(2.4525) == pytest.approx(25.0)
    assert hc.pressure_from_head_bar(25.0) == pytest.approx(2.4525)
    assert hc.differential_pressure_bar(3.6525, 1.2) == pytest.approx(2.4525)


def test_efficiency_and_sec():
    assert hc.efficiency_percent(2.04375, 2.919643) == pytest.approx(70.0, rel=1e-5)
    assert hc.efficiency_percent(1.0, 0.0) == 0.0
    assert hc.specific_energy_kwh_per_m3(3.318, 30) == pytest.approx(0.1106, rel=1e-3)
    assert math.isnan(hc.specific_energy_kwh_per_m3(3.0, 0.0))


def test_affinity_laws():
    assert hc.affinity_flow(30, 1450, 1160) == pytest.approx(24.0)
    assert hc.affinity_head(25, 1450, 1160) == pytest.approx(16.0)
    assert hc.affinity_power(2.92, 1450, 1160) == pytest.approx(2.92 * 0.512)


def test_npsh_available_hand_calc():
    # (1.2 + 1.013 - 0.032) bar = 218100 Pa / 9810 = 22.232 m
    assert hc.npsh_available_m(1.2, 1.013, 0.032) == pytest.approx(22.2324, rel=1e-4)


# ---------------------------------------------------------------- pump model
def test_rated_operating_point(warm):
    s = warm.step(1.0)
    assert s["flow_m3h"] == pytest.approx(30.0, rel=1e-6)
    assert s["head_m"] == pytest.approx(25.0, rel=1e-6)
    assert s["pump_efficiency"] == pytest.approx(0.70, rel=1e-6)
    assert s["shaft_power_kw"] == pytest.approx(2.04375 / 0.7, rel=1e-6)
    assert s["motor_power_kw"] == pytest.approx(2.04375 / 0.7 / 0.88, rel=1e-6)
    assert s["discharge_pressure_bar"] == pytest.approx(1.2 + 2.4525, rel=1e-6)
    assert s["bearing_temp_c"] == pytest.approx(55.0, abs=0.01)


def test_motor_current_hand_calc(warm):
    # P_in = 3.3180 kW; I_act = 3318/(sqrt3*400) = 4.789 A
    # I_rated = 8522.7/(sqrt3*400*0.85) = 14.473 A, I_r,rated = 7.624 A, I_mag = 5.066 A
    # load = 0.3893 -> I_r = 5.066 + 2.558*0.3893^2 = 5.454 A -> I = 7.258 A
    s = warm.step(1.0)
    assert s["motor_current_a"] == pytest.approx(7.258, abs=0.005)
    assert 0.6 < s["power_factor"] < 0.7


def test_pump_curve_obeys_affinity_laws():
    pump = PumpModel()
    r = 0.8
    for q in (0.0, 10.0, 25.0):
        assert pump.pump_head_m(q * r, 1450 * r) == pytest.approx(pump.pump_head_m(q, 1450) * r ** 2)


def test_shaft_power_scales_with_speed_cubed_on_similar_points():
    # Pure-friction system (no static head) keeps operating points similar
    pump = PumpModel()
    pump.cfg.system.static_head_m = 0.0
    pump.k_sys = 25.0 / 30.0 ** 2
    hi = pump.operating_point(1450)
    lo = pump.operating_point(1160)
    assert lo["flow_m3h"] == pytest.approx(hi["flow_m3h"] * 0.8, rel=1e-6)
    assert lo["shaft_power_kw"] == pytest.approx(hi["shaft_power_kw"] * 0.8 ** 3, rel=1e-6)


def test_throttling_reduces_flow_and_raises_discharge_pressure(warm):
    base = warm.step(1.0)
    warm.set_valve_opening(0.5)
    throttled = warm.step(1.0)
    assert throttled["flow_m3h"] < base["flow_m3h"]
    assert throttled["discharge_pressure_bar"] > base["discharge_pressure_bar"]
    assert throttled["shaft_power_kw"] < base["shaft_power_kw"]   # radial pump


def test_deadhead_at_low_speed():
    pump = PumpModel()
    op = pump.operating_point(400)   # shut-off head 31.25*(400/1450)^2 = 2.4 m < 10 m static
    assert op["flow_m3h"] == 0.0
    assert op["shaft_power_kw"] > 0.0


def test_startup_and_shutdown_sequence():
    pump = PumpModel()
    pump.start()
    states = [pump.step(1.0)["state"] for _ in range(6)]
    assert states[0] == PumpState.STARTING.value
    assert states[-1] == PumpState.RUNNING.value
    assert pump.speed_rpm == pytest.approx(1450)
    pump.stop()
    for _ in range(10):
        s = pump.step(1.0)
    assert s["state"] == PumpState.STOPPED.value
    assert s["speed_rpm"] == 0.0 and s["flow_m3h"] == 0.0 and s["motor_current_a"] == 0.0


def test_emergency_stop_latches_until_reset(warm):
    warm.emergency_stop()
    s = warm.step(0.5)
    assert s["motor_power_kw"] == 0.0            # motor de-energised
    for _ in range(40):
        s = warm.step(0.5)
    assert s["speed_rpm"] == 0.0
    assert s["state"] == PumpState.EMERGENCY_STOP.value
    warm.start()                                 # must not restart while latched
    assert warm.step(0.5)["state"] == PumpState.EMERGENCY_STOP.value
    warm.reset()
    assert warm.state == PumpState.STOPPED


def test_low_suction_pressure_causes_cavitation_head_loss():
    pump = PumpModel()
    healthy = pump.operating_point(1450)
    starved = pump.operating_point(1450, effects=FaultEffects(suction_pressure_drop_bar=2.0))
    assert starved["npsh_available_m"] < starved["npsh_required_m"] * 1.1
    assert starved["cavitation_head_factor"] < 0.97
    assert starved["flow_m3h"] < healthy["flow_m3h"]


def test_bearing_temperature_first_order_response():
    pump = PumpModel()
    pump.start()
    tau = pump.cfg.thermal.bearing_time_constant_s
    for _ in range(int(tau)):
        s = pump.step(1.0)
    # After one time constant ~63 % of the 30 degC rise (start transient is short)
    assert 25 + 0.55 * 30 < s["bearing_temp_c"] < 25 + 0.68 * 30


def test_fault_effects_combine():
    a = FaultEffects(head_factor=0.9, extra_friction_kw=0.2, vibration_add_mm_s=3.0)
    b = FaultEffects(head_factor=0.8, extra_friction_kw=0.1, vibration_add_mm_s=4.0)
    c = a.combine(b)
    assert c.head_factor == pytest.approx(0.72)
    assert c.extra_friction_kw == pytest.approx(0.3)
    assert c.vibration_add_mm_s == pytest.approx(5.0)


def test_friction_fault_raises_current_and_bearing_temp(warm):
    base = warm.step(1.0)
    hot = PumpModel(warm_start=True)
    for _ in range(3000):
        s = hot.step(1.0, FaultEffects(extra_friction_kw=0.5))
    assert s["motor_current_a"] > base["motor_current_a"]
    assert s["bearing_temp_c"] == pytest.approx(55 + 30 * 0.5, abs=0.5)
