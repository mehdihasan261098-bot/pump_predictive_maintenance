"""Task 1.5 tests: KPIs from measured data vs hand calculations + fault discrimination."""
import math

import numpy as np
import pandas as pd
import pytest

from analysis import hydraulic_calculations as hc
from config.pump_config import get_config
from simulation.fault_simulator import FaultScenario, FaultType, generate_fault_dataset

CFG = get_config()


def row(**kw):
    base = dict(suction_pressure_bar=1.2, discharge_pressure_bar=3.6525, flow_m3h=30.0,
                motor_power_kw=3.318, speed_rpm=1450.0)
    base.update(kw)
    return pd.DataFrame([base])


def test_rated_point_hand_calculation():
    k = hc.compute_kpis(row(), CFG).iloc[0]
    assert k.diff_pressure_bar == pytest.approx(2.4525)
    assert k.head_m == pytest.approx(25.0)                       # 245250 / 9810
    assert k.hydraulic_power_kw == pytest.approx(2.04375)        # 1000*9.81*(30/3600)*25
    assert k.wire_to_water_eff_pct == pytest.approx(100 * 2.04375 / 3.318)   # 61.596 %
    assert k.pump_eff_est_pct == pytest.approx(100 * 2.04375 / (3.318 * 0.88))  # 69.995 %
    assert k.sec_kwh_m3 == pytest.approx(3.318 / 30)             # 0.1106 kWh/m3
    assert k.motor_load_pct == pytest.approx(100 * 3.318 * 0.88 / 7.5)
    assert k.bep_ratio == pytest.approx(1.0)
    assert k.expected_flow_m3h == pytest.approx(30.0)
    assert k.flow_deficit_m3h == pytest.approx(0.0, abs=1e-9)
    assert k.head_deviation_pct == pytest.approx(0.0, abs=1e-9)
    assert k.npsh_available_m == pytest.approx(218100 / 9810)    # 22.232 m
    assert k.npsh_required_m == pytest.approx(3.0)
    assert k.npsh_margin == pytest.approx(218100 / 9810 / 3.0)


def test_pump_curve_inversion_roundtrip():
    for q in (0.0, 12.0, 30.0, 40.0):
        for n in (1450.0, 1160.0):
            h = hc.expected_head_from_flow_m(q, n, CFG)
            if h > 0:
                assert hc.expected_flow_from_head_m3h(h, n, CFG) == pytest.approx(q, abs=1e-9)


def test_head_above_shutoff_gives_zero_expected_flow():
    assert hc.expected_flow_from_head_m3h(40.0, 1450, CFG) == 0.0


def test_stopped_pump_gives_nan_curve_kpis():
    k = hc.compute_kpis(row(flow_m3h=0.0, motor_power_kw=0.0, speed_rpm=0.0,
                            discharge_pressure_bar=1.3, suction_pressure_bar=1.3), CFG).iloc[0]
    for col in ("wire_to_water_eff_pct", "sec_kwh_m3", "bep_ratio", "expected_flow_m3h",
                "flow_deficit_m3h", "npsh_margin"):
        assert math.isnan(k[col]), col
    assert k.head_m == pytest.approx(0.0)
    assert not math.isnan(k.npsh_available_m)


def test_missing_values_propagate_without_error():
    k = hc.compute_kpis(row(flow_m3h=np.nan), CFG).iloc[0]
    assert math.isnan(k.hydraulic_power_kw) and math.isnan(k.sec_kwh_m3) and math.isnan(k.flow_deficit_m3h)
    assert k.head_m == pytest.approx(25.0)


def test_zero_flow_running_deadhead():
    k = hc.compute_kpis(row(flow_m3h=0.0, discharge_pressure_bar=1.2 + 31.25 * 0.0981), CFG).iloc[0]
    assert k.hydraulic_power_kw == 0.0
    assert math.isnan(k.sec_kwh_m3)
    assert k.wire_to_water_eff_pct == 0.0


def test_performance_curve_passes_rated_point():
    c = hc.performance_curve(CFG)
    i = (c.flow_m3h - 30).abs().idxmin()
    assert c.head_m.iloc[0] == pytest.approx(31.25)
    assert c.efficiency_pct.max() == pytest.approx(70.0, abs=0.1)
    assert c.loc[i, "head_m"] == pytest.approx(25.0, abs=0.3)


def test_performance_curve_obeys_affinity():
    hi, lo = hc.performance_curve(CFG, 1450), hc.performance_curve(CFG, 1160)
    np.testing.assert_allclose(lo.flow_m3h, hi.flow_m3h * 0.8)
    np.testing.assert_allclose(lo.head_m, hi.head_m * 0.64, atol=1e-9)
    np.testing.assert_allclose(lo.shaft_power_kw, hi.shaft_power_kw * 0.512, atol=1e-9)


def _late_kpis(fault):
    df = generate_fault_dataset([FaultScenario(fault, shape="step")], duration_s=600, seed=9, out_dir=None)
    return hc.compute_kpis(df[df.time_s > 300], CFG).mean()


def test_kpis_discriminate_faults():
    healthy = hc.compute_kpis(generate_fault_dataset([], 600, seed=9, out_dir=None), CFG).mean()
    seal = _late_kpis(FaultType.SEAL_LEAKAGE)
    cav = _late_kpis(FaultType.CAVITATION)
    restr = _late_kpis(FaultType.FLOW_RESTRICTION)
    bearing = _late_kpis(FaultType.BEARING_DEGRADATION)
    assert abs(healthy.flow_deficit_m3h) < 0.3 and abs(healthy.head_deviation_pct) < 1
    assert seal.flow_deficit_m3h == pytest.approx(3.0, abs=0.3)
    assert cav.npsh_margin < 1.1 and cav.head_deviation_pct < -10
    assert restr.bep_ratio < 0.5 and abs(restr.flow_deficit_m3h) < 0.5
    assert restr.sec_kwh_m3 > 1.5 * healthy.sec_kwh_m3
    assert bearing.pump_eff_est_pct < healthy.pump_eff_est_pct - 5


def test_system_resistance_ratio_separates_throttling_from_blockage():
    from simulation.sensor_simulator import OperatingProfile
    throttled = generate_fault_dataset([], 300, seed=2, out_dir=None,
                                       profile=OperatingProfile(events=[(0, "valve", 0.55), (0, "speed", 1300.0)]))
    blocked = generate_fault_dataset([FaultScenario(FaultType.FLOW_RESTRICTION, shape="step", max_severity=0.2)],
                                     300, seed=2, out_dir=None)
    r_thr = hc.compute_kpis(throttled[throttled.time_s > 60], CFG).system_resistance_ratio.mean()
    r_blk = hc.compute_kpis(blocked[blocked.time_s > 60], CFG).system_resistance_ratio.mean()
    assert r_thr == pytest.approx(1.0, abs=0.03)          # deliberate throttling is not a fault
    assert r_blk == pytest.approx(1 + 9 * 0.2, rel=0.05)  # = simulated resistance factor
    assert math.isnan(hc.compute_kpis(row(), CFG).system_resistance_ratio.iloc[0])   # no valve info
