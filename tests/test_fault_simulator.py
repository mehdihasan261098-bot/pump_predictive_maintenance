"""Task 1.4 tests: each fault must move the physically relevant sensors."""
import pytest

from simulation.fault_simulator import (FAULT_TYPES, FaultScenario, FaultSimulator, FaultType,
                                        fault_effects, generate_fault_dataset)
from simulation.pump_model import HEALTHY, PumpModel


def settle(effects=None, seconds=4000):
    """True steady-state of a warm pump under constant fault effects."""
    pump = PumpModel(warm_start=True)
    for _ in range(seconds // 10):
        s = pump.step(10.0, effects)
    return s


@pytest.fixture(scope="module")
def healthy():
    return settle()


def full(fault):
    return settle(fault_effects(fault, 1.0))


def test_zero_severity_is_healthy():
    for f in FAULT_TYPES:
        assert fault_effects(f, 0.0) == HEALTHY


def test_healthy_reference(healthy):
    assert healthy["flow_m3h"] == pytest.approx(30.0, rel=1e-3)
    assert healthy["vibration_rms_mm_s"] == pytest.approx(1.8, rel=1e-3)
    assert healthy["bearing_temp_c"] == pytest.approx(55.0, abs=0.1)


def test_bearing_degradation(healthy):
    s = full(FaultType.BEARING_DEGRADATION)
    assert s["vibration_accel_m_s2"] > 10 * healthy["vibration_accel_m_s2"]   # HF impacts
    assert s["bearing_temp_c"] > healthy["bearing_temp_c"] + 10
    assert s["motor_current_a"] > healthy["motor_current_a"]
    assert s["flow_m3h"] == pytest.approx(healthy["flow_m3h"], rel=1e-3)      # hydraulics unaffected


def test_bearing_early_stage_shows_in_acceleration_first(healthy):
    s = settle(fault_effects(FaultType.BEARING_DEGRADATION, 0.3))
    accel_ratio = s["vibration_accel_m_s2"] / healthy["vibration_accel_m_s2"]
    vel_ratio = s["vibration_rms_mm_s"] / healthy["vibration_rms_mm_s"]
    assert accel_ratio > 4 and vel_ratio < 1.15


def test_imbalance(healthy):
    s = full(FaultType.IMBALANCE)
    assert s["vibration_rms_mm_s"] > 7.1          # ISO 10816 zone D territory
    assert s["bearing_temp_c"] - healthy["bearing_temp_c"] < 6
    assert s["flow_m3h"] == pytest.approx(healthy["flow_m3h"], rel=1e-3)


def test_misalignment(healthy):
    s = full(FaultType.MISALIGNMENT)
    assert s["vibration_rms_mm_s"] > 3 * healthy["vibration_rms_mm_s"]
    assert s["motor_power_kw"] > healthy["motor_power_kw"] * 1.05            # coupling losses
    assert s["bearing_temp_c"] > healthy["bearing_temp_c"] + 10


def test_cavitation(healthy):
    s = full(FaultType.CAVITATION)
    assert s["suction_pressure_bar"] < 0                                     # vacuum at inlet
    assert s["npsh_available_m"] < 1.1 * s["npsh_required_m"]
    assert s["head_m"] < 0.9 * healthy["head_m"]
    assert s["flow_m3h"] < healthy["flow_m3h"]
    assert s["vibration_accel_m_s2"] > 20 * healthy["vibration_accel_m_s2"]


def test_mild_suction_loss_gives_no_cavitation(healthy):
    s = settle(fault_effects(FaultType.CAVITATION, 0.3))
    assert s["suction_pressure_bar"] < healthy["suction_pressure_bar"] - 0.5
    assert s["vibration_accel_m_s2"] == pytest.approx(healthy["vibration_accel_m_s2"], rel=0.01)


def test_flow_restriction(healthy):
    s = full(FaultType.FLOW_RESTRICTION)
    assert s["flow_m3h"] < 0.5 * healthy["flow_m3h"]
    assert s["discharge_pressure_bar"] > healthy["discharge_pressure_bar"]
    assert s["motor_power_kw"] < healthy["motor_power_kw"]
    assert s["vibration_rms_mm_s"] > healthy["vibration_rms_mm_s"]            # off-BEP


def test_seal_leakage(healthy):
    s = full(FaultType.SEAL_LEAKAGE)
    assert s["pump_flow_m3h"] - s["flow_m3h"] == pytest.approx(3.0)
    assert s["flow_m3h"] < healthy["flow_m3h"]
    assert s["pump_flow_m3h"] > healthy["pump_flow_m3h"]
    assert s["casing_temp_c"] > healthy["casing_temp_c"] + 4


def test_overheating(healthy):
    s = full(FaultType.OVERHEATING)
    assert s["bearing_temp_c"] > healthy["bearing_temp_c"] + 30
    assert s["casing_temp_c"] > healthy["casing_temp_c"] + 3
    assert s["vibration_rms_mm_s"] == pytest.approx(healthy["vibration_rms_mm_s"], rel=0.01)


@pytest.mark.parametrize("fault", FAULT_TYPES)
def test_severity_is_monotonic(fault, healthy):
    key = {FaultType.BEARING_DEGRADATION: "vibration_accel_m_s2", FaultType.IMBALANCE: "vibration_rms_mm_s",
           FaultType.MISALIGNMENT: "vibration_rms_mm_s", FaultType.CAVITATION: "suction_pressure_bar",
           FaultType.FLOW_RESTRICTION: "flow_m3h", FaultType.SEAL_LEAKAGE: "flow_m3h",
           FaultType.OVERHEATING: "bearing_temp_c"}[fault]
    sign = -1 if key in ("suction_pressure_bar", "flow_m3h") else 1
    vals = [sign * settle(fault_effects(fault, s), 3000)[key] for s in (0.0, 0.33, 0.66, 1.0)]
    assert vals == sorted(vals) and vals[-1] > vals[0]


def test_multiple_simultaneous_faults(healthy):
    eff = fault_effects(FaultType.IMBALANCE, 0.8).combine(fault_effects(FaultType.FLOW_RESTRICTION, 0.8))
    s = settle(eff)
    assert s["vibration_rms_mm_s"] > 5
    assert s["flow_m3h"] < 0.6 * healthy["flow_m3h"]


def test_scenario_shapes():
    lin = FaultScenario(FaultType.IMBALANCE, onset_s=100, ramp_s=100, shape="linear")
    exp = FaultScenario(FaultType.IMBALANCE, onset_s=100, ramp_s=100, shape="exponential")
    step = FaultScenario(FaultType.IMBALANCE, onset_s=100, shape="step", max_severity=0.6, end_s=300)
    assert lin.severity(50) == 0 and lin.severity(150) == pytest.approx(0.5) and lin.severity(500) == 1.0
    assert exp.severity(150) < 0.5 and exp.severity(200) == pytest.approx(1.0)
    assert step.severity(100) == 0.6 and step.severity(300) == 0.0          # repaired
    with pytest.raises(ValueError):
        FaultScenario(FaultType.IMBALANCE, shape="zigzag")


def test_manual_switching_and_labels():
    fs = FaultSimulator()
    assert fs.label(0) == ("healthy", 0.0, "healthy")
    fs.set_fault("cavitation", 0.4)
    fs.set_fault(FaultType.IMBALANCE, 0.7)
    assert fs.label(0) == ("imbalance", 0.7, "cavitation+imbalance")
    fs.set_fault("imbalance", 0)
    assert fs.label(0)[0] == "cavitation"
    fs.clear()
    assert fs.effects(0) == HEALTHY


def test_generated_dataset_labels_follow_scenario():
    df = generate_fault_dataset([FaultScenario(FaultType.FLOW_RESTRICTION, onset_s=100, ramp_s=200)],
                                duration_s=400, seed=5, out_dir=None)
    assert (df[df.time_s <= 100].fault_label == "healthy").all()
    assert (df[df.time_s > 320].fault_label == "flow_restriction").all()
    assert df.sev_flow_restriction.is_monotonic_increasing
    assert df[df.time_s > 350].flow_m3h.mean() < 0.6 * df[df.time_s < 90].flow_m3h.mean()
