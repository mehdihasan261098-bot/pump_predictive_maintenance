"""Task 2.4 tests: rule-based diagnosis on simulated windows (known faults)."""
import dataclasses

import pandas as pd
import pytest

from analysis.feature_extraction import extract_features
from config.pump_config import get_config
from detection.rule_based import CRITICAL, NORMAL, WARNING, RuleBasedDiagnoser, evaluate
from simulation.fault_simulator import FaultScenario, FaultType, generate_fault_dataset
from simulation.sensor_simulator import OperatingProfile

DX = RuleBasedDiagnoser()


def last_window(scenarios, duration=600, seed=11, profile=None):
    df = generate_fault_dataset(scenarios, duration, seed=seed, out_dir=None, profile=profile)
    return extract_features(df, seed=seed).iloc[-1].to_dict()


def step(fault, sev=1.0):
    return [FaultScenario(fault, shape="step", max_severity=sev)]


@pytest.mark.parametrize("profile", [None,
                                     OperatingProfile(events=[(0, "speed", 1300.0), (0, "valve", 0.75)]),
                                     OperatingProfile(events=[(0, "valve", 0.8)], demand_amplitude=0.04)])
def test_healthy_gives_normal_at_various_operating_points(profile):
    d = DX.diagnose(last_window([], duration=1800, profile=profile))
    assert d.running and d.overall_level == NORMAL and d.primary_fault == "healthy"


@pytest.mark.parametrize("fault,duration", [
    (FaultType.BEARING_DEGRADATION, 600), (FaultType.IMBALANCE, 600), (FaultType.MISALIGNMENT, 600),
    (FaultType.CAVITATION, 600), (FaultType.FLOW_RESTRICTION, 600), (FaultType.SEAL_LEAKAGE, 600),
    (FaultType.OVERHEATING, 2400)])
def test_each_fault_detected_as_primary(fault, duration):
    d = DX.diagnose(last_window(step(fault), duration))
    assert d.primary_fault == fault.value
    assert d.active[0].level in (WARNING, CRITICAL)


@pytest.mark.parametrize("fault", list(FaultType))
def test_results_carry_evidence_reason_and_action(fault):
    d = DX.diagnose(last_window(step(fault), 2400 if fault is FaultType.OVERHEATING else 600))
    top = d.active[0]
    assert top.evidence and all(isinstance(v, float) for v in top.evidence.values())
    assert len(top.reason) > 20 and len(top.recommendation) > 20
    assert 0.5 <= top.confidence <= 1.0


def test_severity_levels_escalate():
    mild = DX.diagnose(last_window(step(FaultType.SEAL_LEAKAGE, 0.4)))
    severe = DX.diagnose(last_window(step(FaultType.SEAL_LEAKAGE, 1.0)))
    get = lambda d: next(r for r in d.results if r.fault == "seal_leakage")
    assert get(mild).level == WARNING and get(severe).level == CRITICAL


def test_multiple_simultaneous_faults():
    d = DX.diagnose(last_window(step(FaultType.IMBALANCE, 0.9) + step(FaultType.FLOW_RESTRICTION, 0.8)))
    found = {r.fault for r in d.active}
    assert {"imbalance", "flow_restriction"} <= found


def test_specific_fault_ranked_above_generic_symptom():
    d = DX.diagnose(last_window(step(FaultType.IMBALANCE, 1.0)))
    faults = [r.fault for r in d.active]
    assert "excessive_vibration" in faults
    assert faults.index("imbalance") < faults.index("excessive_vibration")


def test_cavitation_does_not_trigger_seal_leakage():
    d = DX.diagnose(last_window(step(FaultType.CAVITATION, 1.0)))
    assert next(r for r in d.results if r.fault == "seal_leakage").level == NORMAL


def test_mechanical_heat_not_reported_as_overheating():
    d = DX.diagnose(last_window(step(FaultType.MISALIGNMENT, 1.0), 2400))
    assert next(r for r in d.results if r.fault == "overheating").level == NORMAL


def test_not_running_returns_normal():
    d = DX.diagnose({"speed_rpm_mean": 0.0})
    assert not d.running and d.overall_level == NORMAL


def test_missing_features_do_not_crash():
    d = DX.diagnose({"speed_rpm_mean": 1450.0})
    assert d.running and d.overall_level == NORMAL


def test_thresholds_are_configurable():
    cfg = get_config()
    cfg.rules = dataclasses.replace(cfg.rules, velocity_warning=1.0)
    d = RuleBasedDiagnoser(cfg).diagnose(last_window([]))
    assert next(r for r in d.results if r.fault == "excessive_vibration").level == WARNING


def test_evaluate_reports_fp_fn():
    rows = [last_window([], seed=1), last_window(step(FaultType.IMBALANCE), seed=2),
            last_window(step(FaultType.IMBALANCE, 0.1), seed=3)]
    feats = pd.DataFrame(rows)
    out = evaluate(feats)
    s = out["summary"]
    assert s["false_positive_rate"] == 0.0
    assert s["windows"] == 3 and 0.0 <= s["false_negative_rate"] <= 0.5
    assert "imbalance" in out["confusion"].index
    assert len(evaluate(feats, min_severity=0.3)["table"]) == 2
