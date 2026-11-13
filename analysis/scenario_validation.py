"""
End-to-end fault-scenario validation (Tasks 4.1 / 4.3).

Every scenario runs through the COMPLETE dashboard pipeline (SimulationEngine:
pump -> sensors -> faults -> acquisition/QC -> SQLite -> features -> rules ->
ML -> health index -> alarms) with an in-memory database:

  * 10 min healthy, then the fault ramps linearly to its target severity over
    30 min, then held for 20 min (60 min simulated per scenario)
  * healthy scenarios at three operating points (false-positive check)
  * one two-fault scenario

Recorded per scenario: detection delay and severity at first CORRECT rule
diagnosis, first correct ML prediction, first non-Healthy health index, first
Warning/Critical alarm; wrong-fault diagnoses (rules / ML) after onset; any
abnormal output before onset (false positives); final HI, condition, urgency.

Run:  python -m analysis.scenario_validation
Outputs: reports/validation_report.md, reports/validation_results.csv
Synthetic data only - this is NOT industrial validation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from dashboard.engine import SimulationEngine

ONSET_S, RAMP_S, TOTAL_S = 600.0, 1800.0, 3600.0


@dataclass
class Scenario:
    name: str
    faults: dict = field(default_factory=dict)       # fault -> target severity
    speed: float = 1450.0
    valve: float = 1.0


SCENARIOS = [
    Scenario("healthy, rated point"),
    Scenario("healthy, 1300 rpm / valve 70 %", speed=1300.0, valve=0.7),
    Scenario("healthy, 1450 rpm / valve 60 %", valve=0.6),
    Scenario("bearing degradation", {"bearing_degradation": 0.8}),
    Scenario("imbalance", {"imbalance": 0.8}),
    Scenario("misalignment", {"misalignment": 0.8}),
    Scenario("cavitation", {"cavitation": 0.9}),
    Scenario("flow restriction", {"flow_restriction": 0.8}),
    Scenario("seal leakage", {"seal_leakage": 0.8}),
    Scenario("overheating", {"overheating": 0.8}),
    Scenario("multi: imbalance + seal leakage", {"imbalance": 0.7, "seal_leakage": 0.7}),
]


def _first(df: pd.DataFrame, mask) -> pd.Timestamp | None:
    sel = df[mask]
    return None if sel.empty else sel.timestamp.min()


def run_scenario(sc: Scenario, seed: int = 0) -> dict:
    eng = SimulationEngine(db_path=":memory:", start_time="2026-06-01 08:00:00", seed=seed)
    eng.set_speed(sc.speed)
    eng.set_valve(sc.valve)
    eng.advance(ONSET_S)
    t_onset = eng.now
    for f, s in sc.faults.items():
        eng.ramp_fault(f, s, RAMP_S)
    eng.advance(TOTAL_S - ONSET_S)
    db = eng.db
    diag, health, alarms = db.get_diagnoses(), db.get_health_history(), db.get_alarms()
    rules = diag[diag.method == "rule"]
    ml = diag[diag.method == "ml_classifier"]
    pre, post = (lambda d: d[d.timestamp <= t_onset]), (lambda d: d[d.timestamp > t_onset])
    truth = set(sc.faults)
    generic = {"excessive_vibration", "high_bearing_temperature"}
    sev_at = lambda ts: None if ts is None else round(min(max(((ts - t_onset).total_seconds()) / RAMP_S, 0), 1) *
                                                         max(sc.faults.values()), 2)
    delay = lambda ts: None if ts is None else round((ts - t_onset).total_seconds() / 60, 1)

    t_rule = _first(post(rules), post(rules).fault.isin(truth)) if truth else None
    t_ml = _first(post(ml), post(ml).fault.isin(truth)) if truth else None
    t_hi = _first(post(health), post(health).condition != "Healthy") if truth else None
    abn_alarms = alarms[alarms.priority.isin(["Warning", "Critical"])]
    t_alarm = None if abn_alarms.empty else abn_alarms.raised_at.min()

    pre_rule_fp = int(len(pre(rules)))
    pre_ml_fp = int((pre(ml).fault != "healthy").sum())
    pre_hi_fp = int((pre(health).condition != "Healthy").sum())
    pre_alarm_fp = int((abn_alarms.raised_at <= t_onset).sum())
    if not truth:            # whole run is healthy -> everything is a potential false positive
        pre_rule_fp, pre_ml_fp = int(len(rules)), int((ml.fault != "healthy").sum())
        pre_hi_fp, pre_alarm_fp = int((health.condition != "Healthy").sum()), int(len(abn_alarms))

    wrong_rule = post(rules)[~post(rules).fault.isin(truth | generic)] if truth else rules.iloc[0:0]
    wrong_ml = post(ml)[~post(ml).fault.isin(truth | {"healthy"})] if truth else ml.iloc[0:0]
    last = health.iloc[-1] if not health.empty else None
    eng.close()
    return {
        "scenario": sc.name, "faults": ", ".join(f"{k} {v}" for k, v in sc.faults.items()) or "—",
        "rule_delay_min": delay(t_rule), "rule_sev": sev_at(t_rule),
        "ml_delay_min": delay(t_ml), "ml_sev": sev_at(t_ml),
        "hi_delay_min": delay(t_hi), "alarm_delay_min": delay(t_alarm) if truth else None,
        "fp_rule_windows": pre_rule_fp, "fp_ml_windows": pre_ml_fp, "fp_hi_windows": pre_hi_fp,
        "fp_alarms": pre_alarm_fp,
        "wrong_rule_windows": int(len(wrong_rule)), "wrong_rule_faults": ", ".join(sorted(set(wrong_rule.fault))),
        "wrong_ml_windows": int(len(wrong_ml)), "wrong_ml_faults": ", ".join(sorted(set(wrong_ml.fault))),
        "final_hi": None if last is None else round(float(last.health_index), 1),
        "final_condition": None if last is None else last.condition,
        "rule_windows_post": int(len(post(rules))), "ml_windows_post": int(len(post(ml))),
    }


def to_markdown(df: pd.DataFrame, runtime_s: float) -> str:
    d = df.copy()
    na = lambda v: "—" if v is None or (isinstance(v, float) and pd.isna(v)) else v
    rows = ["| Scenario | Rule detection: delay min (severity) | ML detection: delay min (severity) | "
            "HI leaves Healthy (min) | First W/C alarm (min) | Wrong rule / ML windows | Final HI |",
            "|---|---|---|---|---|---|---|"]
    for r in d.itertuples():
        if r.faults == "—":
            rows.append(f"| {r.scenario} | FP windows: {r.fp_rule_windows} | FP windows: {r.fp_ml_windows} | "
                        f"FP windows: {r.fp_hi_windows} | FP alarms: {r.fp_alarms} | — | {na(r.final_hi)} ({r.final_condition}) |")
        else:
            rows.append(f"| {r.scenario} | {na(r.rule_delay_min)} ({na(r.rule_sev)}) | {na(r.ml_delay_min)} ({na(r.ml_sev)}) | "
                        f"{na(r.hi_delay_min)} | {na(r.alarm_delay_min)} | {r.wrong_rule_windows} / {r.wrong_ml_windows}"
                        f"{' (' + r.wrong_ml_faults + ')' if r.wrong_ml_faults else ''} | {na(r.final_hi)} ({r.final_condition}) |")
    faulty = d[d.faults != "—"]
    healthy = d[d.faults == "—"]
    return "\n".join([
        "# End-to-end fault scenario validation", "",
        f"Generated by `python -m analysis.scenario_validation` ({runtime_s:.0f} s). Each scenario runs the complete "
        "dashboard pipeline (SimulationEngine) for 60 simulated minutes: healthy for 10 min, fault ramps linearly to the "
        "target severity over 30 min, then held. Delays are minutes after fault onset; severity in brackets is the fault "
        "severity at the moment of first correct detection. One analysis window = 60 s.", "",
        *rows, "",
        "## Summary", "",
        f"* Faulty scenarios: {len(faulty)} · rules identified the correct fault in "
        f"{int(faulty.rule_delay_min.notna().sum())}/{len(faulty)}, ML in {int(faulty.ml_delay_min.notna().sum())}/{len(faulty)}.",
        f"* False positives before onset / in healthy runs: rule windows {int(d.fp_rule_windows.sum())}, ML windows "
        f"{int(d.fp_ml_windows.sum())}, health-index windows {int(d.fp_hi_windows.sum())}, Warning/Critical alarms {int(d.fp_alarms.sum())}.",
        f"* Healthy runs at three operating points: rule FP {int(healthy.fp_rule_windows.sum())}, ML FP "
        f"{int(healthy.fp_ml_windows.sum())}, HI FP {int(healthy.fp_hi_windows.sum())}, alarm FP {int(healthy.fp_alarms.sum())}.",
        f"* Wrong-fault outputs after onset: rules {int(faulty.wrong_rule_windows.sum())} windows, ML "
        f"{int(faulty.wrong_ml_windows.sum())} windows (generic symptoms such as excessive vibration are not counted as wrong).",
        "",
        "## Interpretation and limitations", "",
        "* Detection delay is dominated by physics: a fault must change the measured signals beyond the healthy noise and "
        "operating-point variability (e.g. overheating is limited by the 10-min bearing thermal time constant).",
        "* Simulator, features, rules and ML share the same modelling assumptions; results on synthetic data **do not "
        "establish industrial diagnostic accuracy**. Real deployment requires machine-specific baselines and field data.",
        "", "Full numbers: `reports/validation_results.csv`.",
    ])


if __name__ == "__main__":
    t0 = time.perf_counter()
    results = []
    for i, sc in enumerate(SCENARIOS):
        t1 = time.perf_counter()
        results.append(run_scenario(sc, seed=900 + i))
        print(f"{sc.name:35s} done in {time.perf_counter() - t1:5.1f} s")
    df = pd.DataFrame(results)
    runtime = time.perf_counter() - t0
    df.to_csv("reports/validation_results.csv", index=False)
    md = to_markdown(df, runtime)
    open("reports/validation_report.md", "w", encoding="utf-8").write(md)
    pd.set_option("display.width", 250)
    print(df[["scenario", "rule_delay_min", "rule_sev", "ml_delay_min", "ml_sev", "hi_delay_min", "alarm_delay_min",
              "fp_rule_windows", "fp_ml_windows", "fp_hi_windows", "fp_alarms", "wrong_rule_windows",
              "wrong_ml_windows", "wrong_ml_faults", "final_hi", "final_condition"]].to_string(index=False))
