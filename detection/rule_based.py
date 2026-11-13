"""
Rule-based (expert-system) fault diagnosis.

Input: one window feature dict (analysis.feature_extraction.FEATURE_COLUMNS).
Each rule combines several pieces of evidence instead of a single threshold,
returns a level (Normal / Warning / Critical), a confidence, the sensor
evidence it used, a human-readable reason and a recommended action.

Specific fault rules:  bearing_degradation, imbalance, misalignment, cavitation,
                       flow_restriction, seal_leakage, overheating
Generic condition rules (symptoms without a specific cause):
                       excessive_vibration, high_bearing_temperature
When a specific rule explains a symptom, the generic rule is reported but
ranked below it.

Thresholds: config.pump_config.RuleThresholds (illustrative, configurable).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from config.pump_config import AppConfig, get_config

NORMAL, WARNING, CRITICAL = "Normal", "Warning", "Critical"
LEVEL_RANK = {NORMAL: 0, WARNING: 1, CRITICAL: 2}
GENERIC_RULES = ("excessive_vibration", "high_bearing_temperature")

RECOMMENDATIONS = {
    "bearing_degradation": {
        WARNING: "Plan bearing inspection; check lubrication (grease quantity/condition); trend envelope spectrum daily.",
        CRITICAL: "Replace the bearing at the earliest opportunity; reduce load if possible; avoid unplanned failure.",
    },
    "imbalance": {
        WARNING: "Inspect impeller for deposits/erosion; plan field balancing.",
        CRITICAL: "Stop and balance the rotor (ISO 21940); inspect impeller for damage or missing material.",
    },
    "misalignment": {
        WARNING: "Check shaft alignment (laser) and coupling condition at next stop; check soft foot and pipe strain.",
        CRITICAL: "Re-align pump and motor; inspect coupling elements and bearings for secondary damage.",
    },
    "cavitation": {
        WARNING: "Check suction strainer, suction valve fully open and tank level; reduce flow if possible.",
        CRITICAL: "Reduce flow/speed immediately, clear suction restriction; inspect impeller for cavitation erosion.",
    },
    "flow_restriction": {
        WARNING: "Check discharge valve position, strainers/filters and for pipe fouling; avoid prolonged low-flow operation.",
        CRITICAL: "Remove restriction or open bypass; prolonged operation far below BEP overheats liquid and damages bearings/seal.",
    },
    "seal_leakage": {
        WARNING: "Inspect mechanical seal area for visible leakage; check flush plan; plan seal replacement.",
        CRITICAL: "Replace mechanical seal; check shaft/sleeve and bearings for liquid ingress.",
    },
    "overheating": {
        WARNING: "Check lubricant level/condition, cooling (fan, cooling water) and ambient temperature.",
        CRITICAL: "Reduce load or stop; restore cooling/lubrication before restart.",
    },
    "excessive_vibration": {
        WARNING: "Perform detailed vibration analysis (spectrum, phase); check mounting bolts and foundation.",
        CRITICAL: "Vibration in ISO 10816 zone D territory: stop for investigation to avoid damage.",
    },
    "high_bearing_temperature": {
        WARNING: "Check lubrication and bearing load; verify temperature sensor.",
        CRITICAL: "Bearing temperature critical: stop and inspect bearing and lubrication.",
    },
}


@dataclass
class RuleResult:
    fault: str
    level: str = NORMAL
    confidence: float = 0.0
    evidence: dict = field(default_factory=dict)
    reason: str = ""
    recommendation: str = ""

    @property
    def rank(self) -> int:
        return LEVEL_RANK[self.level]


@dataclass
class Diagnosis:
    results: list[RuleResult]
    running: bool

    @property
    def overall_level(self) -> str:
        return max((r.level for r in self.results), key=LEVEL_RANK.get, default=NORMAL)

    @property
    def active(self) -> list[RuleResult]:
        """Abnormal results, most severe first; specific faults before generic
        symptoms at the same level; then by confidence."""
        act = [r for r in self.results if r.level != NORMAL]
        return sorted(act, key=lambda r: (-r.rank, r.fault in GENERIC_RULES, -r.confidence))

    @property
    def primary_fault(self) -> str:
        return self.active[0].fault if self.active else "healthy"

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.results])


def _g(f: dict, key: str, default: float = math.nan) -> float:
    v = f.get(key, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else v


def _conf(value: float, threshold: float, span: float) -> float:
    """Confidence 0.5 at threshold rising to 1.0 one `span` beyond it."""
    return max(0.5, min(1.0, 0.5 + 0.5 * (value - threshold) / span)) if span else 1.0


def _result(fault, level, conf, evidence, reason) -> RuleResult:
    rec = RECOMMENDATIONS[fault].get(level, "No action required.") if level != NORMAL else "No action required."
    return RuleResult(fault, level, round(conf, 3), evidence, reason, rec)


class RuleBasedDiagnoser:
    def __init__(self, config: AppConfig | None = None):
        self.cfg = config or get_config()
        self.th = self.cfg.rules

    # ---------------------------------------------------------------- rules
    def _excessive_vibration(self, f):
        v = _g(f, "vibration_rms_mm_s_mean", 0.0)
        ev = {"vibration_rms_mm_s": v}
        if v >= self.th.velocity_critical:
            return _result("excessive_vibration", CRITICAL, _conf(v, self.th.velocity_critical, 3),
                           ev, f"Velocity RMS {v:.1f} mm/s >= critical {self.th.velocity_critical}")
        if v >= self.th.velocity_warning:
            return _result("excessive_vibration", WARNING, _conf(v, self.th.velocity_warning, 2.6),
                           ev, f"Velocity RMS {v:.1f} mm/s >= warning {self.th.velocity_warning}")
        return _result("excessive_vibration", NORMAL, 0, ev, "Velocity within limits")

    def _high_bearing_temperature(self, f):
        t = _g(f, "bearing_temp_c_mean", 0.0)
        ev = {"bearing_temp_c": t}
        if t >= self.th.bearing_temp_critical:
            return _result("high_bearing_temperature", CRITICAL, _conf(t, self.th.bearing_temp_critical, 10),
                           ev, f"Bearing temperature {t:.1f} degC >= {self.th.bearing_temp_critical}")
        if t >= self.th.bearing_temp_warning:
            return _result("high_bearing_temperature", WARNING, _conf(t, self.th.bearing_temp_warning, 15),
                           ev, f"Bearing temperature {t:.1f} degC >= {self.th.bearing_temp_warning}")
        return _result("high_bearing_temperature", NORMAL, 0, ev, "Bearing temperature normal")

    def _imbalance(self, f):
        a1, ratio = _g(f, "vib_amp_1x_mm_s", 0.0), _g(f, "vib_ratio_2x_1x", 0.0)
        order = _g(f, "vib_dominant_order", 0.0)
        ev = {"amp_1x_mm_s": a1, "ratio_2x_1x": ratio, "dominant_order": order,
              "vibration_rms_mm_s": _g(f, "vibration_rms_mm_s_mean", 0.0)}
        one_x = abs(order - 1.0) < 0.1 and ratio < self.th.imbalance_max_2x_1x
        if one_x and a1 >= self.th.imbalance_1x_critical:
            return _result("imbalance", CRITICAL, _conf(a1, self.th.imbalance_1x_critical, 4), ev,
                           f"Dominant 1X component {a1:.1f} mm/s with low 2X/1X ({ratio:.2f}): strong unbalance")
        if one_x and a1 >= self.th.imbalance_1x_warning:
            return _result("imbalance", WARNING, _conf(a1, self.th.imbalance_1x_warning, 4), ev,
                           f"1X component {a1:.1f} mm/s dominant, 2X/1X {ratio:.2f}: unbalance developing")
        return _result("imbalance", NORMAL, 0, ev, "1X component normal")

    def _misalignment(self, f):
        ratio, a2 = _g(f, "vib_ratio_2x_1x", 0.0), _g(f, "vib_amp_2x_mm_s", 0.0)
        v, rise = _g(f, "vibration_rms_mm_s_mean", 0.0), _g(f, "bearing_temp_rise_c", 0.0)
        ev = {"ratio_2x_1x": ratio, "amp_2x_mm_s": a2, "vibration_rms_mm_s": v, "bearing_temp_rise_c": rise}
        if a2 < self.th.misalignment_min_2x:
            return _result("misalignment", NORMAL, 0, ev, "2X component small")
        if ratio >= self.th.misalignment_ratio_critical and (v >= self.th.velocity_warning
                                                            or rise >= self.th.bearing_rise_warning):
            return _result("misalignment", CRITICAL, _conf(ratio, self.th.misalignment_ratio_critical, 0.5), ev,
                           f"2X/1X = {ratio:.2f} with velocity {v:.1f} mm/s / bearing rise {rise:.0f} degC: "
                           "coupling forces at twice running speed")
        if ratio >= self.th.misalignment_ratio_warning:
            return _result("misalignment", WARNING, _conf(ratio, self.th.misalignment_ratio_warning, 0.6), ev,
                           f"2X/1X = {ratio:.2f} (healthy ~0.3): misalignment pattern")
        return _result("misalignment", NORMAL, 0, ev, "2X/1X ratio normal")

    def _bearing(self, f):
        env = max(_g(f, "vib_env_bpfo_ratio", 0.0), _g(f, "vib_env_bpfi_ratio", 0.0))
        hf, kurt = _g(f, "vib_acc_hf_band_rms", 0.0), _g(f, "vib_acc_kurtosis", 0.0)
        temp = _g(f, "bearing_temp_c_mean", 0.0)
        ev = {"envelope_defect_ratio": env, "acc_hf_band_rms": hf, "acc_kurtosis": kurt, "bearing_temp_c": temp}
        if env < self.th.bearing_env_ratio:
            return _result("bearing_degradation", NORMAL, 0, ev, "No bearing defect frequency in envelope")
        if hf >= self.th.bearing_hf_critical or temp >= self.th.bearing_temp_warning:
            return _result("bearing_degradation", CRITICAL, _conf(hf, self.th.bearing_hf_critical, 4), ev,
                           f"Defect frequency in envelope (SNR {env:.0f}) with HF acceleration {hf:.1f} m/s2 "
                           f"/ bearing {temp:.0f} degC: advanced bearing damage")
        if hf >= self.th.bearing_hf_warning or kurt >= self.th.bearing_kurtosis:
            return _result("bearing_degradation", WARNING, _conf(env, self.th.bearing_env_ratio, 20), ev,
                           f"Bearing defect frequency in envelope spectrum (SNR {env:.0f}), HF {hf:.2f} m/s2, "
                           f"kurtosis {kurt:.1f}: early bearing defect")
        return _result("bearing_degradation", NORMAL, 0, ev, "Envelope peak without HF/impulsive support")

    def _cavitation(self, f):
        m, ps = _g(f, "npsh_margin_mean", math.inf), _g(f, "suction_pressure_bar_mean", 9.0)
        hf, hd = _g(f, "vib_acc_hf_band_rms", 0.0), _g(f, "head_deviation_pct_mean", 0.0)
        ev = {"npsh_margin": m, "suction_pressure_bar": ps, "acc_hf_band_rms": hf, "head_deviation_pct": hd}
        if m < self.th.npsh_margin_critical or (m < 2.5 and (hf >= self.th.cavitation_hf
                                                             or hd <= self.th.cavitation_head_drop_pct)):
            return _result("cavitation", CRITICAL, _conf(self.th.npsh_margin_warning - m, 2.5, 2), ev,
                           f"NPSH margin {m:.2f} with HF {hf:.1f} m/s2 and head deviation {hd:.1f} %: "
                           "active cavitation")
        if m < self.th.npsh_margin_warning or ps < self.th.suction_pressure_low_bar:
            return _result("cavitation", WARNING, _conf(self.th.npsh_margin_warning - m, 0, 2), ev,
                           f"Low suction pressure {ps:.2f} bar / NPSH margin {m:.2f}: cavitation risk")
        return _result("cavitation", NORMAL, 0, ev, "NPSH margin adequate")

    def _flow_restriction(self, f):
        bep, hd = _g(f, "bep_ratio_mean", 1.0), _g(f, "head_deviation_pct_mean", 0.0)
        eff, sec = _g(f, "pump_eff_est_pct_mean", 0.0), _g(f, "sec_kwh_m3_mean", 0.0)
        rr = _g(f, "system_resistance_ratio_mean")
        ev = {"system_resistance_ratio": rr, "bep_ratio": bep, "head_deviation_pct": hd, "pump_eff_est_pct": eff,
              "sec_kwh_m3": sec}
        on_curve = abs(hd) < 5.0          # pump healthy, operating point moved along its curve
        if not math.isnan(rr):
            # valve position known: deliberate throttling is NOT a fault, an unexpected rise of
            # system resistance is
            if on_curve and rr >= self.th.resistance_ratio_critical:
                return _result("flow_restriction", CRITICAL, _conf(rr, self.th.resistance_ratio_critical, 3), ev,
                               f"System resistance {rr:.1f}x the value expected for the valve position, flow "
                               f"{100 * bep:.0f} % of BEP: severe blockage in discharge line")
            if on_curve and rr >= self.th.resistance_ratio_warning:
                return _result("flow_restriction", WARNING, _conf(rr, self.th.resistance_ratio_warning, 1.4), ev,
                               f"System resistance {rr:.2f}x expected for valve position (flow {100 * bep:.0f} % "
                               f"of BEP, efficiency {eff:.0f} %): partial restriction / fouling")
            return _result("flow_restriction", NORMAL, 0, ev, "System resistance consistent with valve position")
        if on_curve and bep < self.th.bep_ratio_critical:
            return _result("flow_restriction", CRITICAL, _conf(self.th.bep_ratio_warning - bep, 0.2, 0.2), ev,
                           f"Flow only {100 * bep:.0f} % of BEP flow with pump on its curve: severe "
                           "discharge restriction / throttling")
        if on_curve and bep < self.th.bep_ratio_warning:
            return _result("flow_restriction", WARNING, _conf(self.th.bep_ratio_warning - bep, 0, 0.2), ev,
                           f"Flow {100 * bep:.0f} % of BEP flow, efficiency {eff:.0f} %: restricted discharge")
        return _result("flow_restriction", NORMAL, 0, ev, "Operating near BEP")

    def _seal_leakage(self, f):
        d, m = _g(f, "flow_deficit_m3h_mean", 0.0), _g(f, "npsh_margin_mean", math.inf)
        crise = _g(f, "casing_temp_rise_c", 0.0)
        ev = {"flow_deficit_m3h": d, "casing_temp_rise_c": crise, "npsh_margin": m}
        if m < 2.5:          # head loss from cavitation also creates an apparent deficit
            return _result("seal_leakage", NORMAL, 0, ev, "Flow deficit explained by cavitation")
        if d >= self.th.flow_deficit_critical:
            return _result("seal_leakage", CRITICAL, _conf(d, self.th.flow_deficit_critical, 1.5), ev,
                           f"Measured flow {d:.2f} m3/h below pump-curve flow at measured head "
                           f"(casing rise {crise:.1f} degC): liquid lost before flow meter")
        if d >= self.th.flow_deficit_warning:
            return _result("seal_leakage", WARNING, _conf(d, self.th.flow_deficit_warning, 1.2), ev,
                           f"Flow deficit {d:.2f} m3/h vs pump curve: possible seal leakage")
        return _result("seal_leakage", NORMAL, 0, ev, "Flow consistent with pump curve")

    def expected_bearing_rise(self, f) -> float:
        """Healthy bearing temperature rise for the current speed and load.
        Baseline thermal model: rise = rise_rated * (0.5 n/n0 + 0.5 P_shaft/P_shaft,rated).
        On a real machine this baseline would be regressed from healthy history."""
        p, th = self.cfg.pump, self.cfg.thermal
        r = _g(f, "speed_rpm_mean", 0.0) / p.rated_speed_rpm
        p_shaft = _g(f, "motor_power_kw_mean", 0.0) * self.cfg.motor.efficiency
        p_rated = p.fluid_density * 9.81 * p.rated_flow_m3h / 3600 * p.rated_head_m / p.rated_efficiency / 1000
        return (th.bearing_temp_nominal_c - th.ambient_temp_c) * (0.5 * r + 0.5 * p_shaft / p_rated)

    def _overheating(self, f, mech_cause: bool):
        rise = _g(f, "bearing_temp_rise_c", 0.0)
        expected = self.expected_bearing_rise(f)
        resid = rise - expected
        slope = _g(f, "bearing_temp_slope_c_min", 0.0)
        ev = {"bearing_temp_rise_c": rise, "expected_rise_c": expected, "thermal_residual_c": resid,
              "bearing_temp_slope_c_min": slope, "casing_temp_rise_c": _g(f, "casing_temp_rise_c", 0.0)}
        if mech_cause:
            return _result("overheating", NORMAL, 0, ev, "Temperature rise explained by mechanical fault")
        not_cooling = slope >= self.th.thermal_min_slope_c_min
        if resid >= self.th.thermal_residual_critical and not_cooling:
            return _result("overheating", CRITICAL, _conf(resid, self.th.thermal_residual_critical, 10), ev,
                           f"Bearing {resid:.1f} degC hotter than expected for current load (rise {rise:.0f} "
                           f"vs {expected:.0f} degC), no vibration cause: cooling/lubrication failure")
        if resid >= self.th.thermal_residual_warning and not_cooling:
            return _result("overheating", WARNING, _conf(resid, self.th.thermal_residual_warning, 9), ev,
                           f"Bearing {resid:.1f} degC above load-expected temperature with normal vibration: "
                           "reduced heat removal")
        return _result("overheating", NORMAL, 0, ev, "Temperatures consistent with load")

    # ---------------------------------------------------------------- driver
    def diagnose(self, features: dict) -> Diagnosis:
        f = dict(features)
        speed_ratio = _g(f, "speed_rpm_mean", 0.0) / self.cfg.pump.rated_speed_rpm
        if speed_ratio < self.th.min_running_speed_ratio:
            return Diagnosis([RuleResult("pump_not_running", NORMAL, 0.0, {"speed_ratio": speed_ratio},
                                         "Rules evaluated only while running", "No action required.")], False)
        res = [self._bearing(f), self._imbalance(f), self._misalignment(f), self._cavitation(f),
               self._flow_restriction(f), self._seal_leakage(f)]
        mech = any(r.level != NORMAL for r in res[:3])
        res += [self._overheating(f, mech), self._excessive_vibration(f), self._high_bearing_temperature(f)]
        return Diagnosis(res, True)


# --------------------------------------------------------------------------
# Evaluation against simulated ground truth
# --------------------------------------------------------------------------
def evaluate(features: pd.DataFrame, config: AppConfig | None = None, min_severity: float = 0.0) -> dict:
    """Compare rule diagnosis with fault_label. Windows of faulty runs below
    `min_severity` are excluded (physically indistinguishable from healthy)."""
    dx = RuleBasedDiagnoser(config)
    df = features[(features.fault_label == "healthy") | (features.fault_severity >= min_severity)].copy()
    diags = [dx.diagnose(r) for r in df.to_dict("records")]
    df["rule_primary"] = [d.primary_fault for d in diags]
    df["rule_level"] = [d.overall_level for d in diags]
    df["rule_faults"] = ["+".join(r.fault for r in d.active) for d in diags]
    truth_faulty = df.fault_label != "healthy"
    detected = df.rule_level != NORMAL
    correct = [lbl in faults.split("+") for lbl, faults in zip(df.fault_label, df.rule_faults)]
    df["rule_correct_top"] = df.rule_primary == df.fault_label
    df["rule_correct_any"] = correct
    summary = {
        "windows": len(df),
        "false_positive_rate": float((detected & ~truth_faulty).sum() / max((~truth_faulty).sum(), 1)),
        "false_negative_rate": float((~detected & truth_faulty).sum() / max(truth_faulty.sum(), 1)),
        "detection_rate": float((detected & truth_faulty).sum() / max(truth_faulty.sum(), 1)),
        "top1_accuracy": float(df.rule_correct_top.mean()),
    }
    confusion = pd.crosstab(df.fault_label, df.rule_primary)
    per_class = df.groupby("fault_label").agg(windows=("rule_correct_top", "size"),
                                              top1=("rule_correct_top", "mean"),
                                              any_match=("rule_correct_any", "mean"),
                                              detected=("rule_level", lambda s: (s != NORMAL).mean()))
    return {"summary": summary, "confusion": confusion, "per_class": per_class, "table": df}


if __name__ == "__main__":
    feats = pd.read_csv("data/processed/features.csv")
    pd.set_option("display.width", 250)
    for sev in (0.0, 0.3):
        out = evaluate(feats, min_severity=sev)
        print(f"\n=== Rule-based evaluation (fault windows with severity >= {sev}) ===")
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in out["summary"].items()})
        print(out["per_class"].round(3).to_string())
        if sev == 0.3:
            print("\nConfusion matrix (rows = true, cols = rule primary):")
            print(out["confusion"].to_string())
            out["table"][["run_id", "window_end_s", "fault_label", "fault_severity", "rule_primary", "rule_level",
                          "rule_faults"]].to_csv("reports/rule_evaluation.csv", index=False)
    example = feats[(feats.fault_label == "cavitation")].sort_values("fault_severity").iloc[-1].to_dict()
    d = RuleBasedDiagnoser().diagnose(example)
    top = d.active[0]
    print(f"\nExample diagnosis: {top.fault} [{top.level}, conf {top.confidence}]\n  reason: {top.reason}\n"
          f"  evidence: { {k: round(v, 2) for k, v in top.evidence.items()} }\n  action: {top.recommendation}")
