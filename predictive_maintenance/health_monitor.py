"""
Equipment health index, degradation trend, simulated RUL and maintenance
recommendations.

Health index (explainable, configurable)
----------------------------------------
1. Each INDICATOR (physical quantity) is mapped to a damage value 0..1 by a
   piecewise-linear function through three anchors taken from the rule limits:
       good value -> 0,   warning limit -> 0.45,   critical limit -> 0.8,
       1.0 one further (critical - warning) step beyond critical.
2. COMPONENT damage = worst of its indicators:
       vibration  velocity RMS, 1X amplitude, 2X/1X ratio
       bearing    HF acceleration, envelope defect ratio
       thermal    load-compensated bearing temperature residual, absolute temp
       hydraulic  flow deficit, BEP ratio (low), head deviation (negative)
       suction    NPSH margin (low), suction pressure (low)
3. Overall damage D = 0.7 * worst component + 0.3 * weighted mean, so one bad
   component cannot be averaged away ("weakest link") while the overall
   state still contributes.   HI = 100 (1 - D), EWMA-smoothed.
Bands (config): >=85 Healthy, 65-84 Caution, 40-64 Degraded, <40 Critical.
These are demonstration thresholds, not certified industrial standards.

Remaining useful life (SIMULATED ESTIMATE)
------------------------------------------
Over the last N evaluations two models are fitted: HI linear in time and
log(damage) linear in time (exponential wear). The better fit (R^2) is
extrapolated to the Critical threshold (HI = 40). An estimate is produced only
when HI is falling faster than a minimum rate with R^2 above a minimum;
otherwise RUL is "not estimated". It is a trend extrapolation on simulated
data, NOT a validated industrial prognosis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.pump_config import AppConfig, get_config
from detection.rule_based import CRITICAL, NORMAL, WARNING, Diagnosis, RuleBasedDiagnoser

URGENCY_ORDER = ["Routine", "Planned", "Urgent", "Immediate"]
FAULT_COMPONENT = {
    "bearing_degradation": "Drive-end bearing", "imbalance": "Impeller / rotor balance",
    "misalignment": "Coupling / shaft alignment", "cavitation": "Suction system",
    "flow_restriction": "Discharge line / valve", "seal_leakage": "Mechanical seal",
    "overheating": "Lubrication / cooling", "excessive_vibration": "Pump-motor unit",
    "high_bearing_temperature": "Bearings / lubrication",
}
COMPONENT_ADVICE = {
    "vibration": "Elevated vibration: perform spectrum analysis and check mounting.",
    "bearing": "Bearing condition indicators rising: check lubrication, trend envelope spectrum.",
    "thermal": "Temperature above load-expected value: check lubrication and cooling.",
    "hydraulic": "Hydraulic performance off-design: check valves, strainers, seal leakage.",
    "suction": "Suction conditions degraded: check strainer, suction valve, tank level.",
}


def damage(value: float, good: float, warning: float, critical: float, cfg: AppConfig | None = None) -> float:
    """Piecewise-linear indicator damage (works for rising or falling limits)."""
    if value is None or not np.isfinite(value):
        return 0.0
    h = (cfg or get_config()).health
    dw, dc = h.damage_at_warning, h.damage_at_critical
    sign = 1.0 if critical > good else -1.0
    x, g, w, c = sign * value, sign * good, sign * warning, sign * critical
    if x <= g:
        return 0.0
    if x <= w:
        return dw * (x - g) / (w - g)
    if x <= c:
        return dw + (dc - dw) * (x - w) / (c - w)
    return min(1.0, dc + (1 - dc) * (x - c) / (c - w))


@dataclass
class HealthRecord:
    timestamp: pd.Timestamp
    health_index: float
    health_index_raw: float
    condition: str
    components: dict
    indicators: dict
    trend_per_hour: float | None
    rul_hours: float | None
    rul_model: str
    urgency: str
    recommendations: list = field(default_factory=list)
    suggested_date: pd.Timestamp | None = None
    primary_fault: str = "healthy"


class HealthMonitor:
    def __init__(self, config: AppConfig | None = None):
        self.cfg = config or get_config()
        self.h = self.cfg.health
        self.rules = RuleBasedDiagnoser(self.cfg)
        self.history: list[HealthRecord] = []

    # ------------------------------------------------------------------
    def condition(self, hi: float) -> str:
        if hi >= self.h.healthy_min:
            return "Healthy"
        if hi >= self.h.caution_min:
            return "Caution"
        if hi >= self.h.degraded_min:
            return "Degraded"
        return "Critical"

    def indicator_damage(self, f: dict) -> tuple[dict, dict]:
        """Returns (component damages, indicator details)."""
        th, cfg = self.cfg.rules, self.cfg
        g = lambda k, d=np.nan: float(f.get(k, d)) if f.get(k, d) is not None else np.nan
        resid = g("bearing_temp_rise_c") - self.rules.expected_bearing_rise(f)
        if g("bearing_temp_slope_c_min", 0.0) < th.thermal_min_slope_c_min:
            resid = 0.0      # cooling towards the new (lower-load) equilibrium: transient, not a fault
        ind = {
            "vibration": {
                "velocity_rms_mm_s": (g("vibration_rms_mm_s_mean"), 2.2, th.velocity_warning, th.velocity_critical),
                "amp_1x_mm_s": (g("vib_amp_1x_mm_s"), 2.6, th.imbalance_1x_warning, th.imbalance_1x_critical),
                "ratio_2x_1x": (g("vib_ratio_2x_1x"), 0.5, th.misalignment_ratio_warning,
                                th.misalignment_ratio_critical),
            },
            "bearing": {
                "acc_hf_band_rms": (g("vib_acc_hf_band_rms"), 0.45, th.bearing_hf_warning, th.bearing_hf_critical),
                "envelope_ratio": (max(g("vib_env_bpfo_ratio", 0), g("vib_env_bpfi_ratio", 0)), 3.0,
                                   th.bearing_env_ratio, 30.0),
            },
            "thermal": {
                "thermal_residual_c": (resid, 2.0, th.thermal_residual_warning, th.thermal_residual_critical),
                "bearing_temp_c": (g("bearing_temp_c_mean"), 60.0, th.bearing_temp_warning, th.bearing_temp_critical),
            },
            "hydraulic": {
                "flow_deficit_m3h": (g("flow_deficit_m3h_mean"), 0.4, th.flow_deficit_warning,
                                     th.flow_deficit_critical),
                # system-resistance ratio when valve position is known, else BEP ratio fallback
                **({"system_resistance_ratio": (g("system_resistance_ratio_mean"), 1.2, th.resistance_ratio_warning,
                                                th.resistance_ratio_critical)}
                   if np.isfinite(g("system_resistance_ratio_mean")) else
                   {"bep_ratio": (g("bep_ratio_mean"), 0.7, th.bep_ratio_warning, th.bep_ratio_critical)}),
                "head_deviation_pct": (g("head_deviation_pct_mean"), -1.5, th.cavitation_head_drop_pct, -10.0),
            },
            "suction": {
                "npsh_margin": (g("npsh_margin_mean"), 5.0, th.npsh_margin_warning, th.npsh_margin_critical),
                "suction_pressure_bar": (g("suction_pressure_bar_mean"), 1.0, th.suction_pressure_low_bar, 0.3),
            },
        }
        comps, details = {}, {}
        for comp, items in ind.items():
            dmg = {k: damage(v, *anchors, cfg=cfg) for k, (v, *anchors) in items.items()}
            if "envelope_ratio" in dmg:
                # envelope SNR shows that a defect EXISTS (saturates early), not how
                # severe it is -> capped at warning level; severity via HF amplitude
                dmg["envelope_ratio"] = min(dmg["envelope_ratio"], self.h.damage_at_warning)
            comps[comp] = max(dmg.values())
            details.update({k: {"value": items[k][0], "damage": d} for k, d in dmg.items()})
        return comps, details

    def health_index(self, f: dict) -> tuple[float, dict, dict]:
        comps, details = self.indicator_damage(f)
        w = self.h.weights
        mean = sum(w[c] * d for c, d in comps.items()) / sum(w.values())
        dmg = self.h.worst_share * max(comps.values()) + (1 - self.h.worst_share) * mean
        return 100.0 * (1.0 - dmg), {c: round(100 * (1 - d), 1) for c, d in comps.items()}, details

    # ------------------------------------------------------------------
    def _trend_and_rul(self, times_h: np.ndarray, hi: np.ndarray) -> tuple[float | None, float | None, str]:
        n = len(hi)
        if n < max(5, self.h.trend_windows // 2):
            return None, None, "insufficient history"
        t = times_h - times_h[-1]
        slope, icpt = np.polyfit(t, hi, 1)
        if hi[-1] < self.h.degraded_min:
            return float(slope), 0.0, "already critical"
        if slope > -self.h.rul_min_slope_per_h:
            return float(slope), None, "no significant degradation trend"

        def r2(y, yhat):
            ss = np.sum((y - y.mean()) ** 2)
            return 1 - np.sum((y - yhat) ** 2) / ss if ss > 0 else 0.0

        fits = {"linear": (r2(hi, slope * t + icpt), None)}
        d = np.clip(1 - hi / 100.0, 1e-3, None)
        b, a = np.polyfit(t, np.log(d), 1)
        if b > 0:
            fits["exponential"] = (r2(np.log(d), b * t + a), (a, b))
        model = max(fits, key=lambda k: fits[k][0])
        if fits[model][0] < self.h.rul_min_r2:
            return float(slope), None, f"trend too noisy (R2 {fits[model][0]:.2f})"
        target_d = 1 - self.h.degraded_min / 100.0
        if model == "linear":
            t_cross = (self.h.degraded_min - icpt) / slope
        else:
            a, b = fits[model][1]
            t_cross = (math.log(target_d) - a) / b
        return float(slope), float(max(t_cross, 0.0)), model

    def _urgency(self, hi: float, diag: Diagnosis | None, rul: float | None) -> str:
        lvl = diag.overall_level if diag else NORMAL
        if hi < self.h.degraded_min or lvl == CRITICAL or (rul is not None and rul < 1.0):
            return "Immediate"
        if hi < self.h.caution_min or (rul is not None and rul < 24.0):
            return "Urgent"
        if hi < self.h.healthy_min or lvl == WARNING or (rul is not None and rul < 24.0 * 7):
            return "Planned"
        return "Routine"

    def _schedule(self, ts: pd.Timestamp, urgency: str, rul: float | None) -> pd.Timestamp:
        horizon_h = {"Immediate": 0.0, "Urgent": 24.0, "Planned": 24.0 * 7, "Routine": 24.0 * 30}[urgency]
        if rul is not None:
            horizon_h = min(horizon_h, 0.5 * rul)          # act well before predicted critical state
        return ts + pd.to_timedelta(horizon_h, unit="h")

    # ------------------------------------------------------------------
    def update(self, timestamp, features: dict, diagnosis: Diagnosis | None = None) -> HealthRecord:
        ts = pd.Timestamp(timestamp)
        diagnosis = diagnosis or self.rules.diagnose(features)
        raw, comps, details = self.health_index(features)
        prev = self.history[-1].health_index if self.history else raw
        hi = self.h.ewma_alpha * raw + (1 - self.h.ewma_alpha) * prev
        recent = self.history[-(self.h.trend_windows - 1):]
        times = np.array([(r.timestamp - ts).total_seconds() / 3600 for r in recent] + [0.0])
        values = np.array([r.health_index for r in recent] + [hi])
        slope, rul, model = self._trend_and_rul(times, values)
        urgency = self._urgency(hi, diagnosis, rul)
        recs = list(dict.fromkeys(r.recommendation for r in diagnosis.active))
        if not recs:
            recs = [COMPONENT_ADVICE[c] for c, v in comps.items() if v < self.h.healthy_min]
        if not recs:
            recs = ["Continue routine condition monitoring."]
        rec = HealthRecord(ts, round(hi, 2), round(raw, 2), self.condition(hi), comps, details, slope,
                           None if rul is None else round(rul, 3), model, urgency, recs,
                           self._schedule(ts, urgency, rul), diagnosis.primary_fault)
        self.history.append(rec)
        return rec

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"timestamp": r.timestamp, "health_index": r.health_index,
                              "health_index_raw": r.health_index_raw, "condition": r.condition,
                              "trend_per_hour": r.trend_per_hour, "rul_hours": r.rul_hours, "rul_model": r.rul_model,
                              "urgency": r.urgency, "primary_fault": r.primary_fault,
                              **{f"hi_{k}": v for k, v in r.components.items()}} for r in self.history])

    # ------------------------------------------------------------------
    def persist(self, record: HealthRecord, db, pump_id: str = "P-101") -> int | None:
        """Store the health record; open a maintenance work order when needed
        (one open order per component - duplicates suppressed)."""
        db.insert_health(record.timestamp, record.health_index, record.condition, record.rul_hours,
                         record.components, pump_id=pump_id)
        if URGENCY_ORDER.index(record.urgency) < URGENCY_ORDER.index("Planned"):
            return None
        component = FAULT_COMPONENT.get(record.primary_fault, "Pump-motor unit")
        open_orders = db.get_maintenance(pump_id=pump_id)
        if not open_orders.empty:
            same = open_orders[(open_orders.component == component)
                               & open_orders.status.isin(["open", "scheduled"])]
            if not same.empty:
                return None
        return db.add_maintenance(component, record.recommendations[0], record.urgency, created_at=record.timestamp,
                                  scheduled_for=record.suggested_date, status="scheduled",
                                  notes=f"Auto-generated: HI {record.health_index:.0f} ({record.condition}), "
                                        f"diagnosis: {record.primary_fault if record.primary_fault != 'healthy' else 'no specific fault identified yet'}, RUL "
                                        f"{'n/a' if record.rul_hours is None else f'{record.rul_hours:.2f} h'} "
                                        f"(simulated estimate)", pump_id=pump_id)


# --------------------------------------------------------------------------
# Validation on simulated degradation runs
# --------------------------------------------------------------------------
def run_monitor(features: pd.DataFrame, config: AppConfig | None = None) -> pd.DataFrame:
    mon = HealthMonitor(config)
    for row in features.to_dict("records"):
        mon.update(row["timestamp"], row)
    out = mon.history_frame()
    out["window_end_s"] = features["window_end_s"].to_numpy()
    out["fault_severity"] = features["fault_severity"].to_numpy()
    return out


def rul_validation(hist: pd.DataFrame, threshold: float = 40.0) -> pd.DataFrame:
    """Compare each RUL estimate with the actual remaining time until the
    smoothed HI first fell below `threshold` in the same run."""
    below = hist[hist.health_index < threshold]
    if below.empty:
        return pd.DataFrame()
    t_fail = below.window_end_s.iloc[0]
    est = hist[hist.rul_hours.notna() & (hist.window_end_s < t_fail)].copy()
    est["actual_rul_h"] = (t_fail - est.window_end_s) / 3600.0
    est["error_h"] = est.rul_hours - est.actual_rul_h
    return est[["window_end_s", "health_index", "rul_model", "rul_hours", "actual_rul_h", "error_h"]]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from analysis.feature_extraction import extract_features
    from database.database_manager import DatabaseManager
    from simulation.fault_simulator import FaultScenario, FaultType, generate_fault_dataset

    cases = {
        "healthy": [],
        "bearing (exponential)": [FaultScenario(FaultType.BEARING_DEGRADATION, onset_s=600, ramp_s=4800,
                                                shape="exponential")],
        "imbalance (linear)": [FaultScenario(FaultType.IMBALANCE, onset_s=600, ramp_s=4800)],
        "seal leakage (linear)": [FaultScenario(FaultType.SEAL_LEAKAGE, onset_s=600, ramp_s=4800)],
        "overheating (linear)": [FaultScenario(FaultType.OVERHEATING, onset_s=600, ramp_s=4800)],
    }
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for i, (name, scen) in enumerate(cases.items()):
        df = generate_fault_dataset(scen, 6000, seed=500 + i, out_dir=None)
        hist = run_monitor(extract_features(df, seed=i))
        t_min = hist.window_end_s / 60
        axes[0].plot(t_min, hist.health_index, label=name)
        axes[1].plot(t_min, hist.rul_hours * 60, ".", ms=3, label=f"{name} (estimated)")
        val = rul_validation(hist)
        first = hist[hist.condition != "Healthy"].window_end_s.min() / 60
        print(f"\n{name}: min HI {hist.health_index.min():.1f}, final condition {hist.condition.iloc[-1]}, "
              f"first non-Healthy at {first:.0f} min, final urgency {hist.urgency.iloc[-1]}")
        print("  HI by severity:", hist.groupby(pd.cut(hist.fault_severity, [-0.01, 0.05, 0.3, 0.6, 1.0]),
                                                     observed=True).health_index.mean().round(1).to_dict())
        if not val.empty:
            axes[1].plot(val.window_end_s / 60, val.actual_rul_h * 60, "k-", lw=0.8)
            late = val[val.actual_rul_h <= val.actual_rul_h.max() / 2]
            print(f"  RUL estimates: {len(val)}, MAE {val.error_h.abs().mean() * 60:.1f} min overall, "
                  f"MAE {late.error_h.abs().mean() * 60:.1f} min in final half; models {val.rul_model.value_counts().to_dict()}")
    for lvl, c in ((85, "g"), (65, "orange"), (40, "r")):
        axes[0].axhline(lvl, color=c, ls="--", lw=0.8)
    axes[0].set(ylabel="Health index", ylim=(0, 102), title="Health index under gradual degradation (simulated)")
    axes[1].set(ylabel="RUL to HI=40 [min]", xlabel="time [min]", ylim=(0, 120),
                title="Simulated RUL estimate (dots) vs actual remaining time (black lines)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("reports/health_index_demo.png", dpi=85)

    # persistence demo
    with DatabaseManager(":memory:") as db:
        mon = HealthMonitor()
        df = generate_fault_dataset(cases["bearing (exponential)"], 6000, seed=501, out_dir=None)
        orders = [mon.persist(mon.update(r["timestamp"], r), db) for r in extract_features(df).to_dict("records")]
        print("\nDB: health rows", db.counts()["health_history"], "| work orders created",
              sum(o is not None for o in orders))
        print(db.get_maintenance()[["component", "urgency", "status", "scheduled_for", "notes"]].to_string())
