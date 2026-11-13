"""
Pump fault simulation.

Each fault type maps a severity s in [0, 1] (0 = healthy, 1 = end of useful
condition) to a :class:`FaultEffects` object. The pump model then turns these
physical changes into sensor responses, so secondary effects (operating-point
shift, motor current, heating, off-BEP vibration) emerge from the physics
rather than being painted onto the sensors.

Physical basis per fault (full table in docs/engineering_assumptions.md):

BEARING_DEGRADATION  Spalls on races -> impacts at bearing defect frequencies.
    Early: high-frequency acceleration rises first (linear in s). Late: friction
    (s^2) heats the bearing and broadband velocity grows (s^2).
IMBALANCE            Residual unbalance m*e -> 1X centrifugal force ~ m e w^2.
    Velocity RMS grows linearly with s; slightly higher bearing load/temperature.
MISALIGNMENT         Coupling forces at 1X/2X (radial + axial), coupling and
    bearing reaction losses -> extra friction, current and bearing temperature.
CAVITATION           Suction-side restriction (clogged strainer / falling tank)
    lowers suction pressure -> NPSHa falls. Noise and head loss are produced
    by the pump model from the resulting NPSH margin (incipient < 2.5, 3 % head
    drop at margin 1.0) - they are NOT imposed directly.
FLOW_RESTRICTION     Discharge-side blockage / fouling raises system resistance
    k (x10 at s=1) -> flow falls, discharge pressure rises, power falls,
    pump moves left of BEP (vibration, liquid heating).
SEAL_LEAKAGE         Worn mechanical-seal faces leak liquid to atmosphere (up to
    10 % of rated flow) -> delivered flow below the value the pump curve
    implies for the measured head; seal-face friction heats the casing.
OVERHEATING          Loss of cooling / lubrication degradation / hot ambient:
    heat-removal capacity halves at s=1, so all temperature rises double;
    lower oil viscosity adds a little friction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from simulation.pump_model import HEALTHY, FaultEffects


class FaultType(str, Enum):
    BEARING_DEGRADATION = "bearing_degradation"
    IMBALANCE = "imbalance"
    MISALIGNMENT = "misalignment"
    CAVITATION = "cavitation"
    FLOW_RESTRICTION = "flow_restriction"
    SEAL_LEAKAGE = "seal_leakage"
    OVERHEATING = "overheating"


FAULT_TYPES = list(FaultType)
HEALTHY_LABEL = "healthy"
LABEL_THRESHOLD = 0.05      # below this severity a fault is treated as absent


def fault_effects(fault: FaultType | str, severity: float) -> FaultEffects:
    """Physical effects of one fault at a given severity (0-1)."""
    fault = FaultType(fault)
    s = min(max(float(severity), 0.0), 1.0)
    if s == 0.0:
        return HEALTHY
    if fault is FaultType.BEARING_DEGRADATION:
        return FaultEffects(extra_friction_kw=0.40 * s ** 2, bearing_temp_add_c=8.0 * s ** 2,
                            hf_accel_add_m_s2=8.0 * s, vibration_add_mm_s=4.0 * s ** 2)
    if fault is FaultType.IMBALANCE:
        return FaultEffects(vibration_add_mm_s=9.0 * s, bearing_temp_add_c=3.0 * s,
                            extra_friction_kw=0.05 * s, hf_accel_add_m_s2=0.5 * s)
    if fault is FaultType.MISALIGNMENT:
        return FaultEffects(vibration_add_mm_s=7.0 * s, extra_friction_kw=0.30 * s,
                            bearing_temp_add_c=8.0 * s, hf_accel_add_m_s2=1.0 * s)
    if fault is FaultType.CAVITATION:
        return FaultEffects(suction_pressure_drop_bar=2.1 * s)
    if fault is FaultType.FLOW_RESTRICTION:
        return FaultEffects(system_resistance_factor=1.0 + 9.0 * s)
    if fault is FaultType.SEAL_LEAKAGE:
        return FaultEffects(seal_leakage_m3h=3.0 * s, casing_temp_add_c=6.0 * s)
    if fault is FaultType.OVERHEATING:
        return FaultEffects(cooling_factor=1.0 - 0.5 * s, bearing_temp_add_c=5.0 * s,
                            extra_friction_kw=0.05 * s)
    raise ValueError(fault)


@dataclass
class FaultScenario:
    """Time-varying severity of one fault.

    shape: 'step'        -> max_severity from onset_s
           'linear'      -> 0 -> max_severity over ramp_s
           'exponential' -> slow start, accelerating (typical wear-out):
                            s = max * (e^(k x) - 1)/(e^k - 1), x = (t-onset)/ramp_s
    end_s: time at which the fault is repaired (severity -> 0).
    """
    fault: FaultType
    onset_s: float = 0.0
    ramp_s: float = 0.0
    max_severity: float = 1.0
    shape: str = "linear"
    end_s: float = math.inf
    growth: float = 3.0

    SHAPES = ("step", "linear", "exponential")

    def __post_init__(self):
        self.fault = FaultType(self.fault)
        if self.shape not in self.SHAPES:
            raise ValueError(f"unknown shape '{self.shape}'")
        if not 0.0 <= self.max_severity <= 1.0:
            raise ValueError("max_severity must be in [0, 1]")

    def severity(self, t: float) -> float:
        if t < self.onset_s or t >= self.end_s:
            return 0.0
        if self.shape == "step" or self.ramp_s <= 0:
            return self.max_severity
        x = min((t - self.onset_s) / self.ramp_s, 1.0)
        if self.shape == "linear":
            return self.max_severity * x
        return self.max_severity * (math.exp(self.growth * x) - 1) / (math.exp(self.growth) - 1)


class FaultSimulator:
    """Combines scheduled scenarios and manual (dashboard) fault settings."""

    def __init__(self, scenarios: list[FaultScenario] | None = None):
        self.scenarios = list(scenarios or [])
        self.manual: dict[FaultType, float] = {}

    # --- manual control (dashboard sliders) ---
    def set_fault(self, fault: FaultType | str, severity: float):
        fault = FaultType(fault)
        if severity <= 0:
            self.manual.pop(fault, None)
        else:
            self.manual[fault] = min(float(severity), 1.0)

    def clear(self):
        """Return to healthy operation (scheduled scenarios and manual faults)."""
        self.manual.clear()
        self.scenarios.clear()

    # --- evaluation ---
    def severities(self, t: float) -> dict[FaultType, float]:
        sev = {f: 0.0 for f in FAULT_TYPES}
        for sc in self.scenarios:
            sev[sc.fault] = max(sev[sc.fault], sc.severity(t))
        for f, s in self.manual.items():
            sev[f] = max(sev[f], s)
        return sev

    def effects(self, t: float) -> FaultEffects:
        total = HEALTHY
        for f, s in self.severities(t).items():
            if s > 0:
                total = total.combine(fault_effects(f, s))
        return total

    def __call__(self, t: float) -> FaultEffects:
        """Allows passing the simulator directly as SensorSimulator effects_fn."""
        return self.effects(t)

    def label(self, t: float) -> tuple[str, float, str]:
        """(dominant fault label, its severity, '+'-joined list of active faults)."""
        active = {f.value: s for f, s in self.severities(t).items() if s >= LABEL_THRESHOLD}
        if not active:
            return HEALTHY_LABEL, 0.0, HEALTHY_LABEL
        dominant = max(active, key=active.get)
        return dominant, active[dominant], "+".join(sorted(active))

    def annotate(self, df: pd.DataFrame, dt: float) -> pd.DataFrame:
        """Add ground-truth label columns to a simulated DataFrame.
        Effects for a record were evaluated at (time_s - dt)."""
        t_eff = df["time_s"] - dt
        labels = [self.label(t) for t in t_eff]
        df = df.copy()
        df["fault_label"] = [lb[0] for lb in labels]
        df["fault_severity"] = [lb[1] for lb in labels]
        df["active_faults"] = [lb[2] for lb in labels]
        for f in FAULT_TYPES:
            df[f"sev_{f.value}"] = [self.severities(t)[f] for t in t_eff]
        return df


def generate_fault_dataset(scenarios: list[FaultScenario], duration_s: float = 3600.0, seed: int = 0,
                           name: str | None = None, out_dir: str | None = "data/raw",
                           profile=None) -> pd.DataFrame:
    """Simulate a warm-started pump under the given fault scenario(s).
    `profile` (OperatingProfile) defaults to continuous operation at rated point."""
    from simulation.sensor_simulator import OperatingProfile, SensorSimulator

    sim = SensorSimulator(seed=seed, warm_start=True, dropout_probability=0.002)
    faults = FaultSimulator(scenarios)
    df = sim.run(duration_s, profile or OperatingProfile.continuous(), effects_fn=faults, include_truth=True)
    df = faults.annotate(df, sim.cfg.sensors.sampling_interval_s)
    df["run_id"] = name or "_".join(sc.fault.value for sc in scenarios) or HEALTHY_LABEL
    if out_dir and name:
        df.to_csv(f"{out_dir}/{name}.csv", index=False)
    return df


def standard_scenario(fault: FaultType | str, duration_s: float = 3600.0) -> FaultScenario:
    """Healthy for the first 10 min, then degradation to full severity at 80 %
    of the run, then held. Wear faults grow exponentially, others linearly."""
    fault = FaultType(fault)
    shape = "exponential" if fault in (FaultType.BEARING_DEGRADATION, FaultType.SEAL_LEAKAGE) else "linear"
    return FaultScenario(fault, onset_s=600.0, ramp_s=0.8 * duration_s - 600.0, shape=shape)


def generate_all_datasets(duration_s: float = 3600.0, out_dir: str = "data/raw") -> dict[str, pd.DataFrame]:
    out = {}
    for i, f in enumerate(FAULT_TYPES):
        out[f.value] = generate_fault_dataset([standard_scenario(f, duration_s)], duration_s,
                                              seed=100 + i, name=f"fault_{f.value}", out_dir=out_dir)
    combo = [standard_scenario(FaultType.BEARING_DEGRADATION, duration_s),
             FaultScenario(FaultType.MISALIGNMENT, onset_s=1200, ramp_s=1500, max_severity=0.7)]
    out["multiple"] = generate_fault_dataset(combo, duration_s, seed=200,
                                             name="fault_multiple_bearing_misalignment", out_dir=out_dir)
    out["healthy_steady"] = generate_fault_dataset([], duration_s, seed=300,
                                                   name="healthy_steady", out_dir=out_dir)
    return out


def plot_signatures(datasets: dict[str, pd.DataFrame], path: str):
    """Heat-map of % change of each true sensor value at full severity vs healthy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from simulation.sensor_simulator import SENSOR_NAMES

    healthy = datasets["healthy_steady"]
    # same time window as the faulty runs -> same process-demand phase
    ref = healthy[healthy.time_s > healthy.time_s.max() - 300][[f"true_{s}" for s in SENSOR_NAMES]].mean()
    rows, names = [], []
    for key, df in datasets.items():
        if key == "healthy_steady":
            continue
        last = df[df.time_s > df.time_s.max() - 300][[f"true_{s}" for s in SENSOR_NAMES]].mean()
        rows.append(100.0 * (last - ref) / ref.abs())
        names.append(key)
    mat = np.array(rows)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    lim = 100
    im = ax.imshow(np.clip(mat, -lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(SENSOR_NAMES)), SENSOR_NAMES, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.0f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(mat[i, j]) > 60 else "black")
    fig.colorbar(im, ax=ax, label="% change vs healthy (clipped at ±100)")
    ax.set_title("Fault signatures at full severity (true values, last 5 min)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return pd.DataFrame(mat, index=names, columns=SENSOR_NAMES)


if __name__ == "__main__":
    data = generate_all_datasets()
    table = plot_signatures(data, "reports/fault_signatures.png")
    pd.set_option("display.width", 200)
    print("% change of true sensor values at full severity vs healthy:")
    print(table.round(1).to_string())
    print("\nLabel counts (multiple-fault run):")
    print(data["multiple"]["active_faults"].value_counts().to_string())
