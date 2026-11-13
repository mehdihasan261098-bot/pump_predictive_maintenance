"""
Virtual sensor system.

Wraps :class:`PumpModel` and turns its TRUE physical state into timestamped
MEASUREMENTS, adding what real instruments add:

  * Gaussian measurement noise (1-sigma per sensor, from config)
  * instrument behaviour: flow-meter low-flow cut-off, encoder zero at standstill,
    non-negative magnitudes
  * random missing readings (dropout probability)
  * scheduled sensor failures: dead (NaN), stuck, drift, offset (bias), spikes

Because every sensor is derived from the same physical state, the readings stay
physically consistent (e.g. throttling lowers flow, raises discharge pressure
and lowers motor current at the same time).

An :class:`OperatingProfile` scripts operator actions (start, stop, valve and
speed changes, E-stop) so complete scenarios - startup, steady state, load
variation, shutdown - can be generated reproducibly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from config.pump_config import AppConfig, get_config
from simulation.pump_model import FaultEffects, PumpModel

# sensor name -> (pump-state key, unit, plausible physical range for validation)
SENSORS: dict[str, tuple[str, str, tuple[float, float]]] = {
    "bearing_temp_c":         ("bearing_temp_c",         "degC",  (-20.0, 150.0)),
    "casing_temp_c":          ("casing_temp_c",          "degC",  (-20.0, 150.0)),
    "vibration_rms_mm_s":     ("vibration_rms_mm_s",     "mm/s",  (0.0, 50.0)),
    "vibration_accel_m_s2":   ("vibration_accel_m_s2",   "m/s2",  (0.0, 200.0)),
    "suction_pressure_bar":   ("suction_pressure_bar",   "bar",   (-1.0, 10.0)),
    "discharge_pressure_bar": ("discharge_pressure_bar", "bar",   (-1.0, 16.0)),
    "flow_m3h":               ("flow_m3h",               "m3/h",  (0.0, 80.0)),
    "motor_current_a":        ("motor_current_a",        "A",     (0.0, 60.0)),
    "motor_power_kw":         ("motor_power_kw",         "kW",    (0.0, 15.0)),
    "speed_rpm":              ("speed_rpm",              "rpm",   (0.0, 1800.0)),
}
SENSOR_NAMES = list(SENSORS)
CONDITION_COLUMNS = ["state", "speed_command_rpm", "valve_opening"]

FLOW_CUTOFF_M3H = 0.5      # typical mag-flow-meter low-flow cut-off
SPEED_CUTOFF_RPM = 10.0    # encoder reads 0 below this


# ---------------------------------------------------------------------------
# Sensor failures
# ---------------------------------------------------------------------------
@dataclass
class SensorFailure:
    """A fault of the INSTRUMENT (not of the pump).

    mode: 'dead'   -> NaN
          'stuck'  -> frozen at the last good value
          'drift'  -> + rate * (t - start)   (rate in sensor unit per second)
          'offset' -> + magnitude            (calibration bias)
          'spike'  -> with probability `probability` add +/- magnitude
    """
    sensor: str
    mode: str
    start_s: float
    end_s: float = math.inf
    magnitude: float = 0.0
    rate: float = 0.0
    probability: float = 0.05

    MODES = ("dead", "stuck", "drift", "offset", "spike")

    def __post_init__(self):
        if self.sensor not in SENSORS:
            raise ValueError(f"unknown sensor '{self.sensor}'")
        if self.mode not in self.MODES:
            raise ValueError(f"unknown failure mode '{self.mode}'")

    def active(self, t: float) -> bool:
        return self.start_s <= t < self.end_s


# ---------------------------------------------------------------------------
# Operating profile
# ---------------------------------------------------------------------------
@dataclass
class OperatingProfile:
    """Time-ordered operator actions: (time_s, action, value).

    actions: 'start', 'stop', 'estop', 'reset', 'valve' (0.05-1), 'speed' (rpm)
    Optional slow process-demand variation superimposes a sinusoid on the
    valve opening (amplitude, period) to mimic changing plant demand.
    """
    events: list[tuple[float, str, float]] = field(default_factory=list)
    demand_amplitude: float = 0.0
    demand_period_s: float = 900.0

    ACTIONS = ("start", "stop", "estop", "reset", "valve", "speed")

    def __post_init__(self):
        for _, action, _ in self.events:
            if action not in self.ACTIONS:
                raise ValueError(f"unknown action '{action}'")
        self.events = sorted(self.events, key=lambda e: e[0])

    @classmethod
    def standard(cls, duration_s: float = 3600.0, demand_amplitude: float = 0.04) -> "OperatingProfile":
        """Startup -> steady -> load variation (valve, speed) -> shutdown."""
        d = duration_s
        return cls(events=[
            (10.0, "start", 0.0),
            (0.40 * d, "valve", 0.70),     # reduced plant demand (throttling)
            (0.55 * d, "valve", 1.00),
            (0.65 * d, "speed", 1300.0),   # VFD turndown
            (0.80 * d, "speed", 1450.0),
            (d - 60.0, "stop", 0.0),
        ], demand_amplitude=demand_amplitude)

    @classmethod
    def continuous(cls, demand_amplitude: float = 0.04) -> "OperatingProfile":
        """Pump already running (use with warm_start=True); demand variation only."""
        return cls(events=[], demand_amplitude=demand_amplitude)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class SensorSimulator:
    """Generates timestamped sensor records from a virtual pump."""

    def __init__(self, config: AppConfig | None = None, seed: int | None = None,
                 warm_start: bool = False, start_time: str | pd.Timestamp = "2026-01-01 08:00:00",
                 failures: list[SensorFailure] | None = None, dropout_probability: float | None = None):
        self.cfg = config or get_config()
        self.rng = np.random.default_rng(self.cfg.random_seed if seed is None else seed)
        self.pump = PumpModel(self.cfg, warm_start=warm_start)
        self.start_time = pd.Timestamp(start_time)
        self.failures = list(failures or [])
        self.dropout_probability = (self.cfg.sensors.dropout_probability
                                    if dropout_probability is None else dropout_probability)
        self._last_good: dict[str, float] = {}
        self._base_valve = self.pump.valve_opening
        self._event_idx = 0

    def set_base_valve(self, opening: float):
        """Operator valve setpoint (demand variation of the profile is added on top)."""
        self._base_valve = min(max(float(opening), 0.05), 1.0)

    @property
    def base_valve(self) -> float:
        return self._base_valve

    # -- profile handling --------------------------------------------------
    def _apply_profile(self, profile: OperatingProfile, t: float):
        while self._event_idx < len(profile.events) and profile.events[self._event_idx][0] <= t:
            _, action, value = profile.events[self._event_idx]
            if action == "start":
                self.pump.start()
            elif action == "stop":
                self.pump.stop()
            elif action == "estop":
                self.pump.emergency_stop()
            elif action == "reset":
                self.pump.reset()
            elif action == "valve":
                self._base_valve = value
            elif action == "speed":
                self.pump.set_speed_command(value)
            self._event_idx += 1
        demand = profile.demand_amplitude * math.sin(2 * math.pi * t / profile.demand_period_s)
        self.pump.set_valve_opening(self._base_valve + demand)

    # -- measurement model -------------------------------------------------
    def measure(self, truth: dict) -> dict[str, float]:
        """Convert a true pump state into noisy instrument readings."""
        noise = self.cfg.sensors.noise_std
        t = truth["time_s"]
        out = {}
        for name, (key, _, _) in SENSORS.items():
            value = truth[key] + self.rng.normal(0.0, noise.get(name, 0.0))
            if name == "flow_m3h" and value < FLOW_CUTOFF_M3H:
                value = 0.0
            elif name == "speed_rpm" and truth[key] < SPEED_CUTOFF_RPM:
                value = 0.0
            elif name in ("vibration_rms_mm_s", "vibration_accel_m_s2"):
                value = abs(value)       # RMS of signal + noise is never negative
            elif name in ("motor_current_a", "motor_power_kw"):
                value = max(value, 0.0)
            out[name] = value

        for f in self.failures:
            if not f.active(t):
                continue
            if f.mode == "dead":
                out[f.sensor] = math.nan
            elif f.mode == "stuck":
                out[f.sensor] = self._last_good.get(f.sensor, out[f.sensor])
            elif f.mode == "drift":
                out[f.sensor] += f.rate * (t - f.start_s)
            elif f.mode == "offset":
                out[f.sensor] += f.magnitude
            elif f.mode == "spike" and self.rng.random() < f.probability:
                out[f.sensor] += f.magnitude * self.rng.choice((-1.0, 1.0))

        stuck = {f.sensor for f in self.failures if f.mode == "stuck" and f.active(t)}
        for name in SENSOR_NAMES:
            if name not in stuck and not math.isnan(out[name]):
                self._last_good[name] = out[name]
            if self.dropout_probability > 0 and self.rng.random() < self.dropout_probability:
                out[name] = math.nan
        return out

    def step(self, profile: OperatingProfile | None = None, effects: FaultEffects | None = None,
             include_truth: bool = False) -> dict:
        """Advance one sampling interval and return one timestamped record."""
        dt = self.cfg.sensors.sampling_interval_s
        if profile is not None:
            self._apply_profile(profile, self.pump.time_s)
        truth = self.pump.step(dt, effects)
        record = {"timestamp": self.start_time + pd.to_timedelta(float(truth["time_s"]), unit="s"),
                  "time_s": truth["time_s"]}
        record.update(self.measure(truth))
        record.update({c: truth[c] for c in CONDITION_COLUMNS})
        if include_truth:
            record.update({f"true_{n}": truth[SENSORS[n][0]] for n in SENSOR_NAMES})
        return record

    def run(self, duration_s: float, profile: OperatingProfile | None = None,
            effects_fn: Callable[[float], FaultEffects] | None = None,
            include_truth: bool = False) -> pd.DataFrame:
        """Simulate `duration_s` seconds. `effects_fn(t)` may inject pump faults."""
        n = int(round(duration_s / self.cfg.sensors.sampling_interval_s))
        rows = []
        for _ in range(n):
            eff = effects_fn(self.pump.time_s) if effects_fn else None
            rows.append(self.step(profile, eff, include_truth))
        return pd.DataFrame(rows)


def generate_healthy_baseline(duration_s: float = 3600.0, seed: int = 42,
                              path: str | None = "data/raw/healthy_baseline.csv") -> pd.DataFrame:
    """Healthy reference dataset: cold start, steady running, load changes, shutdown."""
    sim = SensorSimulator(seed=seed, dropout_probability=0.002)
    df = sim.run(duration_s, OperatingProfile.standard(duration_s), include_truth=True)
    df["fault_label"] = "healthy"
    df["fault_severity"] = 0.0
    if path:
        df.to_csv(path, index=False)
    return df


def plot_overview(df: pd.DataFrame, path: str, title: str = "Sensor overview"):
    """Save a multi-panel time-series plot of all sensors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 2, figsize=(14, 13), sharex=True)
    minutes = df["time_s"] / 60.0
    for ax, name in zip(axes.ravel(), SENSOR_NAMES):
        ax.plot(minutes, df[name], lw=0.6, label="measured")
        if f"true_{name}" in df:
            ax.plot(minutes, df[f"true_{name}"], lw=1.0, color="k", alpha=0.7, label="true")
        ax.set_ylabel(f"{name}\n[{SENSORS[name][1]}]", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time [min]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    df = generate_healthy_baseline()
    plot_overview(df, "reports/healthy_baseline_overview.png", "Healthy baseline (1 h, 1 s sampling)")
    print(f"rows={len(df)}  columns={len(df.columns)}  missing readings={int(df[SENSOR_NAMES].isna().sum().sum())}")
    print("\nTime per state [s]:", df["state"].value_counts().to_dict())
    steady = df[(df.state == "Running") & (df.time_s.between(1200, 1400))]
    print("\nSteady state (t=1200-1400 s), measured mean / std:")
    print(steady[SENSOR_NAMES].agg(["mean", "std"]).T.round(3).to_string())
