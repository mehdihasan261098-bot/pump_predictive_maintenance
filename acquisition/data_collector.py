"""
Data acquisition layer.

    MeasurementSource  ->  DataCollector  ->  (buffer, KPIs, CSV, sinks e.g. SQLite)

* ``MeasurementSource`` is the hardware-abstraction boundary. ``SimulatedSource``
  wraps the virtual pump; a real installation would implement the same
  ``read()`` method on top of a Modbus/OPC UA client (see
  docs/industrial_integration.md) and nothing downstream would change.
* ``DataCollector`` performs the classic DAQ quality-control steps on every
  record: duplicate-timestamp rejection, missing-value detection, plausibility
  (range) check, rate-of-change (spike) check, short-gap forward fill with
  flags, then computes the engineering KPIs and forwards each batch to sinks.
* Acquisition is idempotent per timestamp and can be driven "up to a target
  time", so Streamlit reruns cannot create duplicate or extra samples.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd

from analysis.hydraulic_calculations import compute_kpis
from config.pump_config import AppConfig, get_config
from simulation.fault_simulator import FaultSimulator
from simulation.sensor_simulator import CONDITION_COLUMNS, SENSOR_NAMES, SENSORS, OperatingProfile, SensorSimulator

# Maximum physically credible change between two consecutive samples (per s).
# Temperatures have large thermal inertia, so a jump of several degC in 1 s is
# an instrument problem, not process behaviour.
DEFAULT_MAX_RATE_PER_S = {"bearing_temp_c": 3.0, "casing_temp_c": 3.0}

QC_OK, QC_MISSING, QC_RANGE, QC_RATE = "ok", "missing", "range", "rate"


class MeasurementSource(Protocol):
    """Anything that can deliver one timestamped record of all sensors."""

    sample_interval_s: float

    def read(self) -> dict:
        ...


class SimulatedSource:
    """Virtual pump + sensors + faults behind the MeasurementSource interface."""

    def __init__(self, simulator: SensorSimulator | None = None, faults: FaultSimulator | None = None,
                 profile: OperatingProfile | None = None):
        self.simulator = simulator or SensorSimulator(warm_start=True)
        self.faults = faults or FaultSimulator()
        self.profile = profile or OperatingProfile.continuous()
        self.sample_interval_s = self.simulator.cfg.sensors.sampling_interval_s

    @property
    def time_s(self) -> float:
        return self.simulator.pump.time_s

    def read(self) -> dict:
        t = self.time_s
        rec = self.simulator.step(self.profile, self.faults.effects(t))
        label, severity, active = self.faults.label(t)
        rec.update(fault_label=label, fault_severity=severity, active_faults=active)
        return rec


class DataCollector:
    """Validates, enriches, buffers and distributes measurement records."""

    def __init__(self, source: MeasurementSource | None = None, config: AppConfig | None = None,
                 max_gap_fill: int = 3, max_rate_per_s: dict | None = None,
                 buffer_seconds: float | None = 24 * 3600.0):
        self.cfg = config or get_config()
        self.source = source or SimulatedSource()
        self.max_gap_fill = max_gap_fill
        self.max_rate = dict(DEFAULT_MAX_RATE_PER_S if max_rate_per_s is None else max_rate_per_s)
        self.buffer_seconds = buffer_seconds
        self.sinks: list[Callable[[pd.DataFrame], None]] = []
        self._frames: list[pd.DataFrame] = []
        self._cache: pd.DataFrame | None = None
        self._seen: set[pd.Timestamp] = set()
        self._last_valid: dict[str, tuple[float, float]] = {}   # sensor -> (time_s, value)
        self._gap: dict[str, int] = {s: 0 for s in SENSOR_NAMES}
        self.stats = {"records": 0, "duplicates": 0, "missing": 0, "range": 0, "rate": 0, "filled": 0}

    # ------------------------------------------------------------------
    # Quality control of one record
    # ------------------------------------------------------------------
    def validate(self, rec: dict) -> dict | None:
        """Return the cleaned record, or None if it is a duplicate."""
        ts = pd.Timestamp(rec["timestamp"])
        if ts in self._seen:
            self.stats["duplicates"] += 1
            return None
        self._seen.add(ts)
        out = dict(rec)
        out["timestamp"] = ts
        t = float(rec["time_s"])
        flags = []
        for name in SENSOR_NAMES:
            value = rec.get(name, math.nan)
            value = math.nan if value is None else float(value)
            issue = QC_OK
            lo, hi = SENSORS[name][2]
            if math.isnan(value):
                issue = QC_MISSING
            elif not lo <= value <= hi:
                issue = QC_RANGE
            elif name in self.max_rate and name in self._last_valid:
                t_prev, v_prev = self._last_valid[name]
                if abs(value - v_prev) > self.max_rate[name] * max(t - t_prev, 1e-9):
                    issue = QC_RATE
                    if self._gap[name] >= self.max_gap_fill:
                        # persistent new level, not a spike: accept and re-baseline
                        self.stats[QC_RATE] += 1
                        flags.append(f"{name}:rate_step")
                        issue = QC_OK
            if issue == QC_OK:
                self._last_valid[name] = (t, value)
                self._gap[name] = 0
            else:
                self.stats[issue] += 1
                self._gap[name] += 1
                if self._gap[name] <= self.max_gap_fill and name in self._last_valid:
                    value = self._last_valid[name][1]
                    issue += "+filled"
                    self.stats["filled"] += 1
                else:
                    value = math.nan
                flags.append(f"{name}:{issue}")
            out[name] = value
        flags = sorted(set(flags), key=flags.index)
        out["qc_flags"] = ";".join(flags)
        out["qc_issue_count"] = len(flags)
        self.stats["records"] += 1
        return out

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------
    def ingest(self, records: list[dict]) -> pd.DataFrame:
        """Validate external records (e.g. replayed CSV) and store them."""
        rows = [r for r in (self.validate(rec) for rec in records) if r is not None]
        if not rows:
            return pd.DataFrame()
        batch = pd.DataFrame(rows)
        batch = pd.concat([batch, compute_kpis(batch, self.cfg)], axis=1)
        self._frames.append(batch)
        self._cache = None
        self._trim()
        for sink in self.sinks:
            sink(batch)
        return batch

    def acquire(self, n_samples: int = 1) -> pd.DataFrame:
        """Read n new samples from the source."""
        return self.ingest([self.source.read() for _ in range(max(int(n_samples), 0))])

    def acquire_to_time(self, target_time_s: float) -> pd.DataFrame:
        """Read samples until the source clock reaches target_time_s.
        Calling again with the same target adds nothing (rerun-safe)."""
        dt = self.source.sample_interval_s
        n = int(math.floor((target_time_s - self.source.time_s) / dt + 1e-9))
        return self.acquire(n)

    def _trim(self):
        if self.buffer_seconds is None or not self._frames:
            return
        latest = self._frames[-1]["time_s"].iloc[-1]
        while len(self._frames) > 1 and self._frames[0]["time_s"].iloc[-1] < latest - self.buffer_seconds:
            self._frames.pop(0)
            self._cache = None

    # ------------------------------------------------------------------
    # Access / export
    # ------------------------------------------------------------------
    @property
    def data(self) -> pd.DataFrame:
        if self._cache is None:
            self._cache = (pd.concat(self._frames, ignore_index=True) if self._frames else pd.DataFrame())
        return self._cache

    def latest(self) -> dict | None:
        return None if not self._frames else self._frames[-1].iloc[-1].to_dict()

    def export_csv(self, path: str | Path | None = None) -> Path:
        df = self.data
        if path is None:
            stamp = df["timestamp"].iloc[-1].strftime("%Y%m%d_%H%M%S") if len(df) else "empty"
            path = Path("data/processed") / f"acquisition_{stamp}.csv"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path


def clean_dataset(df: pd.DataFrame, config: AppConfig | None = None, max_gap_fill: int = 3) -> tuple[pd.DataFrame, dict]:
    """Offline cleaning of a recorded dataset with the same QC rules.
    Sorts by timestamp, removes duplicates, flags/fills, adds KPIs."""
    col = DataCollector(source=SimulatedSource(), config=config, max_gap_fill=max_gap_fill, buffer_seconds=None)
    ordered = df.assign(timestamp=pd.to_datetime(df["timestamp"])).sort_values("timestamp", kind="stable")
    keep = [c for c in ordered.columns if not c.startswith("true_")]
    cleaned = col.ingest(ordered[keep].to_dict("records"))
    return cleaned, dict(col.stats)


if __name__ == "__main__":
    from simulation.fault_simulator import FaultScenario, FaultType
    from simulation.sensor_simulator import SensorFailure

    sim = SensorSimulator(seed=11, warm_start=True, dropout_probability=0.01, failures=[
        SensorFailure("bearing_temp_c", "spike", start_s=300, end_s=600, magnitude=25.0, probability=0.05),
        SensorFailure("flow_m3h", "dead", start_s=900, end_s=960),
        SensorFailure("discharge_pressure_bar", "offset", start_s=1200, magnitude=30.0),
    ])
    faults = FaultSimulator([FaultScenario(FaultType.IMBALANCE, onset_s=600, ramp_s=900)])
    collector = DataCollector(SimulatedSource(sim, faults))
    stored = []
    collector.sinks.append(lambda b: stored.append(len(b)))

    collector.acquire_to_time(1800)
    collector.acquire_to_time(1800)                 # simulated Streamlit rerun -> no new rows
    dup = collector.ingest([collector.data.iloc[-1].to_dict()])  # duplicate timestamp
    df = collector.data
    path = collector.export_csv()

    print(f"rows={len(df)} (expected 1800), duplicate batch rows={len(dup)}, sink batches={stored}")
    print("QC stats:", collector.stats)
    print("rows with QC issues:", int((df.qc_issue_count > 0).sum()))
    print("example flags:", df.loc[df.qc_issue_count > 0, "qc_flags"].head(4).tolist())
    print("bearing temp max after spike filter:", round(df.bearing_temp_c.max(), 2))
    print("flow NaN rows during 60 s dead sensor:", int(df.flow_m3h.isna().sum()))
    print("discharge pressure NaN after +30 bar offset (out of range):",
          int(df[df.time_s > 1201].discharge_pressure_bar.isna().sum()))
    print("label counts:", df.fault_label.value_counts().to_dict())
    print("KPI columns present:", all(c in df for c in ("head_m", "sec_kwh_m3", "npsh_margin")))
    print("exported:", path)
