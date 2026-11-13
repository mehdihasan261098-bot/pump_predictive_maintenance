"""
SimulationEngine: the single stateful object behind the dashboard.

    pump + sensors + faults -> DataCollector (QC, KPIs) -> SQLite
                                   |
          every 60 s of simulated time (one window):
          features -> rule diagnosis -> ML -> health index -> alarms -> SQLite

Rerun safety (Streamlit re-executes the script on every interaction):
* The engine lives in st.session_state / st.cache_resource, so it is created
  once per session.
* Simulation advances to a TARGET simulated time computed from the wall clock
  (elapsed wall time x speed-up). Re-running the script within the same instant
  produces no new samples; `DataCollector.acquire_to_time` and the database
  UNIQUE constraints make duplicate records impossible.
* A lock serialises concurrent reruns.
"""
from __future__ import annotations

import math
import threading
import time

import pandas as pd

from acquisition.data_collector import DataCollector, SimulatedSource
from alarms.alarm_manager import AlarmManager
from analysis.feature_extraction import window_features
from config.pump_config import AppConfig, get_config
from database.database_manager import DatabaseManager
from detection.ml_detection import MLDetector
from detection.rule_based import RuleBasedDiagnoser
from predictive_maintenance.health_monitor import HealthMonitor
from simulation.fault_simulator import FaultScenario, FaultSimulator, FaultType
from simulation.sensor_simulator import OperatingProfile, SensorSimulator
from simulation.vibration_signal import generate_vibration

WINDOW_S = 60.0
MAX_CATCH_UP_S = 1800.0        # cap per tick (e.g. after the browser tab slept)


class SimulationEngine:
    def __init__(self, db_path: str | None = None, config: AppConfig | None = None, seed: int = 42,
                 speedup: float = 10.0, warm_start: bool = True, start_time=None):
        self.cfg = config or get_config()
        self.db = DatabaseManager(db_path or self.cfg.database_path)
        last = self.db.get_measurements(limit=1, latest_first=True, columns=["time_s"])
        start = pd.Timestamp(start_time) if start_time is not None else pd.Timestamp.now().floor("s")
        if not last.empty:
            start = max(start, last.timestamp.iloc[0].ceil("s") + pd.to_timedelta(1, unit="s"))
        self.sim = SensorSimulator(self.cfg, seed=seed, warm_start=warm_start, start_time=start,
                                   dropout_probability=0.001)
        self.faults = FaultSimulator()
        self.profile = OperatingProfile.continuous(demand_amplitude=0.02)
        self.source = SimulatedSource(self.sim, self.faults, self.profile)
        self.collector = DataCollector(self.source, self.cfg, buffer_seconds=6 * 3600.0)
        self.collector.sinks.append(self.db.as_sink())
        self.rules = RuleBasedDiagnoser(self.cfg)
        self.health = HealthMonitor(self.cfg)
        self.ml = MLDetector() if MLDetector.available() else None
        self.alarms = AlarmManager(self.db, self.cfg)
        self.speedup = speedup
        self.paused = False
        self._lock = threading.RLock()
        self._last_wall: float | None = None
        self._next_window_end = self.source.time_s + WINDOW_S
        self.latest_features: dict | None = None
        self.latest_diagnosis = None
        self.latest_health = None
        self.latest_ml: dict | None = None
        self.events: list[dict] = []
        self._vib_cache: tuple | None = None

    # ------------------------------------------------------------------ time
    @property
    def sim_time_s(self) -> float:
        return self.source.time_s

    @property
    def now(self) -> pd.Timestamp:
        return self.sim.start_time + pd.to_timedelta(self.sim_time_s, unit="s")

    def tick(self, wall_time: float | None = None) -> int:
        """Advance according to elapsed wall time x speed-up. Returns new samples."""
        wall = time.time() if wall_time is None else wall_time
        with self._lock:
            if self._last_wall is None or self.paused:
                self._last_wall = wall
                return 0
            dt = max(wall - self._last_wall, 0.0) * self.speedup
            self._last_wall = wall
            return self._advance_to(self.sim_time_s + min(dt, MAX_CATCH_UP_S))

    def advance(self, seconds: float) -> int:
        """Fast-forward simulated time (processed in chunks)."""
        with self._lock:
            n, target = 0, self.sim_time_s + seconds
            while self.sim_time_s < target - 1e-9:
                n += self._advance_to(min(self.sim_time_s + 600.0, target))
            self._last_wall = time.time()
            return n

    def _advance_to(self, target_s: float) -> int:
        batch = self.collector.acquire_to_time(target_s)
        if batch.empty:
            return 0
        while self._next_window_end <= self.sim_time_s + 1e-9:
            self._process_window(self._next_window_end)
            self._next_window_end += WINDOW_S
        last = batch.iloc[-1].to_dict()
        self.events += self.alarms.evaluate(last["timestamp"], last, running=last["state"] == "Running",
                                            state=last["state"])
        self.events = self.events[-200:]
        return len(batch)

    # ------------------------------------------------------------------ analytics per window
    def _process_window(self, end_s: float):
        data = self.collector.data
        win = data[(data.time_s > end_s - WINDOW_S) & (data.time_s <= end_s)]
        if len(win) < 0.9 * WINDOW_S / self.cfg.sensors.sampling_interval_s:
            return
        ts = win.timestamp.iloc[-1]
        if not (win.state == "Running").all():
            self.latest_diagnosis = self.rules.diagnose({"speed_rpm_mean": float(win.speed_rpm.mean())})
            return
        sev = {f: s for f, s in self.faults.severities(end_s - 1.0).items() if s > 0}
        f = window_features(win, win, sev, self.cfg, seed=int(end_s))
        diag = self.rules.diagnose(f)
        ml = None
        if self.ml is not None:
            p = self.ml.predict(f).iloc[0].to_dict()
            ml = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in p.items()}
            self.db.insert_diagnosis(ts, "ml_classifier", ml["predicted_fault"],
                                     "Normal" if ml["predicted_fault"] == "healthy" else "Warning",
                                     ml["confidence"], {k: v for k, v in ml.items() if k.startswith("p_")},
                                     "Random Forest classification of window features", "")
            self.db.insert_diagnosis(ts, "ml_anomaly", "anomaly" if ml["is_anomaly"] else "normal",
                                     "Warning" if ml["is_anomaly"] else "Normal", ml["anomaly_score"],
                                     {"threshold": self.ml.iforest["threshold"]},
                                     "Isolation Forest anomaly score vs healthy baseline", "")
        for r in diag.active:
            self.db.insert_diagnosis(ts, "rule", r.fault, r.level, r.confidence, r.evidence, r.reason,
                                     r.recommendation)
        hrec = self.health.update(ts, f, diag)
        self.health.persist(hrec, self.db)
        self.events += self.alarms.evaluate(ts, None, diagnosis=diag, health=hrec, ml=ml)
        self.latest_features, self.latest_diagnosis, self.latest_health, self.latest_ml = f, diag, hrec, ml

    # ------------------------------------------------------------------ operator commands
    def start(self):
        with self._lock:
            self.sim.pump.start()

    def stop(self):
        with self._lock:
            self.sim.pump.stop()

    def emergency_stop(self):
        """SIMULATED ONLY - not a safety function."""
        with self._lock:
            self.sim.pump.emergency_stop()

    def reset_emergency_stop(self) -> bool:
        with self._lock:
            self.sim.pump.reset()
            return self.sim.pump.state.value != "Emergency stop"

    def set_speed(self, rpm: float):
        with self._lock:
            self.sim.pump.set_speed_command(rpm)

    def set_valve(self, opening: float):
        with self._lock:
            self.sim.set_base_valve(opening)

    def set_fault(self, fault: FaultType | str, severity: float):
        with self._lock:
            self.faults.set_fault(fault, severity)

    def ramp_fault(self, fault: FaultType | str, target: float, ramp_s: float, shape: str = "linear"):
        """Gradual degradation starting now (simulated time)."""
        with self._lock:
            self.faults.scenarios.append(FaultScenario(FaultType(fault), onset_s=self.sim_time_s, ramp_s=ramp_s,
                                                       max_severity=target, shape=shape))

    def clear_faults(self):
        with self._lock:
            self.faults.clear()

    # ------------------------------------------------------------------ views
    @property
    def state(self) -> str:
        return self.sim.pump.state.value

    def fault_severities(self) -> dict:
        return {f.value: s for f, s in self.faults.severities(self.sim_time_s).items()}

    def latest(self) -> dict | None:
        return self.collector.latest()

    def recent(self, minutes: float = 30.0) -> pd.DataFrame:
        d = self.collector.data
        return d if d.empty else d[d.time_s > self.sim_time_s - minutes * 60.0]

    def vibration_snapshot(self, healthy_reference: bool = False):
        """1 s waveform for the current state (cached per simulated second)."""
        rec = self.latest()
        if rec is None:
            return None
        key = (round(self.sim_time_s), healthy_reference)
        if self._vib_cache and self._vib_cache[0] == key:
            return self._vib_cache[1]
        kpi_margin = rec.get("npsh_margin", math.inf)
        margin = kpi_margin if isinstance(kpi_margin, float) and math.isfinite(kpi_margin) else math.inf
        speed = rec.get("speed_rpm") or 0.0
        if healthy_reference:
            r = speed / self.cfg.pump.rated_speed_rpm
            sig = generate_vibration(speed, self.cfg.vibration.baseline_rms_mm_s * r ** 2, None, seed=1)
        else:
            sev = {f: s for f, s in self.faults.severities(self.sim_time_s).items() if s > 0}
            v = rec.get("vibration_rms_mm_s")
            sig = generate_vibration(speed, v if v == v and v is not None else 0.0, sev, margin,
                                     rec.get("bep_ratio") if rec.get("bep_ratio") == rec.get("bep_ratio") else 1.0,
                                     seed=int(self.sim_time_s))
        self._vib_cache = (key, sig)
        return sig

    def close(self):
        self.db.close()
