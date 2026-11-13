"""
Alarm management (simplified ISA-18.2 style lifecycle).

    condition true            -> ACTIVE (unacknowledged)
    operator acknowledges     -> ACKNOWLEDGED
    condition returns normal  -> CLEARED  (latched: stays visible until reset)
    operator resets (cleared) -> RESET    (closed; leaves the active list)
    condition returns while CLEARED -> re-activated, occurrences + 1

* Priorities: Info / Warning / Critical. Info alarms are events: they reset
  automatically when their condition clears.
* Duplicate suppression: at most ONE open instance per alarm key; a persisting
  condition does not create new records.
* Hysteresis (deadband) prevents chattering around a threshold.
* Escalation: a diagnosis alarm going Warning -> Critical updates the priority
  of the open instance instead of creating a second alarm.
* Sources: sensor limits, rule-based diagnoses, health index, equipment state
  (emergency stop), ML anomaly flag, sensor quality (QC) flags.

The simulated emergency stop is NOT safety-rated and must never be used as a
real industrial safety function.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from config.pump_config import AppConfig, get_config
from simulation.sensor_simulator import SENSOR_NAMES

INFO, WARNING, CRITICAL = "Info", "Warning", "Critical"
PRIORITY_RANK = {INFO: 0, WARNING: 1, CRITICAL: 2}
OPEN_STATUSES = ("active", "acknowledged", "cleared")


@dataclass
class AlarmDefinition:
    key: str
    priority: str
    message: str
    column: str
    op: str = ">"                 # '>' high alarm, '<' low alarm
    threshold: float = 0.0
    deadband: float = 0.0
    require_running: bool = False
    unit: str = ""

    def condition(self, value, currently_on: bool) -> bool | None:
        """True/False, or None when the value is missing (state unchanged)."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if self.op == ">":
            return value > (self.threshold - self.deadband if currently_on else self.threshold)
        return value < (self.threshold + self.deadband if currently_on else self.threshold)


def default_definitions(cfg: AppConfig | None = None) -> list[AlarmDefinition]:
    cfg = cfg or get_config()
    r = cfg.rules
    D = AlarmDefinition
    return [
        D("bearing_temp_hi", WARNING, "Bearing temperature high", "bearing_temp_c", ">", r.bearing_temp_warning, 2.0, unit="°C"),
        D("bearing_temp_hihi", CRITICAL, "Bearing temperature very high", "bearing_temp_c", ">", r.bearing_temp_critical, 2.0, unit="°C"),
        D("casing_temp_hi", WARNING, "Pump casing temperature high", "casing_temp_c", ">", 45.0, 1.0, unit="°C"),
        D("vibration_hi", WARNING, "Vibration velocity high", "vibration_rms_mm_s", ">", r.velocity_warning, 0.3, True, "mm/s"),
        D("vibration_hihi", CRITICAL, "Vibration velocity very high", "vibration_rms_mm_s", ">", r.velocity_critical, 0.3, True, "mm/s"),
        D("accel_hf_hi", WARNING, "HF acceleration high (bearing/cavitation)", "vibration_accel_m_s2", ">", 2.0, 0.2, True, "m/s²"),
        D("accel_hf_hihi", CRITICAL, "HF acceleration very high", "vibration_accel_m_s2", ">", 8.0, 0.5, True, "m/s²"),
        D("suction_lo", WARNING, "Suction pressure low", "suction_pressure_bar", "<", r.suction_pressure_low_bar, 0.05, True, "bar"),
        D("suction_lolo", CRITICAL, "Suction pressure very low (cavitation risk)", "suction_pressure_bar", "<", 0.3, 0.05, True, "bar"),
        D("discharge_hi", WARNING, "Discharge pressure high (restriction)", "discharge_pressure_bar", ">", 4.2, 0.05, True, "bar"),
        D("flow_lo", WARNING, "Flow low", "flow_m3h", "<", 15.0, 1.0, True, "m³/h"),
        D("current_hi", WARNING, "Motor current high", "motor_current_a", ">", 0.9 * cfg.motor.rated_current_a, 0.2, unit="A"),
        D("current_hihi", CRITICAL, "Motor current above rated", "motor_current_a", ">", cfg.motor.rated_current_a, 0.2, unit="A"),
    ]


class AlarmManager:
    def __init__(self, db, config: AppConfig | None = None, definitions: list[AlarmDefinition] | None = None,
                 pump_id: str = "P-101"):
        self.db = db
        self.cfg = config or get_config()
        self.defs = definitions or default_definitions(self.cfg)
        self.pump_id = pump_id
        self._open: dict[str, dict] = {}
        existing = db.get_alarms(statuses=OPEN_STATUSES, pump_id=pump_id)
        known = {d.key for d in self.defs}
        managed = lambda k: k in known or k.startswith(("diag_", "sensor_", "health_")) or k in (
            "emergency_stop", "ml_anomaly")
        for row in existing.sort_values("raised_at").itertuples():
            if not managed(row.alarm_key) and row.status != "cleared":
                # no definition evaluates this key any more -> it could never clear by itself
                db.update_alarm(int(row.id), status="cleared", cleared_at=pd.Timestamp.now())
                row = row._replace(status="cleared")
            self._open[row.alarm_key] = {"id": int(row.id), "status": row.status, "priority": row.priority,
                                         "occurrences": int(row.occurrences)}

    # ------------------------------------------------------------------ core
    def _raise(self, key, priority, message, ts, value=None, threshold=None) -> dict | None:
        a = self._open.get(key)
        if a is None:
            aid = self.db.insert_alarm(key, priority, message, ts, value, threshold, pump_id=self.pump_id)
            self._open[key] = {"id": aid, "status": "active", "priority": priority, "occurrences": 1}
            return {"event": "raised", "key": key, "id": aid, "priority": priority, "message": message}
        if a["status"] == "cleared":                       # condition returned before reset
            a.update(status="active", occurrences=a["occurrences"] + 1, priority=priority)
            self.db.update_alarm(a["id"], status="active", cleared_at=None, acknowledged_at=None,
                                 acknowledged_by=None, occurrences=a["occurrences"], priority=priority,
                                 message=message, value=value)
            return {"event": "reactivated", "key": key, "id": a["id"], "priority": priority, "message": message}
        if PRIORITY_RANK[priority] > PRIORITY_RANK[a["priority"]]:   # escalation
            a.update(priority=priority, status="active")
            self.db.update_alarm(a["id"], priority=priority, message=message, value=value, status="active",
                                 acknowledged_at=None, acknowledged_by=None)
            return {"event": "escalated", "key": key, "id": a["id"], "priority": priority, "message": message}
        return None                                         # duplicate suppressed

    def _clear(self, key, ts) -> dict | None:
        a = self._open.get(key)
        if a is None or a["status"] == "cleared":
            return None
        if a["priority"] == INFO:                           # events reset automatically
            self.db.update_alarm(a["id"], status="reset", cleared_at=ts, reset_at=ts)
            del self._open[key]
            return {"event": "auto_reset", "key": key, "id": a["id"]}
        a["status"] = "cleared"
        self.db.update_alarm(a["id"], status="cleared", cleared_at=ts)
        return {"event": "cleared", "key": key, "id": a["id"]}

    def _set(self, on: bool | None, key, priority, message, ts, value=None, threshold=None):
        if on is None:
            return None
        return self._raise(key, priority, message, ts, value, threshold) if on else self._clear(key, ts)

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, ts, record: dict | None = None, running: bool = True, diagnosis=None, health=None,
                 state: str | None = None, ml: dict | None = None) -> list[dict]:
        ts = pd.Timestamp(ts)
        events = []
        if record is not None:
            for d in self.defs:
                on_now = d.key in self._open and self._open[d.key]["status"] != "cleared"
                value = record.get(d.column)
                cond = False if (d.require_running and not running) else d.condition(value, on_now)
                msg = f"{d.message}: {value:.2f} {d.unit} (limit {d.threshold:g})" if cond else d.message
                events.append(self._set(cond, d.key, d.priority, msg, ts, value, d.threshold))
            for s in SENSOR_NAMES:
                v = record.get(s, math.nan)
                bad = v is None or (isinstance(v, float) and math.isnan(v))   # unavailable after QC/gap-fill
                events.append(self._set(bad, f"sensor_{s}", INFO, f"Sensor reading unavailable / invalid: {s}", ts))
        if diagnosis is not None and diagnosis.running:
            for r in diagnosis.results:
                on = r.level in (WARNING, CRITICAL)
                events.append(self._set(on, f"diag_{r.fault}", r.level if on else WARNING,
                                        f"Diagnosis: {r.fault.replace('_', ' ')} — {r.reason}", ts, r.confidence))
        if health is not None:
            hi = health.health_index
            h = self.cfg.health
            events.append(self._set(hi < h.healthy_min, "health_caution", INFO, f"Health index {hi:.0f}: Caution", ts, hi, h.healthy_min))
            events.append(self._set(hi < h.caution_min, "health_degraded", WARNING, f"Health index {hi:.0f}: Degraded", ts, hi, h.caution_min))
            events.append(self._set(hi < h.degraded_min, "health_critical", CRITICAL, f"Health index {hi:.0f}: Critical", ts, hi, h.degraded_min))
        if state is not None:
            events.append(self._set(state == "Emergency stop", "emergency_stop", CRITICAL,
                                    "EMERGENCY STOP activated (simulated, not safety-rated)", ts))
        if ml is not None:
            events.append(self._set(bool(ml.get("is_anomaly")), "ml_anomaly", INFO,
                                    f"ML anomaly detected (score {ml.get('anomaly_score', float('nan')):.3f})", ts,
                                    ml.get("anomaly_score")))
        return [e for e in events if e]

    # ------------------------------------------------------------------ operator actions
    def acknowledge(self, alarm_id: int, ts, user: str = "operator") -> bool:
        for a in self._open.values():
            if a["id"] == alarm_id and a["status"] == "active":
                a["status"] = "acknowledged"
                return self.db.update_alarm(alarm_id, status="acknowledged", acknowledged_at=ts, acknowledged_by=user)
            if a["id"] == alarm_id and a["status"] == "cleared":
                return self.db.update_alarm(alarm_id, acknowledged_at=ts, acknowledged_by=user)
        return False

    def acknowledge_all(self, ts, user: str = "operator") -> int:
        return sum(self.acknowledge(a["id"], ts, user) for a in list(self._open.values()))

    def reset(self, alarm_id: int, ts, user: str = "operator") -> bool:
        """Only alarms whose condition has CLEARED can be reset."""
        for key, a in list(self._open.items()):
            if a["id"] == alarm_id and a["status"] == "cleared":
                row = self.db.get_alarms(pump_id=self.pump_id)
                acked = row.loc[row.id == alarm_id, "acknowledged_at"].notna().any()
                fields = dict(status="reset", reset_at=ts)
                if not acked:
                    fields.update(acknowledged_at=ts, acknowledged_by=user)
                self.db.update_alarm(alarm_id, **fields)
                del self._open[key]
                return True
        return False

    def reset_all_cleared(self, ts, user: str = "operator") -> int:
        return sum(self.reset(a["id"], ts, user) for a in list(self._open.values()) if a["status"] == "cleared")

    # ------------------------------------------------------------------ views
    def active(self) -> pd.DataFrame:
        df = self.db.get_alarms(statuses=OPEN_STATUSES, pump_id=self.pump_id)
        if df.empty:
            return df
        df["rank"] = df.priority.map(PRIORITY_RANK)
        return df.sort_values(["rank", "raised_at"], ascending=[False, False]).drop(columns="rank")

    def counts(self) -> dict:
        act = self.active()
        return {p: int(((act.priority == p) & (act.status != "cleared")).sum()) if not act.empty else 0
                for p in (CRITICAL, WARNING, INFO)} | {"unacknowledged": int((act.status == "active").sum())
                                                       if not act.empty else 0}

    def history(self, start=None, end=None) -> pd.DataFrame:
        return self.db.get_alarms(start=start, end=end, pump_id=self.pump_id)
