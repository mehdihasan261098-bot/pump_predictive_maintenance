"""
SQLite persistence layer.

Tables
  measurements         timestamped sensor values + operating condition + QC + KPIs
                       UNIQUE(pump_id, timestamp) -> duplicate inserts are ignored
  operating_status     state transitions only (Stopped/Starting/Running/...)
  diagnoses            fault-detection results (rule-based and ML)
  health_history       health index time series (UNIQUE per timestamp)
  alarms               alarm lifecycle: raised -> acknowledged -> cleared -> reset
  maintenance_records  planned / completed maintenance work

Timestamps are stored as fixed-width ISO text 'YYYY-MM-DD HH:MM:SS.ffffff' so
that string comparison equals chronological comparison (date filters use the
index). NaN values are stored as NULL. Enumerated fields are protected with
CHECK constraints; update helpers only accept existing column names.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
from pathlib import Path

import pandas as pd

from config.pump_config import get_config, project_path
from simulation.sensor_simulator import SENSOR_NAMES

DEFAULT_PUMP_ID = "P-101"
TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

KPI_COLUMNS = ["head_m", "hydraulic_power_kw", "wire_to_water_eff_pct", "pump_eff_est_pct", "sec_kwh_m3", "system_resistance_ratio",
               "bep_ratio", "flow_deficit_m3h", "head_deviation_pct", "npsh_available_m", "npsh_margin"]
MEASUREMENT_COLUMNS = (["time_s", *SENSOR_NAMES, "state", "speed_command_rpm", "valve_opening",
                        "fault_label", "fault_severity", "active_faults", "qc_flags", "qc_issue_count"]
                       + KPI_COLUMNS)
_TEXT_COLUMNS = {"state", "fault_label", "active_faults", "qc_flags"}

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pump_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    {", ".join(f"{c} {'TEXT' if c in _TEXT_COLUMNS else 'REAL'}" for c in MEASUREMENT_COLUMNS)},
    UNIQUE (pump_id, timestamp)
);
CREATE INDEX IF NOT EXISTS ix_meas_ts ON measurements (pump_id, timestamp);

CREATE TABLE IF NOT EXISTS operating_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pump_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('Stopped','Starting','Running','Shutdown','Emergency stop')),
    previous_state TEXT,
    note TEXT,
    UNIQUE (pump_id, timestamp, state)
);

CREATE TABLE IF NOT EXISTS diagnoses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pump_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    method TEXT NOT NULL CHECK (method IN ('rule','ml_anomaly','ml_classifier')),
    fault TEXT NOT NULL,
    level TEXT NOT NULL CHECK (level IN ('Normal','Warning','Critical')),
    confidence REAL,
    evidence TEXT,
    reason TEXT,
    recommendation TEXT,
    UNIQUE (pump_id, timestamp, method, fault)
);
CREATE INDEX IF NOT EXISTS ix_diag_ts ON diagnoses (pump_id, timestamp);

CREATE TABLE IF NOT EXISTS health_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pump_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    health_index REAL NOT NULL CHECK (health_index BETWEEN 0 AND 100),
    condition TEXT NOT NULL CHECK (condition IN ('Healthy','Caution','Degraded','Critical')),
    rul_hours REAL,
    components TEXT,
    UNIQUE (pump_id, timestamp)
);

CREATE TABLE IF NOT EXISTS alarms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pump_id TEXT NOT NULL,
    alarm_key TEXT NOT NULL,
    priority TEXT NOT NULL CHECK (priority IN ('Info','Warning','Critical')),
    message TEXT NOT NULL,
    value REAL,
    threshold REAL,
    raised_at TEXT NOT NULL,
    acknowledged_at TEXT,
    acknowledged_by TEXT,
    cleared_at TEXT,
    reset_at TEXT,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active','acknowledged','cleared','reset')),
    occurrences INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_alarm_status ON alarms (pump_id, status);

CREATE TABLE IF NOT EXISTS maintenance_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pump_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    component TEXT NOT NULL,
    action TEXT NOT NULL,
    urgency TEXT NOT NULL CHECK (urgency IN ('Routine','Planned','Urgent','Immediate')),
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','scheduled','completed','cancelled')),
    scheduled_for TEXT,
    completed_at TEXT,
    technician TEXT,
    notes TEXT,
    cost REAL
);
"""

TABLES = ("measurements", "operating_status", "diagnoses", "health_history", "alarms", "maintenance_records")


def ts(value) -> str | None:
    """Normalise any timestamp-like value to the stored text format."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return pd.Timestamp(value).strftime(TS_FORMAT)


def _clean(v):
    """NaN/NaT -> None, numpy scalars -> Python."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):
        v = v.item()
        if isinstance(v, float) and math.isnan(v):
            return None
    return v


class DatabaseManager:
    """Thread-safe (single connection + lock) SQLite access for the app."""

    def __init__(self, path: str | Path | None = None):
        self.path = str(path or get_config().database_path)
        if self.path != ":memory:":
            self.path = str(project_path(self.path))
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            if self.path != ":memory:":
                self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            # migration for databases created before the 'occurrences' column existed
            alarm_cols = [r["name"] for r in self.conn.execute("PRAGMA table_info(alarms)")]
            if "occurrences" not in alarm_cols:
                self.conn.execute("ALTER TABLE alarms ADD COLUMN occurrences INTEGER NOT NULL DEFAULT 1")
            meas_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(measurements)")}
            for c in MEASUREMENT_COLUMNS:
                if c not in meas_cols:      # columns added in later versions
                    self.conn.execute(f"ALTER TABLE measurements ADD COLUMN {c} "
                                      f"{'TEXT' if c in _TEXT_COLUMNS else 'REAL'}")
            self.conn.commit()
        self._columns = {t: [r["name"] for r in self.conn.execute(f"PRAGMA table_info({t})")] for t in TABLES}

    # -- lifecycle ---------------------------------------------------------
    def close(self):
        with self._lock:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- generic helpers ---------------------------------------------------
    def _execute(self, sql: str, params=()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def query(self, sql: str, params=()) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql_query(sql, self.conn, params=params)

    def _insert(self, table: str, row: dict, ignore: bool = False) -> int | None:
        cols = [c for c in row if c in self._columns[table]]
        # ON CONFLICT DO NOTHING ignores only UNIQUE conflicts; unlike
        # INSERT OR IGNORE it still raises on CHECK / NOT NULL violations.
        sql = (f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
               f"{' ON CONFLICT DO NOTHING' if ignore else ''}")
        cur = self._execute(sql, [_clean(row[c]) for c in cols])
        return cur.lastrowid if cur.rowcount else None

    def _update(self, table: str, row_id: int, fields: dict) -> bool:
        bad = set(fields) - set(self._columns[table]) | ({"id"} & set(fields))
        if bad:
            raise ValueError(f"invalid column(s) for {table}: {sorted(bad)}")
        sets = ", ".join(f"{c} = ?" for c in fields)
        cur = self._execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*map(_clean, fields.values()), row_id])
        return cur.rowcount == 1

    def _select(self, table: str, time_col: str, pump_id: str | None, start, end,
                where: str = "", params: tuple = (), order: str = "ASC", limit: int | None = None,
                columns: list[str] | None = None) -> pd.DataFrame:
        cols = "*"
        if columns:
            bad = set(columns) - set(self._columns[table])
            if bad:
                raise ValueError(f"invalid column(s) for {table}: {sorted(bad)}")
            cols = ", ".join(dict.fromkeys(["timestamp" if time_col == "timestamp" else time_col, *columns]))
        clauses, args = [], []
        if pump_id is not None:
            clauses.append("pump_id = ?")
            args.append(pump_id)
        if start is not None:
            clauses.append(f"{time_col} >= ?")
            args.append(ts(start))
        if end is not None:
            clauses.append(f"{time_col} <= ?")
            args.append(ts(end))
        if where:
            clauses.append(where)
            args.extend(params)
        sql = f"SELECT {cols} FROM {table}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += f" ORDER BY {time_col} {order}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        df = self.query(sql, args)
        for c in df.columns:
            if c in ("timestamp", "raised_at", "acknowledged_at", "cleared_at", "reset_at",
                     "created_at", "scheduled_for", "completed_at"):
                df[c] = pd.to_datetime(df[c], format=TS_FORMAT, errors="coerce")
        return df

    # -- measurements --------------------------------------------------------
    def insert_measurements(self, df: pd.DataFrame, pump_id: str = DEFAULT_PUMP_ID) -> int:
        """Bulk insert; rows with an existing (pump_id, timestamp) are ignored.
        Returns the number of rows actually inserted."""
        if df is None or df.empty:
            return 0
        cols = [c for c in MEASUREMENT_COLUMNS if c in df.columns]
        sql = (f"INSERT INTO measurements (pump_id, timestamp, {', '.join(cols)}) "
               f"VALUES (?, ?, {', '.join('?' * len(cols))}) ON CONFLICT DO NOTHING")
        stamps = pd.to_datetime(df["timestamp"]).dt.strftime(TS_FORMAT).tolist()
        values = df[cols].astype(object).where(df[cols].notna(), None).to_numpy().tolist()
        rows = [(pump_id, t, *map(_clean, v)) for t, v in zip(stamps, values)]
        with self._lock:
            before = self.conn.total_changes
            self.conn.executemany(sql, rows)
            self.conn.commit()
            return self.conn.total_changes - before

    def as_sink(self, pump_id: str = DEFAULT_PUMP_ID):
        """Callable for DataCollector.sinks: stores every acquired batch and
        logs operating-state transitions."""
        def sink(batch: pd.DataFrame):
            self.insert_measurements(batch, pump_id)
            for t, state in zip(batch["timestamp"], batch["state"]):
                self.log_status(t, state, pump_id=pump_id)
        return sink

    def get_measurements(self, start=None, end=None, pump_id: str = DEFAULT_PUMP_ID,
                         columns: list[str] | None = None, limit: int | None = None,
                         latest_first: bool = False) -> pd.DataFrame:
        return self._select("measurements", "timestamp", pump_id, start, end, columns=columns,
                            limit=limit, order="DESC" if latest_first else "ASC")

    # -- operating status ----------------------------------------------------
    def log_status(self, timestamp, state: str, note: str = "", pump_id: str = DEFAULT_PUMP_ID) -> int | None:
        """Insert only when the state differs from the state in force at that
        time (replaying the same history therefore adds nothing)."""
        t = ts(timestamp)
        with self._lock:
            last = self.conn.execute("SELECT state FROM operating_status WHERE pump_id = ? AND timestamp <= ? "
                                     "ORDER BY timestamp DESC, id DESC LIMIT 1", (pump_id, t)).fetchone()
        prev = last["state"] if last else None
        if prev == state:
            return None
        return self._insert("operating_status", {"pump_id": pump_id, "timestamp": t, "state": state,
                                                 "previous_state": prev, "note": note}, ignore=True)

    def get_status_history(self, start=None, end=None, pump_id: str = DEFAULT_PUMP_ID) -> pd.DataFrame:
        return self._select("operating_status", "timestamp", pump_id, start, end)

    # -- diagnoses -----------------------------------------------------------
    def insert_diagnosis(self, timestamp, method: str, fault: str, level: str, confidence: float | None = None,
                         evidence: dict | str | None = None, reason: str = "", recommendation: str = "",
                         pump_id: str = DEFAULT_PUMP_ID) -> int | None:
        if isinstance(evidence, dict):
            evidence = json.dumps(evidence, default=float)
        return self._insert("diagnoses", dict(pump_id=pump_id, timestamp=ts(timestamp), method=method, fault=fault,
                                              level=level, confidence=confidence, evidence=evidence, reason=reason,
                                              recommendation=recommendation), ignore=True)

    def get_diagnoses(self, start=None, end=None, method: str | None = None,
                      pump_id: str = DEFAULT_PUMP_ID) -> pd.DataFrame:
        return self._select("diagnoses", "timestamp", pump_id, start, end,
                            where="method = ?" if method else "", params=(method,) if method else ())

    # -- health ---------------------------------------------------------------
    def insert_health(self, timestamp, health_index: float, condition: str, rul_hours: float | None = None,
                      components: dict | None = None, pump_id: str = DEFAULT_PUMP_ID) -> int | None:
        return self._insert("health_history", dict(
            pump_id=pump_id, timestamp=ts(timestamp), health_index=float(health_index), condition=condition,
            rul_hours=rul_hours, components=json.dumps(components, default=float) if components else None),
            ignore=True)

    def get_health_history(self, start=None, end=None, pump_id: str = DEFAULT_PUMP_ID) -> pd.DataFrame:
        return self._select("health_history", "timestamp", pump_id, start, end)

    # -- alarms ---------------------------------------------------------------
    def insert_alarm(self, alarm_key: str, priority: str, message: str, raised_at, value: float | None = None,
                     threshold: float | None = None, pump_id: str = DEFAULT_PUMP_ID) -> int:
        return self._insert("alarms", dict(pump_id=pump_id, alarm_key=alarm_key, priority=priority, message=message,
                                           value=value, threshold=threshold, raised_at=ts(raised_at)))

    def update_alarm(self, alarm_id: int, **fields) -> bool:
        for k in ("acknowledged_at", "cleared_at", "reset_at"):
            if k in fields:
                fields[k] = ts(fields[k])
        return self._update("alarms", alarm_id, fields)

    def get_alarms(self, start=None, end=None, statuses: tuple[str, ...] | None = None,
                   pump_id: str = DEFAULT_PUMP_ID) -> pd.DataFrame:
        where, params = "", ()
        if statuses:
            where = f"status IN ({', '.join('?' * len(statuses))})"
            params = tuple(statuses)
        return self._select("alarms", "raised_at", pump_id, start, end, where=where, params=params, order="DESC")

    # -- maintenance ------------------------------------------------------------
    def add_maintenance(self, component: str, action: str, urgency: str, created_at=None, scheduled_for=None,
                        status: str = "open", technician: str = "", notes: str = "", cost: float | None = None,
                        pump_id: str = DEFAULT_PUMP_ID) -> int:
        return self._insert("maintenance_records", dict(
            pump_id=pump_id, created_at=ts(created_at or pd.Timestamp.now()), component=component, action=action,
            urgency=urgency, status=status, scheduled_for=ts(scheduled_for), technician=technician, notes=notes,
            cost=cost))

    def update_maintenance(self, record_id: int, **fields) -> bool:
        for k in ("scheduled_for", "completed_at"):
            if k in fields:
                fields[k] = ts(fields[k])
        return self._update("maintenance_records", record_id, fields)

    def get_maintenance(self, start=None, end=None, status: str | None = None,
                        pump_id: str = DEFAULT_PUMP_ID) -> pd.DataFrame:
        return self._select("maintenance_records", "created_at", pump_id, start, end,
                            where="status = ?" if status else "", params=(status,) if status else (), order="DESC")

    # -- export / summary --------------------------------------------------------
    def export_csv(self, table: str, path: str | Path, start=None, end=None,
                   pump_id: str = DEFAULT_PUMP_ID) -> Path:
        getters = {"measurements": self.get_measurements, "operating_status": self.get_status_history,
                   "diagnoses": self.get_diagnoses, "health_history": self.get_health_history,
                   "alarms": self.get_alarms, "maintenance_records": self.get_maintenance}
        if table not in getters:
            raise ValueError(f"unknown table '{table}'")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        getters[table](start=start, end=end, pump_id=pump_id).to_csv(path, index=False)
        return path

    def counts(self) -> dict[str, int]:
        return {t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}


if __name__ == "__main__":
    from acquisition.data_collector import DataCollector, SimulatedSource
    from simulation.fault_simulator import FaultScenario, FaultSimulator, FaultType
    from simulation.sensor_simulator import OperatingProfile, SensorSimulator

    db = DatabaseManager("data/demo_database.db")   # separate from the dashboard database
    sim = SensorSimulator(seed=21, dropout_probability=0.002)
    faults = FaultSimulator([FaultScenario(FaultType.BEARING_DEGRADATION, onset_s=600, ramp_s=2400,
                                           shape="exponential")])
    col = DataCollector(SimulatedSource(sim, faults, OperatingProfile.standard(3600)))
    col.sinks.append(db.as_sink())
    for t in range(300, 3601, 300):             # acquire in 5-min batches
        col.acquire_to_time(t)
    print("first run, rows inserted :", db.counts()["measurements"])
    print("re-insert same data      :", db.insert_measurements(col.data), "rows inserted (duplicates ignored)")

    last = col.latest()
    db.insert_diagnosis(last["timestamp"], "rule", "bearing_degradation", "Warning", 0.8,
                        {"vibration_accel_m_s2": last["vibration_accel_m_s2"]},
                        "HF acceleration above warning limit", "Plan bearing inspection")
    db.insert_health(last["timestamp"], 62.5, "Degraded", None, {"vibration": 55.0, "thermal": 80.0})
    aid = db.insert_alarm("accel_high", "Warning", "Vibration acceleration high", last["timestamp"],
                          last["vibration_accel_m_s2"], 2.0)
    db.update_alarm(aid, status="acknowledged", acknowledged_at=last["timestamp"], acknowledged_by="operator")
    mid = db.add_maintenance("DE bearing", "Inspect / replace bearing", "Planned", created_at=last["timestamp"])
    db.update_maintenance(mid, status="scheduled", scheduled_for=pd.Timestamp(last["timestamp"]) + pd.to_timedelta(2, unit="D"))

    window = db.get_measurements(start="2026-01-01 08:30:00", end="2026-01-01 08:35:00",
                                 columns=["vibration_accel_m_s2", "bearing_temp_c", "fault_label"])
    print("date-filtered rows 08:30-08:35:", len(window), "| labels:", window.fault_label.unique().tolist())
    print("status transitions:", db.get_status_history()[["timestamp", "previous_state", "state"]]
          .astype(str).values.tolist())
    print("alarm:", db.get_alarms().iloc[0][["alarm_key", "status", "acknowledged_by"]].to_dict())
    print("maintenance:", db.get_maintenance().iloc[0][["component", "status", "scheduled_for"]].astype(str).to_dict())
    print("NULL flow values stored  :", int(db.query("SELECT COUNT(*) n FROM measurements WHERE flow_m3h IS NULL").n[0]))
    print("export:", db.export_csv("measurements", "reports/measurements_export.csv",
                                   start="2026-01-01 08:50:00"))
    print("table counts:", db.counts())
    db.close()
