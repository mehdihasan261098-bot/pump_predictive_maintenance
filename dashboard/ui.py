"""Shared dashboard helpers: engine access, styling, small widgets."""
from __future__ import annotations

import math
import os

import plotly.graph_objects as go
import streamlit as st

from dashboard.engine import SimulationEngine
from simulation.sensor_simulator import SENSORS

LEVEL_COLORS = {"Normal": "#2EAD5B", "Healthy": "#2EAD5B", "Info": "#2E9BFF", "Caution": "#E0B000",
                "Warning": "#F08C00", "Degraded": "#F08C00", "Critical": "#E03131", "Routine": "#2EAD5B",
                "Planned": "#2E9BFF", "Urgent": "#F08C00", "Immediate": "#E03131"}
STATE_COLORS = {"Running": "#2EAD5B", "Starting": "#2E9BFF", "Shutdown": "#E0B000", "Stopped": "#8A94A6",
                "Emergency stop": "#E03131"}
SENSOR_LABELS = {
    "bearing_temp_c": "Bearing temperature", "casing_temp_c": "Casing temperature",
    "vibration_rms_mm_s": "Vibration velocity RMS", "vibration_accel_m_s2": "HF acceleration RMS",
    "suction_pressure_bar": "Suction pressure", "discharge_pressure_bar": "Discharge pressure",
    "flow_m3h": "Flow rate", "motor_current_a": "Motor current", "motor_power_kw": "Motor power",
    "speed_rpm": "Speed",
}
UNITS = {k: v[1].replace("degC", "°C").replace("m3/h", "m³/h").replace("m/s2", "m/s²") for k, v in SENSORS.items()}

CSS = """
<style>
.block-container {padding-top: 1.2rem;}
.card {background:#17222F; border:1px solid #26364A; border-radius:8px; padding:10px 14px; height:100%;}
.card .lbl {font-size:0.78rem; color:#8A94A6; text-transform:uppercase; letter-spacing:.04em;}
.card .val {font-size:1.3rem; font-weight:600; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.card .sub {font-size:0.8rem; color:#B8C2CF;}
.pill {display:inline-block; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.85rem; color:#fff;}
.banner {border-radius:6px; padding:8px 14px; margin-bottom:10px; font-weight:600; color:#fff;}
</style>
"""


def demo_mode() -> bool:
    """Public demo (e.g. Streamlit Community Cloud): set PUMP_DEMO_MODE = "1" as an
    environment variable or in the app's secrets."""
    if os.environ.get("PUMP_DEMO_MODE", "").strip() in ("1", "true", "True"):
        return True
    try:
        return str(st.secrets.get("PUMP_DEMO_MODE", "")).strip() in ("1", "true", "True")
    except Exception:            # no secrets file
        return False


@st.cache_resource
def _shared_ml():
    """ML models loaded once per server and shared by all sessions (read-only)."""
    from detection.ml_detection import MLDetector
    return MLDetector() if MLDetector.available() else None


@st.cache_resource
def _shared_engine() -> SimulationEngine:
    """Normal mode: one engine per server process (shared like a real SCADA server).
    PUMP_DB_PATH overrides the database file (used by tests)."""
    return SimulationEngine(db_path=os.environ.get("PUMP_DB_PATH"))


def get_engine() -> SimulationEngine:
    """Demo mode: every visitor gets a PRIVATE simulation with an in-memory
    database (one visitor's E-stop or fault never affects another), pre-filled
    with 10 minutes of healthy history."""
    if not demo_mode():
        return _shared_engine()
    if "engine" not in st.session_state:
        eng = SimulationEngine(db_path=":memory:", ml_detector=_shared_ml())
        eng.advance(600)
        st.session_state["engine"] = eng
    return st.session_state["engine"]


get_engine.clear = _shared_engine.clear      # used by tests


def setup_page():
    st.markdown(CSS, unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "", color: str | None = None):
    style = f"color:{color};" if color else ""
    st.markdown(f"<div class='card'><div class='lbl'>{label}</div><div class='val' style='{style}'>{value}</div>"
                f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def pill(text: str, color: str | None = None) -> str:
    return f"<span class='pill' style='background:{color or LEVEL_COLORS.get(text, '#8A94A6')}'>{text}</span>"


def fmt(v, digits=2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:,.{digits}f}"


def alarm_banner(engine: SimulationEngine):
    c = engine.alarms.counts()
    if c["Critical"]:
        st.markdown(f"<div class='banner' style='background:#E03131'>⚠ {c['Critical']} CRITICAL alarm(s) active"
                    f" — {c['unacknowledged']} unacknowledged</div>", unsafe_allow_html=True)
    elif c["Warning"]:
        st.markdown(f"<div class='banner' style='background:#F08C00'>⚠ {c['Warning']} warning alarm(s) active"
                    f" — {c['unacknowledged']} unacknowledged</div>", unsafe_allow_html=True)


def gauge(value: float, title: str, bands=(40, 65, 85), height: int = 220) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={"text": title, "font": {"size": 14}},
        number={"suffix": "", "font": {"size": 34}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#E6EDF3", "thickness": 0.25},
               "steps": [{"range": [0, bands[0]], "color": "#5C1F22"}, {"range": [bands[0], bands[1]], "color": "#5C3A12"},
                         {"range": [bands[1], bands[2]], "color": "#4D4512"}, {"range": [bands[2], 100], "color": "#1D4A2E"}]}))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=40, b=10), template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def style_fig(fig: go.Figure, height: int = 320, title: str | None = None) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=50, r=20, t=40 if title else 20, b=40),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111A24", title=title,
                      legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified")
    return fig


def live(engine: SimulationEngine, body):
    """Run `body(engine)` as an auto-refreshing fragment when live mode is on."""
    interval = st.session_state.get("refresh_s", 2) if st.session_state.get("live", True) else None

    @st.fragment(run_every=interval)
    def _frag():
        engine.tick()
        body(engine)

    _frag()
