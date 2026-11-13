"""Page 2 — Real-time trends of all process sensors."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.ui import alarm_banner, get_engine, live, style_fig

PANELS = [
    ("Temperatures [°C]", [("bearing_temp_c", "Bearing"), ("casing_temp_c", "Casing")], None),
    ("Vibration", [("vibration_rms_mm_s", "Velocity RMS [mm/s]")], ("vibration_accel_m_s2", "HF accel [m/s²]")),
    ("Pressures [bar]", [("suction_pressure_bar", "Suction"), ("discharge_pressure_bar", "Discharge")], None),
    ("Flow [m³/h]", [("flow_m3h", "Flow")], None),
    ("Motor", [("motor_current_a", "Current [A]")], ("motor_power_kw", "Power [kW]")),
    ("Speed [rpm]", [("speed_rpm", "Speed")], None),
]


def _body(engine):
    alarm_banner(engine)
    minutes = st.session_state.get("rt_minutes", 15)
    df = engine.recent(minutes)
    if df.empty:
        st.info("No data yet — the simulation is starting.")
        return
    t = df.timestamp
    fig = make_subplots(rows=3, cols=2, subplot_titles=[p[0] for p in PANELS], vertical_spacing=0.1,
                        horizontal_spacing=0.07, specs=[[{"secondary_y": True}] * 2] * 3)
    for i, (title, prim, sec) in enumerate(PANELS):
        r, c = i // 2 + 1, i % 2 + 1
        for col, name in prim:
            fig.add_trace(go.Scattergl(x=t, y=df[col], name=name, mode="lines", line=dict(width=1.2)),
                          row=r, col=c, secondary_y=False)
        if sec:
            fig.add_trace(go.Scattergl(x=t, y=df[sec[0]], name=sec[1], mode="lines",
                                       line=dict(width=1.2, dash="dot")), row=r, col=c, secondary_y=True)
    style_fig(fig, height=820)
    fig.update_layout(hovermode="x", legend=dict(orientation="h", y=1.09, yanchor="bottom"),
                      margin=dict(t=90))
    st.plotly_chart(fig, width="stretch", key="rt_chart")
    last = df.iloc[-1]
    st.caption(f"{len(df)} samples · last {last.timestamp:%H:%M:%S} · state {last.state} · "
               f"valve {last.valve_opening * 100:.0f} % · speed cmd {last.speed_command_rpm:.0f} rpm · "
               f"QC issues in window: {int((df.qc_issue_count > 0).sum())}")


def render():
    engine = get_engine()
    st.title("Real-Time Monitoring")
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.selectbox("Time window [sim min]", [5, 15, 30, 60, 180, 360], index=1, key="rt_minutes")
    c2.metric("Simulation speed-up", f"{engine.speedup:.0f}×")
    c3.caption("Controls (start/stop, speed, valve, faults, speed-up, update interval) are in the sidebar. "
               "Each rerun advances the simulation to a wall-clock target time, so refreshes never duplicate "
               "records.")
    live(engine, _body)
