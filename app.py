"""
Centrifugal Pump Monitoring & Predictive Maintenance — SCADA-style dashboard.

Run:  streamlit run app.py
All equipment is SIMULATED. The emergency stop is not a safety function.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Pump PdM SCADA", page_icon="⚙️", layout="wide")

from dashboard import fault_analysis, historical_data, maintenance, overview, realtime, vibration  # noqa: E402
from dashboard.ui import STATE_COLORS, get_engine, pill, setup_page  # noqa: E402
from simulation.fault_simulator import FAULT_TYPES  # noqa: E402

engine = get_engine()
setup_page()


def sidebar():
    sb = st.sidebar
    sb.markdown("### ⚙️ P-101 Centrifugal Pump")
    sb.markdown(f"{pill(engine.state, STATE_COLORS.get(engine.state))} &nbsp; "
                f"<small>sim time {engine.now:%Y-%m-%d %H:%M:%S}</small>", unsafe_allow_html=True)

    sb.markdown("**Operation**")
    c1, c2, c3 = sb.columns(3)
    if c1.button("▶ Start", width="stretch"):
        engine.start()
    if c2.button("■ Stop", width="stretch"):
        engine.stop()
    if c3.button("E-STOP", type="primary", width="stretch"):
        engine.emergency_stop()
    if engine.state == "Emergency stop":
        if sb.button("Reset emergency stop", width="stretch"):
            if not engine.reset_emergency_stop():
                sb.warning("Shaft still turning — wait until speed is zero.")
    sb.caption("Simulated E-stop — NOT safety-rated, never use as a real safety function.")

    sb.slider("Speed command [rpm]", 900, 1500, int(engine.sim.pump.speed_command_rpm), 25, key="speed_cmd",
              on_change=lambda: engine.set_speed(st.session_state.speed_cmd))
    sb.slider("Discharge valve opening [%]", 20, 100, int(round(engine.sim.base_valve * 100)), 5, key="valve_cmd",
              on_change=lambda: engine.set_valve(st.session_state.valve_cmd / 100))

    with sb.expander("Fault injection", expanded=False):
        names = [f.value for f in FAULT_TYPES]
        fault = st.selectbox("Fault", names, format_func=lambda s: s.replace("_", " ").capitalize())
        sev = st.slider("Target severity", 0.0, 1.0, 0.6, 0.05)
        mode = st.radio("Mode", ["Instant", "Gradual"], horizontal=True)
        ramp = st.number_input("Ramp duration [sim min]", 1, 600, 30) if mode == "Gradual" else 0
        a, b = st.columns(2)
        if a.button("Apply", width="stretch"):
            if mode == "Instant":
                engine.set_fault(fault, sev)
            else:
                engine.ramp_fault(fault, sev, ramp * 60.0)
        if b.button("Clear all", width="stretch"):
            engine.clear_faults()
        active = {k: v for k, v in engine.fault_severities().items() if v > 0}
        st.caption("Active: " + (", ".join(f"{k.replace('_', ' ')} {v:.2f}" for k, v in active.items())
                                 if active else "none (healthy)"))

    with sb.expander("Simulation", expanded=False):
        engine.speedup = float(st.select_slider("Speed-up (sim s per wall s)", [1, 5, 10, 30, 60, 120, 300],
                                                value=int(engine.speedup)))
        st.toggle("Live update", value=True, key="live")
        st.select_slider("Update interval [s]", [1, 2, 3, 5, 10], value=2, key="refresh_s")
        engine.paused = st.toggle("Pause simulation", value=engine.paused)
        a, b = st.columns(2)
        if a.button("+5 min", width="stretch"):
            engine.advance(300)
        if b.button("+30 min", width="stretch"):
            with st.spinner("Simulating 30 min…"):
                engine.advance(1800)


sidebar()
pages = st.navigation([
    st.Page(overview.render, title="Equipment Overview", icon="🏭", url_path="overview", default=True),
    st.Page(realtime.render, title="Real-Time Monitoring", icon="📈", url_path="realtime"),
    st.Page(vibration.render, title="Vibration Analysis", icon="〰️", url_path="vibration"),
    st.Page(fault_analysis.render, title="Fault Detection", icon="🔍", url_path="faults"),
    st.Page(maintenance.render, title="Predictive Maintenance", icon="🛠️", url_path="maintenance"),
    st.Page(historical_data.render, title="Historical Data & Reports", icon="🗄️", url_path="history"),
])
pages.run()
