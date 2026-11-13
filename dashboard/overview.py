"""Page 1 — Equipment overview: status, all sensors, KPIs, health, active alarms."""
from __future__ import annotations

import streamlit as st

from dashboard.ui import (LEVEL_COLORS, SENSOR_LABELS, STATE_COLORS, UNITS, alarm_banner, card, fmt, gauge,
                          get_engine, live)
from simulation.sensor_simulator import SENSOR_NAMES


def _body(engine):
    alarm_banner(engine)
    rec = engine.latest() or {}
    hrec, diag, ml = engine.latest_health, engine.latest_diagnosis, engine.latest_ml
    counts = engine.alarms.counts()

    c = st.columns([1.2, 1, 1, 1, 1.2])
    with c[0]:
        st.plotly_chart(gauge(hrec.health_index if hrec else 100.0, "Health index"), width="stretch",
                        key="ov_gauge")
    with c[1]:
        card("Equipment state", engine.state, f"speed cmd {engine.sim.pump.speed_command_rpm:.0f} rpm · "
             f"valve {engine.sim.base_valve * 100:.0f} %", STATE_COLORS.get(engine.state))
    with c[2]:
        cond = hrec.condition if hrec else "Healthy"
        card("Condition", cond, f"urgency: {hrec.urgency if hrec else 'Routine'}", LEVEL_COLORS.get(cond))
    with c[3]:
        prim = diag.primary_fault if diag and diag.running else "healthy"
        lvl = diag.overall_level if diag else "Normal"
        card("Rule diagnosis", prim.replace("_", " ").capitalize(),
             f"ML: {ml['predicted_fault'].replace('_', ' ') if ml else 'n/a'}", LEVEL_COLORS.get(lvl))
    with c[4]:
        overall = "CRITICAL" if counts["Critical"] else "WARNING" if counts["Warning"] else "NORMAL"
        col = LEVEL_COLORS["Critical" if counts["Critical"] else "Warning" if counts["Warning"] else "Normal"]
        card("Overall system status", overall, f"alarms: {counts['Critical']} critical · {counts['Warning']} warning · "
             f"{counts['Info']} info · {counts['unacknowledged']} unack.", col)

    st.markdown("#### Sensor readings")
    prev = engine.recent(2.0)
    prev = prev.iloc[0].to_dict() if len(prev) > 60 else {}
    cols = st.columns(5)
    for i, s in enumerate(SENSOR_NAMES):
        v, p = rec.get(s), prev.get(s)
        delta = None if v is None or p is None or v != v or p != p else f"{v - p:+.2f} /2 min"
        cols[i % 5].metric(f"{SENSOR_LABELS[s]} [{UNITS[s]}]", fmt(v, 0 if s == "speed_rpm" else 2), delta,
                           delta_color="off")

    st.markdown("#### Performance KPIs")
    k = st.columns(6)
    k[0].metric("Head [m]", fmt(rec.get("head_m"), 1))
    k[1].metric("Hydraulic power [kW]", fmt(rec.get("hydraulic_power_kw"), 2))
    k[2].metric("Pump efficiency (est.) [%]", fmt(rec.get("pump_eff_est_pct"), 1))
    k[3].metric("Specific energy [kWh/m³]", fmt(rec.get("sec_kwh_m3"), 3))
    k[4].metric("Flow / BEP flow [%]", fmt(100 * rec["bep_ratio"] if rec.get("bep_ratio") == rec.get("bep_ratio")
                                           and rec.get("bep_ratio") is not None else None, 0))
    k[5].metric("NPSH margin [-]", fmt(rec.get("npsh_margin"), 1))

    st.markdown("#### Active alarms")
    act = engine.alarms.active()
    if act.empty:
        st.success("No active alarms.")
        return
    show = act[["id", "priority", "status", "message", "raised_at", "occurrences", "acknowledged_by"]].copy()
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"raised_at": st.column_config.DatetimeColumn("raised", format="HH:mm:ss")})


def _alarm_actions(engine):
    act = engine.alarms.active()
    if act.empty:
        return
    a, b, c, d = st.columns([2, 1, 1, 1])
    sel = a.selectbox("Alarm", act.id.tolist(), format_func=lambda i: f"#{i} {act.set_index('id').loc[i, 'message'][:60]}",
                      label_visibility="collapsed")
    if b.button("Acknowledge", width="stretch"):
        engine.alarms.acknowledge(int(sel), engine.now)
        st.rerun()
    if c.button("Ack all", width="stretch"):
        engine.alarms.acknowledge_all(engine.now)
        st.rerun()
    if d.button("Reset cleared", width="stretch"):
        n = engine.alarms.reset_all_cleared(engine.now)
        st.toast(f"{n} cleared alarm(s) reset")
        st.rerun()
    st.caption("Alarms latch until their condition has cleared AND they are reset. Duplicates are suppressed "
               "(one open alarm per key, re-occurrences counted).")


def render():
    engine = get_engine()
    st.title("Equipment Overview")
    live(engine, _body)
    _alarm_actions(engine)
