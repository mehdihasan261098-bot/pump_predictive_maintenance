"""Page 5 — Predictive maintenance: health index, trends, RUL, recommendations, work orders."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.ui import LEVEL_COLORS, alarm_banner, card, gauge, get_engine, live, style_fig


def _body(engine):
    alarm_banner(engine)
    h = engine.latest_health
    if h is None:
        st.info("Waiting for the first 60 s analysis window while the pump is running…")
        return
    c = st.columns([1.3, 1, 1, 1, 1])
    c[0].plotly_chart(gauge(h.health_index, "Health index (0–100)"), width="stretch", key="pm_gauge")
    with c[1]:
        card("Condition", h.condition, f"raw HI {h.health_index_raw:.1f}", LEVEL_COLORS.get(h.condition))
    with c[2]:
        card("Maintenance urgency", h.urgency, f"suggested by {h.suggested_date:%Y-%m-%d %H:%M}",
             LEVEL_COLORS.get(h.urgency))
    with c[3]:
        rul = "—" if h.rul_hours is None else (f"{h.rul_hours * 60:.0f} min" if h.rul_hours < 2 else f"{h.rul_hours:.1f} h")
        card("Simulated RUL", rul, f"to HI 40 · {h.rul_model}")
    with c[4]:
        tr = "—" if h.trend_per_hour is None else f"{h.trend_per_hour:+.1f} /h"
        card("Degradation trend", tr, f"over last {engine.cfg.health.trend_windows} windows")

    left, right = st.columns([1.6, 1])
    hist = engine.db.get_health_history(start=engine.now - pd.to_timedelta(6, unit="h"))
    fig = go.Figure()
    for lo, hi_, col in ((0, 40, "rgba(224,49,49,0.12)"), (40, 65, "rgba(240,140,0,0.12)"),
                         (65, 85, "rgba(224,176,0,0.10)"), (85, 100, "rgba(46,173,91,0.10)")):
        fig.add_hrect(y0=lo, y1=hi_, fillcolor=col, line_width=0)
    if not hist.empty:
        fig.add_trace(go.Scatter(x=hist.timestamp, y=hist.health_index, name="health index", line=dict(width=2)))
    fig.update_yaxes(range=[0, 101], title="HI")
    left.plotly_chart(style_fig(fig, 320, "Health index history (bands: Healthy ≥85, Caution ≥65, Degraded ≥40)"),
                      width="stretch", key="pm_hist")
    comp = pd.Series(h.components)
    bar = go.Figure(go.Bar(x=comp.values, y=comp.index, orientation="h",
                           marker_color=[LEVEL_COLORS[engine.health.condition(v)] for v in comp.values],
                           text=[f"{v:.0f}" for v in comp.values], textposition="auto"))
    bar.update_xaxes(range=[0, 100])
    right.plotly_chart(style_fig(bar, 320, "Component health"), width="stretch", key="pm_comp")

    st.markdown("#### Recommended maintenance actions")
    for r in h.recommendations:
        st.markdown(f"- {r}")
    worst = sorted(h.indicators.items(), key=lambda kv: -kv[1]["damage"])[:4]
    st.caption("Main contributors: " + " · ".join(f"{k} = {v['value']:.2f} (damage {v['damage']:.2f})"
                                                   for k, v in worst if v["damage"] > 0) or "none")
    st.caption("RUL is a SIMULATED trend extrapolation (linear or exponential fit of the recent health index to "
               "HI = 40), not a validated industrial prognosis. Health bands are configurable demonstration values.")


def _work_orders(engine):
    st.markdown("#### Maintenance records / work orders")
    mr = engine.db.get_maintenance()
    if mr.empty:
        st.caption("No maintenance records yet. Work orders are created automatically when urgency ≥ Planned.")
    else:
        st.dataframe(mr[["id", "created_at", "component", "action", "urgency", "status", "scheduled_for",
                         "completed_at", "technician", "cost", "notes"]], hide_index=True, width="stretch")
    open_ = mr[mr.status.isin(["open", "scheduled"])] if not mr.empty else mr
    c1, c2 = st.columns(2)
    with c1.form("complete_wo", clear_on_submit=True):
        st.markdown("**Complete a work order**")
        wo = st.selectbox("Work order", open_.id.tolist() if not open_.empty else [],
                          format_func=lambda i: f"#{i} {open_.set_index('id').loc[i, 'component']}")
        tech = st.text_input("Technician")
        cost = st.number_input("Cost", 0.0, 1e6, 0.0, 50.0)
        repaired = st.checkbox("Fault repaired (clear simulated faults)", value=True)
        if st.form_submit_button("Mark completed") and wo is not None:
            engine.db.update_maintenance(int(wo), status="completed", completed_at=engine.now, technician=tech, cost=cost)
            if repaired:
                engine.clear_faults()
            st.toast(f"Work order #{wo} completed")
    with c2.form("new_wo", clear_on_submit=True):
        st.markdown("**Add a maintenance record**")
        comp = st.text_input("Component", "Mechanical seal")
        action = st.text_input("Action", "Inspection")
        urg = st.selectbox("Urgency", ["Routine", "Planned", "Urgent", "Immediate"])
        when = st.date_input("Scheduled for", engine.now.date())
        if st.form_submit_button("Add record"):
            engine.db.add_maintenance(comp, action, urg, created_at=engine.now, scheduled_for=pd.Timestamp(when),
                                      status="scheduled", notes="manual entry")
            st.toast("Maintenance record added")


def render():
    engine = get_engine()
    st.title("Predictive Maintenance")
    live(engine, _body)
    _work_orders(engine)
