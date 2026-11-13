"""Page 6 — Historical data, performance, fault/alarm/maintenance history, exports and reports."""
from __future__ import annotations

from datetime import datetime, time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.hydraulic_calculations import performance_curve
from dashboard.ui import SENSOR_LABELS, get_engine, style_fig
from simulation.sensor_simulator import SENSOR_NAMES


def build_report(engine, meas: pd.DataFrame, diag: pd.DataFrame, alarms: pd.DataFrame, health: pd.DataFrame,
                 maint: pd.DataFrame, start, end) -> str:
    """Self-contained HTML summary report for the selected period."""
    run = meas[meas.state == "Running"] if not meas.empty else meas
    stats = run[SENSOR_NAMES + ["head_m", "pump_eff_est_pct", "sec_kwh_m3"]].describe().T[["mean", "min", "max"]] \
        if not run.empty else pd.DataFrame()
    energy = float((meas.motor_power_kw.fillna(0) / 3600).sum()) if not meas.empty else 0.0
    volume = float((meas.flow_m3h.fillna(0) / 3600).sum()) if not meas.empty else 0.0
    t = lambda df: df.to_html(index=False, border=0, classes="t") if not df.empty else "<p>none</p>"
    parts = [
        "<html><head><meta charset='utf-8'><title>Pump P-101 report</title><style>body{font-family:Arial;margin:24px;}"
        "table.t{border-collapse:collapse;font-size:12px}table.t td,table.t th{border:1px solid #ccc;padding:3px 6px}"
        "</style></head><body>",
        f"<h1>Pump P-101 — condition monitoring report</h1><p>Period: {start} — {end}<br>Generated: "
        f"{datetime.now():%Y-%m-%d %H:%M} · data: SIMULATED</p>",
        f"<h2>Operation</h2><p>Samples: {len(meas)} · running: {len(run)} · energy: {energy:.2f} kWh · "
        f"volume pumped: {volume:.1f} m³ · SEC: {energy / volume if volume else float('nan'):.3f} kWh/m³</p>",
        "<h2>Sensor and KPI statistics (running)</h2>", stats.round(3).to_html(border=0, classes="t") if not stats.empty else "",
        "<h2>Health</h2>",
        f"<p>Latest HI: {health.health_index.iloc[-1]:.1f} ({health.condition.iloc[-1]}) · min HI: "
        f"{health.health_index.min():.1f}</p>" if not health.empty else "<p>no health records</p>",
        "<h2>Diagnosed faults (rule-based)</h2>",
        t(diag[diag.method == "rule"].groupby(["fault", "level"]).size().reset_index(name="windows")) if not diag.empty else "<p>none</p>",
        "<h2>Alarms</h2>", t(alarms[["raised_at", "priority", "message", "status", "occurrences"]]) if not alarms.empty else "<p>none</p>",
        "<h2>Maintenance</h2>", t(maint[["created_at", "component", "action", "urgency", "status", "scheduled_for"]]) if not maint.empty else "<p>none</p>",
        "<p><i>Simulated data. Health thresholds and RUL are demonstration values, not certified industrial "
        "limits or validated prognoses.</i></p></body></html>",
    ]
    return "".join(parts)


def render():
    engine = get_engine()
    db = engine.db
    st.title("Historical Data & Reports")
    now = engine.now
    first = db.get_measurements(limit=1, columns=["time_s"])
    t_min = first.timestamp.iloc[0] if not first.empty else now
    c = st.columns(4)
    d0 = c[0].date_input("From date", max(t_min, now - pd.to_timedelta(2, unit="h")).date(), min_value=t_min.date())
    h0 = c[1].time_input("From time", (now - pd.to_timedelta(2, unit="h")).time().replace(microsecond=0)
                         if d0 == now.date() else time(0, 0), step=300)
    d1 = c[2].date_input("To date", now.date(), min_value=t_min.date())
    h1 = c[3].time_input("To time", time(23, 59), step=300)
    start, end = pd.Timestamp.combine(d0, h0), pd.Timestamp.combine(d1, h1)
    meas = db.get_measurements(start=start, end=end)
    diag = db.get_diagnoses(start=start, end=end)
    alarms = db.get_alarms(start=start, end=end)
    health = db.get_health_history(start=start, end=end)
    maint = db.get_maintenance()
    st.caption(f"{len(meas)} measurements · {len(diag)} diagnoses · {len(alarms)} alarms · "
               f"{len(health)} health records in the selected period")

    tabs = st.tabs(["Sensor trends", "Operating performance", "Fault history", "Alarm history", "Maintenance", "Export"])
    with tabs[0]:
        sel = st.multiselect("Sensors", SENSOR_NAMES, ["bearing_temp_c", "vibration_rms_mm_s", "flow_m3h"],
                             format_func=lambda s: SENSOR_LABELS[s])
        if meas.empty or not sel:
            st.info("No data in this period.")
        else:
            step = max(1, len(meas) // 5000)          # keep the browser responsive
            m = meas.iloc[::step]
            fig = go.Figure([go.Scattergl(x=m.timestamp, y=m[s], name=SENSOR_LABELS[s], mode="lines") for s in sel])
            st.plotly_chart(style_fig(fig, 420), width="stretch", key="hist_trend")
            if step > 1:
                st.caption(f"Displayed every {step}th sample ({len(m)} points).")
    with tabs[1]:
        run = meas[(meas.state == "Running")] if not meas.empty else meas
        if run.empty:
            st.info("No running data in this period.")
        else:
            curve = performance_curve(engine.cfg)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve.flow_m3h, y=curve.head_m, name="pump curve 1450 rpm", line=dict(color="#E6EDF3")))
            fig.add_trace(go.Scattergl(x=run.flow_m3h, y=run.head_m, mode="markers", name="measured",
                                       marker=dict(size=3, color=run.time_s, colorscale="Viridis", opacity=0.6)))
            fig.update_xaxes(title="Flow [m³/h]", range=[0, 45])
            fig.update_yaxes(title="Head [m]", range=[0, 35])
            a, b = st.columns(2)
            a.plotly_chart(style_fig(fig, 380, "H–Q: measured operating points vs healthy curve"), width="stretch", key="hist_hq")
            k = run[["head_m", "pump_eff_est_pct", "wire_to_water_eff_pct", "sec_kwh_m3", "flow_deficit_m3h",
                     "npsh_margin"]].describe().T[["mean", "min", "max"]].round(3)
            b.dataframe(k, width="stretch")
            energy = float((meas.motor_power_kw.fillna(0) / 3600).sum())
            volume = float((meas.flow_m3h.fillna(0) / 3600).sum())
            b.metric("Energy used [kWh]", f"{energy:.2f}")
            b.metric("Volume pumped [m³]", f"{volume:.1f}")
    with tabs[2]:
        if diag.empty:
            st.info("No diagnoses in this period.")
        else:
            rules = diag[diag.method == "rule"]
            st.dataframe(rules.groupby(["fault", "level"]).size().reset_index(name="windows"), hide_index=True)
            st.dataframe(diag[["timestamp", "method", "fault", "level", "confidence", "reason"]], hide_index=True,
                         width="stretch")
    with tabs[3]:
        if alarms.empty:
            st.info("No alarms in this period.")
        else:
            st.dataframe(alarms.groupby("priority").size().rename("alarms"))
            st.dataframe(alarms[["id", "raised_at", "priority", "alarm_key", "message", "status", "acknowledged_at",
                                 "acknowledged_by", "cleared_at", "reset_at", "occurrences"]], hide_index=True,
                         width="stretch")
    with tabs[4]:
        st.dataframe(maint, hide_index=True, width="stretch") if not maint.empty else st.info("No records.")
    with tabs[5]:
        stamp = f"{start:%Y%m%d_%H%M}_{end:%Y%m%d_%H%M}"
        c = st.columns(3)
        for i, (name, df) in enumerate((("measurements", meas), ("diagnoses", diag), ("alarms", alarms),
                                        ("health_history", health), ("maintenance_records", maint))):
            c[i % 3].download_button(f"⬇ {name}.csv ({len(df)} rows)", df.to_csv(index=False).encode(),
                                     f"{name}_{stamp}.csv", "text/csv", width="stretch",
                                     disabled=df.empty)
        report = build_report(engine, meas, diag, alarms, health, maint, start, end)
        st.download_button("⬇ Summary report (HTML)", report.encode(), f"pump_report_{stamp}.html", "text/html",
                           type="primary")
        with st.expander("Report preview"):
            st.html(report)
