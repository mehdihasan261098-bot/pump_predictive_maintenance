"""Page 4 — Fault detection: rule-based diagnosis vs machine learning."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.ui import LEVEL_COLORS, alarm_banner, card, fmt, get_engine, live, pill, style_fig


def _rules(engine):
    diag = engine.latest_diagnosis
    st.subheader("A · Rule-based diagnosis (expert rules, explainable)")
    if diag is None:
        st.info("Waiting for the first 60 s analysis window…")
        return
    if not diag.running:
        st.info("Pump not running — rules are evaluated only during operation.")
        return
    act = diag.active
    if not act:
        st.success("All rules Normal — no fault detected.")
    for r in act:
        with st.container(border=True):
            st.markdown(f"{pill(r.level)} &nbsp; **{r.fault.replace('_', ' ').capitalize()}** "
                        f"&nbsp; <small>confidence {r.confidence:.2f}</small>", unsafe_allow_html=True)
            st.markdown(f"**Reason:** {r.reason}")
            st.markdown(f"**Recommended action:** {r.recommendation}")
            st.caption("Sensor evidence: " + " · ".join(f"{k} = {fmt(v)}" for k, v in r.evidence.items()))
    with st.expander("All rule results"):
        st.dataframe(pd.DataFrame([{"rule": r.fault, "level": r.level, "confidence": r.confidence,
                                    "reason": r.reason} for r in diag.results]), hide_index=True,
                     width="stretch")


def _ml(engine):
    st.subheader("B · Machine learning (data-driven)")
    ml = engine.latest_ml
    if engine.ml is None:
        st.warning("ML models not found. Train them with:  python -m detection.ml_detection")
        return
    if ml is None:
        st.info("Waiting for the first running analysis window…")
        return
    c = st.columns(3)
    with c[0]:
        card("Random Forest prediction", ml["predicted_fault"].replace("_", " ").capitalize(),
             f"class probability {ml['confidence']:.2f}")
    with c[1]:
        an = bool(ml["is_anomaly"])
        card("Isolation Forest", "ANOMALY" if an else "Normal",
             f"score {ml['anomaly_score']:.3f} / threshold {engine.ml.iforest['threshold']:.3f}",
             LEVEL_COLORS["Warning" if an else "Normal"])
    rule = engine.latest_diagnosis.primary_fault if engine.latest_diagnosis else "healthy"
    agree = rule == ml["predicted_fault"]
    with c[2]:
        card("Rules vs ML", "Agree" if agree else "Disagree", f"rules: {rule.replace('_', ' ')} · ML: "
             f"{ml['predicted_fault'].replace('_', ' ')}", LEVEL_COLORS["Normal" if agree else "Caution"])
    probs = {k[2:]: v for k, v in ml.items() if k.startswith("p_")}
    fig = go.Figure(go.Bar(x=list(probs.values()), y=[k.replace("_", " ") for k in probs], orientation="h",
                           marker_color=["#2E9BFF" if k == ml["predicted_fault"] else "#4A5A70" for k in probs]))
    fig.update_xaxes(range=[0, 1], title="probability")
    st.plotly_chart(style_fig(fig, 300, "Random Forest class probabilities"), width="stretch", key="ml_prob")


def _history(engine):
    st.subheader("Diagnosis history (last 2 h simulated)")
    d = engine.db.get_diagnoses(start=engine.now - pd.to_timedelta(2, unit="h"))
    if d.empty:
        st.caption("No diagnoses recorded yet.")
        return
    d = d[(d.method != "ml_anomaly") & (d.fault != "healthy")]
    if d.empty:
        st.caption("No faults diagnosed in this period.")
        return
    fig = go.Figure()
    for method, sym in (("rule", "circle"), ("ml_classifier", "diamond")):
        m = d[d.method == method]
        fig.add_trace(go.Scatter(x=m.timestamp, y=m.fault, mode="markers", name=method,
                                 marker=dict(symbol=sym, size=8, color=[LEVEL_COLORS.get(lv, "#888") for lv in m.level])))
    st.plotly_chart(style_fig(fig, 300), width="stretch", key="diag_hist")


def _body(engine):
    alarm_banner(engine)
    left, right = st.columns([1.1, 1])
    with left:
        _rules(engine)
    with right:
        _ml(engine)
    _history(engine)
    st.caption("Rule-based results come from configurable engineering limits with combined evidence; ML results "
               "come from models trained on SIMULATED data (validation: RF 96.9 % accuracy at severity ≥ 0.3, "
               "IF ROC-AUC 0.90). Performance on synthetic data does not establish industrial diagnostic accuracy.")


def render():
    engine = get_engine()
    st.title("Fault Detection")
    live(engine, _body)
