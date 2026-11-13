"""Page 3 — Vibration analysis: waveform, FFT, envelope, healthy vs current."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.vibration_analysis import amplitude_spectrum, analyze_vibration, bearing_frequencies, envelope_spectrum
from dashboard.ui import alarm_banner, fmt, get_engine, live, style_fig


def _body(engine):
    alarm_banner(engine)
    cur, ref = engine.vibration_snapshot(), engine.vibration_snapshot(healthy_reference=True)
    if cur is None or cur.speed_rpm < 10:
        st.info("Pump not running — no vibration to analyse.")
        return
    cfg = engine.cfg
    f = analyze_vibration(cur.velocity_mm_s, cur.acceleration_m_s2, cur.fs, cur.speed_rpm, cfg)
    fr = analyze_vibration(ref.velocity_mm_s, ref.acceleration_m_s2, ref.fs, ref.speed_rpm, cfg)
    f1 = cur.speed_rpm / 60.0

    m = st.columns(8)
    m[0].metric("Velocity RMS [mm/s]", fmt(f["vel_rms"]), f"{f['vel_rms'] - fr['vel_rms']:+.2f} vs healthy", delta_color="inverse")
    m[1].metric("Velocity peak [mm/s]", fmt(f["vel_peak"]))
    m[2].metric("Crest factor", fmt(f["vel_crest_factor"]))
    m[3].metric("Accel kurtosis", fmt(f["acc_kurtosis"]), f"{f['acc_kurtosis'] - fr['acc_kurtosis']:+.1f}", delta_color="inverse")
    m[4].metric("Dominant freq [Hz]", fmt(f["dominant_freq_hz"], 1), f"{f['dominant_order']:.2f}× running speed", delta_color="off")
    m[5].metric("2X / 1X", fmt(f["ratio_2x_1x"]))
    m[6].metric("HF accel band [m/s²]", fmt(f["acc_hf_band_rms"]))
    m[7].metric("Envelope BPFO SNR", fmt(f["env_bpfo_ratio"], 1))

    view = st.session_state.get("vib_ms", 200)
    n = int(view / 1000 * cur.fs)
    w = go.Figure()
    w.add_trace(go.Scatter(x=cur.t[:n] * 1000, y=ref.velocity_mm_s[:n], name="healthy reference", line=dict(width=1, color="#6B7A8F")))
    w.add_trace(go.Scatter(x=cur.t[:n] * 1000, y=cur.velocity_mm_s[:n], name="current", line=dict(width=1.3, color="#2E9BFF")))
    style_fig(w, 280, "Velocity waveform [mm/s] vs time [ms]")
    st.plotly_chart(w, width="stretch", key="vib_wave")

    c1, c2 = st.columns(2)
    fq, amp = amplitude_spectrum(cur.velocity_mm_s, cur.fs)
    _, amp_r = amplitude_spectrum(ref.velocity_mm_s, ref.fs)
    s = go.Figure()
    s.add_trace(go.Scatter(x=fq, y=amp_r, name="healthy", line=dict(width=1, color="#6B7A8F")))
    s.add_trace(go.Scatter(x=fq, y=amp, name="current", line=dict(width=1.3, color="#2E9BFF")))
    for k, lbl in ((1, "1X"), (2, "2X"), (3, "3X"), (cfg.pump.impeller_vanes, "VPF")):
        s.add_vline(x=k * f1, line=dict(dash="dot", width=1, color="#E0B000"), annotation_text=lbl)
    s.update_xaxes(range=[0, 300], title="frequency [Hz]")
    s.update_yaxes(title="amplitude [mm/s]")
    c1.plotly_chart(style_fig(s, 340, "Velocity FFT spectrum (Hann, 1 Hz resolution)"), width="stretch", key="vib_fft")

    fe, env = envelope_spectrum(cur.acceleration_m_s2, cur.fs)
    _, env_r = envelope_spectrum(ref.acceleration_m_s2, ref.fs)
    e = go.Figure()
    e.add_trace(go.Scatter(x=fe, y=env_r, name="healthy", line=dict(width=1, color="#6B7A8F")))
    e.add_trace(go.Scatter(x=fe, y=env, name="current", line=dict(width=1.3, color="#E03131")))
    bf = bearing_frequencies(cur.speed_rpm, cfg)
    for name, color in (("BPFO", "#F08C00"), ("BPFI", "#2EAD5B"), ("BSF", "#B197FC"), ("FTF", "#8A94A6")):
        fx = 2 * bf[name] if name == "BSF" else bf[name]
        e.add_vline(x=fx, line=dict(dash="dot", width=1, color=color), annotation_text=name if name != "BSF" else "2×BSF")
    e.update_xaxes(range=[0, 300], title="frequency [Hz]")
    e.update_yaxes(title="envelope [m/s²]")
    c2.plotly_chart(style_fig(e, 340, "Acceleration envelope spectrum (1.0–2.4 kHz band)"), width="stretch", key="vib_env")

    c3, c4 = st.columns([1, 1])
    c3.markdown("**Fault-frequency indicators**")
    c3.dataframe(pd.DataFrame({
        "indicator": ["1X amplitude [mm/s]", "2X amplitude [mm/s]", "3X amplitude [mm/s]", "Vane pass amplitude [mm/s]",
                      "Envelope BPFO SNR", "Envelope BPFI SNR", "Envelope 2×BSF SNR", "Envelope FTF SNR"],
        "current": [f["amp_1x_mm_s"], f["amp_2x_mm_s"], f["amp_3x_mm_s"], f["amp_vpf_mm_s"], f["env_bpfo_ratio"],
                    f["env_bpfi_ratio"], f["env_bsf_ratio"], f["env_ftf_ratio"]],
        "healthy": [fr["amp_1x_mm_s"], fr["amp_2x_mm_s"], fr["amp_3x_mm_s"], fr["amp_vpf_mm_s"], fr["env_bpfo_ratio"],
                    fr["env_bpfi_ratio"], fr["env_bsf_ratio"], fr["env_ftf_ratio"]],
    }).round(2), hide_index=True, width="stretch")
    c4.markdown("**Bearing characteristic frequencies** (6309-type, kinematic, no slip)")
    c4.dataframe(pd.DataFrame({"frequency": list(bf), "Hz": [round(v, 2) for v in bf.values()],
                               "order (× 1X)": [round(v / f1, 3) for v in bf.values()]}),
                 hide_index=True, width="stretch")
    c4.caption(f"Sampling fs = {cur.fs:.0f} Hz → Nyquist {cur.fs / 2:.0f} Hz; record 1 s → Δf = 1 Hz. "
               "Real defect frequencies deviate 1–2 % (slip), so peaks are searched in ±2 % bands.")


def render():
    engine = get_engine()
    st.title("Vibration Analysis")
    st.select_slider("Waveform window [ms]", [50, 100, 200, 500, 1000], value=200, key="vib_ms")
    live(engine, _body)
