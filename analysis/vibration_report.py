"""
Healthy vs faulty vibration comparison (waveform, FFT, envelope spectrum).

Run:  python -m analysis.vibration_report
"""
from __future__ import annotations

import pandas as pd

from analysis.vibration_analysis import amplitude_spectrum, analyze_vibration, bearing_frequencies, envelope_spectrum
from simulation.fault_simulator import FaultType, fault_effects
from simulation.pump_model import HEALTHY, PumpModel
from simulation.vibration_signal import from_state

CASES = {
    "healthy": {},
    "imbalance (s=0.8)": {FaultType.IMBALANCE: 0.8},
    "misalignment (s=0.8)": {FaultType.MISALIGNMENT: 0.8},
    "bearing (s=0.5)": {FaultType.BEARING_DEGRADATION: 0.5},
    "cavitation (s=0.95)": {FaultType.CAVITATION: 0.95},
}


def steady_state(severities: dict) -> dict:
    eff = HEALTHY
    for f, s in severities.items():
        eff = eff.combine(fault_effects(f, s))
    pump = PumpModel(warm_start=True)
    return pump.step(1.0, eff)


def build(seed: int = 0):
    signals, rows = {}, {}
    for i, (name, sev) in enumerate(CASES.items()):
        st = steady_state(sev)
        sig = from_state(st, sev, seed=seed + i)
        signals[name] = sig
        rows[name] = analyze_vibration(sig.velocity_mm_s, sig.acceleration_m_s2, sig.fs, sig.speed_rpm)
        rows[name]["model_vel_rms"] = st["vibration_rms_mm_s"]
        rows[name]["model_acc_rms"] = st["vibration_accel_m_s2"]
    return signals, pd.DataFrame(rows).T


def plot(signals, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(signals), 3, figsize=(16, 2.6 * len(signals)))
    for row, (name, sig) in zip(axes, signals.items()):
        f1 = sig.speed_rpm / 60
        m = sig.t < 0.25
        row[0].plot(sig.t[m] * 1000, sig.acceleration_m_s2[m], lw=0.5, color="tab:red", label="accel [m/s²]")
        ax2 = row[0].twinx()
        ax2.plot(sig.t[m] * 1000, sig.velocity_mm_s[m], lw=0.8, label="velocity [mm/s]")
        row[0].set_ylabel(name, fontsize=9, fontweight="bold")
        row[0].set_xlabel("time [ms]")
        f, a = amplitude_spectrum(sig.velocity_mm_s, sig.fs)
        row[1].plot(f, a, lw=0.8)
        for k, c in ((1, "g"), (2, "orange"), (6, "purple")):
            row[1].axvline(k * f1, color=c, ls="--", lw=0.7, label=f"{k}X" if k < 6 else "VPF 6X")
        row[1].set(xlim=(0, 300), xlabel="frequency [Hz]", ylabel="velocity amp [mm/s]")
        fe, ae = envelope_spectrum(sig.acceleration_m_s2, sig.fs)
        row[2].plot(fe, ae, lw=0.8, color="tab:red")
        bf = bearing_frequencies(sig.speed_rpm)
        for k in (1, 2, 3):
            row[2].axvline(k * bf["BPFO"], color="k", ls=":", lw=0.8, label="BPFO harmonics" if k == 1 else None)
        row[2].set(xlim=(0, 300), xlabel="frequency [Hz]", ylabel="envelope amp [m/s²]")
    axes[0, 1].legend(fontsize=7)
    axes[0, 2].legend(fontsize=7)
    axes[0, 0].set_title("Waveform (first 250 ms)")
    axes[0, 1].set_title("Velocity FFT spectrum")
    axes[0, 2].set_title("Acceleration envelope spectrum (1.0–2.4 kHz band)")
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    plt.close(fig)


if __name__ == "__main__":
    sigs, table = build()
    plot(sigs, "reports/vibration_comparison.png")
    print("Bearing frequencies @1450 rpm [Hz]:",
          {k: round(v, 2) for k, v in bearing_frequencies(1450).items()})
    cols = ["model_vel_rms", "vel_rms", "model_acc_rms", "acc_hf_band_rms", "acc_rms", "vel_crest_factor", "acc_kurtosis",
            "dominant_freq_hz", "dominant_order", "amp_1x_mm_s", "amp_2x_mm_s", "ratio_2x_1x", "amp_vpf_mm_s",
            "env_bpfo_ratio"]
    pd.set_option("display.width", 250)
    print(table[cols].astype(float).round(2).to_string())
