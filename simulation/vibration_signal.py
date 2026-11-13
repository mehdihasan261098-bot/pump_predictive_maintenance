"""
Synthetic vibration waveforms (Task 2.1).

The pump model produces OVERALL levels (velocity RMS, acceleration RMS). This
module turns the same operating state and fault severities into time
waveforms whose spectral content reflects the fault mechanism:

Velocity (mm/s, 10-1000 Hz machinery band)
  healthy         1X (residual unbalance), small 2X/3X, vane pass 6X, broadband 10-300 Hz
  imbalance       1X dominant (rotating centrifugal force at running speed)
  misalignment    strong 2X (> 1X) plus 1X and 3X (coupling forces twice per rev)
  bearing (late)  broadband + raised floor
  cavitation      broadband random + vane pass
  off-BEP flow    vane pass and low-frequency broadband (recirculation)
Acceleration (m/s^2)
  derivative of the velocity components (a = 2 pi f v)
  + broadband HF floor
  + bearing outer-race defect: impact train at BPFO (1 % timing jitter for
    slip) exciting a 1.8 kHz structural resonance (zeta = 0.05)
  + cavitation: bursty band-limited noise 500-2400 Hz

Scaling: velocity components are allocated per source (healthy / each fault /
cavitation) using the same contributions as the pump model, and the waveform
is scaled so its RMS equals the overall velocity reading. The HF acceleration
RMS uses the model's HF terms; the total acceleration RMS therefore agrees
closely, but not exactly, with the model's overall estimate (which assumes
all velocity at 1X).

Assumption: outer-race defect is used for bearing degradation (most common
fault location because the load zone is stationary).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal

from analysis.vibration_analysis import bearing_frequencies
from config.pump_config import AppConfig, get_config
from simulation.fault_simulator import FaultType, fault_effects
from simulation.pump_model import PumpModel

RESONANCE_HZ = 1800.0
RESONANCE_ZETA = 0.05

# relative RMS weights of velocity components per source (normalised below)
_VEL_PATTERNS = {
    "healthy": {"1x": 0.75, "2x": 0.25, "3x": 0.10, "vpf": 0.40, "noise": 0.30},
    FaultType.IMBALANCE: {"1x": 0.97, "2x": 0.10, "noise": 0.15},
    FaultType.MISALIGNMENT: {"1x": 0.50, "2x": 0.80, "3x": 0.30, "noise": 0.10},
    FaultType.BEARING_DEGRADATION: {"noise": 0.90, "1x": 0.20, "bpfo": 0.30},
    "cavitation": {"noise": 0.90, "vpf": 0.40},
}
_ORDERS = {"1x": 1, "2x": 2, "3x": 3}


@dataclass
class VibrationSignal:
    t: np.ndarray
    velocity_mm_s: np.ndarray
    acceleration_m_s2: np.ndarray
    fs: float
    speed_rpm: float


def _normalise(w: dict) -> dict:
    total = math.sqrt(sum(v * v for v in w.values()))
    return {k: v / total for k, v in w.items()}


def _band_noise(rng, n, fs, lo, hi):
    nyq = fs / 2
    sos = signal.butter(4, [lo / nyq, min(hi, 0.98 * nyq) / nyq], btype="bandpass", output="sos")
    x = signal.sosfilt(sos, rng.standard_normal(n + 2000))[2000:]
    return x / np.sqrt(np.mean(x ** 2))


def _impact_train(rng, n, fs, rate_hz, jitter=0.01):
    """Unit-RMS train of decaying resonance responses at `rate_hz`."""
    x = np.zeros(n)
    t, period = rng.uniform(0, 1 / rate_hz), 1 / rate_hz
    while t < n / fs:
        x[int(t * fs)] += rng.uniform(0.8, 1.2)
        t += period * (1 + rng.normal(0, jitter))
    wn = 2 * math.pi * RESONANCE_HZ
    th = np.arange(int(6 / (RESONANCE_ZETA * wn) * fs)) / fs
    h = np.exp(-RESONANCE_ZETA * wn * th) * np.sin(wn * math.sqrt(1 - RESONANCE_ZETA ** 2) * th)
    y = np.convolve(x, h)[:n]
    return y / np.sqrt(np.mean(y ** 2))


def generate_vibration(speed_rpm: float, v_rms_mm_s: float, severities: dict | None = None,
                       npsh_margin: float = math.inf, bep_ratio: float = 1.0, cfg: AppConfig | None = None,
                       duration_s: float | None = None, seed: int | None = None) -> VibrationSignal:
    cfg = cfg or get_config()
    vib = cfg.vibration
    fs = vib.sampling_rate_hz
    n = int(round((duration_s or vib.window_seconds) * fs))
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    severities = {FaultType(k): float(v) for k, v in (severities or {}).items() if v > 0}
    r = speed_rpm / cfg.pump.rated_speed_rpm
    noise_floor = 0.02 * rng.standard_normal(n)              # instrument noise (mm/s)

    if r <= 0.01:
        return VibrationSignal(t, noise_floor, 0.002 * rng.standard_normal(n), fs, speed_rpm)

    f1 = speed_rpm / 60.0
    freqs = {"1x": f1, "2x": 2 * f1, "3x": 3 * f1, "vpf": cfg.pump.impeller_vanes * f1,
             "bpfo": bearing_frequencies(speed_rpm, cfg)["BPFO"]}

    # --- RMS contribution of every source (same terms as PumpModel) ---
    sources = {"healthy": (vib.baseline_rms_mm_s + vib.off_bep_vibration_mm_s * (bep_ratio - 1) ** 2) * r ** 2}
    for f, s in severities.items():
        if f in _VEL_PATTERNS:
            sources[f] = fault_effects(f, s).vibration_add_mm_s * r ** 2
    cav_i = PumpModel.cavitation_intensity(npsh_margin)
    sources["cavitation"] = PumpModel.CAVITATION_VIBRATION_MM_S * cav_i * r ** 2

    # --- accumulate component amplitudes (RMS^2 adds for the same component) ---
    comp_ms = {}
    for src, rms in sources.items():
        for comp, w in _normalise(_VEL_PATTERNS[src]).items():
            comp_ms[comp] = comp_ms.get(comp, 0.0) + (rms * w) ** 2

    vel = np.zeros(n)
    acc_lf = np.zeros(n)
    for comp, ms in comp_ms.items():
        a_rms = math.sqrt(ms)
        if comp == "noise":
            x = _band_noise(rng, n, fs, 10.0, 300.0) * a_rms
            vel += x
            acc_lf += np.gradient(x, 1 / fs) / 1000.0
        else:
            ph = rng.uniform(0, 2 * math.pi)
            w = 2 * math.pi * freqs[comp]
            vel += math.sqrt(2) * a_rms * np.sin(w * t + ph)
            acc_lf += math.sqrt(2) * a_rms * w * np.cos(w * t + ph) / 1000.0

    scale = v_rms_mm_s / math.sqrt(np.mean(vel ** 2)) if v_rms_mm_s > 0 else 0.0
    vel = vel * scale + noise_floor
    acc = acc_lf * scale

    # --- high-frequency acceleration sources ---
    acc += vib.baseline_hf_accel_m_s2 * r ** 2 * _band_noise(rng, n, fs, 500.0, 2400.0)
    for f, s in severities.items():
        hf = fault_effects(f, s).hf_accel_add_m_s2 * r ** 2
        if hf <= 0:
            continue
        if f is FaultType.BEARING_DEGRADATION:
            acc += hf * _impact_train(rng, n, fs, freqs["bpfo"])
        else:
            acc += hf * _band_noise(rng, n, fs, 500.0, 2400.0)
    if cav_i > 0:
        bursts = np.abs(_band_noise(rng, n, fs, 2.0, 40.0))
        cav = _band_noise(rng, n, fs, 500.0, 2400.0) * bursts
        acc += PumpModel.CAVITATION_HF_ACCEL_M_S2 * cav_i * r ** 2 * cav / np.sqrt(np.mean(cav ** 2))
    return VibrationSignal(t, vel, acc, fs, speed_rpm)


def from_state(state: dict, severities: dict | None = None, cfg: AppConfig | None = None,
               seed: int | None = None) -> VibrationSignal:
    """Waveform consistent with a PumpModel/collector record."""
    return generate_vibration(state["speed_rpm"], state["vibration_rms_mm_s"], severities,
                              state.get("npsh_margin", math.inf), state.get("bep_ratio", 1.0), cfg, seed=seed)
