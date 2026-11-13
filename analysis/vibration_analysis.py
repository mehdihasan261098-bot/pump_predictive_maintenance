"""
Vibration signal analysis.

Sampling and the Nyquist criterion
----------------------------------
A sampled signal can only represent frequencies below fs/2 (Nyquist frequency).
Content above fs/2 folds back (aliases) onto lower frequencies and cannot be
separated afterwards, so real systems use an analogue anti-alias filter and
sample at >= 2.56 x the highest frequency of interest. Here fs = 5 kHz ->
Nyquist 2.5 kHz, which covers 1X (24.2 Hz) ... vane-pass (145 Hz), the bearing
defect frequencies (9-120 Hz) and their harmonics, and a simulated structural
resonance at 1.8 kHz used for envelope analysis. Frequency resolution of an
FFT is df = 1 / T (T = record length), e.g. 1 Hz for a 1 s record.

Contents
  time-domain : RMS, peak, peak-to-peak, crest factor, kurtosis, skewness
  frequency   : single-sided amplitude spectrum (Hann window, amplitude
                corrected), dominant frequency, harmonic orders, band RMS
  bearings    : BPFO / BPFI / BSF / FTF (rigid-body kinematic formulas),
                envelope (demodulation) spectrum and defect-frequency ratios
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal, stats

from config.pump_config import AppConfig, get_config


# --------------------------------------------------------------------------
# Time domain
# --------------------------------------------------------------------------
def time_features(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(np.max(x) - np.min(x)),
        "crest_factor": peak / rms if rms > 0 else 0.0,
        # Pearson kurtosis: 3.0 for Gaussian noise, 1.5 for a pure sine,
        # >> 3 for impulsive signals (bearing impacts)
        "kurtosis": float(stats.kurtosis(x, fisher=False)) if rms > 0 else 0.0,
        "skewness": float(stats.skew(x)) if rms > 0 else 0.0,
    }


# --------------------------------------------------------------------------
# Frequency domain
# --------------------------------------------------------------------------
def amplitude_spectrum(x: np.ndarray, fs: float, window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    """Single-sided PEAK amplitude spectrum.

    The window reduces leakage; dividing by its coherent gain (sum(w)/N, 0.5
    for Hann) restores the true amplitude of a sinusoid at a bin centre.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    w = signal.get_window(window, n) if window else np.ones(n)
    spec = np.fft.rfft((x - x.mean()) * w)
    amp = 2.0 * np.abs(spec) / np.sum(w)
    amp[0] /= 2.0
    if n % 2 == 0:
        amp[-1] /= 2.0
    return np.fft.rfftfreq(n, 1.0 / fs), amp


def dominant_frequency(freqs: np.ndarray, amp: np.ndarray, fmin: float = 2.0) -> tuple[float, float]:
    mask = freqs >= fmin
    i = int(np.argmax(amp[mask]))
    return float(freqs[mask][i]), float(amp[mask][i])


def peak_near(freqs: np.ndarray, amp: np.ndarray, f_target: float, tol_hz: float | None = None) -> tuple[float, float]:
    """Largest spectral line within +/- tol of f_target -> (freq, amplitude)."""
    df = freqs[1] - freqs[0]
    tol = max(2.0 * df, 0.02 * f_target) if tol_hz is None else tol_hz
    mask = np.abs(freqs - f_target) <= tol
    if not mask.any():
        return f_target, 0.0
    i = int(np.argmax(amp[mask]))
    return float(freqs[mask][i]), float(amp[mask][i])


def harmonic_amplitudes(freqs, amp, f1: float, orders=(1, 2, 3)) -> dict[str, float]:
    return {f"{o}x": peak_near(freqs, amp, o * f1)[1] for o in orders}


def band_rms(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """RMS of the signal content between f_lo and f_hi (Parseval on the FFT)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    mask = (f >= f_lo) & (f <= f_hi)
    power = 2.0 * np.sum(np.abs(spec[mask]) ** 2) / n ** 2
    return float(math.sqrt(power))


# --------------------------------------------------------------------------
# Rolling-element bearing kinematics
# --------------------------------------------------------------------------
def bearing_frequencies(speed_rpm: float, cfg: AppConfig | None = None) -> dict[str, float]:
    """Characteristic defect frequencies (Hz) of a bearing with stationary
    outer ring and rotating inner ring.

        BPFO = n/2 * fr * (1 - d/D cos a)      outer-race defect
        BPFI = n/2 * fr * (1 + d/D cos a)      inner-race defect
        BSF  = D/(2d) * fr * (1 - (d/D cos a)^2)   ball spin (defect hits at 2xBSF)
        FTF  = fr/2 * (1 - d/D cos a)          cage

    Assumptions: pure rolling (no slip), rigid elements, constant contact
    angle. Real frequencies deviate 1-2 % because of slip and load-dependent
    contact angle, so analysis must search a tolerance band.
    """
    b = (cfg or get_config()).bearing
    fr = speed_rpm / 60.0
    ratio = b.ball_diameter_mm / b.pitch_diameter_mm * math.cos(math.radians(b.contact_angle_deg))
    return {
        "BPFO": b.n_balls / 2 * fr * (1 - ratio),
        "BPFI": b.n_balls / 2 * fr * (1 + ratio),
        "BSF": b.pitch_diameter_mm / (2 * b.ball_diameter_mm) * fr * (1 - ratio ** 2),
        "FTF": fr / 2 * (1 - ratio),
    }


def envelope_spectrum(x: np.ndarray, fs: float, band: tuple[float, float] = (1000.0, 2400.0)
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Demodulation: band-pass around the structural resonance excited by the
    impacts, take the Hilbert envelope, and compute its spectrum. Repetition
    rates of impacts (BPFO, BPFI ...) appear as clear lines."""
    nyq = fs / 2.0
    hi = min(band[1], 0.98 * nyq)
    sos = signal.butter(4, [band[0] / nyq, hi / nyq], btype="bandpass", output="sos")
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, np.asarray(x, dtype=float))))
    return amplitude_spectrum(env - env.mean(), fs)


def bearing_defect_indicators(acceleration: np.ndarray, fs: float, speed_rpm: float,
                              cfg: AppConfig | None = None) -> dict[str, float]:
    """For each defect frequency: envelope amplitude at f and 2f relative to
    the median envelope level 5-250 Hz (signal-to-noise ratio, ~1 if absent)."""
    freqs, env = envelope_spectrum(acceleration, fs)
    floor = float(np.median(env[(freqs > 5) & (freqs < 250)])) + 1e-12
    out = {}
    for name, f in bearing_frequencies(speed_rpm, cfg).items():
        f_eff = 2 * f if name == "BSF" else f
        a1 = peak_near(freqs, env, f_eff)[1]
        a2 = peak_near(freqs, env, 2 * f_eff)[1]
        out[f"env_{name.lower()}_ratio"] = (a1 + a2) / (2 * floor)
    return out


# --------------------------------------------------------------------------
# Complete feature set for one vibration snapshot
# --------------------------------------------------------------------------
def analyze_vibration(velocity_mm_s: np.ndarray, acceleration_m_s2: np.ndarray, fs: float,
                      speed_rpm: float, cfg: AppConfig | None = None) -> dict[str, float]:
    cfg = cfg or get_config()
    out: dict[str, float] = {}
    for prefix, x in (("vel", velocity_mm_s), ("acc", acceleration_m_s2)):
        for k, v in time_features(x).items():
            out[f"{prefix}_{k}"] = v
    freqs, amp = amplitude_spectrum(velocity_mm_s, fs)
    out["dominant_freq_hz"], out["dominant_amp_mm_s"] = dominant_frequency(freqs, amp)
    f1 = speed_rpm / 60.0
    out["running_freq_hz"] = f1
    if f1 > 1.0:
        out["dominant_order"] = out["dominant_freq_hz"] / f1
        h = harmonic_amplitudes(freqs, amp, f1, (1, 2, 3, cfg.pump.impeller_vanes))
        out["amp_1x_mm_s"] = h["1x"]
        out["amp_2x_mm_s"] = h["2x"]
        out["amp_3x_mm_s"] = h["3x"]
        out["amp_vpf_mm_s"] = h[f"{cfg.pump.impeller_vanes}x"]
        out["ratio_2x_1x"] = h["2x"] / h["1x"] if h["1x"] > 1e-9 else 0.0
        out.update(bearing_defect_indicators(acceleration_m_s2, fs, speed_rpm, cfg))
    else:
        out.update(dominant_order=0.0, amp_1x_mm_s=0.0, amp_2x_mm_s=0.0, amp_3x_mm_s=0.0,
                   amp_vpf_mm_s=0.0, ratio_2x_1x=0.0, env_bpfo_ratio=0.0, env_bpfi_ratio=0.0,
                   env_bsf_ratio=0.0, env_ftf_ratio=0.0)
    out["acc_hf_band_rms"] = band_rms(acceleration_m_s2, fs, 500.0, 2400.0)   # = acceleration sensor band
    out["vel_broadband_rms"] = band_rms(velocity_mm_s, fs, 200.0, 1000.0)
    return out
