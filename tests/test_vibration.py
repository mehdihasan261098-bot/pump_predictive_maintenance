"""Tasks 2.1/2.2 tests: vibration analysis on KNOWN synthetic signals and
fault-specific content of the generated waveforms."""
import math

import numpy as np
import pytest

from analysis import vibration_analysis as va
from simulation.fault_simulator import FaultType
from simulation.vibration_signal import generate_vibration

FS = 5000.0
T = np.arange(int(FS)) / FS            # 1 s -> 1 Hz resolution


def sine(f, a, phase=0.0):
    return a * np.sin(2 * np.pi * f * T + phase)


# ------------------------------------------------------------ time domain
def test_pure_sine_time_features():
    f = va.time_features(sine(50, 3.0))
    assert f["rms"] == pytest.approx(3 / math.sqrt(2), rel=1e-4)
    assert f["peak"] == pytest.approx(3.0, rel=1e-3)
    assert f["peak_to_peak"] == pytest.approx(6.0, rel=1e-3)
    assert f["crest_factor"] == pytest.approx(math.sqrt(2), rel=1e-3)
    assert f["kurtosis"] == pytest.approx(1.5, abs=0.01)


def test_gaussian_noise_kurtosis_is_three():
    x = np.random.default_rng(0).standard_normal(200_000)
    assert va.time_features(x)["kurtosis"] == pytest.approx(3.0, abs=0.05)


def test_impulsive_signal_high_kurtosis_and_crest():
    x = np.zeros(5000)
    x[::200] = 10.0
    f = va.time_features(x + 0.1 * np.random.default_rng(1).standard_normal(5000))
    assert f["kurtosis"] > 20 and f["crest_factor"] > 5


def test_zero_signal_is_safe():
    f = va.time_features(np.zeros(100))
    assert f["rms"] == 0 and f["crest_factor"] == 0 and f["kurtosis"] == 0


# ------------------------------------------------------------ FFT
def test_fft_recovers_frequency_and_amplitude():
    freqs, amp = va.amplitude_spectrum(sine(50, 3.0, 0.7), FS)
    assert freqs[1] - freqs[0] == pytest.approx(1.0)           # df = 1/T
    f, a = va.dominant_frequency(freqs, amp)
    assert f == 50.0 and a == pytest.approx(3.0, rel=1e-3)


def test_fft_two_tones_and_harmonics():
    x = sine(24, 2.0) + sine(48, 0.5) + sine(72, 0.25)
    freqs, amp = va.amplitude_spectrum(x, FS)
    h = va.harmonic_amplitudes(freqs, amp, 24.0)
    assert h["1x"] == pytest.approx(2.0, rel=1e-3)
    assert h["2x"] == pytest.approx(0.5, rel=1e-3)
    assert h["3x"] == pytest.approx(0.25, rel=1e-3)


def test_off_bin_frequency_found_within_tolerance():
    freqs, amp = va.amplitude_spectrum(sine(24.17, 2.0), FS)
    f, a = va.peak_near(freqs, amp, 24.17)
    assert abs(f - 24.17) <= 1.0 and a > 1.6                   # Hann scalloping < 1.5 dB


def test_aliasing_above_nyquist():
    # 3000 Hz sampled at 5 kHz appears at 5000 - 3000 = 2000 Hz
    freqs, amp = va.amplitude_spectrum(sine(3000, 1.0), FS)
    assert va.dominant_frequency(freqs, amp)[0] == 2000.0


def test_band_rms_parseval():
    x = sine(100, 2.0) + sine(1500, 1.0)
    assert va.band_rms(x, FS, 50, 200) == pytest.approx(2 / math.sqrt(2), rel=1e-3)
    assert va.band_rms(x, FS, 1000, 2000) == pytest.approx(1 / math.sqrt(2), rel=1e-3)
    assert va.band_rms(x, FS, 0, 2500) == pytest.approx(math.sqrt(2 + 0.5), rel=1e-3)


# ------------------------------------------------------------ bearings
def test_bearing_frequency_hand_calculation():
    # 6309-type: n=8, d=17.5, D=72.5, alpha=0, fr=24.1667 Hz, d/D=0.241379
    b = va.bearing_frequencies(1450)
    assert b["BPFO"] == pytest.approx(4 * 24.16667 * (1 - 0.241379), rel=1e-5)   # 73.33
    assert b["BPFI"] == pytest.approx(4 * 24.16667 * (1 + 0.241379), rel=1e-5)   # 120.00
    assert b["BSF"] == pytest.approx(72.5 / 35 * 24.16667 * (1 - 0.241379 ** 2), rel=1e-5)  # 47.14
    assert b["FTF"] == pytest.approx(0.5 * 24.16667 * (1 - 0.241379), rel=1e-5)  # 9.17
    assert b["BPFO"] + b["BPFI"] == pytest.approx(8 * 24.16667, rel=1e-6)      # n * fr identity


def test_envelope_recovers_modulation_frequency():
    carrier = sine(1800, 1.0)
    mod = 1 + 0.8 * np.sin(2 * np.pi * 73 * T)
    freqs, env = va.envelope_spectrum(carrier * mod, FS)
    assert va.dominant_frequency(freqs, env)[0] == 73.0


# ------------------------------------------------------------ generated waveforms
def features(sev=None, v=1.8, margin=math.inf, seed=0):
    sig = generate_vibration(1450, v, sev, npsh_margin=margin, seed=seed)
    return va.analyze_vibration(sig.velocity_mm_s, sig.acceleration_m_s2, sig.fs, sig.speed_rpm)


def test_velocity_rms_matches_requested_overall_level():
    for v in (1.8, 4.0, 9.0):
        assert features(v=v)["vel_rms"] == pytest.approx(v, rel=0.02)


def test_healthy_dominant_1x_and_hf_band_matches_model():
    f = features()
    assert f["dominant_order"] == pytest.approx(1.0, abs=0.05)
    assert f["acc_hf_band_rms"] == pytest.approx(0.30, rel=0.15)
    assert f["env_bpfo_ratio"] < 4


def test_imbalance_is_1x():
    f = features({FaultType.IMBALANCE: 0.8}, v=7.4)
    assert f["dominant_order"] == pytest.approx(1.0, abs=0.05)
    assert f["ratio_2x_1x"] < 0.3 and f["amp_1x_mm_s"] > 8


def test_misalignment_is_2x():
    f = features({FaultType.MISALIGNMENT: 0.8}, v=5.9)
    assert f["dominant_order"] == pytest.approx(2.0, abs=0.05)
    assert f["ratio_2x_1x"] > 1.0


def test_bearing_defect_detected_in_envelope():
    f = features({FaultType.BEARING_DEGRADATION: 0.5}, v=2.06)
    assert f["env_bpfo_ratio"] > 10
    assert f["acc_kurtosis"] > 4
    assert f["env_bpfi_ratio"] < f["env_bpfo_ratio"] / 3


def test_cavitation_broadband_hf():
    f = features({FaultType.CAVITATION: 0.95}, v=4.7, margin=0.5)
    assert f["acc_hf_band_rms"] > 10
    assert f["vel_broadband_rms"] > features()["vel_broadband_rms"] * 1.5 or f["amp_vpf_mm_s"] > 2


def test_stopped_pump_gives_near_zero_signal():
    sig = generate_vibration(0.0, 0.0, seed=1)
    assert np.sqrt(np.mean(sig.velocity_mm_s ** 2)) < 0.05
    f = va.analyze_vibration(sig.velocity_mm_s, sig.acceleration_m_s2, sig.fs, 0.0)
    assert f["amp_1x_mm_s"] == 0.0


def test_reproducible_with_seed():
    a = generate_vibration(1450, 2.0, {FaultType.BEARING_DEGRADATION: 0.4}, seed=5)
    b = generate_vibration(1450, 2.0, {FaultType.BEARING_DEGRADATION: 0.4}, seed=5)
    np.testing.assert_array_equal(a.acceleration_m_s2, b.acceleration_m_s2)
