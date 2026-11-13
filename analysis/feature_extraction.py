"""
Feature extraction for diagnosis and machine learning.

One feature vector is computed per non-overlapping time window (default 60 s)
of RUNNING operation. It combines

  process statistics   mean / std of every sensor in the window
  trends               slope of bearing & casing temperature and velocity
                       (per minute) - captures developing faults
  engineering KPIs     head, flow deficit, head deviation, BEP ratio, pump
                       efficiency, NPSH margin, SEC, temperature rises
  vibration features   a 1 s vibration snapshot generated at the window end
                       (speed, measured velocity level, NPSH margin, BEP ratio
                       and the simulated physical fault state) and analysed
                       with analysis.vibration_analysis (prefix vib_)

Leakage protection: ground-truth columns (true_*, sev_*, fault_*) are never
used as features; they are carried only as labels/metadata. ML splits must be
done by `run_id` (whole simulation runs), never by random windows.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analysis.hydraulic_calculations import compute_kpis
from analysis.vibration_analysis import analyze_vibration
from config.pump_config import AppConfig, get_config
from simulation.fault_simulator import FAULT_TYPES, FaultScenario, FaultType, generate_fault_dataset
from simulation.sensor_simulator import SENSOR_NAMES, OperatingProfile
from simulation.vibration_signal import generate_vibration

WINDOW_S = 60.0
KPI_FEATURES = ["head_m", "flow_deficit_m3h", "head_deviation_pct", "bep_ratio", "pump_eff_est_pct",
                "npsh_margin", "sec_kwh_m3", "motor_load_pct", "system_resistance_ratio"]
VIB_FEATURES = ["vel_crest_factor", "vel_kurtosis", "acc_rms", "acc_crest_factor", "acc_kurtosis",
                "dominant_order", "amp_1x_mm_s", "amp_2x_mm_s", "amp_3x_mm_s", "amp_vpf_mm_s", "ratio_2x_1x",
                "env_bpfo_ratio", "env_bpfi_ratio", "env_bsf_ratio", "env_ftf_ratio", "acc_hf_band_rms",
                "vel_broadband_rms"]
META_COLUMNS = ["run_id", "window_start_s", "window_end_s", "timestamp", "fault_label", "fault_severity",
                "active_faults", *[f"sev_{f.value}" for f in FAULT_TYPES]]


def _slope_per_min(t_s: np.ndarray, y: np.ndarray) -> float:
    ok = ~np.isnan(y)
    if ok.sum() < 3:
        return math.nan
    return float(np.polyfit(t_s[ok] / 60.0, y[ok], 1)[0])


def feature_columns() -> list[str]:
    cols = []
    for s in SENSOR_NAMES:
        cols += [f"{s}_mean", f"{s}_std"]
    cols += ["bearing_temp_slope_c_min", "casing_temp_slope_c_min", "vibration_rms_slope_mm_s_min",
             "bearing_temp_rise_c", "casing_temp_rise_c"]
    cols += [f"{k}_mean" for k in KPI_FEATURES]
    cols += [f"vib_{k}" for k in VIB_FEATURES]
    return cols


FEATURE_COLUMNS = feature_columns()


def window_features(win: pd.DataFrame, kpis: pd.DataFrame | None = None, severities: dict | None = None,
                    cfg: AppConfig | None = None, seed: int | None = None) -> dict:
    """Features of ONE window of measured records (also used live by the app).

    severities: physical fault state used only to synthesise the vibration
    waveform the accelerometer would record (not a feature itself).
    """
    cfg = cfg or get_config()
    kpis = compute_kpis(win, cfg) if kpis is None else kpis
    t = win["time_s"].to_numpy(dtype=float)
    f: dict[str, float] = {}
    for s in SENSOR_NAMES:
        v = win[s].to_numpy(dtype=float)
        f[f"{s}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else math.nan
        f[f"{s}_std"] = float(np.nanstd(v)) if np.isfinite(v).sum() > 1 else math.nan
    f["bearing_temp_slope_c_min"] = _slope_per_min(t, win["bearing_temp_c"].to_numpy(dtype=float))
    f["casing_temp_slope_c_min"] = _slope_per_min(t, win["casing_temp_c"].to_numpy(dtype=float))
    f["vibration_rms_slope_mm_s_min"] = _slope_per_min(t, win["vibration_rms_mm_s"].to_numpy(dtype=float))
    f["bearing_temp_rise_c"] = f["bearing_temp_c_mean"] - cfg.thermal.ambient_temp_c
    f["casing_temp_rise_c"] = f["casing_temp_c_mean"] - cfg.thermal.fluid_temp_c
    for k in KPI_FEATURES:
        col = kpis[k].to_numpy(dtype=float)
        f[f"{k}_mean"] = float(np.nanmean(col)) if np.isfinite(col).any() else math.nan

    speed = f["speed_rpm_mean"]
    margin = f["npsh_margin_mean"]
    sig = generate_vibration(speed if np.isfinite(speed) else 0.0,
                             f["vibration_rms_mm_s_mean"] if np.isfinite(f["vibration_rms_mm_s_mean"]) else 0.0,
                             severities, npsh_margin=margin if np.isfinite(margin) else math.inf,
                             bep_ratio=f["bep_ratio_mean"] if np.isfinite(f["bep_ratio_mean"]) else 1.0,
                             cfg=cfg, seed=seed)
    vib = analyze_vibration(sig.velocity_mm_s, sig.acceleration_m_s2, sig.fs, sig.speed_rpm, cfg)
    for k in VIB_FEATURES:
        f[f"vib_{k}"] = vib[k]
    return f


def extract_features(df: pd.DataFrame, window_s: float = WINDOW_S, cfg: AppConfig | None = None,
                     seed: int = 0, running_only: bool = True) -> pd.DataFrame:
    """Window features for a whole simulated/recorded run (labels carried
    along when present)."""
    cfg = cfg or get_config()
    kpis = compute_kpis(df, cfg)
    win_id = np.floor((df["time_s"].to_numpy() - df["time_s"].iloc[0]) / window_s).astype(int)
    rows = []
    rng = np.random.default_rng(seed)
    for w, idx in pd.Series(np.arange(len(df))).groupby(win_id):
        win = df.iloc[idx.to_numpy()]
        if len(win) < 0.9 * window_s / cfg.sensors.sampling_interval_s:
            continue                                    # incomplete trailing window
        if running_only and "state" in win and not (win["state"] == "Running").all():
            continue
        end = win.iloc[-1]
        sev = {f: float(end[f"sev_{f.value}"]) for f in FAULT_TYPES if f"sev_{f.value}" in win}
        feats = window_features(win, kpis.iloc[idx.to_numpy()], sev, cfg, seed=int(rng.integers(1 << 31)))
        meta = {"run_id": end.get("run_id", "run"), "window_start_s": float(win["time_s"].iloc[0]),
                "window_end_s": float(end["time_s"]), "timestamp": end.get("timestamp")}
        for c in META_COLUMNS[4:]:
            if c in win:
                meta[c] = end[c]
        rows.append({**meta, **feats})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Training dataset: many independent runs with varied operating conditions
# --------------------------------------------------------------------------
def training_run_specs(runs_per_class: int = 6, duration_s: float = 2400.0, seed: int = 1000) -> list[dict]:
    """Run definitions: each fault and healthy, with random operating point
    (valve 0.55-1.0, speed 1300-1450 rpm), max severity 0.5-1.0, onset and
    growth shape; plus a few two-fault combinations."""
    rng = np.random.default_rng(seed)
    specs = []
    classes = ["healthy", *[f.value for f in FAULT_TYPES]]
    for cls in classes:
        for k in range(runs_per_class):
            onset = float(rng.uniform(300, 700))
            scen = []
            if cls != "healthy":
                f = FaultType(cls)
                scen.append(FaultScenario(f, onset_s=onset, ramp_s=float(rng.uniform(0.4, 0.7)) * duration_s,
                                          max_severity=float(rng.uniform(0.5, 1.0)),
                                          shape=str(rng.choice(["linear", "exponential"]))))
            specs.append(dict(run_id=f"{cls}_{k:02d}", scenarios=scen, seed=int(rng.integers(1 << 30)),
                              valve=float(rng.uniform(0.55, 1.0)), speed=float(rng.choice([1300, 1375, 1450])),
                              duration_s=duration_s))
    combos = [(FaultType.BEARING_DEGRADATION, FaultType.MISALIGNMENT), (FaultType.IMBALANCE, FaultType.CAVITATION),
              (FaultType.SEAL_LEAKAGE, FaultType.OVERHEATING)]
    for i, (a, b) in enumerate(combos):
        scen = [FaultScenario(a, onset_s=400, ramp_s=1200, max_severity=0.9),
                FaultScenario(b, onset_s=900, ramp_s=1000, max_severity=0.7)]
        specs.append(dict(run_id=f"multi_{a.value}+{b.value}", scenarios=scen, seed=int(rng.integers(1 << 30)),
                          valve=1.0, speed=1450.0, duration_s=duration_s))
    return specs


def simulate_run(spec: dict) -> pd.DataFrame:
    profile = OperatingProfile(events=[(0.0, "valve", spec["valve"]), (0.0, "speed", spec["speed"])],
                               demand_amplitude=0.03)
    df = generate_fault_dataset(spec["scenarios"], spec["duration_s"], seed=spec["seed"],
                                name=None, out_dir=None, profile=profile)
    df["run_id"] = spec["run_id"]
    return df


def build_feature_dataset(runs_per_class: int = 6, duration_s: float = 2400.0, window_s: float = WINDOW_S,
                          path: str | None = "data/processed/features.csv", seed: int = 1000) -> pd.DataFrame:
    frames = []
    for i, spec in enumerate(training_run_specs(runs_per_class, duration_s, seed)):
        feats = extract_features(simulate_run(spec), window_s, seed=spec["seed"])
        feats["run_valve"] = spec["valve"]
        feats["run_speed_cmd"] = spec["speed"]
        frames.append(feats)
    out = pd.concat(frames, ignore_index=True)
    if path:
        out.to_csv(path, index=False)
    return out


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    data = build_feature_dataset()
    print(f"built {len(data)} windows from {data.run_id.nunique()} runs, {len(FEATURE_COLUMNS)} features "
          f"in {time.perf_counter() - t0:.0f} s")
    print("\nwindows per label:", data.fault_label.value_counts().to_dict())
    print("NaN fraction in features:", round(float(data[FEATURE_COLUMNS].isna().mean().mean()), 4))
    late = data[(data.fault_severity >= 0.4) | (data.fault_label == "healthy")]
    key = ["vibration_rms_mm_s_mean", "vib_ratio_2x_1x", "vib_env_bpfo_ratio", "vib_acc_hf_band_rms",
           "bearing_temp_rise_c", "flow_deficit_m3h_mean", "npsh_margin_mean", "bep_ratio_mean",
           "pump_eff_est_pct_mean"]
    pd.set_option("display.width", 250)
    print("\nclass means of key features (healthy + windows with severity >= 0.4):")
    print(late.groupby("fault_label")[key].mean().round(2).to_string())
