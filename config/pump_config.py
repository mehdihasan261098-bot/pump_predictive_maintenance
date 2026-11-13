"""
Central configuration for the centrifugal pump monitoring system.

Edit the values in this file to change the simulated pump. All values are
ILLUSTRATIVE SIMULATION ASSUMPTIONS, not certified industrial limits.

Units are stated in every field name or comment. SI units are used internally;
"engineering" units (m3/h, bar, RPM, kW) are used where operators expect them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project root: relative data/model paths are resolved against it so the app
# works regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


# --------------------------------------------------------------------------
# Physical constants
# --------------------------------------------------------------------------
GRAVITY = 9.81            # m/s^2
WATER_DENSITY = 1000.0    # kg/m^3 (clean water at ~20 degC)


@dataclass
class PumpConfig:
    """Rated (best efficiency point) data of the pump at nominal speed."""
    rated_flow_m3h: float = 30.0          # Q at BEP
    rated_head_m: float = 25.0            # H at BEP
    rated_efficiency: float = 0.70        # hydraulic/pump efficiency at BEP (0-1)
    rated_speed_rpm: float = 1450.0       # 4-pole motor at 50 Hz with slip
    shutoff_head_ratio: float = 1.25      # H(Q=0) / H_rated, typical 1.1-1.3
    impeller_vanes: int = 6               # gives vane-pass frequency = vanes * 1X
    fluid_density: float = WATER_DENSITY
    fluid_vapour_pressure_bar: float = 0.032   # water at ~25 degC (absolute)
    npsh_required_m: float = 3.0          # NPSHr at rated flow (assumed)


@dataclass
class MotorConfig:
    """Induction motor driving the pump."""
    rated_power_kw: float = 7.5           # nameplate output power
    rated_voltage_v: float = 400.0        # line-to-line, 3-phase
    power_factor: float = 0.85
    efficiency: float = 0.88              # motor efficiency (assumed constant)
    supply_frequency_hz: float = 50.0
    poles: int = 4
    rated_current_a: float = 14.5         # typical nameplate for 7.5 kW / 400 V
    # Magnetising (no-load) reactive current as a fraction of rated current.
    # Gives the realistic drop of power factor at part load.
    magnetizing_current_ratio: float = 0.35
    rotor_inertia_kgm2: float = 0.06      # motor + coupling + impeller


@dataclass
class OperationConfig:
    """Start/stop behaviour. The motor is assumed to be VFD-driven with slip
    compensation, so the actual speed follows the speed command."""
    start_ramp_s: float = 5.0             # 0 -> command speed
    stop_ramp_s: float = 8.0              # command speed -> 0 (controlled stop)
    estop_time_constant_s: float = 1.5    # free coast-down after E-stop
    max_speed_ratio: float = 1.05         # max command / rated speed


@dataclass
class ThermalConfig:
    """Thermal behaviour (first-order models)."""
    ambient_temp_c: float = 25.0
    bearing_temp_nominal_c: float = 55.0  # steady-state at rated duty
    casing_temp_nominal_c: float = 35.0   # roughly fluid temperature + margin
    fluid_temp_c: float = 30.0            # pumped-liquid inlet temperature
    fluid_specific_heat: float = 4186.0   # J/(kg K), water
    bearing_time_constant_s: float = 600.0
    casing_time_constant_s: float = 300.0
    bearing_rise_per_friction_kw: float = 30.0   # degC per kW of extra friction


@dataclass
class SystemConfig:
    """Piping system the pump is connected to (operating point)."""
    suction_pressure_bar: float = 1.2     # gauge pressure at pump inlet, rated flow
    suction_loss_bar_at_rated: float = 0.1  # suction-line friction loss at rated flow
    atmospheric_pressure_bar: float = 1.013
    static_head_m: float = 10.0           # static lift part of the system curve
    # The system curve H_sys = static + k*Q^2 is calibrated so that it crosses
    # the pump curve at the rated point (done in the pump model).


@dataclass
class BearingConfig:
    """Rolling-element bearing geometry (approximate, deep-groove type 6309)."""
    n_balls: int = 8
    ball_diameter_mm: float = 17.5
    pitch_diameter_mm: float = 72.5
    contact_angle_deg: float = 0.0


@dataclass
class VibrationConfig:
    sampling_rate_hz: float = 5000.0      # >> 2 x highest frequency of interest
    window_seconds: float = 1.0           # length of one vibration snapshot
    baseline_rms_mm_s: float = 1.8        # healthy pump, ISO 10816 zone A/B
    off_bep_vibration_mm_s: float = 0.8   # extra RMS per (Q/Q_bep - 1)^2
    baseline_hf_accel_m_s2: float = 0.3   # broadband high-frequency acceleration
    noise_level_mm_s: float = 0.15


@dataclass
class SensorConfig:
    sampling_interval_s: float = 1.0      # process-data sampling interval
    # Gaussian measurement noise (1 sigma) per sensor
    noise_std: dict = field(default_factory=lambda: {
        "bearing_temp_c": 0.2,
        "casing_temp_c": 0.2,
        "vibration_rms_mm_s": 0.05,
        "vibration_accel_m_s2": 0.05,
        "suction_pressure_bar": 0.01,
        "discharge_pressure_bar": 0.02,
        "flow_m3h": 0.3,
        "motor_current_a": 0.05,
        "motor_power_kw": 0.03,
        "speed_rpm": 2.0,
    })
    dropout_probability: float = 0.0      # probability of a missing reading


@dataclass
class RuleThresholds:
    """Rule-based diagnosis limits. Illustrative values for THIS simulated
    pump (healthy: 1.8 mm/s, 55 degC bearing, NPSH margin ~7-10); velocity
    limits are inspired by ISO 10816 zone boundaries for small machines.
    Real limits must come from machine baselines and standards."""
    min_running_speed_ratio: float = 0.5        # rules evaluated above this speed
    # general vibration (velocity RMS, mm/s)
    velocity_warning: float = 4.5
    velocity_critical: float = 7.1
    # bearing temperature (absolute, degC) and rise above ambient (degC)
    bearing_temp_warning: float = 70.0
    bearing_temp_critical: float = 85.0
    bearing_rise_warning: float = 38.0          # healthy ~30 at rated duty
    bearing_rise_critical: float = 50.0
    casing_rise_warning: float = 9.0            # healthy ~5
    # load-compensated thermal residual: measured rise - expected rise (degC)
    thermal_residual_warning: float = 6.0
    thermal_residual_critical: float = 15.0
    thermal_min_slope_c_min: float = -0.1       # ignore cooling transients
    # imbalance: 1X peak amplitude (mm/s) with low 2X/1X
    imbalance_1x_warning: float = 3.5           # healthy 1X peak ~2.0
    imbalance_1x_critical: float = 8.0
    imbalance_max_2x_1x: float = 0.5
    # misalignment: 2X/1X ratio and minimum 2X amplitude
    misalignment_ratio_warning: float = 0.8
    misalignment_ratio_critical: float = 1.2
    misalignment_min_2x: float = 1.5
    # rolling-element bearing: envelope SNR at defect frequency + HF / kurtosis
    bearing_env_ratio: float = 5.0
    bearing_hf_warning: float = 0.8             # m/s2, healthy 0.3
    bearing_hf_critical: float = 4.0
    bearing_kurtosis: float = 3.8
    # cavitation
    npsh_margin_warning: float = 4.0
    npsh_margin_critical: float = 1.5
    suction_pressure_low_bar: float = 0.8       # healthy 1.2 +/- 0.05 bar
    cavitation_hf: float = 3.0
    cavitation_head_drop_pct: float = -3.0
    # flow restriction: system resistance vs expected for commanded valve (primary)
    resistance_ratio_warning: float = 1.6
    resistance_ratio_critical: float = 3.0
    # fallback when valve position unknown: flow / BEP flow at current speed
    bep_ratio_warning: float = 0.6
    bep_ratio_critical: float = 0.4
    # seal leakage: flow deficit vs pump curve (m3/h)
    flow_deficit_warning: float = 0.8
    flow_deficit_critical: float = 2.0


@dataclass
class HealthBands:
    """Illustrative health-index bands (0-100). Not industrial standards."""
    healthy_min: float = 85.0
    caution_min: float = 65.0
    degraded_min: float = 40.0            # below this -> Critical
    # component weights for the weighted-mean part of the damage
    weights: dict = field(default_factory=lambda: {
        "vibration": 0.25, "bearing": 0.25, "thermal": 0.20, "hydraulic": 0.15, "suction": 0.15})
    # damage D = worst_share * max(component) + (1 - worst_share) * weighted mean
    worst_share: float = 0.7
    # indicator damage anchors: good -> 0, warning -> 0.45, critical -> 0.8
    damage_at_warning: float = 0.45
    damage_at_critical: float = 0.8
    ewma_alpha: float = 0.3               # smoothing per evaluation window
    trend_windows: int = 15               # history length used for trend / RUL
    rul_min_slope_per_h: float = 1.0      # HI must fall faster than this to estimate RUL
    rul_min_r2: float = 0.6


@dataclass
class AppConfig:
    pump: PumpConfig = field(default_factory=PumpConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    operation: OperationConfig = field(default_factory=OperationConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    bearing: BearingConfig = field(default_factory=BearingConfig)
    vibration: VibrationConfig = field(default_factory=VibrationConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    health: HealthBands = field(default_factory=HealthBands)
    rules: RuleThresholds = field(default_factory=RuleThresholds)
    database_path: str = "data/pump_monitoring.db"
    random_seed: int = 42


def get_config() -> AppConfig:
    """Return a fresh default configuration object."""
    return AppConfig()


if __name__ == "__main__":
    cfg = get_config()
    q = cfg.pump.rated_flow_m3h / 3600.0
    p_h = cfg.pump.fluid_density * GRAVITY * q * cfg.pump.rated_head_m
    p_shaft = p_h / cfg.pump.rated_efficiency
    p_elec = p_shaft / cfg.motor.efficiency
    print(f"Hydraulic power  : {p_h / 1000:.3f} kW")
    print(f"Shaft power      : {p_shaft / 1000:.3f} kW")
    print(f"Electrical input : {p_elec / 1000:.3f} kW")
    print(f"Motor loading    : {100 * p_shaft / 1000 / cfg.motor.rated_power_kw:.1f} % of nameplate")
