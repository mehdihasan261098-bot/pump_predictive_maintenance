"""
Hydraulic engineering calculations for centrifugal pumps.

All functions accept Python floats or NumPy arrays (vectorised) so they can be
used both inside the simulator and on whole DataFrame columns.

Unit conventions (explicit in every argument name):
    flow      m3/h  (converted to m3/s internally: Q[m3/s] = Q[m3/h] / 3600)
    pressure  bar   (1 bar = 100 000 Pa)
    head      m of liquid column
    power     kW
"""
from __future__ import annotations

import numpy as np

from config.pump_config import GRAVITY, WATER_DENSITY

SECONDS_PER_HOUR = 3600.0
PA_PER_BAR = 1.0e5


# --------------------------------------------------------------------------
# Unit conversions
# --------------------------------------------------------------------------
def _scale(x, factor):
    """Multiply scalar or array by factor; scalars stay Python floats."""
    arr = np.asarray(x, dtype=float) * factor
    return float(arr) if arr.ndim == 0 else arr


def m3h_to_m3s(q_m3h):
    return _scale(q_m3h, 1.0 / SECONDS_PER_HOUR)


def m3s_to_m3h(q_m3s):
    return _scale(q_m3s, SECONDS_PER_HOUR)


def bar_to_pa(p_bar):
    return _scale(p_bar, PA_PER_BAR)


def pa_to_bar(p_pa):
    return _scale(p_pa, 1.0 / PA_PER_BAR)


def rpm_to_hz(n_rpm):
    return n_rpm / 60.0


def rpm_to_rad_s(n_rpm):
    return n_rpm * 2.0 * np.pi / 60.0


# --------------------------------------------------------------------------
# Pressure / head
# --------------------------------------------------------------------------
def differential_pressure_bar(discharge_bar, suction_bar):
    """dP = p_discharge - p_suction (both gauge or both absolute)."""
    return np.subtract(discharge_bar, suction_bar)


def head_from_pressure_m(dp_bar, density=WATER_DENSITY):
    """H = dP / (rho g). Velocity-head and elevation differences between the
    two pressure tappings are neglected (same pipe size, same elevation)."""
    return bar_to_pa(dp_bar) / (density * GRAVITY)


def pressure_from_head_bar(head_m, density=WATER_DENSITY):
    """dP = rho g H."""
    return pa_to_bar(np.multiply(head_m, density * GRAVITY))


# --------------------------------------------------------------------------
# Power and efficiency
# --------------------------------------------------------------------------
def hydraulic_power_kw(flow_m3h, head_m, density=WATER_DENSITY):
    """P_h = rho g Q H  with Q in m3/s; result in kW."""
    return _scale(np.multiply(m3h_to_m3s(flow_m3h), head_m), density * GRAVITY / 1000.0)


def efficiency_percent(output_kw, input_kw):
    """eta = P_out / P_in x 100. Returns 0 where input power <= 0."""
    out = np.asarray(output_kw, dtype=float)
    inp = np.asarray(input_kw, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(inp > 0, 100.0 * out / np.where(inp > 0, inp, 1.0), 0.0)
    return float(eta) if eta.ndim == 0 else eta


def pump_efficiency_percent(flow_m3h, head_m, shaft_power_kw, density=WATER_DENSITY):
    """Pump (hydraulic) efficiency = P_h / P_shaft x 100."""
    return efficiency_percent(hydraulic_power_kw(flow_m3h, head_m, density), shaft_power_kw)


def wire_to_water_efficiency_percent(flow_m3h, head_m, electrical_power_kw, density=WATER_DENSITY):
    """Overall efficiency = P_h / P_electrical x 100 (pump x motor)."""
    return efficiency_percent(hydraulic_power_kw(flow_m3h, head_m, density), electrical_power_kw)


def specific_energy_kwh_per_m3(electrical_power_kw, flow_m3h):
    """SEC = P[kW] / Q[m3/h]  ->  kWh per m3 pumped. NaN where flow <= 0."""
    p = np.asarray(electrical_power_kw, dtype=float)
    q = np.asarray(flow_m3h, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sec = np.where(q > 0, p / np.where(q > 0, q, 1.0), np.nan)
    return float(sec) if sec.ndim == 0 else sec


# --------------------------------------------------------------------------
# Affinity laws (same impeller diameter, same fluid, geometrically similar
# operating points; viscous/scale effects neglected)
# --------------------------------------------------------------------------
def affinity_flow(q1, n1, n2):
    """Q2 = Q1 (n2/n1)"""
    return q1 * (n2 / n1)


def affinity_head(h1, n1, n2):
    """H2 = H1 (n2/n1)^2"""
    return h1 * (n2 / n1) ** 2


def affinity_power(p1, n1, n2):
    """P2 = P1 (n2/n1)^3"""
    return p1 * (n2 / n1) ** 3


# --------------------------------------------------------------------------
# Suction side
# --------------------------------------------------------------------------
def npsh_available_m(suction_gauge_bar, atmospheric_bar, vapour_pressure_abs_bar,
                     density=WATER_DENSITY, velocity_head_m=0.0):
    """NPSHa = (p_s,abs - p_v) / (rho g) + v^2/(2g)  measured at pump suction."""
    p_abs = np.add(suction_gauge_bar, atmospheric_bar)
    return head_from_pressure_m(np.subtract(p_abs, vapour_pressure_abs_bar), density) + velocity_head_m


# --------------------------------------------------------------------------
# Pump characteristic (shared with simulation.pump_model)
# --------------------------------------------------------------------------
def pump_curve_coefficients(cfg) -> tuple[float, float]:
    """(H_shut, a) of the parabolic curve H = H_shut r^2 - a Q^2 through the
    rated point, Q in m3/h."""
    p = cfg.pump
    h_shut = p.shutoff_head_ratio * p.rated_head_m
    return h_shut, (h_shut - p.rated_head_m) / p.rated_flow_m3h ** 2


def expected_flow_from_head_m3h(head_m, speed_rpm, cfg):
    """Invert the pump curve: flow the healthy pump delivers at this head and
    speed. Q = sqrt((H_shut r^2 - H) / a). 0 above shut-off head, NaN when stopped."""
    h_shut, a = pump_curve_coefficients(cfg)
    r = np.asarray(speed_rpm, dtype=float) / cfg.pump.rated_speed_rpm
    with np.errstate(invalid="ignore"):
        q = np.sqrt(np.clip((h_shut * r ** 2 - np.asarray(head_m, dtype=float)) / a, 0.0, None))
    q = np.where(r > 0.05, q, np.nan)
    return float(q) if q.ndim == 0 else q


def expected_head_from_flow_m(flow_m3h, speed_rpm, cfg):
    """Healthy pump-curve head at this flow and speed (NaN when stopped)."""
    h_shut, a = pump_curve_coefficients(cfg)
    r = np.asarray(speed_rpm, dtype=float) / cfg.pump.rated_speed_rpm
    h = np.clip(h_shut * r ** 2 - a * np.asarray(flow_m3h, dtype=float) ** 2, 0.0, None)
    h = np.where(r > 0.05, h, np.nan)
    return float(h) if h.ndim == 0 else h


def performance_curve(cfg, speed_rpm: float | None = None, points: int = 61):
    """Healthy H-Q, efficiency-Q and shaft-power-Q curves at one speed.
    Same equations as the simulator (see docs/engineering_assumptions.md)."""
    import pandas as pd

    p = cfg.pump
    n = p.rated_speed_rpm if speed_rpm is None else speed_rpm
    r = n / p.rated_speed_rpm
    h_shut, a = pump_curve_coefficients(cfg)
    q = np.linspace(0.0, np.sqrt(h_shut / a) * r, points)
    h = h_shut * r ** 2 - a * q ** 2
    x = q / (p.rated_flow_m3h * r)
    eta = np.clip(p.rated_efficiency * x * (2 - x), 0.0, None)
    p_shaft = (p.fluid_density * GRAVITY * h * (p.rated_flow_m3h / SECONDS_PER_HOUR) * r
               / (p.rated_efficiency * np.maximum(2 - x, 0.1)) / 1000.0)
    return pd.DataFrame({"flow_m3h": q, "head_m": h, "efficiency_pct": 100 * eta,
                         "shaft_power_kw": p_shaft, "speed_rpm": n})


# --------------------------------------------------------------------------
# KPIs from MEASURED sensor data
# --------------------------------------------------------------------------
STOPPED_SPEED_RPM = 10.0


def compute_kpis(df, cfg=None):
    """Derive engineering KPIs from measured sensor columns.

    Required columns: suction_pressure_bar, discharge_pressure_bar, flow_m3h,
    motor_power_kw, speed_rpm. Missing (NaN) inputs propagate to NaN outputs.
    Curve-based KPIs are NaN while the pump is stopped.

    Returns a new DataFrame with the KPI columns (same index as df).
    """
    import pandas as pd

    from config.pump_config import get_config

    cfg = cfg or get_config()
    p, s, m = cfg.pump, cfg.system, cfg.motor
    speed = df["speed_rpm"].to_numpy(dtype=float)
    running = speed >= STOPPED_SPEED_RPM
    flow = df["flow_m3h"].to_numpy(dtype=float)
    p_el = df["motor_power_kw"].to_numpy(dtype=float)

    out = pd.DataFrame(index=df.index)
    out["diff_pressure_bar"] = differential_pressure_bar(df["discharge_pressure_bar"].to_numpy(dtype=float),
                                                         df["suction_pressure_bar"].to_numpy(dtype=float))
    head = head_from_pressure_m(out["diff_pressure_bar"].to_numpy(), p.fluid_density)
    out["head_m"] = head
    out["hydraulic_power_kw"] = hydraulic_power_kw(flow, head, p.fluid_density)
    w2w = efficiency_percent(out["hydraulic_power_kw"].to_numpy(), p_el)
    out["wire_to_water_eff_pct"] = np.where(running & ~np.isnan(p_el) & ~np.isnan(flow), w2w, np.nan)
    out["pump_eff_est_pct"] = out["wire_to_water_eff_pct"] / m.efficiency
    out["sec_kwh_m3"] = np.where(running, specific_energy_kwh_per_m3(p_el, flow), np.nan)
    out["motor_load_pct"] = 100.0 * p_el * m.efficiency / m.rated_power_kw
    out["speed_ratio"] = speed / p.rated_speed_rpm
    with np.errstate(divide="ignore", invalid="ignore"):
        out["bep_ratio"] = np.where(running, flow / (p.rated_flow_m3h * out["speed_ratio"].to_numpy()), np.nan)

    exp_q = expected_flow_from_head_m3h(head, speed, cfg)
    out["expected_flow_m3h"] = exp_q
    out["flow_deficit_m3h"] = exp_q - flow
    exp_h = expected_head_from_flow_m(flow, speed, cfg)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["head_deviation_pct"] = np.where(exp_h > 0, 100.0 * (head - exp_h) / exp_h, np.nan)

    # System-curve deviation: measured system resistance k = (H - H_static)/Q^2
    # divided by the resistance expected for the commanded valve opening
    # (k_design / opening^2). ~1 at ANY healthy operating point (also when
    # deliberately throttled); > 1 only for an unexpected restriction.
    if "valve_opening" in df:
        k_design = (p.rated_head_m - s.static_head_m) / p.rated_flow_m3h ** 2
        valve = np.clip(df["valve_opening"].to_numpy(dtype=float), 0.05, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            k_meas = (head - s.static_head_m) / flow ** 2
            ratio = k_meas / (k_design / valve ** 2)
        out["system_resistance_ratio"] = np.where(running & (flow > 3.0), ratio, np.nan)
    else:
        out["system_resistance_ratio"] = np.nan

    npsha = npsh_available_m(df["suction_pressure_bar"].to_numpy(dtype=float), s.atmospheric_pressure_bar,
                             p.fluid_vapour_pressure_bar, p.fluid_density)
    r = np.clip(speed / p.rated_speed_rpm, 0.0, None)
    npshr = np.maximum(p.npsh_required_m * (np.nan_to_num(flow) / p.rated_flow_m3h) ** 2,
                       0.3 * p.npsh_required_m * r ** 2)
    out["npsh_available_m"] = npsha
    out["npsh_required_m"] = np.where(running, npshr, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["npsh_margin"] = np.where(running, npsha / npshr, np.nan)
    return out
