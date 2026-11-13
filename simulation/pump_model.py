"""
Virtual centrifugal pump + induction motor model.

Produces the TRUE (noise-free) physical state of the machine every time step.
Measurement noise, dropouts and sensor faults are added later by the sensor
simulator; fault physics enter through a ``FaultEffects`` object.

Main modelling assumptions (documented in docs/engineering_assumptions.md):
  * Pump curve       H(Q,n)  = hf * (H_shut*(n/n0)^2 - a*Q^2)       (parabolic)
  * Efficiency curve eta(x)  = eta_bep * x*(2-x),  x = Q / (Q_bep*n/n0)
    -> shaft power P = rho g H Q_bep (n/n0) / (eta_bep (2-x)), finite at Q = 0.
    Together these obey the affinity laws exactly (Q~n, H~n^2, P~n^3).
  * System curve     H_sys = H_static + k*Q^2 / valve_opening^2 * resistance
  * Operating point  = intersection of pump and system curve.
  * NPSHr(Q) = NPSHr_rated*(Q/Q_bep)^2 (floor at low flow). Head falls once
    NPSHa/NPSHr < 1.1 (3 % head drop at NPSHa = NPSHr, the NPSH3 definition).
  * Motor is VFD-fed with slip compensation: speed follows the command along
    ramps. Constant motor efficiency; reactive current = magnetising current
    plus a load-dependent leakage part -> realistic part-load power factor.
  * Temperatures: first-order lag towards a load-dependent steady state.
  * Vibration velocity RMS ~ (n/n0)^2, increased away from the BEP.
  * Acceleration output = HF-band (0.5-2.4 kHz) RMS: baseline + bearing /
    cavitation / mechanical HF sources (waveforms: simulation/vibration_signal.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import Enum

from analysis.hydraulic_calculations import hydraulic_power_kw, npsh_available_m, pump_curve_coefficients
from config.pump_config import GRAVITY, AppConfig, get_config

SQRT3 = math.sqrt(3.0)


class PumpState(str, Enum):
    STOPPED = "Stopped"
    STARTING = "Starting"
    RUNNING = "Running"
    SHUTDOWN = "Shutdown"
    EMERGENCY_STOP = "Emergency stop"


@dataclass
class FaultEffects:
    """Physical modifiers applied to the healthy model. Defaults = healthy.

    Multiplicative factors are 1.0 when healthy, additive terms 0.0.
    Vibration terms are independent sources and are added in quadrature.
    """
    head_factor: float = 1.0               # impeller damage / wear
    efficiency_factor: float = 1.0         # internal hydraulic losses
    system_resistance_factor: float = 1.0  # discharge blockage (>1)
    suction_pressure_drop_bar: float = 0.0 # suction strainer / low tank level
    seal_leakage_m3h: float = 0.0          # liquid lost through mechanical seal
    extra_friction_kw: float = 0.0         # bearing / misalignment losses
    bearing_temp_add_c: float = 0.0
    casing_temp_add_c: float = 0.0
    cooling_factor: float = 1.0            # <1 -> degraded cooling
    vibration_add_mm_s: float = 0.0        # at rated speed
    hf_accel_add_m_s2: float = 0.0         # high-frequency (bearing/cavitation)

    _MULT = ("head_factor", "efficiency_factor", "system_resistance_factor", "cooling_factor")
    _QUAD = ("vibration_add_mm_s", "hf_accel_add_m_s2")

    def combine(self, other: "FaultEffects") -> "FaultEffects":
        """Superpose two simultaneous faults."""
        out = {}
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if f.name in self._MULT:
                out[f.name] = a * b
            elif f.name in self._QUAD:
                out[f.name] = math.hypot(a, b)
            else:
                out[f.name] = a + b
        return FaultEffects(**out)


HEALTHY = FaultEffects()


class PumpModel:
    """Stateful pump/motor simulator. Call :meth:`step` once per time step."""

    MIN_VALVE_OPENING = 0.05

    def __init__(self, config: AppConfig | None = None, warm_start: bool = False):
        self.cfg = config or get_config()
        p, m, s = self.cfg.pump, self.cfg.motor, self.cfg.system

        # --- pump & system curve coefficients (rated speed) ---
        self.n0 = p.rated_speed_rpm
        self.q0 = p.rated_flow_m3h
        self.h_shut, self.a = pump_curve_coefficients(self.cfg)
        self.k_sys = (p.rated_head_m - s.static_head_m) / self.q0 ** 2
        self.k_suction = s.suction_loss_bar_at_rated / self.q0 ** 2
        self.p_suction_static = s.suction_pressure_bar + s.suction_loss_bar_at_rated
        self.shaft_power_rated_kw = hydraulic_power_kw(self.q0, p.rated_head_m, p.fluid_density) / p.rated_efficiency

        # --- motor electrical constants ---
        self.p_in_rated_kw = m.rated_power_kw / m.efficiency
        self.i_rated_calc = self.p_in_rated_kw * 1000 / (SQRT3 * m.rated_voltage_v * m.power_factor)
        self.i_reactive_rated = self.i_rated_calc * math.sqrt(1 - m.power_factor ** 2)
        self.i_magnetizing = m.magnetizing_current_ratio * self.i_rated_calc

        # --- dynamic state ---
        self.time_s = 0.0
        self.state = PumpState.STOPPED
        self.speed_rpm = 0.0
        self.speed_command_rpm = self.n0
        self.valve_opening = 1.0
        self.bearing_temp_c = self.cfg.thermal.ambient_temp_c
        self.casing_temp_c = self.cfg.thermal.fluid_temp_c
        if warm_start:
            self.state = PumpState.RUNNING
            self.speed_rpm = self.n0
            snap = self._compute(0.0, HEALTHY, accel_power_kw=0.0)
            self.bearing_temp_c = snap["_bearing_ss"]
            self.casing_temp_c = snap["_casing_ss"]

    # ------------------------------------------------------------------
    # Operator commands
    # ------------------------------------------------------------------
    def start(self):
        if self.state in (PumpState.STOPPED, PumpState.SHUTDOWN):
            self.state = PumpState.STARTING

    def stop(self):
        if self.state in (PumpState.STARTING, PumpState.RUNNING):
            self.state = PumpState.SHUTDOWN

    def emergency_stop(self):
        """SIMULATED ONLY - not a safety-rated function."""
        self.state = PumpState.EMERGENCY_STOP

    def reset(self):
        """Clear a latched emergency stop once the shaft is at rest."""
        if self.state == PumpState.EMERGENCY_STOP and self.speed_rpm == 0.0:
            self.state = PumpState.STOPPED

    def set_speed_command(self, rpm: float):
        limit = self.cfg.operation.max_speed_ratio * self.n0
        self.speed_command_rpm = min(max(float(rpm), 0.0), limit)

    def set_valve_opening(self, opening: float):
        self.valve_opening = min(max(float(opening), self.MIN_VALVE_OPENING), 1.0)

    # ------------------------------------------------------------------
    # Static (algebraic) hydraulics
    # ------------------------------------------------------------------
    def pump_head_m(self, flow_m3h, speed_rpm, head_factor=1.0):
        r = speed_rpm / self.n0
        return max(head_factor * (self.h_shut * r ** 2 - self.a * flow_m3h ** 2), 0.0)

    def system_head_m(self, flow_m3h, valve_opening=1.0, resistance_factor=1.0):
        k = self.k_sys * resistance_factor / valve_opening ** 2
        return self.cfg.system.static_head_m + k * flow_m3h ** 2

    def npsh_required_m(self, flow_m3h, speed_rpm):
        r = speed_rpm / self.n0
        npshr0 = self.cfg.pump.npsh_required_m
        return max(npshr0 * (flow_m3h / self.q0) ** 2, 0.3 * npshr0 * r ** 2)

    def operating_point(self, speed_rpm: float, valve_opening: float = 1.0,
                        effects: FaultEffects = HEALTHY) -> dict:
        """Solve pump curve = system curve (incl. cavitation head loss)."""
        p, s = self.cfg.pump, self.cfg.system
        r = speed_rpm / self.n0
        leak = effects.seal_leakage_m3h if r > 0 else 0.0
        kk = self.k_sys * effects.system_resistance_factor / max(valve_opening, self.MIN_VALVE_OPENING) ** 2

        cav = 1.0
        q_pump = 0.0
        for _ in range(40):
            hf = effects.head_factor * cav
            if r <= 0:
                q_pump = 0.0
            elif hf * (self.h_shut * r ** 2 - self.a * leak ** 2) - s.static_head_m <= 0:
                q_pump = leak                     # dead-head: only seal leakage flows
            else:
                # hf(H_shut r^2 - a Qp^2) = Hs + kk (Qp - L)^2
                A = hf * self.a + kk
                B = -2.0 * kk * leak
                C = kk * leak ** 2 + s.static_head_m - hf * self.h_shut * r ** 2
                q_pump = (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)
            p_suc = self.p_suction_static - self.k_suction * q_pump ** 2 - effects.suction_pressure_drop_bar
            npsha = npsh_available_m(p_suc, s.atmospheric_pressure_bar, p.fluid_vapour_pressure_bar, p.fluid_density)
            npshr = self.npsh_required_m(q_pump, speed_rpm) if r > 0 else 0.0
            margin = npsha / npshr if npshr > 0 else float("inf")
            cav_new = 1.0 - min(max(0.3 * (1.1 - margin), 0.0), 0.6)
            if abs(cav_new - cav) < 1e-6:
                break
            cav = 0.5 * (cav + cav_new)           # damped fixed-point iteration

        hf = effects.head_factor * cav
        head = self.pump_head_m(q_pump, speed_rpm, hf) if r > 0 else 0.0
        q_delivered = max(q_pump - leak, 0.0)
        p_h = hydraulic_power_kw(q_pump, head, p.fluid_density)
        if r > 0:
            x = q_pump / (self.q0 * r)
            p_shaft = (p.fluid_density * GRAVITY * head * (self.q0 / 3600.0) * r
                       / (p.rated_efficiency * max(2.0 - x, 0.1)) / 1000.0) / effects.efficiency_factor
        else:
            x, p_shaft = 0.0, 0.0
        return {
            "pump_flow_m3h": q_pump,
            "flow_m3h": q_delivered,
            "head_m": head,
            "suction_pressure_bar": p_suc,
            "discharge_pressure_bar": p_suc + head * p.fluid_density * GRAVITY / 1e5,
            "hydraulic_power_kw": p_h,
            "shaft_power_kw": p_shaft,
            "pump_efficiency": p_h / p_shaft if p_shaft > 0 else 0.0,
            "bep_ratio": x,
            "npsh_available_m": npsha,
            "npsh_required_m": npshr,
            "npsh_margin": margin,
            "cavitation_head_factor": cav,
        }

    # Incipient cavitation (bubble formation/collapse noise) starts well before
    # the 3 % head drop; NPSHi is typically 2-4 x NPSH3. Intensity 0 -> 1 as the
    # NPSH margin falls from INCIPIENT_MARGIN to 0.1.
    INCIPIENT_MARGIN = 2.5
    CAVITATION_VIBRATION_MM_S = 6.0
    CAVITATION_HF_ACCEL_M_S2 = 25.0

    @classmethod
    def cavitation_intensity(cls, npsh_margin: float) -> float:
        return min(max((cls.INCIPIENT_MARGIN - npsh_margin) / (cls.INCIPIENT_MARGIN - 0.1), 0.0), 1.0)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def _update_speed(self, dt: float) -> float:
        """Advance speed according to state; returns d(speed)/dt in rpm/s."""
        op = self.cfg.operation
        old = self.speed_rpm
        ramp_up = max(self.speed_command_rpm, 1.0) / op.start_ramp_s
        if self.state == PumpState.STARTING:
            self.speed_rpm = min(self.speed_rpm + ramp_up * dt, self.speed_command_rpm)
            if self.speed_rpm >= self.speed_command_rpm:
                self.state = PumpState.RUNNING
        elif self.state == PumpState.RUNNING:
            diff = self.speed_command_rpm - self.speed_rpm
            self.speed_rpm += max(min(diff, ramp_up * dt), -ramp_up * dt)
        elif self.state == PumpState.SHUTDOWN:
            self.speed_rpm -= self.n0 / op.stop_ramp_s * dt
            if self.speed_rpm <= 0.0:
                self.speed_rpm, self.state = 0.0, PumpState.STOPPED
        elif self.state == PumpState.EMERGENCY_STOP:
            self.speed_rpm *= math.exp(-dt / op.estop_time_constant_s)
            if self.speed_rpm < 5.0:
                self.speed_rpm = 0.0
        else:
            self.speed_rpm = 0.0
        return (self.speed_rpm - old) / dt if dt > 0 else 0.0

    def _compute(self, dt: float, effects: FaultEffects, accel_power_kw: float) -> dict:
        cfg, m, th, vib = self.cfg, self.cfg.motor, self.cfg.thermal, self.cfg.vibration
        r = self.speed_rpm / self.n0
        op = self.operating_point(self.speed_rpm, self.valve_opening, effects)
        running = r > 0

        # --- motor ---
        friction = effects.extra_friction_kw * r if running else 0.0
        energised = self.state in (PumpState.STARTING, PumpState.RUNNING, PumpState.SHUTDOWN)
        p_out = op["shaft_power_kw"] + friction + max(accel_power_kw, 0.0)
        p_in = p_out / m.efficiency if energised else 0.0
        if energised:
            i_active = p_in * 1000 / (SQRT3 * m.rated_voltage_v)
            load = p_in / self.p_in_rated_kw
            i_reactive = self.i_magnetizing + (self.i_reactive_rated - self.i_magnetizing) * load ** 2
            current = math.hypot(i_active, i_reactive)
            pf = i_active / current
        else:
            current, pf = 0.0, 0.0

        # --- thermal (steady-state targets) ---
        cool = max(effects.cooling_factor, 0.05)
        load_rel = op["shaft_power_kw"] / self.shaft_power_rated_kw
        if running:
            bearing_rise = ((th.bearing_temp_nominal_c - th.ambient_temp_c) * (0.5 * r + 0.5 * load_rel)
                            + th.bearing_rise_per_friction_kw * friction + effects.bearing_temp_add_c)
            mass_flow = cfg.pump.fluid_density * max(op["pump_flow_m3h"], 0.2) / 3600.0
            fluid_rise = (op["shaft_power_kw"] - op["hydraulic_power_kw"]) * 1000 / (mass_flow * th.fluid_specific_heat)
            casing_rise = fluid_rise + (th.casing_temp_nominal_c - th.fluid_temp_c) * load_rel + effects.casing_temp_add_c
        else:
            bearing_rise, casing_rise = 0.0, 0.0
        bearing_ss = th.ambient_temp_c + bearing_rise / cool
        casing_ss = th.fluid_temp_c + casing_rise / cool
        if dt > 0:
            self.bearing_temp_c += (bearing_ss - self.bearing_temp_c) * (1 - math.exp(-dt / th.bearing_time_constant_s))
            self.casing_temp_c += (casing_ss - self.casing_temp_c) * (1 - math.exp(-dt / th.casing_time_constant_s))

        # --- vibration (overall levels; waveforms are generated in Stage 2) ---
        if running:
            v_healthy = vib.baseline_rms_mm_s + vib.off_bep_vibration_mm_s * (op["bep_ratio"] - 1.0) ** 2
            cav_i = self.cavitation_intensity(op["npsh_margin"])
            v_cav = self.CAVITATION_VIBRATION_MM_S * cav_i
            v_rms = math.sqrt(v_healthy ** 2 + effects.vibration_add_mm_s ** 2 + v_cav ** 2) * r ** 2
            # Acceleration sensor = HF-band (0.5-2.4 kHz) acceleration RMS, the
            # usual bearing/cavitation condition band; running-speed content
            # is excluded by the band-pass.
            a_rms = math.sqrt(vib.baseline_hf_accel_m_s2 ** 2 + effects.hf_accel_add_m_s2 ** 2
                              + (self.CAVITATION_HF_ACCEL_M_S2 * cav_i) ** 2) * r ** 2
        else:
            v_rms, a_rms = 0.0, 0.0

        return {
            "time_s": self.time_s,
            "state": self.state.value,
            "speed_rpm": self.speed_rpm,
            "speed_command_rpm": self.speed_command_rpm,
            "valve_opening": self.valve_opening,
            **op,
            "motor_power_kw": p_in,
            "motor_current_a": current,
            "power_factor": pf,
            "bearing_temp_c": self.bearing_temp_c,
            "casing_temp_c": self.casing_temp_c,
            "vibration_rms_mm_s": v_rms,
            "vibration_accel_m_s2": a_rms,
            "_bearing_ss": bearing_ss,
            "_casing_ss": casing_ss,
        }

    def step(self, dt: float, effects: FaultEffects | None = None) -> dict:
        """Advance the simulation by dt seconds and return the true state."""
        effects = effects or HEALTHY
        dn_dt = self._update_speed(dt)
        omega = self.speed_rpm * 2 * math.pi / 60
        accel_power_kw = self.cfg.motor.rotor_inertia_kgm2 * omega * (dn_dt * 2 * math.pi / 60) / 1000
        self.time_s += dt
        snap = self._compute(dt, effects, accel_power_kw)
        return {k: v for k, v in snap.items() if not k.startswith("_")}


if __name__ == "__main__":
    import pandas as pd

    pump = PumpModel(warm_start=True)
    rated = pump.step(1.0)
    print("Rated operating point (warm start):")
    for k in ("flow_m3h", "head_m", "suction_pressure_bar", "discharge_pressure_bar", "hydraulic_power_kw",
              "shaft_power_kw", "pump_efficiency", "motor_power_kw", "motor_current_a", "power_factor",
              "bearing_temp_c", "casing_temp_c", "vibration_rms_mm_s", "vibration_accel_m_s2", "npsh_available_m"):
        print(f"  {k:24s} {rated[k]:9.3f}")

    print("\nCold start -> run -> throttle -> shutdown (selected rows):")
    pump = PumpModel()
    rows = []
    pump.start()
    for t in range(60):
        if t == 30:
            pump.set_valve_opening(0.5)
        if t == 45:
            pump.stop()
        rows.append(pump.step(1.0))
    df = pd.DataFrame(rows)
    cols = ["time_s", "state", "speed_rpm", "flow_m3h", "head_m", "discharge_pressure_bar",
            "motor_power_kw", "motor_current_a", "vibration_rms_mm_s"]
    print(df.loc[[0, 2, 4, 5, 10, 29, 31, 44, 46, 50, 53, 59], cols].round(2).to_string(index=False))
