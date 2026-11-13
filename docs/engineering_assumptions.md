# Engineering Assumptions and Limitations

All values are **illustrative simulation assumptions**. They are chosen to be
physically plausible for a small end-suction centrifugal water pump, but they
are not certified limits and the simulation is **not** a validated model of any
real machine. Results obtained on this synthetic data do not establish
industrial diagnostic accuracy.

## 1. Pump and system (`simulation/pump_model.py`)

| Item | Assumption |
|---|---|
| Rated point | Q = 30 m³/h, H = 25 m, η = 70 %, n = 1450 rpm (water, ρ = 1000 kg/m³) |
| Pump curve | H = hf·(H_shut·r² − a·Q²), H_shut = 1.25·H_rated, r = n/n_rated |
| Efficiency curve | η = η_bep·x(2−x), x = Q/(Q_bep·r) — peak at BEP, zero at shut-off |
| Shaft power | P = ρgHQ_bep·r / (η_bep(2−x)) — algebraically equal to ρgQH/η but finite at Q = 0 (≈62 % of rated at shut-off) |
| Affinity laws | Q ∝ n, H ∝ n², P ∝ n³ hold exactly between similar points (same impeller, same fluid, viscous/scale effects neglected) |
| System curve | H_sys = 10 m static + k·Q²·resistance/valve², calibrated through the rated point |
| Suction | 1.2 bar(g) at rated flow, 0.1 bar suction-line loss ∝ Q², vapour pressure 0.032 bar(a), atmosphere 1.013 bar |
| NPSHr | 3 m at rated flow, ∝ Q² with a low-flow floor |
| Cavitation | Incipient (noise) below NPSHa/NPSHr = 2.5; head drop 0.3·(1.1 − margin) (3 % at margin 1.0 — NPSH3 definition), max 60 % |
| Motor | 7.5 kW, 400 V 3-ph, η = 0.88 const, PF 0.85 at rated load, VFD with slip compensation (speed = command) |
| Motor current | I = √(I_active² + I_reactive²); I_reactive = I_mag (35 % of rated) + leakage part ∝ load² → PF ≈ 0.66 at 39 % load |
| Start/stop | 5 s ramp up, 8 s controlled ramp down, E-stop = de-energised coast, τ = 1.5 s, latched until reset |
| Thermal | First-order lags (bearing τ = 600 s, casing τ = 300 s). Bearing rise 30 °C at rated duty, split 50 % speed / 50 % load, +30 °C per kW extra friction. Casing = fluid (30 °C) + liquid heating by pump losses + load term |
| Vibration (overall) | v = √(v_base² + v_offBEP² + Σv_fault² + v_cav²)·r², v_base = 1.8 mm/s, v_offBEP = 0.8·(x−1)². Acceleration sensor = **HF-band (0.5–2.4 kHz) RMS** = √(0.3² + Σa_HF,fault² + a_cav²)·r² m/s² |

### Vibration waveforms (`simulation/vibration_signal.py`, `analysis/vibration_analysis.py`)

| Item | Assumption |
|---|---|
| Sampling | fs = 5 kHz (Nyquist 2.5 kHz), 1 s records → Δf = 1 Hz, Hann window, amplitude-corrected |
| Healthy velocity | 1X, 2X, 3X, vane-pass 6X (145 Hz), broadband 10–300 Hz |
| Imbalance | 1X dominant (2X/1X ≈ 0.1) |
| Misalignment | 2X > 1X (2X/1X ≈ 1.4) plus 3X; axial direction not simulated (single sensor) |
| Bearing | Outer-race defect assumed: impact train at BPFO (±1 % jitter = slip) exciting a 1.8 kHz resonance, ζ = 0.05 → envelope lines at BPFO and harmonics |
| Bearing frequencies | Kinematic formulas, pure rolling, rigid, α = 0°: BPFO 73.33, BPFI 120.00, BSF 47.14, FTF 9.17 Hz at 1450 rpm. Real values deviate 1–2 % (slip) → peaks are searched in ±2 % bands |
| Cavitation | Bursty band-limited noise 0.5–2.4 kHz (amplitude-modulated 2–40 Hz) + broadband velocity + vane-pass |
| Real structural resonances are typically 2–10 kHz; scaled down to fit fs = 5 kHz | |

The rated duty uses only **39 %** of the 7.5 kW motor nameplate (shaft power
2.92 kW). This follows directly from the specified pump data and was kept.

## 2. Sensors (`simulation/sensor_simulator.py`)

* Gaussian noise (1σ): temperatures 0.2 °C, velocity 0.05 mm/s, acceleration 0.05 m/s², suction 0.01 bar, discharge 0.02 bar, flow 0.3 m³/h, current 0.05 A, power 0.03 kW, speed 2 rpm.
* Flow-meter low-flow cut-off 0.5 m³/h; speed encoder reads 0 below 10 rpm; RMS values non-negative.
* Instrument failures (separate from pump faults): dead (NaN), stuck, drift, offset, spike; random dropouts.

## 3. Fault models (`simulation/fault_simulator.py`)

Severity s ∈ [0, 1]. Effects are applied to the physics; sensor responses then
emerge from the model (operating point, motor load, heat, off-BEP vibration).

| Fault | Physical mechanism | Imposed effect at severity s | Main sensor signature at s = 1 |
|---|---|---|---|
| Bearing degradation | Race/ball spalls → impacts at BPFO/BPFI/BSF; wear raises friction | HF accel +8s m/s²; friction +0.4s² kW; bearing +8s² °C; velocity +4s² mm/s | Accel ×20, bearing +15 °C, velocity +144 %, current +7 % |
| Imbalance | Residual unbalance, force ∝ m·e·ω² at 1X | velocity +9s mm/s (1X); bearing +3s °C | Velocity ×5 (≈9 mm/s); little heat |
| Misalignment | Coupling reaction forces at 1X/2X/axial; coupling losses | velocity +7s; friction +0.3s kW; bearing +8s °C | Velocity ×4, bearing +15 °C, power +10 % |
| Cavitation | Suction-side restriction / low tank level → NPSHa falls | suction pressure −2.1s bar (noise & head loss produced by the model) | Suction < 0 bar(g), head −66 %, flow −12 %, accel ×50 |
| Flow restriction | Discharge blockage / fouling → steeper system curve | system resistance ×(1+9s) | Flow −63 %, discharge +16 %, power −25 %, velocity +18 % |
| Seal leakage | Worn seal faces → liquid lost before the flow meter; face friction heat | leakage 3s m³/h; casing +6s °C | Delivered flow −3 % while pump flow rises; casing +17 % |
| Overheating | Loss of cooling / degraded lubricant / hot ambient | cooling capacity ×(1−0.5s); bearing +5s °C | Bearing +37 °C (≈92 °C), casing +14 % |

Severity time profiles: step, linear ramp, exponential (wear-out, slow start
then accelerating). Simultaneous faults are superposed: multiplicative factors
multiply, additive terms add, vibration sources add in quadrature.

**Distinguishing imbalance from misalignment** is not possible from overall
levels alone (both raise velocity). Stage 2 vibration spectra separate them:
imbalance → dominant 1X; misalignment → strong 2X (and axial 1X).

## 4. Diagnosis, health index and RUL

* **Rule thresholds** (`RuleThresholds`) are set from the simulated healthy baseline with safety margins; velocity limits are inspired by ISO 10816 zone boundaries. Overheating uses a load-compensated thermal residual (expected rise = 30 °C·(0.5·n/n₀ + 0.5·P_shaft/P_shaft,rated)); on real equipment this baseline would be regressed from healthy history.
* **Flow restriction** uses the system-resistance ratio k_meas/k_expected with k_meas = (H − H_static)/Q² and k_expected = k_design/valve². It is ≈ 1 at any healthy operating point (also when deliberately throttled) and rises only for unexpected restriction. This assumes the valve position and static head are known (available in a real SCADA); without valve information the rule falls back to the flow/BEP ratio.
* **Health index**: indicator damage piecewise-linear (good → 0, warning → 0.45, critical → 0.8); component = worst indicator; D = 0.7·worst component + 0.3·weighted mean; HI = 100(1−D), EWMA α = 0.3. The bearing envelope ratio only shows that a defect exists, so its damage is capped at warning level; severity follows HF acceleration. The thermal residual is ignored while the bearing is cooling (load reduction transient).
* **RUL** is a **simulated estimate**: best of linear-HI or exponential-damage fits over the last 15 evaluations, extrapolated to HI = 40, only if HI falls faster than 1 point/h with R² ≥ 0.6. Early estimates on exponential wear are unreliable; estimates are conservative (pessimistic) early in linear degradations. It is not a validated industrial prognosis.
* **ML**: performance on synthetic data does not establish industrial diagnostic accuracy; the simulator and the features share modelling assumptions.

## 5. Known limitations

* Constant motor efficiency; no motor-winding temperature or electrical faults.
* Single-phase water properties; no temperature-dependent density/vapour pressure.
* Seal leakage rate prescribed (not ∝ √pressure).
* Fault effect magnitudes are engineering estimates, not fitted to field data.
* Simulated E-stop is **not** a safety function and must never be used as one.
