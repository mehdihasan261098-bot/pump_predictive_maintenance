# PROJECT STATUS — Pump Monitoring & Predictive Maintenance

Authoritative progress record. A new conversation should read this file first.
Last updated: 2026-11-13

## Environment (verified)
| Item | Value |
|---|---|
| OS | Windows 11 Pro |
| Python | 3.12.10 (system), project venv at `.venv/` |
| Run Python | `.venv\Scripts\python.exe` (global `pip` script is permission-blocked; always use `python -m pip`) |
| Project root | `D:\Pump_project\pump_predictive_maintenance` |
| Installed | numpy 2.5.4, pandas 2.3.3, scipy 1.18.1, matplotlib 3.11.2, plotly 6.9.0, streamlit 1.67.0, **scikit-learn 1.7.2 (pinned)**, joblib 1.6.0, pytest 9.2.0, **pyarrow 21.0.0 (pinned)** |

## Key technical decisions
1. **Project venv** (`.venv`) isolates dependencies from the global Python.
2. **pyarrow pinned to 21.0.0**: pyarrow 26.0.1 `_compute` DLL is blocked by Windows Application Control ("An Application Control policy has blocked this file"), which breaks `import pandas`. 21.0.0 and 19.0.1 load fine; `pip check` clean.
3. **Config as dataclasses** in `config/pump_config.py` (PumpConfig, MotorConfig, ThermalConfig, SystemConfig, BearingConfig, VibrationConfig, SensorConfig, HealthBands). SI internally; m3/h, bar, RPM, kW at interfaces.
4. Derived rated chain (hand-verified): P_h = 2.044 kW, shaft = 2.920 kW, electrical = 3.318 kW → motor at **38.9 %** of 7.5 kW nameplate (the specified pump is lightly loading the motor; kept as given, documented).
5. Assumed motor: 400 V, 3-phase, PF 0.85, η 0.88, 4-pole 50 Hz, 14.5 A nameplate. Bearing: 6309-type (8 balls, Bd 17.5 mm, Pd 72.5 mm). Impeller: 6 vanes. Vibration fs = 5 kHz, 1 s windows.
6. `.streamlit/config.toml`: telemetry disabled, dark industrial theme.

## Stage / task status
| Task | Title | Status |
|---|---|---|
| 1.1 | Environment setup & project directory | **VERIFIED** |
| 1.2 | Pump and motor model | **VERIFIED** |
| 1.3 | Virtual sensor simulator | **VERIFIED** |
| 1.4 | Fault simulation (7 faults) | **VERIFIED** |
| 1.5 | Hydraulic engineering calculations | **VERIFIED** |
| 1.6 | Data acquisition | **VERIFIED** |
| 1.7 | SQLite database | **VERIFIED** — **STAGE 1 COMPLETE** |
| 2.1 | Vibration signal generation | **VERIFIED** |
| 2.2 | Vibration analysis & FFT | **VERIFIED** |
| 2.3 | Feature extraction | **VERIFIED** |
| 2.4 | Rule-based diagnosis | **VERIFIED** |
| 2.5 | ML anomaly detection (Isolation Forest) | **VERIFIED** |
| 2.6 | ML fault classification (Random Forest) | **VERIFIED** |
| 2.7 | Health index | **VERIFIED** |
| 2.8 | Predictive maintenance recommendations | **VERIFIED** — **STAGE 2 COMPLETE** |
| 3.1 | Dashboard architecture (`app.py`) | **VERIFIED** (browser) |
| 3.2 | Equipment overview page | **VERIFIED** (browser) |
| 3.3 | Real-time monitoring page | **VERIFIED** (browser) |
| 3.4 | Vibration analysis page | **VERIFIED** (browser) |
| 3.5 | Fault detection page | **VERIFIED** (browser) |
| 3.6 | Predictive maintenance page | **VERIFIED** (browser) |
| 3.7 | Historical data & reports page | **VERIFIED** (browser; download buttons rendered, files not downloaded) |
| 3.8 | Alarm management | **VERIFIED** — **STAGE 3 COMPLETE** |
| 4.1 | System integration | **VERIFIED** (engine + scenario validation) |
| 4.2 | Automated testing | **VERIFIED** (221 tests) |
| 4.3 | Fault-scenario validation | **VERIFIED** (`reports/validation_report.md`) |
| 4.4 | Debugging & performance | **VERIFIED** |
| 4.5 | Dashboard execution verification | **VERIFIED** (browser, Stage 3) |
| 4.6 | Documentation (README, architecture, assumptions) | **VERIFIED** (Mermaid diagram validated) |
| 4.7 | GitHub preparation | **VERIFIED** (pushed to https://github.com/mehdihasan261098-bot/pump_predictive_maintenance, public) |
| 4.8 | CV & LinkedIn descriptions | **IMPLEMENTED** (`docs/CV_and_LinkedIn.md`; wording for user review) |
| 4.9 | Final backup & handover | **VERIFIED** — **ALL 4 STAGES COMPLETE** |

### Roadmap — optional (Priority 3, DEFERRED until core complete)
Multi-pump monitoring · pump comparison · energy optimisation · maintenance scheduling · Modbus TCP server (pymodbus) · OPC UA server (asyncua) · digital twin view · email notifications · advanced ML.

## Task log

### Task 1.1 — Environment setup — VERIFIED
- **Created:** folder tree (config, simulation, acquisition, analysis, detection, predictive_maintenance, database, alarms, dashboard, models/saved_models, data/raw, data/processed, reports, tests, docs) with `__init__.py` in packages; `requirements.txt`; `.gitignore`; `.streamlit/config.toml`; `config/pump_config.py`; `tests/test_environment.py`; this file.
- **Commands run:**
  - `python --version` → 3.12.10
  - `python -m venv .venv`; `.venv\Scripts\python.exe -m pip install -r requirements.txt`
  - `python -m pip check` → No broken requirements
  - `python -m compileall -q config tests simulation analysis` → OK
  - `python -m config.pump_config` → P_h 2.044 kW / shaft 2.920 kW / elec 3.318 kW
  - `python -m pytest tests/ -q` → **13 passed** (first run: 1 failed — pandas import, see issue I-1)
  - Headless `streamlit run` of a scratch script, `GET /_stcore/health` → `ok`, then stopped.
- **Not yet verifiable:** `streamlit run app.py` (app.py is created in Task 3.1).

### Task 1.2 — Pump and motor model — VERIFIED
- **Created:** `analysis/hydraulic_calculations.py` (unit conversions, dP, H=dP/(rho g), P_h=rho g Q H, efficiency, wire-to-water, SEC, affinity laws, NPSHa — vectorised), `simulation/pump_model.py`, `tests/test_pump_model.py`.
- **Modified:** `config/pump_config.py` (+OperationConfig ramps/E-stop; motor magnetising ratio & inertia; fluid temp, cp, bearing friction-rise; suction loss, atm pressure; off-BEP & HF vibration baselines).
- **Model design (decisions):**
  - `PumpModel.step(dt, FaultEffects)` returns the TRUE state dict; sensors add noise later (1.3), faults are expressed as a `FaultEffects` dataclass (head/efficiency/system-resistance factors, suction drop, seal leakage, extra friction, temp adds, cooling factor, vibration/HF-accel adds; `combine()` superposes simultaneous faults, vibration in quadrature). Task 1.4 maps fault type+severity → FaultEffects.
  - Pump curve H = hf(H_shut r² − aQ²), H_shut = 1.25·H_r; η = η_bep·x(2−x); shaft power in cancelled form (finite at Q=0). Exact affinity-law consistency (tested).
  - System curve Hs=10 m + kQ²/valve²·resistance; quadratic solve incl. seal leakage; dead-head when shut-off head < static head.
  - Cavitation: NPSHr ∝ Q² (floor), head factor falls when NPSHa/NPSHr < 1.1 (3 % at ratio 1.0), damped fixed-point iteration.
  - Motor: VFD with slip compensation (speed = command, ramps 5 s up / 8 s down, E-stop coast τ=1.5 s, latched until `reset()` at zero speed; de-energised). Reactive current = I_mag + leakage·load² → PF 0.66 at rated duty. Acceleration power J·ω·dω/dt included.
  - Thermal first-order lags; bearing rise ∝ (0.5r + 0.5 load) + 30 °C/kW extra friction; casing = fluid 30 °C + fluid heating by losses + load term.
  - Vibration overall: (1.8 + 0.8(x−1)²) mm/s, fault adds in quadrature, all ×r²; accel = √((2πf₁v)² + HF²).
- **Commands:** `python -m simulation.pump_model` (demo), `python -m compileall -q analysis simulation config tests`, `python -m pytest tests/ -q` → **31 passed**.
- **Inspected output (rated, warm start):** Q 30.000 m³/h, H 25.000 m, p_s 1.200 bar, p_d 3.653 bar, P_h 2.044 kW, P_shaft 2.920 kW, η 0.700, P_elec 3.318 kW, I 7.257 A (hand 7.258), PF 0.660, T_bearing 55.0 °C, T_casing 35.0 °C, v 1.80 mm/s, a 0.406 m/s², NPSHa 22.23 m. Start (0→1450 rpm in 5 s, accel power visible), valve 0.5 → Q 16.99 m³/h, H 29.25 m, P_elec 2.71 kW; shutdown → Stopped at t≈53 s.

### Task 1.3 — Virtual sensor simulator — VERIFIED
- **Created:** `simulation/sensor_simulator.py`, `tests/test_sensor_simulator.py`; outputs `data/raw/healthy_baseline.csv` (3600 rows × 27 cols), `reports/healthy_baseline_overview.png`.
- **Design:**
  - `SENSORS` registry: name → (pump-state key, unit, plausible range). Range is for invalid-reading checks in 1.6.
  - `SensorSimulator(seed, warm_start, start_time, failures, dropout_probability)`; `.step(profile, effects, include_truth)` → one timestamped record (10 sensors + `state`, `speed_command_rpm`, `valve_opening`; optional `true_*` columns); `.run(duration_s, profile, effects_fn(t))` → DataFrame. `effects_fn` is the hook for the fault simulator (1.4).
  - Measurement model: Gaussian noise (config σ), flow cut-off 0.5 m³/h, encoder zero < 10 rpm, |RMS|, non-negative current/power, random NaN dropout.
  - `SensorFailure(sensor, mode, start_s, end_s, magnitude, rate, probability)`, modes dead/stuck/drift/offset/spike (instrument faults, distinct from pump faults).
  - `OperatingProfile(events, demand_amplitude, demand_period_s)` actions start/stop/estop/reset/valve/speed; `.standard(d)` = start @10 s → valve 0.7 @40 % → 1.0 @55 % → 1300 rpm @65 % → 1450 rpm @80 % → stop @d−60 s, plus ±0.04 sinusoidal demand (900 s); `.continuous()` for warm-started runs.
  - `generate_healthy_baseline()` adds `fault_label='healthy'`, `fault_severity=0`; `plot_overview()` saves a 10-panel plot.
- **Commands:** `python -m simulation.sensor_simulator`; `python -m pytest tests/ -q -W error::DeprecationWarning` → **46 passed** (first run 45/1: DeprecationWarning, I-2).
- **Inspected:** steady state (1200–1400 s) means: Q 29.98 m³/h, p_d 3.654 bar, I 7.259 A, P 3.316 kW, n 1450.1 rpm; std devs equal configured noise. 74 missing values = 0.2 % dropout as set. Plot reviewed: bearing warm-up τ≈600 s, throttling ↓Q ↓I ↑p_d ↑vibration (off-BEP), VFD turndown ↓all per affinity, shutdown → static suction 1.30 bar.

### Task 1.4 — Fault simulation — VERIFIED
- **Created:** `simulation/fault_simulator.py`, `tests/test_fault_simulator.py`, `docs/engineering_assumptions.md` (pump, sensor, fault physics + limitations).
- **Modified:** `simulation/pump_model.py` — cavitation noise now produced by the physics: `cavitation_intensity(npsh_margin)` (incipient < 2.5) adds up to 6 mm/s velocity and 25 m/s² HF accel; `operating_point` returns `npsh_margin`.
- **Design:**
  - `FaultType` enum (7 faults); `fault_effects(fault, s)` → `FaultEffects` (magnitudes in docs table).
  - `FaultScenario(fault, onset_s, ramp_s, max_severity, shape step|linear|exponential, end_s)` — `end_s` = repair (switch back to healthy).
  - `FaultSimulator(scenarios)`: `set_fault()/clear()` manual control for dashboard; `effects(t)` superposes all active faults; callable → pass directly as `effects_fn`; `label(t)` → (dominant fault, severity, '+'-joined active list; threshold 0.05); `annotate(df, dt)` adds `fault_label`, `fault_severity`, `active_faults`, `sev_<fault>` columns.
  - `generate_fault_dataset(scenarios, duration, seed, name)` warm start + continuous demand profile; adds `run_id` (for leakage-free ML split by run in 2.5/2.6).
  - `standard_scenario`: healthy 0–600 s, ramp to s=1 at 80 % of run (exponential for bearing & seal, linear others), hold.
- **Datasets (data/raw, 1 h @ 1 s each, ≈17 MB total, git-ignored):** `fault_<7 faults>.csv`, `fault_multiple_bearing_misalignment.csv` (bearing exp. + misalignment 0.7 from 1200 s), `healthy_steady.csv`, plus `healthy_baseline.csv` (1.3). Seeds 100–106, 200, 300.
- **Commands:** `python -m simulation.fault_simulator` (datasets + `reports/fault_signatures.png`); `python -m pytest tests/ -q -W error::DeprecationWarning` → **68 passed**.
- **Inspected signature table (% vs healthy, s=1):** bearing: accel +1879, vel +144, T_brg +27, I +7 · imbalance: vel +410 · misalignment: vel +302, T_brg +28, P +10 · cavitation: p_s −173 (vacuum), p_d −66, Q −12, accel +5059 · restriction: Q −63, p_d +16, P −25 · seal: casing +17, Q −3 (pump flow ↑) · overheating: T_brg +67, casing +14. Fixed an artifact (healthy reference window mismatched demand phase → spurious −1.2 % flow) and unreadable heat-map labels.
- **Note for 2.4/2.6:** imbalance vs misalignment differ only in magnitude/heat in overall values → separation must use spectra (1X vs 2X) from Stage 2. Seal leakage is subtle (detect via flow deficit vs pump curve at measured head + casing temperature).

### Task 1.5 — Hydraulic engineering calculations — VERIFIED
- **Modified:** `analysis/hydraulic_calculations.py` (+`pump_curve_coefficients`, `expected_flow_from_head_m3h` (curve inversion), `expected_head_from_flow_m`, `performance_curve(cfg, speed)`, `compute_kpis(df, cfg)`); `simulation/pump_model.py` now uses `pump_curve_coefficients` (single source of curve equations).
- **Created:** `analysis/performance_report.py` (`kpi_summary`, `plot_performance` → `reports/pump_performance.png`), `tests/test_hydraulic_kpis.py`.
- **`compute_kpis` columns (from MEASURED sensors):** diff_pressure_bar, head_m, hydraulic_power_kw, wire_to_water_eff_pct, pump_eff_est_pct (= w2w / η_motor), sec_kwh_m3, motor_load_pct, speed_ratio, bep_ratio, expected_flow_m3h, flow_deficit_m3h, head_deviation_pct, npsh_available_m, npsh_required_m, npsh_margin. Curve-based KPIs NaN when speed < 10 rpm; NaN inputs propagate.
- **Commands:** `python -m analysis.performance_report`; `python -m pytest tests/ -q -W error::DeprecationWarning -W error::RuntimeWarning` → **77 passed** (first run 76/1: float tolerance 4.5e-15 at run-out in affinity test; added atol).
- **Inspected KPI table (last 5 min, s=1):** healthy: H 25.26 m, η_pump 70.1 %, SEC 0.112 kWh/m³, deficit −0.01 m³/h · seal leakage: **deficit 3.005 m³/h** (= simulated leak), head dev −4.9 % · cavitation: **NPSH margin 0.50**, head dev −18.2 % · restriction: BEP ratio 0.36, η 41 %, SEC 0.229 (×2), deficit ≈ 0 · bearing: η 61.5 % (friction). Plot reviewed: restriction moves along the healthy curve; cavitation drops below it along the system curve; baseline sits on 1450/1300 rpm curves.
- **Diagnostic features ready for 2.4:** flow_deficit (seal), npsh_margin + head_deviation (cavitation), bep_ratio + diff-pressure ↑ (restriction), pump_eff_est ↓ (mechanical losses).

### Task 1.6 — Data acquisition — VERIFIED
- **Created:** `acquisition/data_collector.py`, `docs/industrial_integration.md` (sensors, 4–20 mA DAQ, PLC, Modbus RTU/TCP, OPC UA, implementation path), `tests/test_data_collector.py`.
- **Design:**
  - `MeasurementSource` protocol (`read() -> dict`, `sample_interval_s`) = hardware-abstraction boundary; `SimulatedSource(simulator, faults, profile)` implements it and adds `fault_label/fault_severity/active_faults`. Future `ModbusSource`/`OpcUaSource` plug in here.
  - `DataCollector(source, max_gap_fill=3, max_rate_per_s={bearing,casing: 3 °C/s}, buffer_seconds=24 h)`: per record → duplicate-timestamp rejection, missing / range (from `SENSORS`) / rate-of-change checks, forward-fill ≤ 3 samples, `qc_flags` ("sensor:issue[+filled]" joined by ';') and `qc_issue_count`; persistent temperature level change accepted after `max_gap_fill` suspect samples (`rate_step`). Batch → `compute_kpis` columns appended → stored → pushed to `sinks` (callables; SQLite sink in 1.7).
  - `acquire(n)`, `acquire_to_time(t)` (**rerun-safe**: same target adds nothing), `ingest(records)`, `.data`, `.latest()`, `.stats`, `export_csv()` → `data/processed/acquisition_<ts>.csv`; `clean_dataset(df)` offline cleaning with identical rules (sorts, de-duplicates, drops `true_*`).
- **Commands:** `python -m acquisition.data_collector` (demo with spikes, dead sensor, offset, dropout, imbalance fault); `python -m pytest tests/ -q -W error::DeprecationWarning -W error::RuntimeWarning` → **88 passed** (first run 87/1: test used a 10 °C step that the time-scaled rate allowance legitimately accepts; changed test to 30 °C to exercise the re-baseline branch — code unchanged).
- **Inspected demo:** 1800 rows after `acquire_to_time(1800)` twice (rerun added 0), duplicate rejected, 12 temperature spikes removed (max 58.4 °C = real imbalance heating), dead flow 60 s → 3 filled + 57 NaN, +30 bar offset → 598 range-flagged NaN, labels healthy 645 / imbalance 1155, CSV exported.

### Task 1.7 — SQLite database — VERIFIED
- **Created:** `database/database_manager.py`, `tests/test_database.py`. Default DB `data/pump_monitoring.db` (git-ignored, WAL mode, `check_same_thread=False` + RLock for Streamlit).
- **Schema:** `measurements` (sensors + conditions + labels + QC + 10 KPIs, UNIQUE(pump_id,timestamp)), `operating_status` (transitions only, UNIQUE(pump_id,timestamp,state)), `diagnoses` (method rule|ml_anomaly|ml_classifier, level Normal|Warning|Critical, evidence JSON, UNIQUE per ts/method/fault), `health_history` (0–100 CHECK, condition Healthy|Caution|Degraded|Critical, rul_hours, components JSON), `alarms` (priority Info|Warning|Critical; status active|acknowledged|cleared|reset; ack/clear/reset timestamps), `maintenance_records` (urgency Routine|Planned|Urgent|Immediate; status open|scheduled|completed|cancelled). Timestamps fixed-width text `%Y-%m-%d %H:%M:%S.%f` → string order = time order.
- **API:** `insert_measurements(df)` → rows inserted; `as_sink()` for `DataCollector.sinks` (also logs state transitions); `get_measurements(start,end,columns,limit,latest_first)`; `log_status`/`get_status_history`; `insert_diagnosis`/`get_diagnoses(method)`; `insert_health`/`get_health_history`; `insert_alarm`/`update_alarm`/`get_alarms(statuses)`; `add_maintenance`/`update_maintenance`/`get_maintenance(status)`; `export_csv(table, path, start, end)`; `counts()`; `query(sql)`. Column names validated against PRAGMA table_info (injection-safe).
- **Commands:** `python -m database.database_manager` ×2 on a fresh DB (1 h standard profile with exponential bearing fault, 5-min batches); `python -m pytest tests/ -q -W error::DeprecationWarning -W error::RuntimeWarning` → **99 passed**; `python -m compileall -q . -x "\.venv"` → OK.
- **Inspected:** 3600 rows inserted; re-insert → 0; date filter 08:30–08:35 → 301 rows; 5 transitions Stopped→Starting→Running→Shutdown→Stopped; alarm acknowledged; maintenance scheduled +2 days; CSV export `reports/measurements_export.csv`. Second run: measurements/status/diagnosis/health unchanged (alarms/maintenance rows are new events by design — alarm de-dup is the alarm manager's job, 3.8).
- **Bugs found & fixed:** (B-1) status replay added duplicate transitions → compare with state in force at that timestamp + UNIQUE. (B-2) `INSERT OR IGNORE` silently swallowed CHECK violations (health_index=150 accepted as no-op) → switched to `INSERT … ON CONFLICT DO NOTHING` (SQLite 3.49.1) which ignores only UNIQUE conflicts.
- **Stage 1 integration verified:** PumpModel → SensorSimulator → FaultSimulator → SimulatedSource → DataCollector (QC + KPIs) → DatabaseManager sink.

### Tasks 2.1 + 2.2 — Vibration signals & analysis — VERIFIED
- **Created:** `simulation/vibration_signal.py` (`generate_vibration(speed, v_rms, severities, npsh_margin, bep_ratio, seed)`, `from_state(record, severities)` → `VibrationSignal(t, velocity_mm_s, acceleration_m_s2, fs, speed_rpm)`), `analysis/vibration_analysis.py` (`time_features`, `amplitude_spectrum`, `dominant_frequency`, `peak_near`, `harmonic_amplitudes`, `band_rms`, `bearing_frequencies`, `envelope_spectrum`, `bearing_defect_indicators`, `analyze_vibration` → 30 features), `analysis/vibration_report.py` (→ `reports/vibration_comparison.png`), `tests/test_vibration.py`.
- **Modified:** `simulation/pump_model.py` — **acceleration sensor redefined as HF-band (0.5–2.4 kHz) RMS** (healthy 0.30 m/s², was 0.406). Reason: old estimate (2π·f₁·v) assumed all velocity at 1X, giving 4.8× mismatch with realistic waveforms (vane-pass/broadband content). `docs/engineering_assumptions.md` (+waveform section). All datasets in `data/raw`, `reports/fault_signatures.png` and the demo DB regenerated.
- **Design:** velocity components allocated per source (healthy / each fault / cavitation) with the same RMS contributions as PumpModel, waveform scaled to the overall velocity reading (exact match). Outer-race bearing defect = BPFO impact train (1 % jitter) × 1.8 kHz resonance (ζ 0.05); cavitation = bursty 0.5–2.4 kHz noise. Velocity broadband limited to 10–300 Hz so it stays out of the HF band.
- **Commands:** `python -m analysis.vibration_report`; regenerated datasets (`simulation.sensor_simulator`, `simulation.fault_simulator`, `database.database_manager`); `python -m pytest tests/ -q -W error::DeprecationWarning -W error::RuntimeWarning` → **118 passed**.
- **Inspected (report + plot):** model vs waveform HF accel 0.30/0.31, 0.50/0.54, 0.85/0.85, 4.01/3.97, 18.06/17.86 m/s²; velocity RMS exact. Imbalance: dominant 1X 10.1 mm/s, 2X/1X 0.11 · misalignment: dominant 2X, 2X/1X 1.35 · bearing s=0.5: env BPFO ratio 52 (healthy 1.5), accel kurtosis 9.7, clear BPFO/2×/3× envelope lines · cavitation: HF 17.9 m/s², dominant vane-pass 145 Hz. Known-signal tests: sine RMS/peak/crest √2/kurtosis 1.5, noise kurtosis 3.0, FFT amplitude within 0.1 %, aliasing 3 kHz → 2 kHz, Parseval band RMS, envelope recovers 73 Hz modulation, bearing frequencies vs hand calc (BPFO 73.33, BPFI 120.00, BSF 47.14, FTF 9.17 Hz).

### Task 2.3 — Feature extraction — VERIFIED
- **Created:** `analysis/feature_extraction.py`, `tests/test_feature_extraction.py`; output `data/processed/features.csv` (git-ignored, rebuild: `python -m analysis.feature_extraction`, ~30 s).
- **Modified:** `simulation/fault_simulator.py` — `generate_fault_dataset(..., profile=None)` accepts an OperatingProfile.
- **Design:**
  - `FEATURE_COLUMNS` (50): per sensor mean+std (20); bearing/casing temp & velocity slopes per min; bearing/casing temperature rise; KPI means (head, flow_deficit, head_deviation, bep_ratio, pump_eff_est, npsh_margin, sec, motor_load); 17 `vib_*` features from a 1 s waveform synthesised at each window end (speed, measured velocity, NPSH margin, BEP ratio, physical fault state) and analysed with `analyze_vibration`.
  - `META_COLUMNS`: run_id, window_start_s/end_s, timestamp, fault_label, fault_severity, active_faults, sev_* — **never features** (tested).
  - `window_features(win, kpis, severities)` — single window (reusable live in dashboard); `extract_features(df, window_s=60)` — non-overlapping windows, RUNNING only, drops incomplete trailing window.
  - `training_run_specs(runs_per_class=6, duration 2400 s)`: healthy + 7 faults × 6 independent runs with random valve 0.75–1.0, speed 1300/1375/1450 rpm, max severity 0.5–1.0, onset 300–700 s, linear/exponential growth; + 3 two-fault runs (bearing+misalignment, imbalance+cavitation, seal+overheating; label = dominant fault). Unique seeds.
- **Commands:** `python -m analysis.feature_extraction` → 2040 windows, 51 runs, 0 % NaN, 29 s; `python -m pytest tests/ -q -W error::DeprecationWarning -W error::RuntimeWarning` → **133 passed**.
- **Inspected class means (severity ≥ 0.4):** imbalance v 6.0 mm/s, 2X/1X 0.13 · misalignment 2X/1X 1.30, T-rise 34 °C · bearing env BPFO 49, HF 5.1 m/s², η 63.8 % · cavitation NPSH margin 4.0 (healthy 10.5), HF 1.2 · restriction BEP 0.37, η 42 % · seal flow deficit 1.89 m³/h · overheating T-rise 39 °C (healthy 28.6). Windows per label: healthy 749 (incl. pre-onset windows of fault runs), faults 167–206 each.
- **Note for 2.5/2.6:** low-severity windows (just after onset) are physically almost indistinguishable from healthy → expect confusion there; report metrics also by severity band.

### Task 2.4 — Rule-based fault diagnosis — VERIFIED
- **Created:** `detection/rule_based.py`, `tests/test_rule_based.py`; output `reports/rule_evaluation.csv`.
- **Modified:** `config/pump_config.py` (+`RuleThresholds` dataclass → `cfg.rules`).
- **Design:** `RuleBasedDiagnoser(cfg).diagnose(feature_dict)` → `Diagnosis(results, running)` with `.overall_level`, `.active` (sorted: level, specific before generic, confidence), `.primary_fault`, `.to_frame()`. Each `RuleResult(fault, level Normal|Warning|Critical, confidence 0.5–1, evidence dict, reason, recommendation)`. Rules skipped below 50 % speed.
  - Specific (combined evidence): bearing = envelope BPFO/BPFI SNR ≥ 5 AND (HF ≥ 0.8 or kurtosis ≥ 3.8), critical HF ≥ 4 or T ≥ 70 · imbalance = dominant order 1 & 2X/1X < 0.5 & 1X ≥ 3.5 / 8 mm/s · misalignment = 2X ≥ 1.5 mm/s & 2X/1X ≥ 0.8, critical ≥ 1.2 with v ≥ 4.5 or bearing rise ≥ 38 · cavitation = NPSH margin < 4 or p_s < 0.8 bar; critical margin < 1.5 or (< 2.5 with HF ≥ 3 or head ≤ −3 %) · flow restriction = BEP ratio < 0.6 / 0.4 with pump on its curve (|head dev| < 5 %) · seal = flow deficit ≥ 0.8 / 2.0 m³/h unless NPSH margin < 2.5 · overheating = **load-compensated thermal residual** (measured rise − expected rise for speed & shaft power) ≥ 6 / 15 °C, temperature not falling, no mechanical cause.
  - Generic symptoms: excessive_vibration (4.5 / 7.1 mm/s), high_bearing_temperature (70 / 85 °C).
  - `evaluate(features, min_severity)` → summary (FP rate, FN rate, detection rate, top-1 accuracy), confusion matrix, per-class table.
- **Iteration (recorded honestly):** v1 on features.csv (sev ≥ 0.3): FP 0 %, detection 90.6 %, top-1 94.8 %, overheating only 47 % (absolute rise limit ignores load) → added thermal residual + slope guard; imbalance 1X warning 4.0 → 3.5 (healthy 1X peak ≈ 2.0; missed at 1300 rpm); suction low 0.5 → 0.8 bar (healthy 1.2 ± 0.05). v2: FP 0 %, detection 95.6 %, top-1 97.6 %.
- **Independent validation (fresh dataset, seed 2000, 27 runs, never used for tuning):** sev ≥ 0.3 → **FP 0 %, FN 3.5 %, detection 96.5 %, top-1 98.0 %**; all severities → detection 77.4 % (low-severity windows physically ≈ healthy). Per class (fresh, sev ≥ 0.3): bearing/cavitation/restriction/imbalance/misalignment 100 %, seal 97.8 %, overheating 68.6 %. No fault-to-fault confusion; every error is a missed detection (FN), never a false alarm.
- **Overheating detection vs severity:** 0.30–0.45: 0 %, 0.45–0.6: 60 %, 0.6–0.8: 93 %, > 0.8: 100 % (small effect + 600 s thermal lag).
- **Commands:** `python -m detection.rule_based`; fresh-data check (inline script, `build_feature_dataset(runs_per_class=3, seed=2000, path=None)`); `python -m pytest tests/ -q -W error::DeprecationWarning -W error::RuntimeWarning` → **159 passed**.

### Tasks 2.5 + 2.6 — ML anomaly detection & classification — VERIFIED
- **Created:** `detection/ml_detection.py`, `tests/test_ml_detection.py`; models `models/saved_models/isolation_forest.joblib` (3.9 MB), `random_forest.joblib` (1.9 MB) (git-ignored; rebuild `python -m detection.ml_detection`, ~35 s); `data/processed/features_validation.csv` (fresh seed 2000, 27 runs, 1080 windows); reports `ml_metrics.json`, `ml_confusion_matrix.png`, `ml_feature_importance.png`, `ml_training_log.txt`.
- **Modified:** `requirements.txt` (scikit-learn pinned 1.7.2, issue I-3), `tests/test_environment.py` (+compiled-submodule import tests).
- **Design:** `split_by_run` (whole runs, stratified by class, ~33 % runs test, multi-fault runs test-only) → train 32 runs / 1280 windows, test 19 runs / 760 windows; + independent validation set. Isolation Forest = Imputer+Scaler+IF(300) on healthy training windows, threshold = 99th pct healthy score. Random Forest = Imputer+RF(300, min_samples_leaf 2, balanced), training excludes faulty windows with severity < 0.2 (label noise), evaluation uses all. Metrics: accuracy, macro P/R/F1, per-class report, confusion, FP/detection rate, by severity, single-fault vs multi-fault (prediction ∈ active faults); permutation + impurity importance. `MLDetector` loads bundles (with meta: sklearn version, save time, train runs) and returns anomaly_score, is_anomaly, predicted_fault, confidence, p_<class>.
- **Results (VALIDATION, fresh runs):**
  - RF sev ≥ 0.3: accuracy **96.9 %**, macro P/R/F1 0.969/0.958/**0.959**, FP **0 %**, detection 98.6 %; single-fault runs **99.1 %**; multi-fault windows: prediction is an active fault **100 %**. All severities: accuracy 88.8 %, detection 86.2 %, FP 0 %.
  - Rules on same windows: sev ≥ 0.3 top-1 98.0 %, detection 96.5 %; all severities detection 77.4 %. → **Rules slightly more accurate at clear severities; RF detects earlier (86 % vs 77 % over all severities).** Both 0 % FP.
  - IF: FP 1.7 %, ROC-AUC 0.90, detection 56 % (sev ≥ 0.6: 66 %); per class: bearing 100 %, restriction 92 %, misalignment 76 %, imbalance 69 %, seal 24 %, cavitation 9 %, overheating 9 %. Tested a 15-feature normalised subset → worse (43 %), kept all 50. Documented limitation: IF catches multivariate deviations, misses single-feature shifts.
  - Test-split (held-out runs) numbers consistent: RF sev ≥ 0.3 acc 95.2 %, F1 0.928, FP 0 %.
  - Top features (permutation): flow_deficit, head_deviation, env_bpfo_ratio, acc_kurtosis, ratio_2x_1x, amp_1x — engineering features dominate.
  - Investigated imbalance→cavitation confusion: 20/22 wrong non-healthy predictions come from the imbalance+cavitation two-fault run (cavitation genuinely present) → reported separately.
- **Synthetic-data caveat:** simulator and features share modelling assumptions; these metrics do NOT establish industrial diagnostic accuracy.
- **Commands:** `python -m detection.ml_detection`; inline diagnosis script (multi-fault source, IF subset comparison); full module import sweep of sklearn/scipy/numpy/pandas/pyarrow/matplotlib/plotly/streamlit/joblib (no blocked DLLs); `python -m pytest tests/ -q -W error::DeprecationWarning -W error::RuntimeWarning` → **175 passed**.

### Tasks 2.7 + 2.8 — Health index & predictive maintenance — VERIFIED
- **Created:** `predictive_maintenance/health_monitor.py`, `tests/test_health_monitor.py`; `reports/health_index_demo.png`.
- **Modified:** `config/pump_config.py` (`HealthBands`: weights vibration .25 / bearing .25 / thermal .20 / hydraulic .15 / suction .15, worst_share 0.7, damage anchors 0.45/0.8, EWMA 0.3, trend 15 windows, RUL min slope 1 pt/h, min R² 0.6); `docs/engineering_assumptions.md` (§4).
- **Design:** `damage(value, good, warning, critical)` piecewise-linear (rising or falling indicators, NaN → 0). `HealthMonitor.update(ts, features, diagnosis=None)` → `HealthRecord(health_index (EWMA), raw, condition, components (0–100 each), indicators (value+damage), trend_per_hour, rul_hours, rul_model, urgency Routine|Planned|Urgent|Immediate, recommendations, suggested_date, primary_fault)`. Urgency: Immediate if HI < 40 / rule Critical / RUL < 1 h; Urgent HI < 65 or RUL < 24 h; Planned HI < 85 / rule Warning / RUL < 7 d. Schedule: now / +24 h / +7 d / +30 d, capped at 0.5·RUL. `persist(record, db)` → health_history row + auto work order (scheduled) with one open order per component. `run_monitor(features)`, `rul_validation(hist)`.
- **Iterations (bugs found & fixed):** (1) bearing HI fell to 35 at severity < 0.3 because envelope SNR saturates → capped envelope damage at warning; now 88/62/45/22 per severity band. (2) Healthy pump at reduced load showed HI 64 during first 10 min (bearing cooling from previous load = thermal transient) → thermal residual ignored while cooling (same guard as rules). (3) Work-order note said "fault healthy" → "no specific fault identified yet". (4) Own test had malformed assertion + sign error — fixed.
- **Validation (6000 s runs, onset 600 s, 80-min degradation):** healthy HI 100 throughout; every fault ends Critical/Immediate; HI decreases monotonically with severity band. First non-Healthy: bearing 18 min, imbalance 27, seal 29, overheating 35 min. RUL MAE in final half of degradation: imbalance 1.0 min, overheating 2.3, seal 2.6, bearing 9.9 min; overall MAE 8.6–12.2 min (linear faults), 51 min bearing (early exponential extrapolation unreliable). Early linear-fault estimates are pessimistic (safe side). DB demo: 100 health rows, 2 work orders (generic early inspection + bearing).
- **Commands:** `python -m predictive_maintenance.health_monitor`; `python -m pytest tests/ -q -W error::DeprecationWarning -W error::RuntimeWarning` → **190 passed**.

### Stage 3 — Dashboard & alarm management (3.1–3.8) — VERIFIED
- **Created:** `alarms/alarm_manager.py`, `dashboard/engine.py`, `dashboard/ui.py`, `dashboard/{overview,realtime,vibration,fault_analysis,maintenance,historical_data}.py`, `app.py`, `tests/test_alarm_manager.py`, `tests/test_engine.py`, `tests/test_dashboard.py`, `D:\Pump_project\.claude\launch.json` (preview launcher; `cmd /c cd /d pump_predictive_maintenance && .venv\Scripts\python.exe -m streamlit run app.py`).
- **Modified:** `database/database_manager.py` (alarms.occurrences column + automatic migration; relative DB paths resolved against project root; `__main__` demo now uses `data/demo_database.db`), `simulation/sensor_simulator.py` (`set_base_valve`, `base_valve`), `config/pump_config.py` (`PROJECT_ROOT`, `project_path()`), `detection/ml_detection.py` (MODEL_DIR root-relative).
- **Alarm manager:** definitions (13 sensor HI/HIHI/LO/LOLO limits with deadbands, running-only flags) + diagnosis alarms (`diag_<fault>`, Warning/Critical with escalation) + health (Caution Info / Degraded Warning / Critical) + E-stop (Critical, "not safety-rated") + ML anomaly (Info) + sensor unavailable (Info). Lifecycle active → acknowledged → cleared (latched) → reset (only when cleared); Info auto-resets; one open instance per key (duplicate suppression), re-occurrence counted; state reloaded from DB on restart; unknown open keys moved to cleared at startup.
- **Engine:** one `SimulationEngine` per server (`st.cache_resource`, `PUMP_DB_PATH` env override). `tick()` advances to wall-clock × speed-up target (max 1800 s catch-up) → rerun-safe; `advance(s)` fast-forward; every 60 s sim window: features → rules → ML (RF + IF) → health (+ work orders) → alarms → DB. Controls: start/stop/E-stop/reset, speed, valve, instant or gradual (ramp) faults, clear. Start time continues after the last DB timestamp.
- **Dashboard:** sidebar (state pill, sim clock, Start/Stop/E-STOP + disclaimer, speed & valve sliders, fault injection instant/gradual, speed-up, live toggle, update interval, pause, +5/+30 min); live pages via `st.fragment(run_every)`; dark industrial theme; plotly charts. Pages per spec incl. alarm table with acknowledge / ack all / reset cleared, rules vs ML comparison, HI history with bands, component health, work-order completion + manual records, date filters, H–Q performance, CSV exports (5 tables) and downloadable HTML summary report.
- **Verification:**
  - pytest: alarm 13 tests, engine 7, dashboard (AppTest: all 6 pages render without exception, sidebar E-STOP click acts on engine, reruns don't duplicate, HTML report sections) 10 → full suite **220 passed** (`-W error::DeprecationWarning -W error::RuntimeWarning`); `compileall` OK.
  - Browser (Streamlit 1.67 via preview, 1440×900): clean start = NORMAL / HI 100 / no alarms; injected misalignment 0.6 via sidebar → overview: rule & ML "Misalignment", HI 56 Degraded, Critical banner, 6 alarms each occurrences = 1; "Ack all" → all acknowledged by operator; vibration page dominant 48 Hz (2X), 2X/1X 1.31; fault page rules vs ML "Agree", RF p≈1.0, IF anomaly; maintenance page HI 31 Critical, component bars, RUL "already critical"; history page 1443 measurements / 41 diagnoses / 7 alarms, export tab with 5 CSV + HTML report served (HTTP 200); real-time 6 trend panels. Server log: no errors. Console: only connection-refused messages during my intentional server restart and 2 benign 404s (Streamlit health probe under `/vibration/` on direct URL load).
  - **Not executed:** clicking download buttons (would save files on the user's machine) — content covered by unit test.
- **Bugs found & fixed:** (D-1) `use_container_width` deprecated in Streamlit 1.67 → `width="stretch"` everywhere. (D-2) status cards broke words mid-letter → nowrap + ellipsis. (D-3) stale alarm from the Task 1.7 demo (unmanaged key `accel_high`) showed a Warning on a healthy pump → demo uses its own DB, unknown open alarm keys cleared at startup, polluted DB deleted (regenerated automatically). (D-4) DB/model paths depended on the working directory → resolved against project root. (D-5) real-time legend overlapped subplot titles → larger top margin. (D-6) AppTest path relative to test file → absolute path.

### Stage 4 — Integration, validation, documentation, handover — VERIFIED
- **Created:** `analysis/scenario_validation.py` (→ `reports/validation_report.md`, `validation_results.csv`), `build_artifacts.py` (one-command rebuild of all generated artifacts, 117 s), `README.md`, `docs/CV_and_LinkedIn.md`; regression test for the system-resistance ratio.
- **Improvement found by validation (V-1):** healthy pump deliberately throttled to 60 % → ML 20 false windows, HI 9 non-Healthy windows (throttling and blockage are hydraulically identical when only flow is observed; training valve range was 0.75–1.0). Fix: new KPI/feature **`system_resistance_ratio`** = measured (H−H_static)/Q² ÷ expected k_design/valve² (≈ 1.000 ± 0.04 at every healthy operating point, = 1+9s under restriction, 1.0 for seal/cavitation); used by flow-restriction rule (warning 1.6, critical 3.0; BEP fallback without valve info) and health index; training valve range widened to 0.55–1.0; DB auto-migrates new measurement columns. Features 50 → 51; models retrained.
- **End-to-end validation (full engine pipeline, 60 min per scenario, ramp 30 min):** 8/8 fault scenarios identified by rules AND ML; healthy at 3 operating points (rated, 1300 rpm/valve 70 %, valve 60 %) → **0 false positives** (rules, ML, HI, alarms); 0 wrong-fault rule windows; 1 wrong ML window (misalignment → overheating). Rule detection delay / severity: bearing 4 min (0.11), flow restriction 4 (0.11, was 12 min/0.32), cavitation 7 (0.21), misalignment 9 (0.24), imbalance 10 (0.27), seal 10 (0.27), multi 11 (0.26), overheating 18 (0.48 — thermal lag). ML: 5–6 min (0.12–0.16) for most, restriction 13 (0.35).
- **Metrics after retraining (fresh validation set, sev ≥ 0.3):** rules FP 0 %, detection 96.5 %, top-1 98.0 %; RF accuracy 96.8 %, macro-F1 0.958, FP 0.6 %; IF ROC-AUC 0.88, FP 1.1 %.
- **Performance (4.4):** engine 1 h simulated in 4.1 s (≈ 870× real time); live tick (20 sim-s) median 7 ms / max 75 ms; DB read 1 h 34 ms; server-side page reruns 45–246 ms, 0 exceptions.
- **GitHub prep (4.7):** `.gitignore` extended (WAL/SHM, secrets); `git init` + initial commit (local only, not pushed). Data, models, DB git-ignored → `python build_artifacts.py` regenerates them.
- **Handover (4.9):** fresh `git clone` (no data/models) → `pytest` **220 passed, 1 skipped** (production-model test skips until `build_artifacts.py` runs; dashboard works without models). Backups (outside repo): `D:\Pump_project\backups\pump_pdm_2026-11-13.bundle` (full repo incl. history, `git bundle verify` OK; restore with `git clone <bundle> pump_pdm`) and `pump_pdm_source_2026-11-13.zip` (source snapshot).
- **Commands:** `python -m analysis.scenario_validation`, `python build_artifacts.py`, perf scripts, `python -m pytest tests/ -q -W error::DeprecationWarning -W error::RuntimeWarning` → **221 passed**, Mermaid validation → valid.

## Known issues / resolutions
- **I-1 (resolved):** `ImportError: DLL load failed while importing _compute: An Application Control policy has blocked this file` from pyarrow 26.0.1. Fix: pin `pyarrow==21.0.0`. If a future machine has no such policy the pin is still compatible.
- **I-2 (resolved):** pandas 2.3.3 + NumPy 2.5 emits `DeprecationWarning: The 'generic' unit for NumPy timedelta is deprecated` for `pd.Timedelta(seconds=<float>)` and for comparing a timedelta Series to a `pd.Timedelta` scalar. Rule: use `pd.to_timedelta(x, unit="s")` and `.dt.total_seconds()` comparisons. Run tests with `-W error::DeprecationWarning` to catch regressions.
- **I-3 (resolved):** `ImportError: DLL load failed while importing _sorting: An Application Control policy has blocked this file` from scikit-learn 1.9.1 (only when importing `sklearn.ensemble`/`metrics`; bare `import sklearn` worked, so Task 1.1 check missed it). Fix: pin `scikit-learn==1.7.2`; swept every compiled module of sklearn, scipy, numpy, pandas, pyarrow, matplotlib, plotly, streamlit, joblib → none blocked. Regression test `test_compiled_submodules_load`.

## Known limitations
- All pump/motor/bearing parameters are illustrative simulation assumptions.
- Motor efficiency constant (no part-load efficiency curve); no speed droop (VFD slip compensation assumed).
- Model rated current is derived (14.47 A) from P/V/PF/η; config `rated_current_a` (14.5 A) is kept for alarm limits.
- Leakage flow treated as a given rate (not ∝ √pressure).

## Next task
**Core project complete.** Optional next steps (Priority 3 roadmap, all DEFERRED):
1. Pushed to GitHub (origin/main). Optional: add a LICENSE of the user's choice, repository description and topics on GitHub.
2. Simulated Modbus TCP server (`pymodbus`) publishing the virtual pump registers + a `ModbusSource` implementing `MeasurementSource`.
3. Simulated OPC UA server (`asyncua`).
4. Multi-pump monitoring / comparison; energy optimisation; e-mail notifications; digital-twin view.
5. Advanced ML (gradient boosting, sequence models, uncertainty-aware RUL).

## Public deployment
- Live demo (Streamlit Community Cloud, deployed by the user from `main`, Python 3.12, secret `PUMP_DEMO_MODE = "1"`): https://pumppredictivemaintenance-mehdi.streamlit.app/
- Verified 2026-11-14: app loads, demo-mode guide shown, ML models load (RF/IF results on Fault Detection page). Redeploys automatically on every push to `main`.
- Demo mode (`dashboard/ui.py`): private in-memory engine per visitor, 10 min prefilled history; models committed.

## How to resume in a new conversation
Read this file first. Environment: `.venv\Scripts\python.exe` (pinned pyarrow 21.0.0, scikit-learn 1.7.2). Regenerate artifacts: `python build_artifacts.py`. Run app: `streamlit run app.py` (from the project folder). Tests: `python -m pytest tests/ -q`.
