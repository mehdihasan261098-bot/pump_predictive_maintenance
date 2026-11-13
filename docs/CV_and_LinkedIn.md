# CV, LinkedIn and Interview Material

All numbers below come from this repository's own tests and reports (simulated data). Keep the word
"simulated" when you quote them — interviewers respect honest framing.

## 1. CV entry (concise)

**Centrifugal Pump Condition Monitoring & AI-Based Predictive Maintenance System** — *Python, personal project, 2026*  
Live demo: https://pumppredictivemaintenance-mehdi.streamlit.app/ · Code: https://github.com/mehdihasan261098-bot/pump_predictive_maintenance

- Built a physics-based digital model of a 7.5 kW centrifugal pump (pump/system curves, affinity laws, NPSH,
  motor current, thermal response) with 10 virtual sensors and 7 simulated faults (bearing, imbalance,
  misalignment, cavitation, flow restriction, seal leakage, overheating).
- Implemented vibration analysis (FFT, harmonics, bearing defect frequencies BPFO/BPFI/BSF/FTF, envelope
  analysis, kurtosis) and hydraulic KPIs (head, efficiency, specific energy, NPSH margin, pump-curve residuals).
- Developed an explainable rule-based diagnosis engine and ML models (Isolation Forest, Random Forest) with a
  leakage-free train/test split by simulation run; on independent simulated validation data: 98 % correct
  fault identification with 0 % false alarms (rules) and 96.8 % accuracy (Random Forest).
- Designed a 0–100 health index with simulated remaining-useful-life estimation, maintenance prioritisation,
  ISA-18.2-style alarm management and a six-page Streamlit/Plotly SCADA-style dashboard backed by SQLite.
- Verified with 221 automated tests (pytest) and end-to-end fault-scenario validation.

*Skills line:* Python · NumPy · Pandas · SciPy · Scikit-learn · Streamlit · Plotly · SQLite · Pytest ·
vibration analysis · condition monitoring · predictive maintenance · pump hydraulics

## 2. LinkedIn project description

**Predictive maintenance for centrifugal pumps — built from first principles in Python**

As a mechanical engineering graduate I wanted to understand how condition monitoring really works, so I
built a complete, software-only predictive-maintenance system for an industrial centrifugal pump.

🔧 **Physics first:** a simulated pump and motor (pump & system curves, affinity laws, NPSH/cavitation,
thermal behaviour) with 10 virtual sensors and seven realistic faults — bearing wear, imbalance, misalignment,
cavitation, flow restriction, seal leakage and overheating.

〰️ **Vibration analysis:** FFT, harmonics, bearing defect frequencies and envelope analysis — imbalance shows up
at 1X, misalignment at 2X, a bearing defect as impacts at BPFO.

🤖 **Diagnosis two ways:** an explainable rule engine (every alarm states its evidence, reason and recommended
action) compared with Isolation Forest and Random Forest models trained without data leakage. On independent
simulated test runs the rules identified the correct fault 98 % of the time with zero false alarms.

📊 **SCADA-style dashboard:** live trends, alarm management with acknowledge/reset, health index and simulated
remaining useful life, maintenance work orders and downloadable reports — Streamlit, Plotly and SQLite.

✅ 221 automated tests and an end-to-end validation of every fault scenario.

One lesson I enjoyed: deliberately throttling a valve and a blocked pipe look identical if you only watch
flow. Comparing the measured system resistance with the value expected for the valve position separates
them — a small piece of engineering that removed all false alarms at part load.

▶ Try the live demo (inject a fault yourself): https://pumppredictivemaintenance-mehdi.streamlit.app/
💻 Code: https://github.com/mehdihasan261098-bot/pump_predictive_maintenance

*All results are from simulated data; real deployment would need field data and machine-specific baselines.*

#PredictiveMaintenance #ConditionMonitoring #MechanicalEngineering #Python #MachineLearning #VibrationAnalysis

## 3. Technical skills demonstrated

| Domain | Skills |
|---|---|
| Mechanical / rotating equipment | Centrifugal pump hydraulics (H–Q, η–Q, system curves, affinity laws), NPSH and cavitation, mechanical seals, rolling-element bearings, shaft alignment, rotor balance, thermal behaviour |
| Vibration & condition monitoring | Time-domain statistics (RMS, crest factor, kurtosis), FFT and spectral analysis, Nyquist/aliasing, harmonics, bearing defect frequencies, envelope demodulation, ISO 10816-style severity limits |
| Reliability & maintenance | Fault signatures, health index design, degradation trending, RUL estimation (and its limits), maintenance urgency and work orders, alarm management (ISA-18.2 concepts) |
| Data & ML | Feature engineering, Isolation Forest, Random Forest, leakage-free group splits, precision/recall/F1, confusion matrices, permutation importance, model serialisation |
| Software engineering | Modular Python architecture, configuration management, SQLite schema design and migrations, data quality control, Streamlit/Plotly dashboards, pytest (221 tests), reproducible build script, Git |
| Industrial automation (concepts) | Sensors and 4–20 mA DAQ, PLCs, Modbus RTU/TCP, OPC UA, SCADA HMI design, hardware-abstraction for data sources |

## 4. Industrial relevance (short explanation)

Centrifugal pumps are among the most common machines in water treatment, oil & gas, chemical, HVAC and
power plants, and unplanned pump failures cause downtime, leaks and safety risks. Predictive maintenance
uses sensor data to detect developing faults early and plan repairs before failure — typically cheaper than
both run-to-failure and fixed-interval maintenance. This project reproduces the full chain used by industrial
condition-monitoring systems: sensors → data acquisition and quality control → engineering KPIs and vibration
analysis → rule-based and ML diagnosis → health index and maintenance planning → alarms and SCADA
visualisation. The data-source interface is designed so that the simulated pump could be replaced by real
sensors via a PLC, Modbus or OPC UA without changing the analysis layers.

## 5. Possible interview questions to prepare

1. Why does imbalance appear at 1X and misalignment at 2X?
2. What is the Nyquist criterion and why did you sample at 5 kHz?
3. How do BPFO and BPFI depend on bearing geometry and speed? Why search in a ±2 % band?
4. Why is a random train/test split of time-series windows misleading?
5. Why did the Isolation Forest miss cavitation and overheating, and why is that acceptable here?
6. How would you validate this system on a real pump?
