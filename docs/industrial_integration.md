# Connecting the System to Real Equipment

This project is entirely simulated. This note explains how the same software
architecture would connect to a real pump. **No hardware integration is
implemented or validated here.**

## Signal chain

```
Pump/motor ─► Sensors/transmitters ─► DAQ / PLC I/O ─► PLC / RTU ─► Fieldbus ─► Edge PC ─► MeasurementSource.read()
                                                                     (Modbus / OPC UA)         └► DataCollector ─► SQLite / dashboard
```

| Measurement | Typical industrial instrument | Typical signal |
|---|---|---|
| Bearing / casing temperature | Pt100 RTD or thermocouple, head-mounted transmitter | 4–20 mA or RTD input card |
| Vibration velocity RMS | Velocity transmitter (ISO 10816 band 10–1000 Hz) | 4–20 mA |
| Vibration acceleration / waveform | IEPE accelerometer (100 mV/g) | Dynamic DAQ, ≥ 5–25 kHz sampling |
| Suction / discharge pressure | Piezoresistive pressure transmitter | 4–20 mA / HART |
| Flow | Electromagnetic flow meter | 4–20 mA, pulse, or fieldbus |
| Motor current / power / speed | VFD internal values or power meter | Modbus registers |

## Data acquisition (DAQ)

* Slow process values (1 Hz) are scanned by PLC analog input cards. The 4–20 mA
  signal is scaled to engineering units, e.g. `p = (I − 4)/16 · span`; a current
  < 3.6 mA indicates a broken wire, which maps to our `missing` QC flag.
* Vibration **waveforms** need a dedicated dynamic DAQ (anti-alias filter,
  ≥ 2.56 × f_max sampling, IEPE excitation). Only the overall values and
  features are normally sent to the PLC/SCADA level.

## PLC

The PLC runs the control logic (start/stop permissives, interlocks, the real
**safety-rated** emergency stop) and exposes measurements to higher levels.
The simulated E-stop in this project is **not** safety-rated.

## Modbus RTU / TCP

* RTU: serial RS-485, master/slave; TCP: Ethernet, port 502.
* Values are read from holding/input registers (16-bit; 32-bit floats occupy
  two registers — word order must match the device).
* Implementation path: a `ModbusSource` class implementing
  `MeasurementSource.read()` with `pymodbus` (`ModbusTcpClient.read_input_registers`)
  and a register map (address, scaling, unit) per sensor. Planned optional
  feature: a simulated Modbus TCP **server** publishing the virtual pump.

## OPC UA

* Platform-independent, information-model based (nodes with browse names,
  engineering units, timestamps, quality codes), built-in security.
* Implementation path: an `OpcUaSource` using `asyncua` to subscribe to the
  sensor nodes; OPC UA status codes map directly onto our QC flags.

## What stays the same

`DataCollector` (QC, gap filling, duplicate rejection, KPIs), the database,
diagnostics, health index and dashboard depend only on the record format of
`MeasurementSource.read()`, so swapping the simulated source for a real one
does not change them.
