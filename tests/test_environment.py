"""Task 1.1 smoke tests: environment and configuration."""
import importlib
import sys

import pytest

from config.pump_config import GRAVITY, get_config

REQUIRED = ["numpy", "pandas", "scipy", "matplotlib", "plotly",
            "streamlit", "sklearn", "joblib", "sqlite3"]


def test_python_version():
    assert sys.version_info >= (3, 11)


@pytest.mark.parametrize("module", REQUIRED)
def test_required_modules_import(module):
    importlib.import_module(module)


def test_config_defaults_match_specification():
    cfg = get_config()
    assert cfg.motor.rated_power_kw == 7.5
    assert cfg.pump.rated_speed_rpm == 1450
    assert cfg.pump.rated_flow_m3h == 30
    assert cfg.pump.rated_head_m == 25
    assert cfg.pump.rated_efficiency == pytest.approx(0.70)
    assert cfg.thermal.bearing_temp_nominal_c == 55


def test_rated_power_chain_hand_calculation():
    # Hand calc: P_h = 1000 * 9.81 * (30/3600) * 25 = 2043.75 W
    cfg = get_config()
    q = cfg.pump.rated_flow_m3h / 3600
    p_h = cfg.pump.fluid_density * GRAVITY * q * cfg.pump.rated_head_m
    assert p_h == pytest.approx(2043.75, rel=1e-9)
    # Shaft power must not exceed motor nameplate
    assert p_h / cfg.pump.rated_efficiency / 1000 < cfg.motor.rated_power_kw


def test_vibration_sampling_satisfies_nyquist():
    cfg = get_config()
    one_x = cfg.pump.rated_speed_rpm / 60
    highest_of_interest = 10 * one_x * cfg.pump.impeller_vanes  # generous
    assert cfg.vibration.sampling_rate_hz > 2 * highest_of_interest


# Compiled submodules that Windows Application Control blocked in newer builds
# (issues I-1, I-3). Importing the top-level package alone does not load them.
@pytest.mark.parametrize("module", [
    "pyarrow.compute", "sklearn.ensemble", "sklearn.metrics", "sklearn.inspection",
    "scipy.signal", "scipy.stats"])
def test_compiled_submodules_load(module):
    importlib.import_module(module)
