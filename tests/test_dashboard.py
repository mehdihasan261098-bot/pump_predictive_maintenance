"""Stage 3 tests: every dashboard page renders without exceptions (Streamlit AppTest),
controls act on the engine, and reruns do not duplicate records."""
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["overview", "realtime", "vibration", "fault_analysis", "maintenance", "historical_data"]


@pytest.fixture(scope="module", autouse=True)
def isolated_db(tmp_path_factory):
    os.environ["PUMP_DB_PATH"] = str(tmp_path_factory.mktemp("dash") / "dash.db")
    from dashboard.ui import get_engine
    get_engine.clear()
    eng = get_engine()
    eng.advance(300)                      # some history so every page has content
    eng.set_fault("imbalance", 0.9)
    eng.advance(180)
    yield eng
    get_engine.clear()
    os.environ.pop("PUMP_DB_PATH", None)


def page_app(name):
    def app(page_name):
        import importlib

        from dashboard.ui import setup_page
        setup_page()
        importlib.import_module(f"dashboard.{page_name}").render()
    return AppTest.from_function(app, args=(name,), default_timeout=60)


@pytest.mark.parametrize("name", PAGES)
def test_page_renders_without_exception(name):
    at = page_app(name)
    at.session_state["live"] = False
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(at.title) == 1


def test_main_app_runs_and_sidebar_controls_work(isolated_db):
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    at.session_state["live"] = False
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    labels = [b.label for b in at.sidebar.button]
    assert {"▶ Start", "■ Stop", "E-STOP"} <= set(labels)
    at.sidebar.button[labels.index("E-STOP")].click().run()
    assert isolated_db.state == "Emergency stop"
    isolated_db.advance(20)
    assert isolated_db.reset_emergency_stop()
    isolated_db.start()
    isolated_db.advance(10)
    assert isolated_db.state == "Running"


def test_overview_shows_diagnosis_and_alarms(isolated_db):
    isolated_db.advance(180)        # ensure complete RUNNING windows after the E-stop test restarted the pump
    at = page_app("overview")
    at.session_state["live"] = False
    at.run()
    md = " ".join(m.value for m in at.markdown)
    assert "Imbalance" in md
    assert any("alarm" in m.value.lower() for m in at.markdown)


def test_reruns_do_not_duplicate_records(isolated_db):
    before = isolated_db.db.counts()["measurements"]
    at = page_app("realtime")
    at.session_state["live"] = False
    for _ in range(5):
        at.run()
    after = isolated_db.db.counts()["measurements"]
    assert after - before <= 5 * isolated_db.speedup * 2       # only wall-clock progress, no replays
    assert isolated_db.collector.data.timestamp.is_unique
    m = isolated_db.db.get_measurements()
    assert m.timestamp.is_unique


def test_html_report_contains_all_sections(isolated_db):
    from dashboard.historical_data import build_report
    db = isolated_db.db
    rep = build_report(isolated_db, db.get_measurements(), db.get_diagnoses(), db.get_alarms(),
                       db.get_health_history(), db.get_maintenance(), "a", "b")
    for section in ("Operation", "Sensor and KPI statistics", "Health", "Diagnosed faults", "Alarms",
                    "Maintenance", "SIMULATED"):
        assert section in rep
    assert "imbalance" in rep and rep.strip().endswith("</html>")


def test_demo_mode_gives_each_session_a_private_engine(monkeypatch):
    """Public-demo mode: separate in-memory simulation per visitor, prefilled history."""
    monkeypatch.setenv("PUMP_DEMO_MODE", "1")
    app_path = str(Path(__file__).resolve().parents[1] / "app.py")
    a = AppTest.from_file(app_path, default_timeout=90)
    b = AppTest.from_file(app_path, default_timeout=90)
    for at in (a, b):
        at.session_state["live"] = False
        at.run()
        assert not at.exception, [e.message for e in at.exception]
    ea, eb = a.session_state["engine"], b.session_state["engine"]
    assert ea is not eb and ea.db.path == ":memory:"
    assert ea.db.counts()["measurements"] >= 600 and ea.latest_health is not None
    ea.emergency_stop()
    ea.advance(10)
    assert ea.state == "Emergency stop" and eb.state == "Running"     # isolation between visitors
    assert any("Public demo" in e.label for e in a.sidebar.expander)
