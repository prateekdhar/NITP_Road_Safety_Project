from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_untrained_app_shows_setup():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Road Damage Inspector"
    app.text_input[0].set_value("runs/train/nonexistent_test_run/weights/best.pt").run()
    assert not app.exception
    assert any("No trained road-damage" in item.value for item in app.info)


def test_registered_pages_round_trip():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    assert not app.exception
    app.switch_page("pages/1_Model_Statistics.py").run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Model statistics & characteristics"
    assert any(metric.value == "70.6%" for metric in app.metric)
    app.switch_page("pages/0_Detection.py").run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Road Damage Inspector"
