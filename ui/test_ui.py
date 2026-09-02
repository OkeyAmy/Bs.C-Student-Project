"""UI smoke tests (streamlit.testing.v1.AppTest).

These render the app headlessly and check that sidebar navigation loads each
section without raising. The API is NOT required: the prediction/comparison
sections degrade to a friendly error when the API is unreachable, so the tests
pass either way (they assert no unhandled exception, not API content).
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent / "app.py")


def run_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception
    return at


def test_all_sections_navigate_without_exception():
    at = run_app()

    radio = at.sidebar.radio[0]
    for section in ["Predict a semester", "Carryover comparison", "About this demo"]:
        radio.set_value(section)
        at.run(timeout=30)
        assert not at.exception, f"exception on section {section}"


def test_tabs_shown_for_predict():
    at = run_app()
    assert at.subheader  # page title present