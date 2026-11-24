from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("page", sorted((ROOT / "pages").glob("*.py")))
def test_page_renders_without_exception(page):
    if not (ROOT / "data/processed/hourly.csv").exists():
        pytest.skip("UI smoke tests require the bundled hourly cache")
    app = AppTest.from_file(str(page)).run(timeout=30)
    assert not app.exception, [str(e.value) for e in app.exception]


def test_navigation_entrypoint():
    if not (ROOT / "data/processed/hourly.csv").exists():
        pytest.skip("UI smoke test requires hourly cache")
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception


def test_training_form_executes():
    if not (ROOT / "data/processed/hourly.csv").exists():
        pytest.skip("UI smoke test requires hourly cache")
    app = AppTest.from_file(str(ROOT / "pages/3_Model_Training.py")).run(timeout=30)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 4
