import sys
import os
import pytest
import numpy as np
from PyQt6.QtWidgets import QApplication

# Ensure the project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture(scope="session")
def qapp():
    """Ensure a QApplication exists for the entire test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    from ui.theme import ThemeManager
    ThemeManager.init(app)
    yield app

# --- Mock Data Fixtures ---

class MockCat:
    categories = ["0", "1", "2"]
    codes = np.array([0, 1, 2])

class MockSeries:
    cat = MockCat()

class MockAdata:
    obs = {"leiden": MockSeries()}
    uns = {}
    
    def __init__(self):
        # Allow dynamic attribute setting for tests
        pass

@pytest.fixture
def mock_adata():
    return MockAdata()
