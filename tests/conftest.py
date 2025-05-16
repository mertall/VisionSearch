import pytest
from src.server.index_store import HNSWIndexSingleton

@pytest.fixture(autouse=True)
def reset_index_singleton():
    HNSWIndexSingleton._index = None
    HNSWIndexSingleton._ready = False
    HNSWIndexSingleton._image_paths = []

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: marks unit tests")
    config.addinivalue_line("markers", "integration: marks integration tests")
