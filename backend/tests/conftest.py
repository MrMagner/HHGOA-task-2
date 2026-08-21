import os
from typing import Generator
import pytest
from fastapi.testclient import TestClient
os.environ["DEMO_MODE"] = "true"
os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["QDRANT_LOCAL_PATH"] = ":memory:"
from backend.main import app
from backend.config.settings import get_settings

@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c
@pytest.fixture
def settings():
    return get_settings()
