import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

# Import your FastAPI app
from server.main import app  # adjust this path as needed

client = TestClient(app)

# ────────────────────────────────────────────────────────────────
# Test GET /index/status
# ────────────────────────────────────────────────────────────────

@patch("server.main.HNSWIndexSingleton")
def test_check_status_ready(mock_index):
    mock_index.is_ready.return_value = True

    response = client.get("/index/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

@patch("server.main.HNSWIndexSingleton")
def test_check_status_building(mock_index):
    mock_index.is_ready.return_value = False

    response = client.get("/index/status")
    assert response.status_code == 200
    assert response.json() == {"status": "building"}


# ────────────────────────────────────────────────────────────────
# Test GET /search
# ────────────────────────────────────────────────────────────────

@patch("server.main.HNSWIndexSingleton")
@patch("server.main.CLIPSageMakerClient")
def test_search_success(mock_clip_client_cls, mock_index):
    mock_index.is_ready.return_value = True
    mock_index.query.return_value = (["/img/1.jpg", "/img/2.jpg"], [0.99, 0.95])

    mock_client = MagicMock()
    mock_client.encode_text.return_value = np.random.rand(512).astype(np.float32)
    mock_clip_client_cls.return_value = mock_client

    response = client.get("/search", params={"query": "a cat", "k": 2})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert results[0]["image_path"] == "/img/1.jpg"
    assert results[0]["score"] > 0

@patch("server.main.HNSWIndexSingleton")
def test_search_index_not_ready(mock_index):
    mock_index.is_ready.return_value = False
    response = client.get("/search", params={"query": "a cat"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Index is still building."

@patch("server.main.HNSWIndexSingleton")
@patch("server.main.CLIPSageMakerClient")
def test_search_encoding_failure(mock_clip_client_cls, mock_index):
    mock_index.is_ready.return_value = True
    mock_client = MagicMock()
    mock_client.encode_text.side_effect = Exception("fail encoding")
    mock_clip_client_cls.return_value = mock_client

    response = client.get("/search", params={"query": "a cat"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to encode query text."


# ────────────────────────────────────────────────────────────────
# Test POST /index/build
# ────────────────────────────────────────────────────────────────

@patch("server.main.build_index")
def test_build_index_success(mock_build_index):
    payload = {
        "dataset_repo": "user/repo",
        "split": "train",
        "image_column": "image"
    }
    response = client.post("/index/build", json=payload)
    assert response.status_code == 200
    assert response.json() == {
        "status": "completed",
        "dataset": "user/repo"
    }

@patch("server.main.build_index")
def test_build_index_failure(mock_build_index):
    mock_build_index.side_effect = Exception("build error")
    payload = {
        "dataset_repo": "user/repo",
        "split": "train",
        "image_column": "image"
    }
    response = client.post("/index/build", json=payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to build index."
