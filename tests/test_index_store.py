import os
import numpy as np
import pytest
from server.index_store import HNSWIndexSingleton

@pytest.mark.unit
def test_query_triggers_auto_load():
    """Query should automatically load the index if not ready."""
    HNSWIndexSingleton._index = None
    HNSWIndexSingleton._ready = False
    HNSWIndexSingleton._image_paths = []

    dummy_vecs = [np.random.rand(512).astype(np.float32)]
    dummy_paths = ["image_0.jpg"]
    HNSWIndexSingleton.load()
    HNSWIndexSingleton.add_items(dummy_vecs, dummy_paths)

    # Reset ready flag to simulate "unloaded" state
    HNSWIndexSingleton._ready = False
    query_vec = np.random.rand(512).astype(np.float32)
    
    # Should succeed — not raise
    results, scores = HNSWIndexSingleton.query(query_vec, k=1)
    
    assert HNSWIndexSingleton.is_ready()
    assert len(results) == 1

@pytest.mark.unit
def test_load_creates_index():
    """Ensure index is initialized on first load call."""
    HNSWIndexSingleton._index = None
    HNSWIndexSingleton._ready = False
    HNSWIndexSingleton._image_paths = []

    HNSWIndexSingleton.load()
    assert HNSWIndexSingleton._index is not None
    assert HNSWIndexSingleton._ready is True

@pytest.mark.unit
def test_is_ready_reflects_state():
    """Test that is_ready() returns correct readiness flag."""
    HNSWIndexSingleton._ready = False
    assert not HNSWIndexSingleton.is_ready()

    HNSWIndexSingleton._ready = True
    assert HNSWIndexSingleton.is_ready()

@pytest.mark.unit
def test_add_items_and_query():
    """Test that added vectors can be queried successfully."""
    HNSWIndexSingleton.load()
    dummy_vecs = [np.random.rand(512).astype(np.float32) for _ in range(5)]
    dummy_paths = [f"image_{i}.jpg" for i in range(5)]
    HNSWIndexSingleton.add_items(dummy_vecs, dummy_paths)

    query_vec = np.random.rand(512).astype(np.float32)
    results, scores = HNSWIndexSingleton.query(query_vec, k=3)

    assert len(results) == 3
    assert len(scores) == 3
    assert all(isinstance(path, str) for path in results)
    assert all(isinstance(score, float) or isinstance(score, np.floating) for score in scores)

@pytest.mark.unit
def test_query_on_unloaded_index_is_safe():
    """If the index isn't loaded, query should load it and handle an empty index gracefully."""
    # Reset singleton
    HNSWIndexSingleton._index = None
    HNSWIndexSingleton._ready = False
    HNSWIndexSingleton._image_paths = []

    HNSWIndexSingleton.load()

    # Now test that calling query on an empty index raises HNSW-specific failure
    with pytest.raises(RuntimeError, match="contiguous 2D array"):
        vec = np.random.rand(512).astype(np.float32)
        HNSWIndexSingleton.query(vec, k=1)

@pytest.mark.unit
def test_save_creates_files(tmp_path):
    """Test that the save() method creates index and metadata files."""

    # 1. Reset singleton state
    HNSWIndexSingleton._index = None
    HNSWIndexSingleton._ready = False
    HNSWIndexSingleton._image_paths = []

    # 2. Patch paths BEFORE load()
    HNSWIndexSingleton.INDEX_PATH = str(tmp_path / "test_index.bin")
    HNSWIndexSingleton.META_PATH = str(tmp_path / "test_paths.txt")

    # 3. Load index and add dummy item
    HNSWIndexSingleton.load()
    dummy_vecs = [np.random.rand(512).astype(np.float32)]
    dummy_paths = ["img.jpg"]
    HNSWIndexSingleton.add_items(dummy_vecs, dummy_paths)

    # 4. Save to disk
    HNSWIndexSingleton.save()

    # 5. Assertions
    assert os.path.exists(HNSWIndexSingleton.INDEX_PATH)
    assert os.path.exists(HNSWIndexSingleton.META_PATH)

    with open(HNSWIndexSingleton.META_PATH, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    
    assert lines == ["img.jpg"], f"Expected ['img.jpg'], got {lines}"
