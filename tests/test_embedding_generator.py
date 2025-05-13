import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from scripts.embedding_generator import generate_image_embeddings


@pytest.mark.integration
def test_single_valid_image_embedding(tmp_path: Path):
    """Generator should yield one embedding for one valid image."""
    img_path = tmp_path / "valid.jpg"
    Image.new("RGB", (224, 224), "blue").save(img_path)

    results = list(generate_image_embeddings([str(img_path)]))
    
    assert len(results) == 1
    vec, path = results[0]
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (512,)
    assert path == str(img_path)


@pytest.mark.integration
def test_single_invalid_image_embedding(tmp_path: Path):
    """Generator should yield nothing for an invalid image path."""
    bad_path = tmp_path / "missing.jpg"  # doesn't exist

    results = list(generate_image_embeddings([str(bad_path)]))
    
    assert results == []


@pytest.mark.integration
def test_all_bad_images_skipped(tmp_path: Path):
    """Generator should skip all invalid images and return nothing."""
    paths = [tmp_path / "missing1.jpg", tmp_path / "missing2.jpg"]

    results = list(generate_image_embeddings([str(p) for p in paths]))

    assert results == []


@pytest.mark.integration
def test_multiple_valid_images(tmp_path: Path):
    """Generator should yield embeddings for all valid images."""
    paths = []
    for i in range(2):
        path = tmp_path / f"valid_{i}.jpg"
        Image.new("RGB", (224, 224), "green").save(path)
        paths.append(str(path))

    results = list(generate_image_embeddings(paths))

    assert len(results) == 2
    for vec, path in results:
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (512,)
        assert path in paths
