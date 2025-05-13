from unittest.mock import patch
import pytest
import numpy as np
from PIL import Image
from server.triton_client import query_triton_image_encoder as image_encoder
import server
@pytest.mark.unit
def test_image_encoder_output_shape():
    """Check that image encoder returns normalized 512-dim vector for a dummy image."""
    image = Image.new("RGB", (224, 224), color="red")
    vec = image_encoder(image)

    assert isinstance(vec, np.ndarray), "Embedding must be a NumPy array"
    assert vec.shape == (512,), "Image embedding must be 512-dim"
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-3, "Embedding should be L2 normalized"


@pytest.mark.unit
def test_image_encoder_with_none_input():
    """Passing None should raise a clear error."""
    with pytest.raises(Exception) as excinfo:
        image_encoder(None)
    assert "None" in str(excinfo.value) or "image" in str(excinfo.value)


@pytest.mark.unit
def test_image_encoder_with_invalid_type():
    """Passing non-image type should raise a ValueError from the processor."""
    with pytest.raises(ValueError) as excinfo:
        image_encoder("not an image")
    assert "flat list of images" in str(excinfo.value)
    
@pytest.mark.unit
@patch("server.triton_client.query_triton_image_encoder")
def test_image_encoder_output_shape_invalid(mock_encoder):
    """Simulate a broken encoder returning an invalid vector shape."""
    mock_encoder.return_value = np.random.rand(300).astype(np.float32)

    image = Image.new("RGB", (224, 224), color="green")
    vec = server.triton_client.query_triton_image_encoder(image)

    assert vec.shape != (512,), f"Expected invalid shape, got {vec.shape}"
