import pytest
from unittest.mock import patch, MagicMock, mock_open
from io import BytesIO
import numpy as np
from PIL import Image
from uuid import uuid4

from scripts.build_index import build_index

# Assuming your build_index function is imported like so:
# from scripts.build_index import build_index

@pytest.fixture
def dummy_dataset():
    # Simulates 2 images: one valid and one missing 'bytes'
    return [
        {"image": {"bytes": Image.new("RGB", (64, 64)).tobytes()}},
        {"image": {}}
    ]

@patch("scripts.build_index.load_dataset")
@patch("scripts.build_index.HNSWIndexSingleton")
@patch("scripts.build_index.CLIPSageMakerClient")
@patch("scripts.build_index.Image")
@patch("os.makedirs")
@patch("os.path.join", side_effect=lambda *args: "/fake/path/" + "_".join(args[1:]))
def test_build_index_happy_path(mock_join, mock_makedirs, mock_image, mock_clip_client_cls, mock_index_singleton, mock_load_dataset, dummy_dataset):
    # Mocks
    mock_load_dataset.return_value = iter(dummy_dataset)
    
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    mock_img.save.return_value = None
    mock_img.close.return_value = None
    mock_img.__enter__.return_value = mock_img
    mock_img.__exit__.return_value = None
    mock_image.open.return_value = mock_img

    mock_clip_client = MagicMock()
    mock_clip_client.encode_image.return_value = np.random.rand(512).astype(np.float32)
    mock_clip_client_cls.return_value = mock_clip_client

    mock_index = MagicMock()
    mock_index.ensure_ready.return_value = None
    mock_index.add_items.return_value = None
    mock_index.save.return_value = None
    mock_index_singleton.ensure_ready.return_value = None
    mock_index_singleton.add_items.return_value = None
    mock_index_singleton.save.return_value = None

    build_index("dummy/repo")

    # Validations
    assert mock_load_dataset.called
    assert mock_clip_client.encode_image.call_count == 1  # One valid image
    assert mock_index_singleton.add_items.call_count == 1
    assert mock_index_singleton.save.called

@patch("scripts.build_index.load_dataset")
@patch("scripts.build_index.HNSWIndexSingleton")
@patch("scripts.build_index.CLIPSageMakerClient")
@patch("scripts.build_index.Image")
def test_build_index_image_error(mock_image, mock_clip_client_cls, mock_index_singleton, mock_load_dataset):
    # Dataset with corrupted image bytes
    mock_load_dataset.return_value = iter([
        {"image": {"bytes": b"not an image"}},
    ])

    mock_image.open.side_effect = Image.UnidentifiedImageError("Invalid image")

    mock_clip_client = MagicMock()
    mock_clip_client_cls.return_value = mock_clip_client

    build_index("dummy/broken")

    assert not mock_clip_client.encode_image.called
    assert not mock_index_singleton.add_items.called
    assert mock_index_singleton.save.called
