import os
import pytest
from unittest.mock import patch, MagicMock
from scripts import build_index
    
@pytest.mark.unit
@patch("scripts.build_index.r")
@patch("scripts.build_index.Image")
@patch("scripts.build_index.load_dataset")
def test_enqueue_hf_dataset_valid(mock_load_dataset, mock_image_module, mock_redis):
    """Test that valid records are correctly processed and enqueued."""
    # Mock image with .convert and .save
    mock_pil_image = MagicMock()
    mock_pil_image.convert.return_value = mock_pil_image
    mock_pil_image.save.return_value = None

    # Dataset returns records with PIL-style image
    mock_dataset = [
        {"image": mock_pil_image},
        {"image": mock_pil_image}
    ]
    mock_load_dataset.return_value = iter(mock_dataset)
    mock_image_module.open.return_value = mock_pil_image

    build_index.enqueue_hf_dataset("dummy/repo", split="train")

    assert mock_redis.rpush.call_count == 2
    args = [call[0][1] for call in mock_redis.rpush.call_args_list]
    assert all("hf_stream_dummy_repo" in arg for arg in args)

@pytest.mark.unit
@patch("scripts.build_index.r")
@patch("scripts.build_index.load_dataset")
def test_enqueue_hf_dataset_skips_missing_column(mock_load_dataset, mock_redis):
    """Test that records missing the image column are skipped."""
    mock_dataset = [
        {"not_image": "invalid"},
        {"image": "invalid_but_skipped"}
    ]
    mock_load_dataset.return_value = iter(mock_dataset)

    build_index.enqueue_hf_dataset("dummy/repo", image_column="image")
    # Should skip both due to bad column or format
    assert mock_redis.rpush.call_count == 0

@pytest.mark.unit
@patch("scripts.build_index.r")
@patch("scripts.build_index.Image")
@patch("scripts.build_index.load_dataset")
def test_enqueue_hf_dataset_handles_invalid_images(mock_load_dataset, mock_image_module, mock_redis):
    """Test that invalid image formats don't crash and are logged/skipped."""
    bad_image = object()  # no .convert or .read
    good_image = MagicMock()
    good_image.convert.return_value = good_image
    good_image.save.return_value = None

    mock_dataset = [
        {"image": bad_image},
        {"image": good_image}
    ]
    mock_load_dataset.return_value = iter(mock_dataset)
    mock_image_module.open.return_value = good_image

    build_index.enqueue_hf_dataset("dummy/repo", image_column="image")

    # Only one call should succeed
    assert mock_redis.rpush.call_count == 1