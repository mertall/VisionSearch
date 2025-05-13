import pytest
import torch
from server.clip_model import CLIPSingleton
from transformers import CLIPModel, CLIPProcessor

@pytest.mark.unit
def test_clip_singleton_load():
    """Test CLIPSingleton loads model, processor, and device correctly."""
    processor, model, device = CLIPSingleton.load()

    assert isinstance(processor, CLIPProcessor), "Processor should be a CLIPProcessor instance"
    assert isinstance(model, CLIPModel), "Model should be a CLIPModel instance"
    assert device in {"cuda", "cpu"}, "Device must be either 'cuda' or 'cpu'"
    assert model.device.type == device, f"Model should be on the {device} device"
