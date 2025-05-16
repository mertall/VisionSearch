import io
import json
import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, Mock
import boto3

from src.server.sage_maker import CLIPSageMakerClient

# Dummy processor to simulate CLIPProcessor
class DummyProcessor:
    def __call__(self, images=None, text=None, return_tensors=None):
        if images is not None:
            return {"pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)}
        if text is not None:
            return {
                "input_ids": np.zeros((1, 10), dtype=np.int64),
                "attention_mask": np.ones((1, 10), dtype=np.int64),
            }
        raise ValueError("Invalid call to DummyProcessor")

@pytest.fixture(autouse=True)
def clear_singleton(monkeypatch, tmp_path):
    CLIPSageMakerClient._instance = None
    monkeypatch.setenv("CLIP_ENDPOINT_NAME", "clip-multimodal-endpoint")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/SageMakerRole")
    monkeypatch.delenv("SAGEMAKER_AUTO_CREATE", raising=False)
    return tmp_path

@patch("transformers.CLIPProcessor.from_pretrained")
@patch("boto3.client")
def test_invalid_json_response(mock_boto_client, mock_processor_loader):
    mock_processor_loader.return_value = DummyProcessor()
    fake_sm = Mock()
    fake_runtime = Mock()
    fake_sm.describe_endpoint.return_value = {}
    fake_runtime.invoke_endpoint.return_value = {"Body": io.BytesIO(b"not json")}
    mock_boto_client.side_effect = lambda svc, **k: fake_sm if svc == "sagemaker" else fake_runtime

    client = CLIPSageMakerClient()
    img = Image.new("RGB", (10, 10))
    with pytest.raises(Exception):
        client.encode_image(img)
    with pytest.raises(Exception):
        client.encode_text("test")
