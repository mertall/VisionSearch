import pytest
from server.triton_client import query_triton_text_encoder

@pytest.mark.unit
def test_text_encoder_output_shape():
    """Ensure Triton text encoder returns a normalized 512-dim vector."""
    query = "a red car on snow"
    vec = query_triton_text_encoder(query)

    assert vec.shape == (512,), "Text embedding must be 512-dim"
    assert abs(vec @ vec - 1.0) < 1e-3, "Embedding should be L2 normalized"

