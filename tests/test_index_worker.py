import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from scripts.index_worker import run_redis_index_worker

class TestStopWorker(Exception):
    """Used to break the infinite loop cleanly from test."""
    pass

@patch("scripts.index_worker.r")
@patch("scripts.index_worker.generate_image_embeddings")
@patch("scripts.index_worker.HNSWIndexSingleton")
def test_run_redis_index_worker(mock_index, mock_generator, mock_redis):
    """Simulate worker receiving one job, then raise to stop."""
    mock_r = MagicMock()
    mock_redis.return_value = mock_r

    call_count = {"n": 0}

    def fake_blpop(*args, **kwargs):
        if call_count["n"] == 0:
            call_count["n"] += 1
            return ("image_jobs", b"/some/image.jpg")
        raise TestStopWorker()

    mock_r.blpop.side_effect = fake_blpop

    mock_vec = np.random.rand(512).astype(np.float32)
    mock_generator.return_value = [(mock_vec, "/some/image.jpg")]

    mock_index._image_paths = []
    mock_index.load.return_value = None
    mock_index.add_items.return_value = None
    mock_index.save.return_value = None

    from scripts import index_worker
    index_worker.r = mock_r

    # Catch test-only exit
    with pytest.raises(TestStopWorker):
        run_redis_index_worker(batch_size=1, persist_every=1)

    mock_generator.assert_called_once()
    mock_index.add_items.assert_called_once()
    mock_index.save.assert_called()
