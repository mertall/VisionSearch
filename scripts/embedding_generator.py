from PIL import Image, UnidentifiedImageError
import numpy as np
import logging
import gc
import torch
from server.triton_client import query_triton_image_encoder

logger = logging.getLogger(__name__)

def generate_image_embeddings(image_paths: list[str]):
    """
    Generator that yields (embedding, image_path) tuples for successful image encodings.
    Gracefully skips and logs failures. Frees memory after each iteration.

    Args:
        image_paths (list[str]): List of image file paths to encode.

    Yields:
        tuple[np.ndarray, str]: A tuple of (512-dim image vector, image path)
    """
    for path in image_paths:
        image = None
        vec = None

        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
            logger.warning(f"❌ Skipping {path}: {e}")
            continue

        try:
            vec = query_triton_image_encoder(image)
            if not isinstance(vec, np.ndarray) or vec.shape != (512,):
                raise ValueError(f"Invalid vector shape for {path}: {vec.shape}")
            yield vec, path
        except Exception as e:
            logger.error(f"❌ Failed to encode {path}: {e}")
        finally:
            # Manual cleanup
            del image
            del vec
            torch.cuda.empty_cache()
            gc.collect()
