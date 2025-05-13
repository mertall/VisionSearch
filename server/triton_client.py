import numpy as np
from PIL import Image
import torch
import logging
from server.clip_model import CLIPSingleton

logger = logging.getLogger(__name__)

def query_triton_text_encoder(text: str) -> np.ndarray:
    """
    Run text through the CLIP text encoder and return normalized 512-dim vector.
    """
    try:
        processor, model, device = CLIPSingleton.load()
        logger.debug(f"Text encoder running on device: {device}")

        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            embedding = model.get_text_features(**inputs)
            embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

        result = embedding.squeeze().cpu().numpy()
        logger.debug(f"Text embedding shape: {result.shape}")
        return result

    except Exception as e:
        logger.error(f"❌ Failed to encode text '{text}': {e}")
        raise


def query_triton_image_encoder(image: Image.Image) -> np.ndarray:
    """
    Run image through the CLIP image encoder and return normalized 512-dim vector.
    """
    try:
        processor, model, device = CLIPSingleton.load()
        logger.debug(f"Image encoder running on device: {device}")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
            embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

        result = embedding.squeeze().cpu().numpy()
        logger.debug(f"Image embedding shape: {result.shape}")
        return result

    except Exception as e:
        logger.error("❌ Failed to encode image: %s", e)
        raise
