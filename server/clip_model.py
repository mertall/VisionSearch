import torch
import logging
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)

class CLIPSingleton:
    _model = None
    _processor = None
#visionsearch-worker  | 
# ERROR:server.triton_client:
# ❌ Failed to encode image: Unable to load weights from pytorch checkpoint file 
# for '/root/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/pytorch_model.bin' 
# at '/root/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/pytorch_model.bin'.
# If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True.
    @classmethod
    def load(cls, model_name="openai/clip-vit-base-patch32", device=None):
        if cls._model is None or cls._processor is None:
            logger.info("🔁 Loading CLIP model...")
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._model = CLIPModel.from_pretrained(model_name).to(device)
            cls._processor = CLIPProcessor.from_pretrained(model_name)
            cls._device = device
            logger.info("✅ CLIP model loaded.")
        return cls._processor, cls._model, cls._device
