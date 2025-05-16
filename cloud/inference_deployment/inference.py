# inference.py

import os
import io
import json
import logging

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def model_fn(*args, **kwargs):
    """
    Load the CLIP model & processor.
    The first positional arg is model_dir; any others are ignored.
    """
    # Extract model_dir from the first positional argument
    model_dir = args[0] if args else kwargs.get("model_dir")
    logger.info(f"Loading CLIP from {model_dir}, extra args={args[1:]}, kwargs={list(kwargs.keys())}")

    model     = CLIPModel.from_pretrained(model_dir)
    processor = CLIPProcessor.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    return {"model": model, "processor": processor, "device": device}


def input_fn(request_body, content_type):
    logger.info(f"Deserializing request with content type: {content_type}")
    if content_type in ("application/x-image", "image/jpeg", "image/png"):
        return Image.open(io.BytesIO(request_body)).convert("RGB")
    if content_type == "application/json":
        data = json.loads(request_body)
        if "inputs" not in data:
            raise ValueError("JSON payload must contain an 'inputs' field")
        return data["inputs"]
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, context):
    model     = context["model"]
    processor = context["processor"]
    device    = context["device"]

    if isinstance(input_data, Image.Image):
        logger.info("Running image through CLIP vision encoder")
        inputs = processor(images=input_data, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
    else:
        texts = input_data if isinstance(input_data, list) else [input_data]
        logger.info(f"Running {len(texts)} texts through CLIP text encoder")
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            embeddings = model.get_text_features(**inputs)

    embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
    return embeddings.cpu().numpy()


def output_fn(prediction: np.ndarray, response_content_type):
    logger.info(f"Serializing prediction as {response_content_type}")
    if response_content_type == "application/json":
        return json.dumps(prediction.tolist()), "application/json"
    raise ValueError(f"Unsupported response content type: {response_content_type}")
