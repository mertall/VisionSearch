from pydantic import BaseModel
from typing import List

class TextEmbeddingRequest(BaseModel):
    text: str

class ImageEmbeddingRequest(BaseModel):
    image_bytes: bytes  # for base64-encoded input if needed

class EmbeddingResponse(BaseModel):
    embedding: List[float]
