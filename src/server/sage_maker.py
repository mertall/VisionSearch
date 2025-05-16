import io
import ast 
import os
import numpy as np
from PIL import Image
import boto3
from transformers import CLIPProcessor
from sagemaker.session import Session
from sagemaker.predictor import Predictor
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import JSONSerializer, IdentitySerializer
import logging

logger = logging.getLogger(__name__)

# **WIP** CLIPSageMakerClient relies on us having a custom inference image, which needs to be built on AWS EC2 or Codebuild. **WIP**
# **WIP** CLIPSageMakerClient will allow for lower latency and easier scalability **WIP**

class CLIPSageMakerClient:
    """
    WIP Singleton class to deploy and use CLIP model via SageMaker using Hugging Face hub.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CLIPSageMakerClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Load config from env
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.endpoint_name = os.getenv("CLIP_ENDPOINT_NAME", "clip-multimodal-endpoint")
        self.model_id = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
        self.role = os.getenv("SAGEMAKER_ROLE_ARN")

        if not self.role:
            raise ValueError("SAGEMAKER_ROLE_ARN must be set as an environment variable")

        # Setup processor and session
        logger.info("🔁 Loading CLIPProcessor...")
        self._processor = CLIPProcessor.from_pretrained(self.model_id)

        boto_sess = boto3.Session(region_name=self.region)
        self.sm_session = Session(boto_session=boto_sess)

        self.json_predictor = Predictor(
            endpoint_name=self.endpoint_name,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
            sagemaker_session=self.sm_session
        )

        self.image_predictor = Predictor(
            endpoint_name=self.endpoint_name,
            serializer=IdentitySerializer(content_type="image/jpeg"),
            deserializer=JSONDeserializer(),
            sagemaker_session=self.sm_session
        )


        logger.info(f"✅ Connected to SageMaker endpoint: {self.endpoint_name}")
        self._initialized = True

    def normalize_embeedding(self, embedding) -> np.ndarray:
        logger.debug(f"[encode_image] 🔁 Raw model response type: {type(embedding)}")

        # ────── Step 1: If response is list with stringified array ──────
        if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], str):
            logger.debug("[encode_image] 📦 Response is list with a stringified array — parsing first item.")
            try:
                embedding = ast.literal_eval(embedding[0])
            except Exception as e:
                logger.error(f"[encode_image] ❌ Failed to parse stringified embedding: {e}")
                raise

        # ────── Step 2: If result is a single nested list ──────
        elif isinstance(embedding, str):
            logger.debug("[encode_image] 🧩 Response is raw string — parsing as literal.")
            try:
                embedding = ast.literal_eval(embedding)
            except Exception as e:
                logger.error(f"[encode_image] ❌ Failed to parse string: {e}")
                raise

        # ────── Step 3: Flatten nested list if needed ──────
        if isinstance(embedding, list) and isinstance(embedding[0], list):
            logger.debug("[encode_image] 🔃 Flattening nested list.")
            embedding = embedding[0]

        logger.info(f"[encode_image] 📐 Parsed embedding vector length: {len(embedding)}")

        # ────── Step 4: Convert to NumPy array ──────
        try:
            embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
        except Exception as e:
            logger.error(f"[encode_image] ❌ Failed to convert embedding to NumPy array: {e}")
            raise

        logger.info(f"[encode_image] ✅ Final embedding shape: {embedding.shape}")
        return embedding
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)

        logger.info("[encode_image] 📨 Sending image payload to endpoint...")
        result = self.image_predictor.predict(buf.read())

        embedding = self.normalize_embeedding(result)

        return embedding

    def encode_text(self, text: str) -> np.ndarray:
        payload = {"inputs": text}
        result = self.json_predictor.predict(payload)
        embedding = self.normalize_embeedding(result)

        return embedding

