import io
import ast 
import json
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
        self.region = os.getenv("AWS_REGION")
        self.endpoint_name = os.getenv("CLIP_ENDPOINT_NAME")
        self.role = os.getenv("SAGEMAKER_ROLE_ARN")

        if not self.role:
            raise ValueError("SAGEMAKER_ROLE_ARN must be set as an environment variable")

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

    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)

        data, _ =  self.image_predictor.predict(buf.read())
        decoded_data = json.loads(data)
        decoded_data = decoded_data[0]
        embedding = np.array(decoded_data, dtype=np.float32).reshape(1, -1)

        return embedding

    def encode_text(self, text: str) -> np.ndarray:
        payload = {"inputs": text}

        data, _ = self.json_predictor.predict(payload)

        decoded_data = json.loads(data)
        decoded_data = decoded_data[0]
        embedding = np.array(decoded_data, dtype=np.float32).reshape(1, -1)


        return embedding

