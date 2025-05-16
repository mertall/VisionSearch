import os
import io
import ast
import logging
from uuid import uuid4
import numpy as np
from PIL import Image, UnidentifiedImageError
from datasets import load_dataset

from server.sage_maker import CLIPSageMakerClient
from server.index_store import HNSWIndexSingleton

logger = logging.getLogger(__name__)
IMG_DIR = os.getenv("IMG_DIR", "/shared/images")

def build_index(repo: str, split: str="train", image_column: str="image"):
    logger.info(f"📦 Starting simplified index build for {repo}/{split}")
    dataset = load_dataset(repo, split=split, streaming=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    clip_client = CLIPSageMakerClient()
    HNSWIndexSingleton.ensure_ready()

    success_count = 0
    fail_count = 0

    for idx, rec in enumerate(dataset):
        try:
            logger.info(f"[{idx}] 🔄 Processing record...")

            img_field = rec.get(image_column)
            if not img_field or not isinstance(img_field, dict) or "bytes" not in img_field:
                logger.warning(f"[{idx}] ⚠️ Invalid or missing 'bytes' field.")
                fail_count += 1
                continue

            img_data = img_field["bytes"]
            img = Image.open(io.BytesIO(img_data)).convert("RGB")

            fname = f"{repo.replace('/', '_')}_{idx}_{uuid4().hex[:6]}.jpg"
            path = os.path.join(IMG_DIR, fname)
            img.save(path, format="JPEG")
            img.close()
            logger.info(f"[{idx}] ✅ Saved image at {path}")

            # Reopen for prediction
            with Image.open(path) as reloaded_img:
                logger.debug(f"[{idx}] 🔍 Reopened image for encoding.")
                embedding = clip_client.encode_image(reloaded_img)

            logger.info(f"[{idx}] ✅ Embedding shape: {embedding.shape}")

            # Add to index
            HNSWIndexSingleton.add_items([embedding], [path])
            logger.info(f"[{idx}] 📌 Embedded and indexed.")
            success_count += 1

        except UnidentifiedImageError:
            logger.warning(f"[{idx}] ❌ Could not identify image, skipping.")
            fail_count += 1

        except Exception as e:
            import traceback
            logger.error(f"[{idx}] ❌ Failed to process image: {e}")
            logger.debug(traceback.format_exc())
            fail_count += 1

    HNSWIndexSingleton.save()
    print(f"🚀 Done. Indexed {success_count}, Failed {fail_count}.")