import os
import time
import logging
from datasets import load_dataset
from server.redis_client import r
from PIL import Image
from io import BytesIO
from uuid import uuid4
from gc import collect as gc_collect

logger = logging.getLogger(__name__)

IMG_DIR = os.getenv("IMG_DIR", "/shared/images")
BATCH_SIZE = int(os.getenv("HF_ENQUEUE_BATCH", 1))  # safer default
QUEUE_LIMIT = int(os.getenv("HF_REDIS_QUEUE_LIMIT", 50))  # throttle if queue is overloaded

def enqueue_hf_dataset(repo: str, split: str = "train", image_column: str = "image"):
    """
    Stream Hugging Face dataset and enqueue image jobs to Redis in batches.
    Includes memory safety, throttling, and cleanup.
    """
    logger.info(f"📦 Streaming dataset: {repo}/{split}")
    ds = load_dataset(repo, split=split, streaming=True)

    os.makedirs(IMG_DIR, exist_ok=True)

    pending_batch = []

    for idx, record in enumerate(ds):
        if image_column not in record:
            logger.warning(f"⚠️ Missing '{image_column}' in record {idx}")
            continue

        try:
            image = record[image_column]
            if hasattr(image, "convert"):
                pil_image = image
            elif hasattr(image, "read"):
                pil_image = Image.open(BytesIO(image.read()))
            else:
                raise ValueError("Unsupported image format")

            path = os.path.join(
                IMG_DIR,
                f"hf_{repo.replace('/', '_')}_{idx}_{uuid4().hex[:6]}.jpg"
            )
            pil_image.save(path)
            pil_image.close()
            pending_batch.append(path)

        except Exception as e:
            logger.error(f"❌ Failed to process image {idx}: {e}")
        finally:
            if 'image' in locals():
                del image
            if 'pil_image' in locals():
                del pil_image
            gc_collect()

        if len(pending_batch) >= BATCH_SIZE:
            while r.llen("image_jobs") > QUEUE_LIMIT:
                logger.warning("⏳ Redis queue too large, sleeping to prevent OOM...")
                time.sleep(5)

            r.rpush("image_jobs", *pending_batch)
            logger.info(f"📤 Enqueued batch of {len(pending_batch)} images")
            pending_batch = []

    if pending_batch:
        r.rpush("image_jobs", *pending_batch)
        logger.info(f"📤 Enqueued final batch of {len(pending_batch)} images")

    logger.info("✅ All dataset images enqueued.")
