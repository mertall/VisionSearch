import os
import time
import logging
from scripts.embedding_generator import generate_image_embeddings
from server.index_store import HNSWIndexSingleton
from server.redis_client import r

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_redis_index_worker(batch_size: int = 8, persist_every: int = 50):
    HNSWIndexSingleton.load()
    logger.info("🔁 Index worker started...")

    batch_vecs, batch_paths = [], []

    while True:
        job = r.blpop("image_jobs", timeout=int(os.getenv("WORKER_TIMEOUT", 5)))
        if not job:
            logger.info("🕒 No new jobs. Sleeping...")
            time.sleep(2)
            continue

        _, path = job
        path = path.decode("utf-8")

        for vec, p in generate_image_embeddings([path]):
            batch_vecs.append(vec)
            batch_paths.append(p)

        if len(batch_vecs) >= batch_size:
            HNSWIndexSingleton.add_items(batch_vecs, batch_paths)
            logger.info(f"✅ Indexed batch of {len(batch_vecs)}")
            batch_vecs, batch_paths = [], []

        if len(HNSWIndexSingleton._image_paths) % persist_every == 0:
            HNSWIndexSingleton.save()

