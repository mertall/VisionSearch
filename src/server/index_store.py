import hnswlib
import numpy as np
import os
import threading
import logging

logger = logging.getLogger(__name__)

class HNSWIndexSingleton:
    _index = None
    _image_paths = []
    _ready = False
    _lock = threading.Lock()

    # File paths
    INDEX_PATH = "data/index/image_index.bin"
    META_PATH = "data/index/image_paths.txt"

    # Configurable HNSW index settings from env
    DIM = int(os.getenv("HNSW_DIM", 512))
    MAX_ELEMENTS = int(os.getenv("HNSW_MAX_ELEMENTS", 100_000))
    SPACE = os.getenv("HNSW_SPACE", "cosine")
    EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", 200))
    M = int(os.getenv("HNSW_M", 16))
    EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", 50))

    @classmethod
    def ensure_ready(cls):
        """Ensure the index is loaded and ready."""
        if not cls._ready:
            logger.warning("⚠️ Index not ready. Loading now...")
            cls.load()

    @classmethod
    def load(cls):
        """
        Load the HNSW index and metadata from disk.
        If not found, initialize an empty index.
        """
        with cls._lock:
            if cls._index is not None:
                cls._ready = True
                return

            cls._index = hnswlib.Index(space=cls.SPACE, dim=cls.DIM)

            if os.path.exists(cls.INDEX_PATH) and os.path.exists(cls.META_PATH):
                cls._index.load_index(cls.INDEX_PATH)
                with open(cls.META_PATH, "r") as f:
                    cls._image_paths = [line.strip() for line in f]
                cls._index.set_ef(cls.EF_SEARCH)
                logger.info(f"✅ Loaded HNSW index with {len(cls._image_paths)} items.")
            else:
                cls._index.init_index(
                    max_elements=cls.MAX_ELEMENTS,
                    ef_construction=cls.EF_CONSTRUCTION,
                    M=cls.M
                )
                logger.info(
                    f"🆕 Initialized new HNSW index with dim={cls.DIM}, ef_construction={cls.EF_CONSTRUCTION}, "
                    f"M={cls.M}, max_elements={cls.MAX_ELEMENTS}"
                )

            cls._ready = True

    @classmethod
    def is_ready(cls):
        return cls._ready

    @classmethod
    def query(cls, vector: np.ndarray, k: int=5):
        """
        Perform a KNN search.
        Returns a list of image paths and similarity scores.
        """
        cls.ensure_ready()

        labels, distances = cls._index.knn_query(vector, k=k)
        results = [cls._image_paths[i] for i in labels[0]]
        scores = [1 - d for d in distances[0]]  # Convert cosine distance to similarity
        logger.debug(f"🔍 Query returned {len(results)} results.")
        return results, scores

    @classmethod
    def add_items(cls, vectors: list[np.ndarray], paths: list[str]):
        """
        Add new image vectors to the index.
        After indexing, deletes image files and frees memory.
        """
        import gc

        with cls._lock:

            cls.ensure_ready()
            cls._ready = False
            start_id = len(cls._image_paths)
            flat_vectors = np.vstack(vectors).astype(np.float32)  # Ensures shape=(N, D)
            cls._index.add_items(np.array(flat_vectors), list(range(start_id, start_id + len(flat_vectors))))
            cls._image_paths.extend(paths)
            logger.info(f"➕ Added {len(paths)} items to index. Total: {len(cls._image_paths)}")

        # 🔥 Clean up memory and delete images from disk
        for p in paths:
            try:
                os.remove(p)
                logger.debug(f"🧹 Deleted image: {p}")
            except Exception as e:
                logger.warning(f"⚠️ Could not delete image {p}: {e}")

        # Clear embedding vectors and paths from memory
        del vectors
        del paths
        gc.collect()
        cls._ready = True

    @classmethod
    def save(cls):
        with cls._lock:
            cls.ensure_ready()
            
            # 🔧 Ensure the directory exists
            os.makedirs(os.path.dirname(cls.INDEX_PATH), exist_ok=True)

            cls._index.save_index(cls.INDEX_PATH)
            with open(cls.META_PATH, "w") as f:
                for path in cls._image_paths:
                    f.write(path + "\n")

            logger.info(f"💾 Saved index to '{cls.INDEX_PATH}' and metadata to '{cls.META_PATH}'")
